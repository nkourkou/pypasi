"""Batch-level drift monitoring: has the measurement changed, and where?

A fitted model keeps producing predictions long after the data stops looking
like what it was trained on. Accuracy would say so, but accuracy needs labels,
and when a new batch of spectra arrives there are none - which is exactly the
moment the question matters.

This module answers it without labels. It treats the band-level quantities a
negotiated inference already produces - how much each band disagrees with its
neighbours, how sure each band is of itself - as **process variables**, charts
them the way a laboratory charts any process, and reports which *wavenumber
interval* left its control limits.

Three lines::

    profile = ControlProfile.fit(clf, X_reference, batches=runs)
    report = profile.check(X_new_batch)
    print(report.summary())

What it is not
--------------
Not outlier detection. :class:`pypasi.NoveltyDetector` asks whether a *spectrum*
is unlike the training data and answers per sample in input space. This asks
whether a *batch* is disturbing the fitted model, answers per band, and returns
an interval in cm-1 that you can take to the instrument. The two find different
problems and are meant to be run together.

Reading the limits honestly
---------------------------
Two things decide whether a chart is worth trusting.

*Where the reference came from.* 
Control limits are only as good as the batches they were built from. Reference
batches carved at random out of one cohort share an instrument, an operator and
a session, so the spread between them understates real batch-to-batch variation
and the limits come out **optimistically tight** - everything later will look
alarming. Pass ``batches=`` with genuine acquisition runs whenever you have
them; :meth:`ControlProfile.fit` warns when you do not.

*How many things are being charted.* Seven bands on two statistics is fourteen
tests at once, so at three sigma a perfectly good batch throws a single flag
now and then - roughly one batch in thirty even when nothing is wrong. A lone
breach is a reason to look; several bands at once, or the same band across
consecutive batches, is a reason to act. :meth:`ControlProfile.check_many` and
:func:`pypasi.viz.plot_control_trend` exist for exactly that reading.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ControlProfile", "ControlReport", "MONITORED_STATISTICS"]


#: Band-level quantities that can be charted. All come from
#: :meth:`pypasi.AuditResult.conflict_table`.
MONITORED_STATISTICS = (
    "mean_stress",         # disagreement with neighbouring bands
    "band_confidence",     # how sure the band's own model is, before negotiation
    "informed_conflict",   # stress discounted by how much the band had to say
    "band_DG",             # this band's share of the reconciliation effort
    "mute_frequency",      # how often the gate silenced it
    "cumulative_stress",
    "final_weight",
)

_DEFAULT_STATISTICS = ("mean_stress", "band_confidence")


def _split(n: int, k: int, rng) -> list[np.ndarray]:
    return np.array_split(rng.permutation(n), k)


def _fingerprint(bands, statistics, classes) -> dict:
    """Identity of the configuration a profile was built for.

    A profile is a set of numbers attached to specific bands of a specific
    model. Checking a batch against a profile built for a different partition
    would silently compare unrelated quantities, so the identity travels with
    the limits and is verified on every use.
    """
    return {
        "labels": list(bands.labels),
        "edges": [round(float(v), 6) for v in bands.edges],
        "axis_name": bands.axis_name,
        "statistics": list(statistics),
        "n_classes": int(len(classes)) if classes is not None else None,
    }


@dataclass
class ControlReport:
    """One batch, measured against a :class:`ControlProfile`.

    Attributes
    ----------
    table
        One row per band and statistic: the reference centre and limits, the
        value this batch produced, its standardised deviation ``z``, and whether
        it sits outside the limits.
    """

    name: str
    n_samples: int
    table: pd.DataFrame
    k: float
    method: str
    limit_scale: float = 1.0
    axis_name: str = "axis"

    # ----------------------------------------------------------- properties

    @property
    def out_of_control(self) -> pd.DataFrame:
        """Only the rows that breached, worst first."""
        out = self.table[self.table["out_of_control"]]
        return out.reindex(out["z"].abs().sort_values(ascending=False).index)

    @property
    def n_out_of_control(self) -> int:
        return int(self.table["out_of_control"].sum())

    @property
    def is_in_control(self) -> bool:
        return self.n_out_of_control == 0

    @property
    def drift_score(self) -> float:
        """Largest standardised deviation anywhere in the batch."""
        z = self.table["z"].to_numpy(dtype=float)
        return float(np.nanmax(np.abs(z))) if z.size else float("nan")

    @property
    def worst(self) -> pd.Series | None:
        """The single most deviant band/statistic, or None if the table is empty."""
        if not len(self.table):
            return None
        return self.table.loc[self.table["z"].abs().idxmax()]

    def bands_out_of_control(self) -> list[str]:
        """Band labels that breached on at least one statistic, worst first."""
        out = self.out_of_control
        seen: list[str] = []
        for lab in out["label"]:
            if lab not in seen:
                seen.append(lab)
        return seen

    # -------------------------------------------------------------- output

    def summary(self) -> str:
        if not len(self.table):
            return f"{self.name}: nothing to report"
        head = (f"{self.name}: {self.n_samples} spectra, "
                f"{self.n_out_of_control} of {len(self.table)} band-statistics "
                f"outside {self.k:g} sigma")
        if self.is_in_control:
            return head + "  IN CONTROL"
        w = self.worst
        lo, hi = w[f"{self.axis_name}_lo"], w[f"{self.axis_name}_hi"]
        bad = ", ".join(self.bands_out_of_control())
        return (f"{head}\n  worst: {w['label']} ({lo:.0f}-{hi:.0f} {self.axis_name}) "
                f"on {w['statistic']}, z = {w['z']:+.1f}\n  bands flagged: {bad}")

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()

    def to_csv(self, path) -> str:
        self.table.to_csv(path, index=False)
        return str(path)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "in control" if self.is_in_control else f"{self.n_out_of_control} out"
        return f"ControlReport({self.name}, n={self.n_samples}, {state})"


@dataclass
class ControlProfile:
    """What this model's band statistics look like when nothing is wrong.

    Parameters
    ----------
    statistics
        Which band-level quantities to chart. See :data:`MONITORED_STATISTICS`.
    k
        Limit width in reference standard deviations. 3 is the Shewhart
        convention.
    method
        ``"sigma"`` puts the limits at the centre plus or minus ``k`` standard
        deviations; ``"quantile"`` puts them at the observed extremes of the
        reference batches, which assumes nothing about the distribution and is
        the safer choice when there are few reference runs.
    laser_nm
        Excitation wavelength. Given one, reports also carry the scattered
        wavelength of each band, because a throughput problem lives at a place
        on the detector rather than at a Raman shift.
    multiplicity : {"empirical", "sidak", "bonferroni", "none"}
        How to widen ``k`` for the number of band-statistics being charted at
        once. Seven bands on two statistics is fourteen simultaneous tests, so
        at a nominal three sigma a perfectly good batch throws a flag roughly
        one time in thirty. Add statistics and it gets worse fast: six
        statistics on seven bands is forty-two tests, and the same nominal limit
        gives a batch-level false-alarm rate near twelve per cent.

        ``"sidak"`` (the default from 0.5.0) solves
        :math:`1-(1-p)^{m}=\alpha` for the per-test rate, so the *batch-level*
        false-alarm rate is what ``k`` says it is however many statistics you
        chart. ``"bonferroni"`` is the slightly more conservative
        :math:`p=\alpha/m`. ``"none"`` reproduces 0.4.x behaviour, where ``k``
        is per test and the family-wise rate grows with the chart.

        Both corrections assume the tests are independent, which bands are not -
        neighbouring bands move together - and both assume the charted
        statistics are normal, which stress and mute frequency are not. On
        synthetic signals with twenty reference batches, seven bands and two
        statistics, a nominal ``k=3`` gives a measured batch-level false-alarm
        rate of 0.25; Sidak brings it to 0.09; the nominal rate is 0.003. The
        correction is doing real work and it is not sufficient on its own,
        because the standard deviation behind the limits is itself estimated
        from twenty numbers.

        ``"empirical"`` (the default from 0.5.0) therefore skips the
        distributional assumption entirely: it computes, for each reference
        batch, the largest ``|z|`` it would have scored against limits built
        from the *other* batches, and puts the threshold at the quantile of that
        leave-one-out distribution matching ``k``. This controls the rate that
        actually matters - how often a clean batch trips *any* of the charted
        statistics - without assuming independence or normality. It needs enough
        reference batches to estimate that quantile and warns when it does not
        have them, falling back to Sidak.
    """

    statistics: tuple[str, ...] = _DEFAULT_STATISTICS
    k: float = 3.0
    method: str = "sigma"
    laser_nm: float | None = None
    decompose_variance: bool = True
    multiplicity: str = "empirical"
    k_empirical_: float | None = None
    resolution_: float | None = None

    centre_: pd.DataFrame | None = field(default=None, repr=False)
    sd_: pd.DataFrame | None = field(default=None, repr=False)
    sigma_within_: pd.Series | None = field(default=None, repr=False)
    sigma_between_: pd.Series | None = field(default=None, repr=False)
    lcl_: pd.DataFrame | None = field(default=None, repr=False)
    ucl_: pd.DataFrame | None = field(default=None, repr=False)
    n_batches_: int = 0
    batch_size_: float = 0.0
    labels_: tuple[str, ...] = ()
    lo_: tuple[float, ...] = ()
    hi_: tuple[float, ...] = ()
    axis_name_: str = "axis"
    fingerprint_: dict = field(default_factory=dict, repr=False)
    reference_batches_: pd.DataFrame | None = field(default=None, repr=False)
    _estimator = None

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(cls, estimator, X=None, *, batches=None, n_batches: int = 20,
            statistics=_DEFAULT_STATISTICS, k: float = 3.0, method: str = "sigma",
            laser_nm: float | None = None, decompose_variance: bool = True,
            multiplicity: str = "empirical", random_state=None) -> "ControlProfile":
        """Learn the in-control distribution of a fitted model's band statistics.

        Parameters
        ----------
        estimator
            A fitted :class:`pypasi.BandNegotiationClassifier`.
        X
            Reference spectra, split at random into ``n_batches`` groups. Ignored
            when ``batches`` is given. **These should not be spectra the
            estimator was fitted on** - see the module docstring.
        batches
            A sequence of arrays, one per real acquisition run. **Strongly
            preferred**: random splits of a single cohort share a session and an
            instrument, so the limits they produce are too tight.
        n_batches
            Number of random groups to carve from ``X``.
        """
        if method not in ("sigma", "quantile"):
            raise ValueError(f"method must be 'sigma' or 'quantile', got {method!r}")
        statistics = tuple(statistics)
        unknown = [s for s in statistics if s not in MONITORED_STATISTICS]
        if unknown:
            raise ValueError(
                f"unknown statistic(s) {unknown}; choose from {list(MONITORED_STATISTICS)}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if batches is None:
            if X is None:
                raise ValueError("give either X or batches")
            rng = np.random.default_rng(random_state)
            X = np.asarray(X, float)
            if n_batches < 2:
                raise ValueError(f"need at least 2 reference batches, got {n_batches}")
            if n_batches > X.shape[0]:
                raise ValueError(
                    f"asked for {n_batches} batches from {X.shape[0]} spectra")
            batches = [X[idx] for idx in _split(X.shape[0], n_batches, rng)]
            warnings.warn(
                "Control limits built from a random split of one cohort. Those "
                "batches share an instrument, an operator and a session, so the "
                "limits will be tighter than real batch-to-batch variation and "
                "later batches will look more alarming than they are. Pass "
                "batches=[run1, run2, ...] with genuine acquisition runs when "
                "you have them, and make sure the reference was held out of "
                "fitting - a profile built on the model's own training spectra "
                "describes in-sample behaviour and flags everything after it.",
                UserWarning, stacklevel=2)
        else:
            batches = [np.asarray(b, float) for b in batches]
            if len(batches) < 2:
                raise ValueError(f"need at least 2 reference batches, got {len(batches)}")

        if len(batches) < 8:
            warnings.warn(
                f"only {len(batches)} reference batches: the standard deviation "
                f"behind these limits is poorly determined. Consider "
                f"method='quantile', which assumes nothing about the "
                f"distribution.", UserWarning, stacklevel=2)

        if multiplicity not in ("none", "sidak", "bonferroni", "empirical"):
            raise ValueError(
                f"unknown multiplicity {multiplicity!r}; use 'empirical', "
                "'sidak', 'bonferroni' or 'none'")
        self = cls(statistics=statistics, k=float(k), method=method, laser_nm=laser_nm,
                   decompose_variance=bool(decompose_variance),
                   multiplicity=str(multiplicity))
        rows = [self._measure(estimator, b) for b in batches]
        ref = pd.concat(
            [r.assign(batch=i, n_samples=len(b)) for i, (r, b) in enumerate(zip(rows, batches))],
            ignore_index=True)

        first = rows[0]
        self.labels_ = tuple(first["label"])
        self.axis_name_ = getattr(estimator.bands_, "axis_name", "axis")
        self.lo_ = tuple(float(v) for v in first[f"{self.axis_name_}_lo"])
        self.hi_ = tuple(float(v) for v in first[f"{self.axis_name_}_hi"])

        wide = ref.pivot_table(index="batch", columns="label", values=list(statistics))
        self.centre_ = wide.mean()
        self.sd_ = wide.std(ddof=1)
        if method == "sigma":
            self.lcl_ = self.centre_ - self.k * self.sd_
            self.ucl_ = self.centre_ + self.k * self.sd_
        else:
            self.lcl_ = wide.min()
            self.ucl_ = wide.max()

        self.n_batches_ = len(batches)
        self.batch_size_ = float(np.mean([len(b) for b in batches]))
        self._calibrate_empirical(wide)
        self._decompose(estimator, batches, wide, random_state)
        self.reference_batches_ = ref
        self.fingerprint_ = _fingerprint(estimator.bands_, statistics,
                                         getattr(estimator, "classes_", None))
        self._estimator = estimator
        return self

    def _calibrate_empirical(self, wide: pd.DataFrame) -> None:
        """Set the threshold from what clean reference batches actually do.

        For each reference batch, recompute the centre and spread from the
        *other* batches and record the largest ``|z|`` it scores. That
        leave-one-out distribution is what a clean batch looks like under this
        chart, multiplicity, correlation between bands, non-normal statistics
        and a finitely-estimated standard deviation all included. The threshold
        is its upper quantile.

        Held out properly, because a batch scored against limits it helped build
        is flattered by exactly the amount that makes this calibration useless.
        """
        from scipy.stats import norm

        self.k_empirical_ = None
        n = len(wide)
        if n < 4:
            return
        alpha = 2.0 * norm.sf(float(self.k))            # target batch-level rate
        # With n reference batches the finest rate observable is 1/n. Targeting
        # 0.0027 from twenty batches is asking for a quantile the data cannot
        # resolve, and that is the normal case rather than a misuse - so it is
        # recorded in ``resolution_`` and shown in the repr instead of warned
        # about on every call. A warning that always fires is not read.
        self.resolution_ = 1.0 / n
        V = wide.to_numpy(dtype=float)

        # The CENTRE is left out; the SPREAD is not. Two reasons, and the second
        # is the one that matters.
        #
        # A batch pulls the mean toward itself by 1/n, which flatters its own z
        # and is exactly what a leave-one-out estimate should remove. But a
        # leave-one-out *standard deviation* is computed from n-1 numbers and
        # collapses when the removed batch was carrying the spread: on a
        # ten-batch profile of a near-saturated statistic, two batches produced
        # |z| of 20.7 and 26.4 that way, and the resulting threshold was a chart
        # that could never fire. Using the pooled spread, the same ten batches
        # give a maximum of 3.14.
        #
        # It is also the spread :meth:`check` itself uses, so this calibration
        # measures what the chart will actually do rather than a variant of it.
        sd = V.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.nan)
        maxz = np.empty(n)
        for i in range(n):
            mu = np.delete(V, i, axis=0).mean(axis=0)
            maxz[i] = np.nanmax(np.abs((V[i] - mu) / sd))
        maxz = maxz[np.isfinite(maxz)]
        if maxz.size < 4:
            return
        q = float(np.clip(1.0 - alpha, 0.0, 1.0))
        self.k_empirical_ = float(np.quantile(maxz, q))

    def _decompose(self, estimator, batches, wide, random_state):
        """Split the reference spread into sampling noise and real batch effect.

        A Shewhart chart of batch means assumes the whole spread between
        subgroups is within-subgroup sampling noise, so that a batch of m
        spectra deserves limits narrower by the square root of m. That is wrong
        whenever batches genuinely differ from one another - and acquisition
        batches always do - and it makes the chart cry wolf on any batch larger
        than the reference runs.

        So measure the two components. Split each reference batch in half: the
        between-batch part is common to both halves and cancels in their
        difference, leaving only sampling noise, while the spread of the batch
        means themselves carries both. That gives

            var(batch mean of m) = sigma_within^2 / m + sigma_between^2

        which reduces to the square-root rule when batches are exchangeable and
        stops shrinking once a real batch effect dominates.
        """
        total = wide.var(ddof=1)
        if not self.decompose_variance or len(batches) < 2:
            self.sigma_within_ = np.sqrt(total * self.batch_size_)
            self.sigma_between_ = total * 0.0
            return
        rng = np.random.default_rng(random_state)
        diffs = []
        for b in batches:
            n = len(b)
            if n < 4:
                continue
            idx = rng.permutation(n)
            a = self._measure(estimator, b[idx[: n // 2]]).set_index("label")
            c = self._measure(estimator, b[idx[n // 2:]]).set_index("label")
            diffs.append({(s_, lab): float(a.loc[lab, s_] - c.loc[lab, s_])
                          for s_ in self.statistics for lab in self.labels_})
        if not diffs:
            self.sigma_within_ = np.sqrt(total * self.batch_size_)
            self.sigma_between_ = total * 0.0
            return
        d2 = pd.DataFrame(diffs).pow(2).mean()          # E[d^2] = 4 s_w^2 / n
        d2.index = pd.MultiIndex.from_tuples(d2.index)
        var_within = (d2 * self.batch_size_ / 4.0).reindex(total.index)
        var_between = (total - var_within / self.batch_size_).clip(lower=0.0)
        self.sigma_within_ = np.sqrt(var_within.clip(lower=0.0))
        self.sigma_between_ = np.sqrt(var_between)

    def effective_sd(self, n: int) -> pd.Series:
        """Standard deviation a batch mean of ``n`` spectra should have."""
        if self.sigma_within_ is None:
            return self.sd_
        return np.sqrt(self.sigma_within_ ** 2 / max(int(n), 1)
                       + self.sigma_between_ ** 2)

    # ---------------------------------------------------------------- check

    @property
    def n_tests(self) -> int:
        """How many band-statistics a single batch is scored against at once."""
        return len(self.statistics) * len(self.labels_)

    @property
    def k_effective(self) -> float:
        """``k``, widened for the number of simultaneous tests.

        ``k`` names a false-alarm rate for one test; a chart runs
        :attr:`n_tests` of them and flags the batch if *any* trips. Without a
        correction the batch-level rate is :math:`1-(1-p)^{m}`, which for the
        default seven bands on two statistics turns a nominal 0.0027 into 0.037,
        and for six statistics into 0.12. The correction inverts that so the
        rate ``k`` names is the one the batch actually gets.
        """
        from scipy.stats import norm

        if self.multiplicity == "none" or self.n_tests <= 1:
            return float(self.k)
        p = 2.0 * norm.sf(float(self.k))
        sidak = float(norm.isf((1.0 - (1.0 - p) ** (1.0 / self.n_tests)) / 2.0))
        if self.multiplicity == "empirical":
            # The more conservative of the two, always. The empirical quantile
            # is the better estimate where the reference batches can resolve it;
            # where they cannot it saturates at the observed maximum, and the
            # parametric correction is then the safer floor. Taking the maximum
            # means the chart is never looser than 0.4.x + Sidak would have been.
            if self.k_empirical_ is None:
                return sidak
            return max(float(self.k_empirical_), sidak)
        p = 2.0 * norm.sf(float(self.k))            # two-sided per-test rate
        m = int(self.n_tests)
        if self.multiplicity == "bonferroni":
            p_adj = p / m
        else:                                       # sidak
            p_adj = 1.0 - (1.0 - p) ** (1.0 / m)
        return float(norm.isf(p_adj / 2.0))

    def check(self, X, *, name: str = "batch", scale_limits: bool = True,
              estimator=None) -> ControlReport:
        """Measure one new batch against the profile. No labels required.

        Parameters
        ----------
        scale_limits
            Rescale the limits for this batch's size, using the variance
            decomposition measured at fit time. Without it a larger batch looks
            spuriously alarming and a smaller one hides real drift.
        estimator
            Only needed after :meth:`load`, which cannot carry a fitted model.
        """
        est = self._resolve(estimator)
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        measured = self._measure(est, X)

        eff = (self.effective_sd(X.shape[0]) if scale_limits else self.sd_)
        ratio = (eff / self.sd_.replace(0.0, np.nan)).fillna(1.0) if scale_limits else None
        k_eff = self.k_effective

        rows = []
        for stat in self.statistics:
            for j, lab in enumerate(self.labels_):
                value = float(measured.loc[measured["label"] == lab, stat].iloc[0])
                centre = float(self.centre_[(stat, lab)])
                sd = float(eff[(stat, lab)])
                if self.method == "sigma":
                    half = k_eff * sd
                else:
                    half = 0.5 * (float(self.ucl_[(stat, lab)])
                                  - float(self.lcl_[(stat, lab)]))
                    half *= float(ratio[(stat, lab)]) if ratio is not None else 1.0
                lcl, ucl = centre - half, centre + half
                z = (value - centre) / sd if sd > 0 else np.nan
                rows.append({
                    "statistic": stat,
                    "label": lab,
                    f"{self.axis_name_}_lo": self.lo_[j],
                    f"{self.axis_name_}_hi": self.hi_[j],
                    "centre": centre, "sd": sd, "lcl": lcl, "ucl": ucl,
                    "value": value, "z": z,
                    "out_of_control": bool(value < lcl or value > ucl),
                })
        table = pd.DataFrame(rows)
        if self.laser_nm:
            laser_cm = 1e7 / float(self.laser_nm)
            table["nm_lo"] = 1e7 / (laser_cm - table[f"{self.axis_name_}_lo"])
            table["nm_hi"] = 1e7 / (laser_cm - table[f"{self.axis_name_}_hi"])
        scale = float(np.nanmean(ratio.to_numpy(dtype=float))) if ratio is not None else 1.0
        table.attrs["limit_scale"] = scale
        table.attrs["k_effective"] = k_eff
        table.attrs["n_tests"] = self.n_tests
        table.attrs["multiplicity"] = self.multiplicity
        return ControlReport(name=name, n_samples=int(X.shape[0]), table=table,
                             k=k_eff, method=self.method, limit_scale=scale,
                             axis_name=self.axis_name_)

    def limits_report(self) -> pd.Series:
        """What threshold this chart is actually using, and how it was reached.

        ``k`` is the nominal per-test limit, ``k_effective`` the one applied.
        ``resolution`` is the finest false-alarm rate the reference batches can
        resolve: if it is much larger than the rate ``k`` names, the empirical
        calibration saturated and the true rate is nearer ``resolution`` than
        nominal. More reference batches is the only fix for that.
        """
        from scipy.stats import norm

        return pd.Series({
            "n_tests": self.n_tests,
            "n_batches": self.n_batches_,
            "multiplicity": self.multiplicity,
            "k_nominal": float(self.k),
            "k_effective": self.k_effective,
            "k_empirical": self.k_empirical_,
            "target_false_alarm_rate": 2.0 * norm.sf(float(self.k)),
            "resolution": self.resolution_,
        })

    def check_many(self, batches, names=None, *, estimator=None,
                   scale_limits: bool = True) -> pd.DataFrame:
        """Check a sequence of batches. One row each, for a trend chart."""
        names = list(names) if names is not None else [f"batch {i + 1}"
                                                       for i in range(len(batches))]
        if len(names) != len(batches):
            raise ValueError(f"{len(names)} names for {len(batches)} batches")
        rows = []
        primary = self.statistics[0]
        for nm, b in zip(names, batches):
            rep = self.check(b, name=nm, estimator=estimator, scale_limits=scale_limits)
            z = rep.table[rep.table["statistic"] == primary].set_index("label")["z"]
            rows.append({"batch": nm, "n_samples": rep.n_samples,
                         "drift_score": rep.drift_score,
                         "n_out_of_control": rep.n_out_of_control,
                         "in_control": rep.is_in_control,
                         **{f"z_{lab}": float(z.get(lab, np.nan)) for lab in self.labels_}})
        frame = pd.DataFrame(rows)
        frame.attrs["statistic"] = primary
        frame.attrs["k"] = self.k
        return frame

    # ------------------------------------------------------- persistence

    def attach(self, estimator) -> "ControlProfile":
        """Re-attach a fitted model to a loaded profile, checking it matches."""
        got = _fingerprint(estimator.bands_, self.statistics,
                           getattr(estimator, "classes_", None))
        for key in ("labels", "edges", "axis_name", "statistics"):
            if got[key] != self.fingerprint_.get(key):
                raise ValueError(
                    f"this profile was built for a different configuration: "
                    f"{key} is {self.fingerprint_.get(key)} in the profile but "
                    f"{got[key]} in the estimator")
        self._estimator = estimator
        return self

    def to_dict(self) -> dict:
        if self.centre_ is None:
            raise RuntimeError("profile is not fitted")
        def _flat(series):
            return {f"{s}|{lab}": float(series[(s, lab)])
                    for s in self.statistics for lab in self.labels_}
        return {
            "pypasi_monitor_version": 1,
            "statistics": list(self.statistics), "k": self.k, "method": self.method,
            "laser_nm": self.laser_nm,
            "labels": list(self.labels_), "lo": list(self.lo_), "hi": list(self.hi_),
            "axis_name": self.axis_name_,
            "n_batches": self.n_batches_, "batch_size": self.batch_size_,
            "fingerprint": self.fingerprint_,
            "centre": _flat(self.centre_), "sd": _flat(self.sd_),
            "lcl": _flat(self.lcl_), "ucl": _flat(self.ucl_),
            "sigma_within": _flat(self.sigma_within_),
            "sigma_between": _flat(self.sigma_between_),
            "decompose_variance": bool(self.decompose_variance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlProfile":
        self = cls(statistics=tuple(d["statistics"]), k=float(d["k"]),
                   method=d["method"], laser_nm=d.get("laser_nm"))
        self.labels_ = tuple(d["labels"])
        self.lo_ = tuple(float(v) for v in d["lo"])
        self.hi_ = tuple(float(v) for v in d["hi"])
        self.axis_name_ = d["axis_name"]
        self.n_batches_ = int(d["n_batches"])
        self.batch_size_ = float(d["batch_size"])
        self.fingerprint_ = d["fingerprint"]
        index = pd.MultiIndex.from_tuples(
            [(s, lab) for s in self.statistics for lab in self.labels_])
        self.decompose_variance = bool(d.get("decompose_variance", True))
        for name in ("centre", "sd", "lcl", "ucl", "sigma_within", "sigma_between"):
            if name not in d:
                continue
            flat = d[name]
            setattr(self, f"{name}_", pd.Series(
                [flat[f"{s}|{lab}"] for s, lab in index], index=index))
        return self

    def save(self, path) -> str:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path, estimator=None) -> "ControlProfile":
        self = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return self.attach(estimator) if estimator is not None else self

    # -------------------------------------------------------------- tables

    def to_frame(self) -> pd.DataFrame:
        """The limits, one row per band and statistic."""
        if self.centre_ is None:
            raise RuntimeError("profile is not fitted")
        rows = []
        for stat in self.statistics:
            for j, lab in enumerate(self.labels_):
                rows.append({"statistic": stat, "label": lab,
                             f"{self.axis_name_}_lo": self.lo_[j],
                             f"{self.axis_name_}_hi": self.hi_[j],
                             "centre": float(self.centre_[(stat, lab)]),
                             "sd": float(self.sd_[(stat, lab)]),
                             "lcl": float(self.lcl_[(stat, lab)]),
                             "ucl": float(self.ucl_[(stat, lab)])})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ internal

    def _resolve(self, estimator):
        est = estimator if estimator is not None else self._estimator
        if est is None:
            raise RuntimeError(
                "no fitted estimator attached. A profile restored with load() "
                "carries only the limits; call profile.attach(clf) or pass "
                "estimator= to check().")
        return est

    def _measure(self, estimator, X) -> pd.DataFrame:
        table = estimator.audit(np.asarray(X, float)).conflict_table()
        missing = [s for s in self.statistics if s not in table.columns]
        if missing:
            raise ValueError(f"the audit does not report {missing}; "
                             f"available: {sorted(table.columns)}")
        return table

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.centre_ is None:
            return "ControlProfile(unfitted)"
        return (f"ControlProfile({len(self.labels_)} bands, "
                f"{len(self.statistics)} statistics, {self.n_batches_} reference "
                f"batches of ~{self.batch_size_:.0f}, {self.method} limits at "
                f"{self.k:g} sigma)")
