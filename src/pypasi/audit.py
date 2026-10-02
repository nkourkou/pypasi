"""Auditing: The negotiation records which bands disagreed, how long they held out and how
far they travelled. :class:`AuditResult` turns that into tables, and - the part
that closes the loop - hands back the raw signal region behind any band the
tables implicate, so a conflict score can be checked against the spectrum it
came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .bands import Band, BandSet
from .negotiate import NegotiationResult

__all__ = ["AuditResult", "BandTrace", "compare_regimes", "band_importance"]


def band_importance(
    bands: BandSet,
    X: np.ndarray,
    y,
    estimator=None,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Per-band discriminative importance from a model fitted on the whole signal.

    This answers a different question from the conflict table, and the pair is
    more informative than either alone. Importance says *where a classifier
    finds information*; conflict says *where inference runs into trouble*. A
    band that scores high on both is informative but unstable - the first place
    to look when predictions fail. A band high on importance and low on conflict
    is a dependable discriminant.

    Parameters
    ----------
    bands
        The band partition to aggregate over.
    X, y
        Signals and labels. The model is fitted on the full feature set, not
        band by band, so that importances are comparable across bands.
    estimator
        Any fitted-or-unfitted estimator exposing ``feature_importances_`` or
        ``coef_`` after fitting. Defaults to
        :class:`~pypasi.chemometrics.PLSDA`, whose VIP scores are the
        conventional choice for vibrational spectra.
    normalize
        Scale importances so the maximum feature importance is 1.

    Returns
    -------
    pandas.DataFrame
        One row per band with ``importance_mean``, ``importance_max`` and
        ``importance_sum``.
    """
    from sklearn.base import clone

    X = np.asarray(X, dtype=float)
    if estimator is None:
        from .chemometrics import PLSDA

        estimator = PLSDA(n_components=10)
    model = clone(estimator)
    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.atleast_2d(model.coef_)).max(axis=0)
    else:
        raise TypeError(
            f"{type(model).__name__} exposes neither feature_importances_ nor "
            "coef_, so per-feature importance cannot be derived"
        )
    if imp.size != X.shape[1]:
        raise ValueError(
            f"importance has {imp.size} entries but X has {X.shape[1]} features"
        )
    if normalize and imp.max() > 0:
        imp = imp / imp.max()

    return pd.DataFrame(
        {
            "band": [b.index for b in bands],
            "label": bands.labels,
            f"{bands.axis_name}_lo": [b.lo for b in bands],
            f"{bands.axis_name}_hi": [b.hi for b in bands],
            "importance_mean": [float(imp[b.features].mean()) for b in bands],
            "importance_max": [float(imp[b.features].max()) for b in bands],
            "importance_sum": [float(imp[b.features].sum()) for b in bands],
        }
    ), imp


@dataclass
class BandTrace:
    """The raw signal behind one band, next to that band's conflict scores.

    Attributes
    ----------
    band
        The band itself, carrying its axis interval and feature indices.
    axis
        Axis values covered by the band.
    X
        Raw signal block, shape ``(n_samples, n_features_in_band)``.
    class_means
        Mean signal per class within the band, shape ``(n_classes, n_features)``.
    conflict
        That band's row of the conflict table.
    """

    band: Band
    axis: np.ndarray
    X: np.ndarray
    class_means: np.ndarray | None
    class_labels: np.ndarray | None
    conflict: pd.Series

    @property
    def mean_signal(self) -> np.ndarray:
        return self.X.mean(axis=0)

    def to_frame(self) -> pd.DataFrame:
        """Long-format table of the raw band, ready for plotting or export."""
        out = {"axis": self.axis, "mean": self.mean_signal, "sd": self.X.std(axis=0)}
        if self.class_means is not None:
            for lab, row in zip(self.class_labels, self.class_means):
                out[f"mean_{lab}"] = row
        return pd.DataFrame(out)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BandTrace({self.band.label}, {self.band.lo:.4g}-{self.band.hi:.4g}, "
            f"{self.X.shape[0]} signals x {self.X.shape[1]} points)"
        )


class AuditResult:
    """Tables and traces derived from one negotiation.

    Parameters
    ----------
    result
        The :class:`~pypasi.negotiate.NegotiationResult` to audit.
    bands
        The band partition used, which supplies the mapping back to raw features.
    X
        The signals that were classified. Optional, but required for
        :meth:`trace_band`.
    y_true
        Ground-truth labels. Optional; without them the conflict table reports
        absolute conflict but cannot contrast correct against incorrect calls.
    classes
        Class labels in the order the model uses internally.
    """

    def __init__(
        self,
        result: NegotiationResult,
        bands: BandSet,
        X: np.ndarray | None = None,
        y_true=None,
        classes=None,
    ):
        self.result = result
        self.bands = bands
        self.X = None if X is None else np.asarray(X, dtype=float)
        self.classes = None if classes is None else np.asarray(classes)
        if y_true is None:
            self.y_true = None
            self.y_pred_labels = None
            self.correct = None
        else:
            self.y_true = np.asarray(y_true)
            if self.classes is not None:
                self.y_pred_labels = self.classes[result.y_pred]
            else:
                self.y_pred_labels = result.y_pred
            self.correct = (self.y_pred_labels == self.y_true).astype(int)

    # ------------------------------------------------------------ properties

    @property
    def n_samples(self) -> int:
        return self.result.n_samples

    @property
    def accuracy(self) -> float | None:
        return None if self.correct is None else float(self.correct.mean())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        acc = "n/a" if self.accuracy is None else f"{self.accuracy:.3f}"
        return (
            f"AuditResult(regime={self.result.regime!r}, n={self.n_samples}, "
            f"bands={self.bands.n_bands}, accuracy={acc})"
        )

    # ---------------------------------------------------------------- tables

    def summary(self, gate_quantile: float = 0.2) -> pd.DataFrame:
        """One row per sample: prediction, geometry, and how much was silenced."""
        r = self.result
        er, delta = r.eredg(gate_quantile=gate_quantile)
        data = {
            "sample": np.arange(self.n_samples),
            "y_pred": self.y_pred_labels if self.y_pred_labels is not None else r.y_pred,
            "confidence": r.proba.max(axis=1),
            "DG": r.dg,
            "DG_early": r.dg_early,
            "REDG": r.redg,
            "eREDG": er,
            "n_muted_bands": (r.mute_frequency > 0).sum(axis=1),
            "mean_mute_fraction": r.mute_frequency.mean(axis=1),
            "max_band_stress": r.mean_stress.max(axis=1),
            "worst_band": [self.bands.labels[i] for i in np.argmax(r.mean_stress, axis=1)],
        }
        if self.y_true is not None:
            data["y_true"] = self.y_true
            data["correct"] = self.correct
        if r.collapsed is not None and np.any(r.collapsed):
            data["collapsed"] = r.collapsed
        df = pd.DataFrame(data)
        df.attrs["activity_threshold"] = delta
        df.attrs["regime"] = r.regime
        return df

    def conflict_table(self) -> pd.DataFrame:
        """One row per band: where disagreement concentrated, and on what.

        Two kinds of band register high stress, and they mean opposite things.
        An *uninformative* band contributes near-uniform evidence, which
        disagrees with any confident consensus without contributing anything. A
        *confidently wrong* band contributes sharp evidence for the wrong class.
        Only the second is a real conflict, and ``band_confidence`` separates
        them: it is the band's own mean maximum class probability before
        negotiation begins. ``informed_conflict`` combines the two, so that
        stress is discounted where the band had nothing to say.

        With ground truth available, the decisive column is ``excess_stress`` -
        cumulative stress on incorrect predictions minus that on correct ones.
        Bands with large positive values are the ones that destabilise the
        classifier when it fails.
        """
        from .divergence import softmax as _softmax

        r = self.result
        confidence = _softmax(r.logits_initial).max(axis=-1).mean(axis=0)
        rows = {
            "band": [b.index for b in self.bands],
            "label": self.bands.labels,
            f"{self.bands.axis_name}_lo": [b.lo for b in self.bands],
            f"{self.bands.axis_name}_hi": [b.hi for b in self.bands],
            "n_features": [b.n_features for b in self.bands],
            "band_confidence": confidence,
            "mean_stress": r.mean_stress.mean(axis=0),
            "cumulative_stress": r.cumulative_stress.mean(axis=0),
            "mute_frequency": r.mute_frequency.mean(axis=0),
            "band_DG": r.band_dg.mean(axis=0),
            "final_weight": r.weights_final.mean(axis=0),
            "degree": r.topology.degrees,
        }
        df = pd.DataFrame(rows)
        df["informed_conflict"] = df["mean_stress"] * df["band_confidence"]

        if self.correct is not None:
            ok = self.correct == 1
            bad = ~ok
            cum = r.cumulative_stress
            bdg = r.band_dg
            df["stress_correct"] = cum[ok].mean(axis=0) if ok.any() else np.nan
            df["stress_incorrect"] = cum[bad].mean(axis=0) if bad.any() else np.nan
            df["excess_stress"] = df["stress_incorrect"] - df["stress_correct"]
            df["band_DG_correct"] = bdg[ok].mean(axis=0) if ok.any() else np.nan
            df["band_DG_incorrect"] = bdg[bad].mean(axis=0) if bad.any() else np.nan
            df["n_correct"] = int(ok.sum())
            df["n_incorrect"] = int(bad.sum())

        # Rank on excess stress when there are errors to contrast against;
        # otherwise fall back to absolute conflict, so the column is never empty.
        basis = "informed_conflict"
        if "excess_stress" in df and df["excess_stress"].notna().any():
            basis = "excess_stress"
        df["conflict_rank"] = df[basis].rank(ascending=False).astype("Int64")
        df.attrs["conflict_rank_basis"] = basis
        return df

    def minority_evidence(self, *, min_support: float = 0.0) -> pd.DataFrame:
        """Bands that were overruled, and what they had been saying.

        H1 silences a band whose disagreement exceeds the gate. That is the
        point of the regime, but it also means the region carrying an unusual
        signal is exactly the region most likely to be suppressed - and once it
        is, the negotiated output carries no trace of what it argued for. This
        table recovers it from the pre-negotiation evidence.

        Parameters
        ----------
        min_support
            Keep only rows where the band's own initial confidence in the class
            it supported reaches this. Raising it isolates *confidently*
            dissenting bands from merely uninformative ones - the distinction
            :meth:`conflict_table` draws with ``band_confidence``, applied per
            band-class instead of per band.

        Returns
        -------
        pandas.DataFrame
            One row per band that was ever muted, with:
            ``supported_class`` - the class its own evidence favoured before
            negotiation, as a label;
            ``consensus_class`` - what the cohort concluded;
            ``dissented`` - whether those differ, i.e. the band was overruled on
            substance rather than merely damped;
            ``initial_support`` - the band's own confidence in its class;
            ``mute_frequency`` - the share of iterations it spent silenced;
            ``n_samples_dissenting`` - how many spectra it dissented on.

        Notes
        -----
        A dissenting band is a lead to follow, not a finding. It says the model
        found regional evidence pointing elsewhere; it does not say that
        evidence is real, and it says nothing at all about what is happening
        chemically. Pair it with :meth:`trace_band` to look at the raw signal
        and with :func:`pypasi.perturb_bands` to ask whether the region actually
        changes the decision.
        """
        from .divergence import softmax as _softmax

        r = self.result
        if r.mute_history is None:
            raise RuntimeError(
                "minority evidence needs the mute history; re-run the audit "
                "under a regime that uses a gate (H1 or H2)")

        P0 = _softmax(r.logits_initial)                  # (n, bands, classes)
        band_class = P0.argmax(axis=-1)                  # (n, bands)
        band_conf = P0.max(axis=-1)                      # (n, bands)
        consensus = r.y_pred[:, None]                    # (n, 1)
        muted_any = r.mute_history.any(axis=1)           # (n, bands)
        dissent = (band_class != consensus) & muted_any
        if self.classes is not None:
            to_label = lambda i: self.classes[i]
        else:
            to_label = lambda i: i

        rows = []
        for k, label in enumerate(self.bands.labels):
            sel = dissent[:, k] & (band_conf[:, k] >= min_support)
            n_dis = int(sel.sum())
            if not muted_any[:, k].any():
                continue
            if n_dis:
                # The modal (supported, consensus) PAIR, not two independent
                # modes. Taking them separately can report a band as dissenting
                # while showing the same class on both sides, because different
                # spectra dissent in different directions.
                pairs = np.stack([band_class[sel, k], r.y_pred[sel]], axis=1)
                uniq, counts = np.unique(pairs, axis=0, return_counts=True)
                sup_i, cons_i = uniq[counts.argmax()]
                supported, consensus_lbl = to_label(sup_i), to_label(cons_i)
                support = float(band_conf[sel, k].mean())
            else:
                supported = consensus_lbl = None
                support = float(band_conf[muted_any[:, k], k].mean())
            rows.append({
                "band": k,
                "label": label,
                f"{self.bands.axis_name}_lo": self.bands[k].lo,
                f"{self.bands.axis_name}_hi": self.bands[k].hi,
                "supported_class": supported,
                "consensus_class": consensus_lbl,
                "dissented": bool(n_dis),
                "initial_support": support,
                "mute_frequency": float(r.mute_frequency[:, k].mean()),
                "n_samples_muted": int(muted_any[:, k].sum()),
                "n_samples_dissenting": n_dis,
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["dissented", "n_samples_dissenting",
                                   "initial_support"], ascending=False)
        return out.reset_index(drop=True)

    def top_conflict_bands(self, n: int = 3) -> list[str]:
        """Labels of the ``n`` bands most implicated in disagreement."""
        tbl = self.conflict_table().sort_values("conflict_rank")
        return tbl["label"].head(n).tolist()

    def per_sample_bands(self, sample: int) -> pd.DataFrame:
        """Band-level detail for a single signal."""
        r = self.result
        return pd.DataFrame(
            {
                "band": [b.index for b in self.bands],
                "label": self.bands.labels,
                f"{self.bands.axis_name}_lo": [b.lo for b in self.bands],
                f"{self.bands.axis_name}_hi": [b.hi for b in self.bands],
                "mean_stress": r.mean_stress[sample],
                "cumulative_stress": r.cumulative_stress[sample],
                "mute_frequency": r.mute_frequency[sample],
                "band_DG": r.band_dg[sample],
                "initial_weight": (r.weight_history[sample, 0] if r.weight_history is not None else np.nan),
                "final_weight": r.weights_final[sample],
            }
        )

    # ------------------------------------------------------------- traceback

    def trace_band(self, band: int | str, *, sample: int | None = None) -> BandTrace:
        """Return the raw signal region behind a band, with its conflict scores.

        This is the step that makes the conflict table actionable: given a band
        the audit blames, look at the spectra themselves over exactly that
        interval and see what is happening.

        Parameters
        ----------
        band
            Band index or label.
        sample
            Restrict to a single signal. Defaults to all of them.
        """
        if self.X is None:
            raise RuntimeError(
                "trace_band needs the signals; build the AuditResult with X= "
                "(BandNegotiationClassifier.audit does this for you)"
            )
        b = self.bands[band]
        block = self.X[:, b.features]
        if sample is not None:
            block = block[[sample], :]

        class_means = None
        class_labels = None
        if self.y_true is not None and sample is None:
            class_labels = np.unique(self.y_true)
            class_means = np.stack(
                [self.X[self.y_true == c][:, b.features].mean(axis=0) for c in class_labels]
            )

        conflict = self.conflict_table().set_index("label").loc[b.label]
        return BandTrace(
            band=b,
            axis=self.bands.axis[b.features],
            X=block,
            class_means=class_means,
            class_labels=class_labels,
            conflict=conflict,
        )

    def trace_top_conflict(self, n: int = 1) -> list[BandTrace]:
        """Traces for the ``n`` most conflicted bands, worst first."""
        return [self.trace_band(lbl) for lbl in self.top_conflict_bands(n)]

    # -------------------------------------------------------------- exports

    def to_csv(self, prefix: str, *, gate_quantile: float = 0.2) -> dict[str, str]:
        """Write the summary, conflict and band tables as CSV files.

        Returns a mapping of table name to the path written.
        """
        paths = {
            "summary": f"{prefix}_summary.csv",
            "conflict": f"{prefix}_band_conflict.csv",
            "bands": f"{prefix}_bands.csv",
        }
        self.summary(gate_quantile).to_csv(paths["summary"], index=False)
        self.conflict_table().to_csv(paths["conflict"], index=False)
        self.bands.to_frame().to_csv(paths["bands"], index=False)
        return paths

    def report(self, path: str, *, sample: int | None = None, title: str | None = None) -> str:
        """Write a self-contained interactive HTML audit report. See :mod:`pypasi.report`."""
        from .report import write_report

        return write_report(self, path, sample=sample, title=title)


def compare_regimes(audits: dict[str, AuditResult], gate_quantile: float = 0.2) -> pd.DataFrame:
    """Side-by-side comparison of several regimes over the same signals.

    Parameters
    ----------
    audits
        Mapping of regime name to :class:`AuditResult`, as returned by
        :meth:`~pypasi.estimator.BandNegotiationClassifier.compare_regimes`.
    """
    rows = []
    for name, a in audits.items():
        r = a.result
        s = a.summary(gate_quantile)
        row = {
            "regime": name,
            "n_samples": a.n_samples,
            "accuracy": a.accuracy,
            "mean_confidence": float(r.proba.max(axis=1).mean()),
            "DG_mean": float(r.dg.mean()),
            "DG_sd": float(r.dg.std(ddof=1)) if a.n_samples > 1 else 0.0,
            "REDG_mean": float(r.redg.mean()),
            "eREDG_median": float(np.nanmedian(s["eREDG"])) if s["eREDG"].notna().any() else np.nan,
            "mute_fraction": float(r.mute_frequency.mean()),
            "n_collapsed": int(r.collapsed.sum()) if r.collapsed is not None else 0,
            "top_conflict_band": a.top_conflict_bands(1)[0],
        }
        if a.correct is not None:
            from sklearn.metrics import roc_auc_score

            active = s.dropna(subset=["eREDG"])
            if active["correct"].nunique() > 1:
                row["eREDG_audit_auc"] = float(
                    roc_auc_score(active["correct"], active["eREDG"])
                )
            else:
                row["eREDG_audit_auc"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
