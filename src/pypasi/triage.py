"""Triage: turning a negotiation into an actionable quality-control decision.

A prediction can be untrustworthy for two unrelated reasons, and conflating them
makes the flag useless.

**The sample is strange.** It does not resemble anything in the training data -
wrong specimen, contamination, substrate change, instrument drift, a cosmic ray.
This is *novelty*, it lives in input space, and chemometrics has measured it for
decades with the squared prediction error and Hotelling's T-squared of a PCA
model. A high-novelty spectrum should be re-acquired or excluded; the model was
never entitled to an opinion about it.

**The sample is ordinary but its evidence contradicts itself.** The spectrum sits
comfortably inside the training distribution, yet its bands argue for different
classes and the negotiation has to force a resolution. This is *conflict*, it
lives in the inference trajectory, and it is what Decision Geometry measures. A
high-conflict spectrum is a real specimen that genuinely looks like two things at
once - the case for a second assay or an expert eye, not for re-measurement.

Crossing the two gives four actions:

==================  ==================  ====================================
novelty             conflict            action
==================  ==================  ====================================
low                 low                 ``accept``
low                 **high**            ``review``   - secondary validation
**high**            low                 ``remeasure`` - specimen or acquisition
**high**            **high**            ``reject``
==================  ==================  ====================================

:class:`Triage` computes both, calibrates each on clean training data so the
referral rate is a quantity you choose rather than one you discover, and
:meth:`Triage.evaluate` produces the evidence needed to defend the claim:

* how *uncorrelated* conflict and novelty actually are on your data;
* whether conflict still separates right from wrong **after restricting to
  in-distribution samples** - if it does, it cannot be novelty detection wearing
  a different hat;
* whether it beats the obvious baseline, the model's own confidence, on a
  risk-coverage curve.

The third is the one that decides whether the machinery earns its place. A flag
that does no better than "the softmax was low" is not worth building.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ACTIONS",
    "NoveltyDetector",
    "Triage",
    "TriageReport",
    "compare_conflict_scores",
    "risk_coverage",
]

ACTIONS = ("accept", "review", "remeasure", "reject")

_ACTION_REASON = {
    "accept": "evidence coherent and the spectrum is typical",
    "review": "typical spectrum, but its bands disagree - secondary validation",
    "remeasure": "spectrum unlike the training data - check specimen or acquisition",
    "reject": "atypical spectrum and incoherent evidence",
}


class NoveltyDetector:
    """Is this spectrum like the ones the model was trained on?

    A PCA model of the training spectra gives the two statistics multivariate
    process control has always used:

    ``spe``
        Squared prediction error, also called the Q residual: how much of the
        spectrum the model cannot reconstruct. Catches features that were never
        in the training data at all.
    ``t2``
        Hotelling's T-squared: how far inside the model the spectrum sits, in
        units of the training variance. Catches spectra built from familiar
        features in unfamiliar proportions.

    The two catch different things and both are reported. Thresholds are
    quantiles of the training distribution, so ``q=0.99`` means one clean
    training spectrum in a hundred would be called novel.

    Parameters
    ----------
    n_components : int
        PCA rank. Clamped to what the data supports.
    q : float
        Quantile of the training distribution used as the threshold.
    scale : bool
        Standardise before PCA.

    Attributes
    ----------
    spe_threshold_, t2_threshold_ : float
    """

    def __init__(self, n_components: int = 10, *, q: float = 0.99, scale: bool = True):
        self.n_components = n_components
        self.q = float(q)
        self.scale = scale

    def fit(self, X) -> "NoveltyDetector":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if not 0.0 < self.q < 1.0:
            raise ValueError("q must lie strictly between 0 and 1")
        self.n_components_ = int(max(1, min(self.n_components, *X.shape)))
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        Xs = self.scaler_.transform(X) if self.scale else X
        self.pca_ = PCA(n_components=self.n_components_).fit(Xs)
        spe, t2 = self._raw(Xs)
        self.spe_threshold_ = float(np.quantile(spe, self.q))
        self.t2_threshold_ = float(np.quantile(t2, self.q))
        return self

    def _raw(self, Xs):
        T = self.pca_.transform(Xs)
        recon = self.pca_.inverse_transform(T)
        spe = np.sum((Xs - recon) ** 2, axis=1)
        var = np.clip(self.pca_.explained_variance_, 1e-12, None)
        t2 = np.sum(T**2 / var, axis=1)
        return spe, t2

    def score(self, X) -> pd.DataFrame:
        """Per-sample ``spe``, ``t2`` and a combined ``novelty``.

        ``novelty`` is each statistic divided by its own threshold, then the
        larger of the two taken - so it exceeds 1 exactly when at least one
        statistic is out of bounds, and the two are commensurable despite
        having completely different units.
        """
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X
        spe, t2 = self._raw(Xs)
        rel_spe = spe / max(self.spe_threshold_, 1e-12)
        rel_t2 = t2 / max(self.t2_threshold_, 1e-12)
        return pd.DataFrame(
            {"spe": spe, "t2": t2, "novelty": np.maximum(rel_spe, rel_t2)}
        )


def _conflict_from_audit(audit, kind: str, gate_quantile: float):
    """Raw conflict per sample, plus a mask of which trajectories were active.

    The mask matters because a trajectory too inactive to have an eREDG had
    nothing to reconcile. Those samples belong at the calm end of the scale
    whichever way round the score turns out to run, so the sentinel cannot be
    baked in here - it has to wait until the orientation is known.
    """
    r = audit.result
    if kind == "eredg":
        vals, _ = r.eredg(gate_quantile=gate_quantile)
        active = np.isfinite(vals)
        return np.where(active, 1.0 - vals, 0.0), active
    if kind == "dg":
        return r.dg, np.ones(r.n_samples, dtype=bool)
    if kind == "stress":
        return r.mean_stress.mean(axis=1), np.ones(r.n_samples, dtype=bool)
    if kind == "mute":
        return r.mute_frequency.mean(axis=1), np.ones(r.n_samples, dtype=bool)
    raise ValueError(
        f"unknown conflict score {kind!r}; use 'eredg', 'dg', 'stress' or 'mute'"
    )


def _orient(raw, active, sign):
    """Apply the learned sign, then park inactive trajectories at the calm end."""
    oriented = sign * np.asarray(raw, dtype=float)
    if active.all():
        return oriented
    calm = oriented[active].min() if active.any() else 0.0
    return np.where(active, oriented, calm)


@dataclass
class TriageReport:
    """Per-sample triage decisions plus the evidence behind the policy."""

    table: pd.DataFrame
    action_counts: pd.Series
    thresholds: dict

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        counts = ", ".join(f"{k} {v}" for k, v in self.action_counts.items())
        return f"TriageReport({len(self.table)} signals: {counts})"


class Triage:
    """A calibrated quality-control policy over a fitted negotiation classifier.

    Parameters
    ----------
    conflict_score : {"eredg", "dg", "stress", "mute"}
        Which trajectory descriptor stands for conflict. ``"eredg"`` is the
        published descriptor and the default.
    review_rate : float
        Share of clean calibration signals that would be sent for review. This
        is the knob a clinic actually has: how many cases can a human look at.
        The conflict threshold is set to hit it.
    novelty_q : float
        Quantile for the novelty thresholds.
    n_components : int
        PCA rank for the novelty model.
    gate_quantile : float
        Activity gate for eREDG, see :func:`pypasi.geometry.eredg`.

    Examples
    --------
    >>> from pypasi import BandNegotiationClassifier, Triage
    >>> from pypasi.datasets import make_conflict_signals
    >>> X, y, axis, _ = make_conflict_signals(n_samples=200, random_state=0)
    >>> clf = BandNegotiationClassifier(axis=axis, n_bands=5, random_state=0).fit(X, y)
    >>> tri = Triage(review_rate=0.1).fit(clf, X)
    >>> report = tri.assess(X)
    >>> sorted(report.action_counts.index) == ['accept']  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        conflict_score: str = "eredg",
        review_rate: float = 0.10,
        novelty_q: float = 0.99,
        n_components: int = 10,
        gate_quantile: float = 0.2,
    ):
        self.conflict_score = conflict_score
        self.review_rate = float(review_rate)
        self.novelty_q = float(novelty_q)
        self.n_components = n_components
        self.gate_quantile = float(gate_quantile)

    # -------------------------------------------------------------------- fit

    def fit(self, classifier, X_cal, y_cal=None) -> "Triage":
        """Calibrate on clean signals the classifier was trained on or validated against.

        Nothing about the classifier is refitted. Calibration only records what
        "normal" looks like, in both senses, so that later thresholds mean
        something.
        """
        if not 0.0 < self.review_rate < 1.0:
            raise ValueError("review_rate must lie strictly between 0 and 1")
        X_cal = np.asarray(X_cal, dtype=float)
        self.classifier_ = classifier
        self.novelty_ = NoveltyDetector(
            n_components=self.n_components, q=self.novelty_q
        ).fit(X_cal)

        audit = classifier.audit(X_cal, y_cal)
        raw, active = _conflict_from_audit(audit, self.conflict_score, self.gate_quantile)

        # Which direction of the trajectory descriptor indicates trouble is not
        # universal - the manuscript reports the eREDG audit as dataset-dependent
        # - so it is learned here rather than assumed. Without labels the
        # published convention is used and recorded as such.
        self.conflict_sign_ = 1
        self.orientation_source_ = "assumed (no calibration labels)"
        if y_cal is not None:
            from sklearn.metrics import roc_auc_score

            pred = (audit.y_pred_labels if audit.y_pred_labels is not None
                    else audit.result.y_pred)
            wrong = (pred != np.asarray(y_cal)).astype(int)
            if len(np.unique(wrong)) > 1:
                auc = float(roc_auc_score(wrong[active], raw[active])) if active.sum() > 2 \
                    else float(roc_auc_score(wrong, raw))
                self.conflict_sign_ = -1 if auc < 0.5 else 1
                self.calibration_auroc_ = max(auc, 1.0 - auc)
                self.orientation_source_ = (
                    f"learned on calibration labels (AUROC {self.calibration_auroc_:.3f})"
                )
        conflict = _orient(raw, active, self.conflict_sign_)
        self.conflict_threshold_ = float(np.quantile(conflict, 1.0 - self.review_rate))
        self.calibration_conflict_ = conflict
        self.calibration_confidence_ = audit.result.proba.max(axis=1)
        self.calibration_wrong_ = None
        if y_cal is not None:
            pred = (audit.y_pred_labels if audit.y_pred_labels is not None
                    else audit.result.y_pred)
            self.calibration_wrong_ = (pred != np.asarray(y_cal)).astype(int)
        return self

    # ----------------------------------------------------------------- assess

    def _scores(self, X, y=None, regime=None):
        if not hasattr(self, "novelty_"):
            raise RuntimeError(
                "Triage must be calibrated first: call fit(classifier, X_cal) "
                "with clean signals before assessing new ones"
            )
        X = np.asarray(X, dtype=float)
        audit = self.classifier_.audit(X, y, regime=regime)
        nov = self.novelty_.score(X)
        raw, active = _conflict_from_audit(audit, self.conflict_score, self.gate_quantile)
        conflict = _orient(raw, active, self.conflict_sign_)
        confidence = audit.result.proba.max(axis=1)
        return audit, nov, conflict, confidence

    def assess(self, X, y=None, regime=None) -> TriageReport:
        """Triage a batch of signals.

        Returns
        -------
        TriageReport
            ``table`` has one row per signal with both scores, both flags, the
            recommended ``action`` and a plain-language ``reason``.
        """
        audit, nov, conflict, confidence = self._scores(X, y, regime)
        novel = nov["novelty"].to_numpy() > 1.0
        conflicted = conflict > self.conflict_threshold_

        action = np.where(
            novel & conflicted, "reject",
            np.where(novel, "remeasure", np.where(conflicted, "review", "accept")),
        )
        table = pd.DataFrame(
            {
                "sample": np.arange(len(conflict)),
                "prediction": audit.y_pred_labels if audit.y_pred_labels is not None
                else audit.result.y_pred,
                "confidence": confidence,
                "spe": nov["spe"].to_numpy(),
                "t2": nov["t2"].to_numpy(),
                "novelty": nov["novelty"].to_numpy(),
                "conflict": conflict,
                "is_novel": novel,
                "is_conflicted": conflicted,
                "action": action,
                "reason": [_ACTION_REASON[a] for a in action],
                "worst_band": [audit.bands.labels[i]
                               for i in np.argmax(audit.result.mean_stress, axis=1)],
            }
        )
        if y is not None:
            table["y_true"] = np.asarray(y)
            table["correct"] = (table["prediction"] == table["y_true"]).astype(int)
        counts = table["action"].value_counts().reindex(ACTIONS, fill_value=0)
        return TriageReport(
            table=table,
            action_counts=counts,
            thresholds={
                "conflict": self.conflict_threshold_,
                "novelty": 1.0,
                "spe": self.novelty_.spe_threshold_,
                "t2": self.novelty_.t2_threshold_,
                "review_rate": self.review_rate,
                "conflict_score": self.conflict_score,
                "conflict_orientation": self.orientation_source_,
            },
        )

    # --------------------------------------------------------------- evidence

    def evaluate(self, X, y, regime=None) -> dict:
        """Does the conflict flag earn its place?

        Three questions, three answers, all on labelled evaluation data.

        ``separation``
            Area under the ROC for each score against the event "the prediction
            was wrong". Includes the model's own confidence, which is the
            baseline any new flag has to beat.
        ``conditional``
            The same for conflict, computed **only on in-distribution signals**.
            If conflict still separates right from wrong once the outliers are
            removed, it is measuring something novelty detection cannot see.
        ``orthogonality``
            Rank correlation between novelty and conflict, and the overlap
            between the two flags. Near zero is the claim.
        ``risk_coverage``
            Accuracy retained as a function of how many signals are kept, for
            each score.

        Returns
        -------
        dict
            With keys ``separation``, ``conditional``, ``orthogonality``,
            ``risk_coverage`` and ``summary``.
        """
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score

        audit, nov, conflict, confidence = self._scores(X, y, regime)
        y = np.asarray(y)
        pred = audit.y_pred_labels if audit.y_pred_labels is not None else audit.result.y_pred
        wrong = (pred != y).astype(int)
        novelty = nov["novelty"].to_numpy()

        scores = {
            "1 - confidence": 1.0 - confidence,
            "novelty": novelty,
            f"conflict ({self.conflict_score})": conflict,
        }

        def _auc(mask=None):
            out = {}
            for name, s in scores.items():
                w = wrong if mask is None else wrong[mask]
                v = s if mask is None else s[mask]
                out[name] = (float(roc_auc_score(w, v))
                             if len(np.unique(w)) > 1 else float("nan"))
            return out

        separation = pd.DataFrame(
            {"score": list(scores), "auroc_error": list(_auc().values())}
        )

        in_dist = novelty <= 1.0
        n_in = int(in_dist.sum())
        cond = _auc(in_dist) if n_in > 2 else {k: float("nan") for k in scores}
        conditional = pd.DataFrame(
            {"score": list(cond), "auroc_error_in_distribution": list(cond.values())}
        )
        conditional.attrs["n_in_distribution"] = n_in
        conditional.attrs["n_total"] = int(len(y))

        # The claim the manuscript actually makes: a conflicted prediction is
        # worth flagging *even when the classifier is confident about it*. Test
        # it where it matters, on the confident half of the cohort.
        confident = confidence >= np.median(confidence)
        n_conf = int(confident.sum())
        conf_auc = (_auc(confident & in_dist)
                    if (confident & in_dist).sum() > 2
                    else {k: float("nan") for k in scores})
        among_confident = pd.DataFrame(
            {"score": list(conf_auc), "auroc_error_confident": list(conf_auc.values())}
        )
        among_confident.attrs["n_confident_in_distribution"] = int((confident & in_dist).sum())
        among_confident.attrs["n_confident"] = n_conf
        among_confident.attrs["n_errors_in_stratum"] = int(wrong[confident & in_dist].sum())

        rho, pval = spearmanr(novelty, conflict)
        both = (novelty > 1.0) & (conflict > self.conflict_threshold_)
        either = (novelty > 1.0) | (conflict > self.conflict_threshold_)
        orthogonality = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "jaccard_overlap": float(both.sum() / either.sum()) if either.any() else 0.0,
            "n_novel_only": int(((novelty > 1.0) & ~(conflict > self.conflict_threshold_)).sum()),
            "n_conflicted_only": int((~(novelty > 1.0) & (conflict > self.conflict_threshold_)).sum()),
            "n_both": int(both.sum()),
        }

        rc = pd.concat(
            [risk_coverage(s, wrong).assign(score=name) for name, s in scores.items()],
            ignore_index=True,
        )

        # Does conflict add anything to confidence, or only restate it? Weighting
        # the two equally would be unfair to whichever is stronger, so the
        # weights are fitted on the calibration data and only applied here.
        conflict_name = f"conflict ({self.conflict_score})"
        auc_conf = (float(roc_auc_score(wrong, 1.0 - confidence))
                    if len(np.unique(wrong)) > 1 else np.nan)
        auc_comb, note = np.nan, "needs calibration labels"
        if (self.calibration_wrong_ is not None
                and len(np.unique(self.calibration_wrong_)) > 1
                and len(np.unique(wrong)) > 1):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            cal = np.column_stack([1.0 - self.calibration_confidence_,
                                   self.calibration_conflict_])
            new = np.column_stack([1.0 - confidence, conflict])
            combiner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            combiner.fit(cal, self.calibration_wrong_)
            auc_comb = float(roc_auc_score(wrong, combiner.predict_proba(new)[:, 1]))
            note = "weights fitted on calibration data"
        incremental = {
            "auroc_confidence_alone": auc_conf,
            "auroc_confidence_plus_conflict": auc_comb,
            "gain": auc_comb - auc_conf,
            "note": note,
        }

        best_uncond = separation.loc[separation["auroc_error"].idxmax(), "score"]
        sep = separation.set_index("score")
        summary = {
            "conflict_beats_confidence": bool(
                sep.loc[conflict_name, "auroc_error"] > sep.loc["1 - confidence", "auroc_error"]
            ),
            "conflict_adds_to_confidence": bool(incremental["gain"] > 0.01),
            "conflict_survives_conditioning": bool(
                conditional.set_index("score").loc[
                    conflict_name, "auroc_error_in_distribution"] > 0.55
            ),
            "conflict_informative_among_confident": bool(
                among_confident.set_index("score").loc[
                    conflict_name, "auroc_error_confident"] > 0.55
            ),
            "novelty_and_conflict_uncorrelated": bool(abs(rho) < 0.3),
            "best_unconditional_score": str(best_uncond),
            "conflict_orientation": self.orientation_source_,
        }
        return {
            "separation": separation,
            "conditional": conditional,
            "among_confident": among_confident,
            "orthogonality": orthogonality,
            "incremental": incremental,
            "risk_coverage": rc,
            "summary": summary,
        }


def risk_coverage(score, wrong, coverages=None) -> pd.DataFrame:
    """Accuracy retained as a function of how much of the cohort is kept.

    Signals are ranked by ``score`` - lower meaning more trustworthy - and the
    least trustworthy are handed off first. A useful flag keeps accuracy high as
    coverage falls; a useless one leaves it flat.

    Parameters
    ----------
    score
        Per-sample untrustworthiness, higher meaning less trustworthy.
    wrong
        1 where the prediction was wrong, 0 where right.
    coverages
        Fractions of the cohort to retain. Defaults to 1.0 down to 0.5.

    Returns
    -------
    pandas.DataFrame
        ``coverage``, ``n_kept``, ``accuracy``, ``errors_remaining``,
        ``errors_caught_fraction``. ``aurc`` (area under the risk-coverage
        curve, lower is better) is attached to ``.attrs``.
    """
    score = np.asarray(score, dtype=float)
    wrong = np.asarray(wrong, dtype=int)
    if score.shape != wrong.shape:
        raise ValueError("score and wrong must have the same shape")
    if coverages is None:
        coverages = np.round(np.arange(1.0, 0.45, -0.1), 3)

    order = np.argsort(score, kind="stable")   # most trustworthy first
    total_errors = int(wrong.sum())
    n = len(score)
    rows = []
    for c in coverages:
        k = max(1, int(round(float(c) * n)))
        kept = order[:k]
        err = int(wrong[kept].sum())
        rows.append({
            "coverage": float(c),
            "n_kept": k,
            "accuracy": 1.0 - err / k,
            "errors_remaining": err,
            "errors_caught_fraction": (
                (total_errors - err) / total_errors if total_errors else np.nan
            ),
        })
    df = pd.DataFrame(rows)
    risks = 1.0 - df["accuracy"].to_numpy()
    df.attrs["aurc"] = float(np.trapezoid(risks[::-1], df["coverage"].to_numpy()[::-1]))
    df.attrs["total_errors"] = total_errors
    return df


def compare_conflict_scores(
    classifier,
    X_cal,
    y_cal,
    X_eval,
    y_eval,
    *,
    scores=("eredg", "dg", "stress", "mute"),
    **triage_kwargs,
) -> pd.DataFrame:
    """Try every conflict descriptor and report which, if any, earns its place.

    This is the first thing to run on a new dataset. The manuscript reports the
    eREDG audit as dataset-dependent, and it is: the descriptor that tracks error
    on one cohort need not on another, and on some cohorts none of them beats the
    model's own confidence.

    Returns
    -------
    pandas.DataFrame
        One row per descriptor: the sign learned during calibration, area under
        the ROC against error overall and among in-distribution signals, the gain
        from adding it to confidence, and ``worth_using``.

    Notes
    -----
    Selecting a descriptor on the same data used to judge it is optimistic. Use a
    third split, or nested resampling, before reporting a number as final.
    """
    rows = []
    for kind in scores:
        tri = Triage(conflict_score=kind, **triage_kwargs).fit(classifier, X_cal, y_cal)
        ev = tri.evaluate(X_eval, y_eval)
        name = f"conflict ({kind})"
        sep = ev["separation"].set_index("score")
        rows.append({
            "conflict_score": kind,
            "sign": tri.conflict_sign_,
            "auroc_error": sep.loc[name, "auroc_error"],
            "auroc_error_in_distribution": ev["conditional"].set_index("score").loc[
                name, "auroc_error_in_distribution"],
            "auroc_confidence": sep.loc["1 - confidence", "auroc_error"],
            "gain_over_confidence": ev["incremental"]["gain"],
            "worth_using": bool(ev["summary"]["conflict_adds_to_confidence"]
                                or ev["summary"]["conflict_beats_confidence"]),
        })
    return pd.DataFrame(rows).sort_values("auroc_error", ascending=False).reset_index(drop=True)
