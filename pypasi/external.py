"""Audit a classifier you already have.

Most laboratories are not going to replace a validated global PLS-DA or SVM
workflow, and they should not have to in order to find out whether today's
spectra are degraded. :class:`ExternalAudit` leaves the existing model's
predictions exactly as they are and fits a band ensemble *alongside* it, purely
as an instrument.

What the report separates, and why it must
------------------------------------------
Three quantities that are easy to conflate:

``host_prediction``
    What the laboratory's own model said. Unchanged, always. This is the answer
    of record.
``band_prediction``
    What the auxiliary band ensemble said. It is a second opinion from a
    *different model*, not a decomposition of the first one.
``conflict``, ``novelty``, ``worst_band``
    Properties of the spectrum and of the auxiliary system.

The negotiation describes the band ensemble. It does not describe the internal
reasoning of the host model, and nothing here should be read as "here is why
your PLS-DA said what it said". A band ensemble fitted on the same data will
often agree, and where it disagrees that is a *flag to investigate*, not a
correction to apply.

What transfers and what does not
--------------------------------
This distinction decides what you may claim.

**Degradation detection transfers.** Whether a spectrum came from a damaged
acquisition is a property of the spectrum. The host model does not enter, so
conflict measured on the auxiliary system says something about the input to the
host model regardless of what the host model is.

**Error prediction does not transfer.** Conflict's relationship to *wrong
predictions* was established against the band ensemble's own errors. A different
host makes different errors, and the relationship has to be re-established
before it can be relied on. :meth:`ExternalAudit.validate` does exactly that on
labelled data, and reports the number rather than assuming it.

Examples
--------
>>> from sklearn.linear_model import LogisticRegression
>>> from pypasi import audit_external
>>> from pypasi.datasets import make_conflict_signals
>>> X, y, axis, *_ = make_conflict_signals(n_samples=120, random_state=0)
>>> host = LogisticRegression(max_iter=1000).fit(X, y)
>>> ext = audit_external(host, X, y, axis=axis, n_bands=5, random_state=0)
>>> report = ext.explain(X[:10])
>>> list(report.columns)[:3]
['host_prediction', 'band_prediction', 'agreement']
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimator import BandNegotiationClassifier
from .triage import NoveltyDetector

__all__ = ["ExternalAudit", "audit_external"]


class ExternalAudit:
    """A band-resolved instrument attached to somebody else's classifier.

    Parameters
    ----------
    host
        Any fitted classifier exposing ``predict``. Never modified, never
        refitted, never consulted for anything but its prediction.
    band_model
        The fitted auxiliary :class:`~pypasi.estimator.BandNegotiationClassifier`.
    novelty
        A fitted :class:`~pypasi.triage.NoveltyDetector`, or ``None``.

    Attributes
    ----------
    bands_ : BandSet
        Delegated from the band model, so a
        :class:`~pypasi.monitor.ControlProfile` can be built on this object
        directly.

    Notes
    -----
    Use :func:`audit_external` to construct one; it fits the auxiliary system
    for you.
    """

    def __init__(self, host, band_model: BandNegotiationClassifier, novelty=None):
        if not hasattr(host, "predict"):
            raise TypeError(
                f"{type(host).__name__} has no predict(); an external audit needs "
                "a fitted classifier to attach to")
        self.host = host
        self.band_model = band_model
        self.novelty = novelty

    # ------------------------------------------------------- delegation

    @property
    def bands_(self):
        return self.band_model.bands_

    @property
    def classes_(self):
        return self.band_model.classes_

    @property
    def evidence_spec_(self):
        return self.band_model.evidence_spec_

    def audit(self, X, y=None, **kwargs):
        """The auxiliary system's audit, so ``ControlProfile`` works unchanged.

        This is the band ensemble's audit and carries the band ensemble's
        prediction. For the host model's prediction alongside it, use
        :meth:`explain`.
        """
        return self.band_model.audit(X, y, **kwargs)

    def predict(self, X):
        """The **host** model's prediction. This class never overrides it."""
        return self.host.predict(X)

    # ------------------------------------------------------------ report

    def explain(self, X, *, gate_quantile: float = 0.2) -> pd.DataFrame:
        """One row per spectrum: the host's answer, and what the audit sees.

        Returns
        -------
        pandas.DataFrame
            ``host_prediction`` - the answer of record, untouched.
            ``band_prediction`` - the auxiliary ensemble's second opinion.
            ``agreement`` - whether the two coincide. Disagreement is a flag to
            investigate, never a correction: the two are different models.
            ``host_confidence`` - the host's own maximum class probability where
            it exposes one, otherwise ``NaN``.
            ``conflict`` - eREDG from the negotiation, the degradation signal.
            ``mean_stress`` - mean band disagreement.
            ``novelty`` - outlier ratio, 1.0 being the calibrated threshold.
            ``worst_band`` / ``worst_band_lo`` / ``worst_band_hi`` - the region
            carrying the most stress for this spectrum, in axis units.
        """
        X = np.asarray(X, dtype=float)
        res = self.band_model.negotiate(X, record_trajectory=True)
        # eREDG runs high for a trajectory that settled early, so the conflict
        # score is its complement - the same orientation Triage uses. A
        # trajectory too inactive to have an eREDG had nothing to reconcile and
        # belongs at the calm end, not at the alarming one.
        raw, _ = res.eredg(gate_quantile=gate_quantile)
        conflict = np.where(np.isfinite(raw), 1.0 - raw, 0.0)
        stress = res.mean_stress                      # (n, bands)
        worst = np.argmax(stress, axis=1)

        host_pred = np.asarray(self.host.predict(X))
        band_pred = self.band_model.classes_[res.y_pred]

        host_conf = np.full(len(X), np.nan)
        if hasattr(self.host, "predict_proba"):
            try:
                host_conf = np.asarray(self.host.predict_proba(X), float).max(axis=1)
            except Exception:                          # pragma: no cover - host's problem
                pass

        labels = self.bands_.labels
        lo = np.array([b.lo for b in self.bands_])
        hi = np.array([b.hi for b in self.bands_])
        out = pd.DataFrame({
            "host_prediction": host_pred,
            "band_prediction": band_pred,
            "agreement": host_pred == band_pred,
            "host_confidence": host_conf,
            "conflict": conflict,
            "mean_stress": stress.mean(axis=1),
            "worst_band": [labels[i] for i in worst],
            f"worst_band_{self.bands_.axis_name}_lo": lo[worst],
            f"worst_band_{self.bands_.axis_name}_hi": hi[worst],
        })
        if self.novelty is not None:
            out.insert(6, "novelty", self.novelty_ratio(X))
        out.attrs["host"] = type(self.host).__name__
        out.attrs["note"] = (
            "conflict and mean_stress describe the auxiliary band ensemble, not "
            "the host model's internal reasoning")
        return out

    def novelty_ratio(self, X) -> np.ndarray:
        """Outlier score scaled so that 1.0 is the calibrated threshold."""
        if self.novelty is None:
            raise RuntimeError("no NoveltyDetector was fitted; pass novelty=True")
        return self.novelty.score(np.asarray(X, float))["novelty"].to_numpy()

    # ---------------------------------------------------------- validation

    def validate(self, X, y, *, gate_quantile: float = 0.2) -> pd.DataFrame:
        """Measure what the audit is worth *for this host*, on labelled data.

        Degradation detection is a property of the spectra and carries over from
        any cohort. The relationship between conflict and a **wrong host
        prediction** does not: it was established against the band ensemble's
        own errors, and a different host makes different errors. This method
        measures it rather than assuming it.

        Returns
        -------
        pandas.DataFrame
            AUROC of each available score against a wrong *host* prediction,
            plus the agreement rate and both models' accuracies. When the host
            is right about everything, or wrong about everything, the AUROC
            columns are ``NaN`` rather than a number computed on nothing.
        """
        from sklearn.metrics import roc_auc_score

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rep = self.explain(X, gate_quantile=gate_quantile)
        wrong = (rep["host_prediction"].to_numpy() != y).astype(int)

        def auc(v):
            v = np.asarray(v, float)
            ok = np.isfinite(v)
            if not (0 < wrong[ok].sum() < ok.sum()):
                return float("nan")
            return float(roc_auc_score(wrong[ok], v[ok]))

        rows = [
            {"score": "conflict", "auroc_vs_host_error": auc(rep["conflict"])},
            {"score": "mean_stress", "auroc_vs_host_error": auc(rep["mean_stress"])},
        ]
        if "novelty" in rep:
            rows.append({"score": "novelty",
                         "auroc_vs_host_error": auc(rep["novelty"])})
        if np.isfinite(rep["host_confidence"]).any():
            rows.append({"score": "1 - host confidence",
                         "auroc_vs_host_error": auc(1.0 - rep["host_confidence"])})

        out = pd.DataFrame(rows)
        out["host_accuracy"] = float((rep["host_prediction"].to_numpy() == y).mean())
        out["band_accuracy"] = float((rep["band_prediction"].to_numpy() == y).mean())
        out["agreement_rate"] = float(rep["agreement"].mean())
        out.attrs["note"] = (
            "these AUROCs are against the HOST model's errors and are specific "
            "to this host; degradation detection is a separate question and "
            "does not depend on the host")
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ExternalAudit(host={type(self.host).__name__}, "
                f"bands={self.bands_.n_bands}, "
                f"novelty={'yes' if self.novelty is not None else 'no'})")


def audit_external(
    host,
    X,
    y,
    *,
    axis=None,
    n_bands: int = 7,
    bands=None,
    base_estimator=None,
    novelty: bool = True,
    n_components: int = 10,
    **kwargs,
) -> ExternalAudit:
    """Attach a band-resolved audit to an already-fitted classifier.

    Parameters
    ----------
    host
        The laboratory's own fitted classifier. Left untouched.
    X, y
        Data to fit the auxiliary band ensemble on. Use the host's own training
        data when you have it: the point is an instrument calibrated to the same
        population, not a competitor trained on something else.
    axis, n_bands, bands, base_estimator, **kwargs
        Passed to :class:`~pypasi.estimator.BandNegotiationClassifier`.
    novelty
        Also fit a :class:`~pypasi.triage.NoveltyDetector` on ``X``.
    n_components
        Components for that detector.

    Returns
    -------
    ExternalAudit

    Notes
    -----
    The auxiliary ensemble is a second model, and it will usually classify
    slightly worse than a well-tuned global one - band partitioning discards the
    cross-band covariance a global model exploits. That does not matter here,
    because its prediction is not the deliverable; its *negotiation* is.
    """
    X = np.asarray(X, dtype=float)
    band_model = BandNegotiationClassifier(
        bands=bands, axis=axis, n_bands=n_bands,
        base_estimator=base_estimator, **kwargs).fit(X, y)
    det = NoveltyDetector(n_components=n_components).fit(X) if novelty else None
    return ExternalAudit(host, band_model, det)
