"""The scikit-learn estimator.

:class:`BandNegotiationClassifier` wraps the whole pipeline - band partitioning,
per-band base learners, negotiated consensus - behind ``fit``/``predict``, so it
drops into ``Pipeline``, ``cross_val_score`` and ``GridSearchCV`` unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from .bands import BandSet
from .divergence import get_divergence, js_divergence
from .evidence import EvidenceSpec, TemperatureCalibrator, to_evidence
from .gates import Gate, QuantileGate, resolve_gate
from .logits import supports_logits
from .negotiate import NegotiationResult, band_stress, initial_weights, negotiate, neighbourhood_consensus
from .regimes import resolve_regime
from .topology import Topology

__all__ = ["BandNegotiationClassifier"]


def _default_base_estimator():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


class BandNegotiationClassifier(ClassifierMixin, BaseEstimator):
    """Classify 1-D signals by negotiated consensus among band-level models.

    Each band of the signal gets its own base classifier. Those models produce
    local class evidence, which is then reconciled through an iterative
    negotiation over an interaction graph. The prediction is the aggregated
    outcome; the trajectory that produced it is available for auditing.

    Parameters
    ----------
    bands : BandSet or None
        Explicit band partition. If ``None``, one is built from ``axis`` using
        ``n_bands`` and ``band_method``.
    axis : array-like or None
        Signal axis, one value per feature (wavenumbers, retention times, ...).
        Defaults to ``arange(n_features)``.
    n_bands : int
        Number of bands to build when ``bands`` is not supplied.
    band_method : {"equal_width", "peak_informed"}
        How to place band boundaries.
    band_range : tuple or None
        ``(lo, hi)`` limits for band construction. Defaults to the axis extent.
    axis_name : str
        Units of the signal axis, e.g. ``"cm-1"``. Used in table columns and
        report labels.
    base_estimator : estimator
        Any scikit-learn-compatible classifier exposing ``decision_function`` or
        ``predict_proba``. Cloned once per band. Defaults to standardised
        multinomial logistic regression.
    topology : Topology or str
        Interaction graph: ``"chain"`` (published default), ``"complete"``,
        ``"knn"``, or a :class:`~pypasi.topology.Topology`.
    regime : {"plain", "H1", "H2"} or Regime
        Regulatory regime.
    gate : Gate, float or str
        Conflict gate. A :class:`~pypasi.gates.QuantileGate` is calibrated
        automatically during ``fit`` on the training signals.
    weight_scheme : {"confidence", "uniform"}
        Initial influence weights.
    alpha, lam, temperature, max_iter, k_early
        Negotiation hyperparameters.
    divergence : {"js", "kl"}
        Disagreement measure.
    calibration : {"auto", "always", "temperature", "per_band", None}
        How to put band evidence on a log-probability scale. Every downstream
        quantity - confidence weights, stress, the gate threshold, DG - is
        computed from ``softmax`` of the band evidence, so an estimator whose
        scores are not a log-odds (an SVM margin, a PLS-DA response) makes those
        quantities incomparable with an estimator whose scores are.

        ``"auto"`` (default) fits a single temperature when, and only when, the
        base estimator's evidence is *not* already an exact log-odds - so for
        logistic regression or LDA it is a no-op and no training data is held
        back. ``"always"`` calibrates regardless, which is worth doing more
        often than it sounds: an exact log-odds is not a *calibrated* one, and a
        regularised logistic regression shrinks its coefficients enough that the
        fitted temperature on band models of synthetic signals comes out near 7.
        The cost is ``calibration_split`` of the training data, so ``"auto"``
        does not spend it on your behalf.

        ``"temperature"`` is ``"always"`` with one shared temperature,
        ``"per_band"`` fits one per band - which makes each band individually
        calibrated at the cost of flattening how much more informative one band
        is than another - and ``None`` disables it and warns, reproducing 0.4.x
        behaviour exactly.
    calibration_split : float
        Fraction of the training data held out to fit the temperature on.
        Calibrating on the band models' own training spectra is the classic
        mistake: an over-confident model looks calibrated on them. Ignored when
        ``calibration_data`` is given.
    calibration_data : tuple or None
        Explicit ``(X, y)`` never seen by the band models. Preferred over
        ``calibration_split`` when you have a held-out split already.
    random_state : int, Generator or None
        Seed for the Metropolis draws and any base estimator that accepts one.

    Attributes
    ----------
    classes_ : ndarray
        Class labels seen during ``fit``.
    bands_ : BandSet
        The band partition actually used.
    topology_ : Topology
        The resolved interaction graph.
    band_estimators_ : list
        One fitted base learner per band.
    gate_ : Gate
        The resolved (and, for quantile gates, calibrated) gate.
    train_stress_ : ndarray
        Per-band stress on the training signals, shape ``(n_samples, n_bands)``.
        This is the distribution a quantile gate is calibrated against.
    evidence_spec_ : EvidenceSpec
        What kind of class evidence the band models produce and on what scale.
        Check ``evidence_spec_.is_probabilistic`` before comparing conflict
        across estimators.
    calibrator_ : TemperatureCalibrator or None
        The fitted temperature, when one was fitted.

    Examples
    --------
    >>> from pypasi import BandNegotiationClassifier, BandSet
    >>> from pypasi.datasets import make_conflict_signals
    >>> X, y, axis = make_conflict_signals(n_samples=60, random_state=0)
    >>> clf = BandNegotiationClassifier(axis=axis, n_bands=5, random_state=0)
    >>> clf.fit(X, y).predict(X).shape
    (60,)
    """

    def __init__(
        self,
        *,
        bands=None,
        axis=None,
        n_bands: int = 7,
        band_method: str = "equal_width",
        band_range=None,
        axis_name: str = "axis",
        base_estimator=None,
        topology="chain",
        regime="H1",
        gate=None,
        weight_scheme: str = "confidence",
        alpha: float = 0.3,
        lam: float = 1.0,
        temperature: float = 0.01,
        max_iter: int = 30,
        k_early: int = 3,
        divergence: str = "js",
        calibration: str | None = "auto",
        calibration_split: float = 0.25,
        calibration_data=None,
        random_state=None,
    ):
        self.bands = bands
        self.axis = axis
        self.n_bands = n_bands
        self.band_method = band_method
        self.band_range = band_range
        self.axis_name = axis_name
        self.base_estimator = base_estimator
        self.topology = topology
        self.regime = regime
        self.gate = gate
        self.weight_scheme = weight_scheme
        self.alpha = alpha
        self.lam = lam
        self.temperature = temperature
        self.max_iter = max_iter
        self.k_early = k_early
        self.divergence = divergence
        self.calibration = calibration
        self.calibration_split = calibration_split
        self.calibration_data = calibration_data
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def _build_bands(self, X: np.ndarray) -> BandSet:
        if self.bands is not None:
            if not isinstance(self.bands, BandSet):
                raise TypeError("bands must be a BandSet or None")
            return self.bands
        axis = np.arange(X.shape[1], dtype=float) if self.axis is None else np.asarray(self.axis, float)
        if axis.size != X.shape[1]:
            raise ValueError(
                f"axis has {axis.size} values but X has {X.shape[1]} features"
            )
        lo, hi = (None, None) if self.band_range is None else self.band_range
        # A band needs at least one feature, so the partition can never be finer
        # than the signal. Silently narrowing beats failing on a short signal.
        n_bands = max(1, min(int(self.n_bands), X.shape[1]))
        if axis.size == 1 or np.ptp(axis) == 0:
            from .bands import Band
            return BandSet(axis, [Band(0, "B1", float(axis[0]), float(axis[0]),
                                       np.arange(axis.size))],
                           axis_name=self.axis_name)
        if self.band_method == "equal_width":
            return BandSet.equal_width(axis, n_bands, lo=lo, hi=hi, axis_name=self.axis_name)
        if self.band_method == "peak_informed":
            return BandSet.peak_informed(axis, X, n_bands, lo=lo, hi=hi, axis_name=self.axis_name)
        raise ValueError(
            f"unknown band_method {self.band_method!r}; use 'equal_width' or 'peak_informed'"
        )

    def fit(self, X, y):
        """Fit one base learner per band and calibrate the conflict gate."""
        X, y = validate_data(self, X, y, accept_sparse=False, dtype="numeric",
                             ensure_min_samples=2, ensure_min_features=1,
                             multi_output=False, y_numeric=False)
        X = np.asarray(X, dtype=float)
        self.classes_ = unique_labels(y)
        if self.classes_.size < 2:
            raise ValueError("at least two classes are required")
        self.n_iter_ = int(self.max_iter)

        self.bands_ = self._build_bands(X)
        self.topology_ = Topology.resolve(self.topology, self.bands_.n_bands)
        self._divergence_ = get_divergence(self.divergence)
        self._regime_ = resolve_regime(self.regime, alpha=self.alpha)

        base = _default_base_estimator() if self.base_estimator is None else self.base_estimator
        if not supports_logits(base):
            raise TypeError(
                f"{type(base).__name__} exposes neither decision_function nor "
                "predict_proba, so it cannot serve as a band model"
            )

        y_idx = np.searchsorted(self.classes_, y)

        # The calibration split is carved out BEFORE the band models are fitted.
        # A temperature fitted on the band models' own training spectra comes out
        # near one however miscalibrated they are, because an over-confident
        # model is over-confident in exactly the right direction there.
        X_fit, y_fit_idx, X_cal, y_cal_idx = self._calibration_split(X, y_idx)

        self.band_estimators_ = []
        for band in self.bands_:
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=self._seed())
            est.fit(X_fit[:, band.features], y_fit_idx)
            self.band_estimators_.append(est)

        self.calibrator_ = None
        self.evidence_spec_ = self._raw_evidence_spec(X[:1])
        mode = self._calibration_mode()
        if mode is not None and X_cal is not None:
            raw = self._band_logits(X_cal, calibrate=False)
            self.calibrator_ = TemperatureCalibrator(mode=mode).fit(raw, y_cal_idx)
        elif self.calibration is None:
            self.evidence_spec_.warn_if_uncalibrated("this model's conflict signal")

        self.gate_ = resolve_gate("quantile" if self.gate is None else self.gate)
        train_logits = self._band_logits(X)
        self.train_stress_ = self._clean_stress(train_logits)
        if isinstance(self.gate_, QuantileGate):
            self.gate_.fit(self.train_stress_)
        return self

    # ------------------------------------------------------------- evidence

    def _raw_evidence_spec(self, X_probe: np.ndarray) -> EvidenceSpec:
        band = self.bands_[0]
        _, spec = to_evidence(self.band_estimators_[0],
                              X_probe[:, band.features], self.classes_.size)
        return spec

    def _calibration_mode(self) -> str | None:
        """Which temperature mode to fit, or None to leave the scale alone."""
        c = self.calibration
        if c is None:
            return None
        if c in ("temperature", "always"):
            return "shared"
        if c == "per_band":
            return "per_band"
        if c == "auto":
            # An exact log-odds needs no repair, and fitting one anyway would
            # spend data to learn T = 1.
            from .evidence import evidence_kind
            base = _default_base_estimator() if self.base_estimator is None else self.base_estimator
            kind, _ = evidence_kind(base)
            return None if kind == "exact_logit" else "shared"
        raise ValueError(
            f"unknown calibration {c!r}; use 'auto', 'always', 'temperature', "
            "'per_band' or None")

    def _calibration_split(self, X, y_idx):
        """Split off held-out data for the temperature, or return the lot unsplit."""
        if self._calibration_mode() is None:
            return X, y_idx, None, None
        if self.calibration_data is not None:
            Xc, yc = self.calibration_data
            Xc = np.asarray(Xc, dtype=float)
            return X, y_idx, Xc, np.searchsorted(self.classes_, np.asarray(yc))

        frac = float(self.calibration_split)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"calibration_split must lie in (0, 1), got {frac}")
        counts = np.bincount(y_idx, minlength=self.classes_.size)
        # Stratification needs at least two of each class on both sides; when the
        # cohort is too small to give that, keep every spectrum for fitting and
        # leave the evidence uncalibrated rather than crippling the band models.
        if counts.min() < 4 or int(round(frac * len(X))) < self.classes_.size:
            warnings.warn(
                f"only {len(X)} training samples with {counts.min()} in the "
                f"smallest class: too few to hold out a calibration split, so "
                f"band evidence is left on its native scale. Pass "
                f"calibration_data=(X, y) to calibrate anyway.",
                UserWarning, stacklevel=3)
            return X, y_idx, None, None
        from sklearn.model_selection import train_test_split
        idx_fit, idx_cal = train_test_split(
            np.arange(len(X)), test_size=frac, stratify=y_idx,
            random_state=self._seed() or 0)
        return X[idx_fit], y_idx[idx_fit], X[idx_cal], y_idx[idx_cal]

    def _seed(self):
        rs = self.random_state
        return rs if isinstance(rs, (int, np.integer)) or rs is None else None

    # ------------------------------------------------------------ transform

    def _band_logits(self, X: np.ndarray, *, calibrate: bool = True) -> np.ndarray:
        """Stack per-band class evidence into ``(n_samples, n_bands, n_classes)``."""
        check_is_fitted(self, "band_estimators_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype="numeric")
        X = np.asarray(X, dtype=float)
        n_classes = self.classes_.size
        blocks = [
            to_evidence(est, X[:, band.features], n_classes)[0]
            for est, band in zip(self.band_estimators_, self.bands_)
        ]
        L = np.stack(blocks, axis=1)
        cal = getattr(self, "calibrator_", None)
        if calibrate and cal is not None:
            L = cal.transform(L)
        return L

    def band_logits(self, X) -> np.ndarray:
        """Public accessor for the per-band class evidence of ``X``.

        Calibrated when a temperature was fitted; see ``evidence_spec_`` for
        what scale the values are on.
        """
        return self._band_logits(X)

    @property
    def evidence_report(self):
        """One row describing the scale the audit is computed on.

        Worth printing before comparing conflict between two base estimators:
        if their ``kind`` differs, or their ``scale`` differs by a large factor,
        the comparison is measuring units as much as information.
        """
        import pandas as pd

        check_is_fitted(self, "band_estimators_")
        spec = self.evidence_spec_
        cal = getattr(self, "calibrator_", None)
        row = {
            "estimator": spec.estimator,
            "source": spec.source,
            "kind": "calibrated_log_proba" if cal is not None else spec.kind,
            "is_probabilistic": spec.is_probabilistic or cal is not None,
            "temperature": float(np.mean(cal.temperature_)) if cal is not None else 1.0,
            "nll_before": None if cal is None else cal.nll_before_,
            "nll_after": None if cal is None else cal.nll_after_,
        }
        return pd.DataFrame([row])

    def _clean_stress(self, band_logits: np.ndarray) -> np.ndarray:
        """Stress at iteration zero, the reference distribution for calibration."""
        w = initial_weights(band_logits, self.weight_scheme)
        consensus = neighbourhood_consensus(band_logits, w, self.topology_)
        return band_stress(band_logits, consensus, self._divergence_)

    # -------------------------------------------------------------- predict

    def negotiate(self, X, *, regime=None, record_trajectory: bool = True, **kwargs) -> NegotiationResult:
        """Run the negotiation on ``X`` and return the full result.

        ``regime`` overrides the configured one, which is how batch comparisons
        of Plain, H1 and H2 reuse a single fit.
        """
        check_is_fitted(self, "band_estimators_")
        logits = self._band_logits(X)
        reg = self._regime_ if regime is None else resolve_regime(regime, alpha=self.alpha)
        params = dict(
            topology=self.topology_,
            regime=reg,
            gate=self.gate_ if reg.uses_gate else None,
            weights=self.weight_scheme,
            lam=self.lam,
            temperature=self.temperature,
            max_iter=self.max_iter,
            divergence=self._divergence_,
            record_trajectory=record_trajectory,
            k_early=self.k_early,
            random_state=self.random_state,
        )
        params.update(kwargs)
        return negotiate(logits, **params)

    def predict_proba(self, X):
        """Class probabilities from the negotiated consensus."""
        return self.negotiate(X, record_trajectory=False).proba

    def predict(self, X):
        """Predicted class labels."""
        check_is_fitted(self, "band_estimators_")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def decision_function(self, X):
        """Aggregated class evidence, before the softmax."""
        res = self.negotiate(X, record_trajectory=False)
        w = res.weights_final
        l = np.einsum("nk,nkc->nc", w, res.logits_final) / (w.sum(axis=1)[:, None] + 1e-12)
        return l[:, 1] - l[:, 0] if self.classes_.size == 2 else l

    # ---------------------------------------------------------------- audit

    def audit(self, X, y=None, **kwargs):
        """Run the negotiation and wrap it in an :class:`~pypasi.audit.AuditResult`."""
        from .audit import AuditResult

        res = self.negotiate(X, record_trajectory=True, **kwargs)
        return AuditResult(res, self.bands_, X=np.asarray(X, dtype=float), y_true=y, classes=self.classes_)

    def band_importance(self, X, y, estimator=None, **kwargs):
        """Per-band discriminative importance, see :func:`pypasi.audit.band_importance`.

        Returns ``(table, per_feature_importance)``. Pair it with
        :meth:`audit` to contrast where the signal is informative against where
        inference is unstable.
        """
        from .audit import band_importance as _bi

        check_is_fitted(self, "bands_")
        return _bi(self.bands_, X, y, estimator, **kwargs)

    def compare_regimes(self, X, y=None, regimes=("plain", "H1", "H2"), **kwargs):
        """Audit the same fit under several regimes.

        Returns
        -------
        dict
            Regime name to :class:`~pypasi.audit.AuditResult`.
        """
        return {str(r): self.audit(X, y, regime=r, **kwargs) for r in regimes}

    def _more_tags(self):  # pragma: no cover - sklearn plumbing
        return {"requires_y": True, "poor_score": True}

    def __sklearn_tags__(self):  # pragma: no cover - sklearn >= 1.6 plumbing
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags
