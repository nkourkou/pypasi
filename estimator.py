"""The scikit-learn estimator.

:class:`BandNegotiationClassifier` wraps the whole pipeline - band partitioning,
per-band base learners, negotiated consensus - behind ``fit``/``predict``, so it
drops into ``Pipeline``, ``cross_val_score`` and ``GridSearchCV`` unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from .bands import BandSet
from .divergence import get_divergence, js_divergence
from .gates import Gate, QuantileGate, resolve_gate
from .logits import supports_logits, to_logits
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
        self.band_estimators_ = []
        for band in self.bands_:
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=self._seed())
            est.fit(X[:, band.features], y_idx)
            self.band_estimators_.append(est)

        self.gate_ = resolve_gate("quantile" if self.gate is None else self.gate)
        train_logits = self._band_logits(X)
        self.train_stress_ = self._clean_stress(train_logits)
        if isinstance(self.gate_, QuantileGate):
            self.gate_.fit(self.train_stress_)
        return self

    def _seed(self):
        rs = self.random_state
        return rs if isinstance(rs, (int, np.integer)) or rs is None else None

    # ------------------------------------------------------------ transform

    def _band_logits(self, X: np.ndarray) -> np.ndarray:
        """Stack per-band class evidence into ``(n_samples, n_bands, n_classes)``."""
        check_is_fitted(self, "band_estimators_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype="numeric")
        X = np.asarray(X, dtype=float)
        n_classes = self.classes_.size
        blocks = [
            to_logits(est, X[:, band.features], n_classes)
            for est, band in zip(self.band_estimators_, self.bands_)
        ]
        return np.stack(blocks, axis=1)

    def band_logits(self, X) -> np.ndarray:
        """Public accessor for the per-band class evidence of ``X``."""
        return self._band_logits(X)

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
