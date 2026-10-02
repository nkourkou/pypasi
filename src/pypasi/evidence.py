"""A typed, comparable interface to class evidence.

Band agents exchange *logits*: real-valued class scores whose differences are
meaningful and whose softmax is a class distribution. Every quantity downstream
of that - confidence weights, stress, the gate threshold, DG - is computed on
that scale, so a scale that means different things for different estimators
makes those quantities incomparable across estimators, and sometimes across
bands.

The problem this module exists to fix
-------------------------------------
:func:`pypasi.logits.to_logits` prefers ``decision_function``. For some
estimators that is exactly a log-odds vector and the softmax reproduces the
model's own probabilities:

* binary and multinomial :class:`~sklearn.linear_model.LogisticRegression`
* :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`, whose
  discriminant is the log posterior up to an additive constant

For others it is not, and softmax over it is a monotone squash with an arbitrary
temperature:

* an SVM margin lives in scaled-feature units and has no probabilistic meaning
* a PLS-DA response is a regression output onto one-hot targets

The difference is not cosmetic. On synthetic conflict signals the mean-centred
logit scale spans a factor of twenty between PLS-DA (0.39) and PCA-LDA (7.86),
and mean band confidence follows it from 0.51 to 0.88. A saturated softmax has
little room left to disagree, so *the estimator with the gentler evidence scale
will look like the one whose bands disagree most informatively* - which is a
statement about units, not about information.

What this module provides
-------------------------
:func:`to_evidence` returns both the logits and an :class:`EvidenceSpec` saying
what kind of thing they are, so callers can refuse, warn, or correct. And
:class:`TemperatureCalibrator` fits a single scalar temperature per estimator on
held-out data, which puts the evidence on a genuine log-probability scale
without flattening the differences *between* bands.

Why one shared temperature rather than per-band calibration
-----------------------------------------------------------
Per-band isotonic or Platt scaling makes each band's probabilities
well-calibrated *conditional on that band* - which normalises away exactly how
informative the band is. "Band 3 is genuinely unsure here" is the signal the
audit consumes, not noise to be removed. A single temperature per estimator puts
every band in the same units while preserving their relative informativeness.
Per-band calibration is available through ``mode="per_band"`` for callers who
want it, and it is not the default.

Empirically, on synthetic conflict signals, temperature calibration cuts the
negative log-likelihood of PLS-DA's band evidence from 0.739 to 0.348 (optimal
temperature 5.4) while leaving PCA-LDA and logistic regression almost untouched
(0.285 to 0.283, 0.323 to 0.301) - which is what you would expect if the first
is not a log-odds scale and the last two are.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EvidenceSpec",
    "EVIDENCE_KINDS",
    "to_evidence",
    "evidence_kind",
    "TemperatureCalibrator",
    "UncalibratedEvidenceWarning",
]

_EPS = 1e-12

#: Evidence kinds, ordered from most to least trustworthy as a log-probability.
#:
#: ``"exact_logit"`` means the softmax of these values reproduces the
#: estimator's own ``predict_proba``. It does **not** mean the estimator is well
#: calibrated. An L2-regularised logistic regression shrinks its coefficients,
#: so its log-odds come out systematically too small - on band models of
#: synthetic conflict signals the optimal temperature is around 7, and the
#: negative log-likelihood falls from 0.66 to 0.17 once it is applied. Exactness
#: is a statement about the relationship between ``decision_function`` and
#: ``predict_proba``; calibration is a statement about the relationship between
#: either of them and reality. Use ``calibration="always"`` to repair the second
#: for an estimator that already satisfies the first.
EVIDENCE_KINDS = (
    "calibrated_log_proba",   # temperature or per-band calibration was fitted
    "log_proba",              # log of the estimator's own predict_proba
    "exact_logit",            # softmax reproduces predict_proba; not calibrated
    "uncalibrated_score",     # decision_function with no probabilistic meaning
)


class UncalibratedEvidenceWarning(UserWarning):
    """Raised when band evidence is used on a scale that is not a log-odds."""


#: Estimator classes whose ``decision_function`` is a log-odds (binary) or a log
#: posterior up to an additive constant (multiclass), so that ``softmax`` of it
#: reproduces ``predict_proba``. Matched by class name to avoid importing
#: optional dependencies.
_EXACT_LOGIT_TYPES = frozenset({
    "LogisticRegression",
    "LogisticRegressionCV",
    "SGDClassifier",                 # only with loss='log_loss'; checked below
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "PCALDA",                        # pypasi: delegates to LinearDiscriminantAnalysis
})

#: Estimator classes whose ``decision_function`` is explicitly NOT a log-odds.
_KNOWN_SCORE_TYPES = frozenset({
    "LinearSVC", "SVC", "NuSVC", "RidgeClassifier", "RidgeClassifierCV",
    "Perceptron", "PassiveAggressiveClassifier", "PLSDA",
})


def _terminal(estimator):
    """The estimator that actually produces the scores, unwrapping pipelines."""
    step = estimator
    for _ in range(8):
        if hasattr(step, "steps"):              # sklearn Pipeline
            step = step.steps[-1][1]
        elif hasattr(step, "best_estimator_"):  # fitted search object
            step = step.best_estimator_
        else:
            break
    return step


@dataclass(frozen=True)
class EvidenceSpec:
    """What kind of class evidence an estimator produced, and on what scale.

    Attributes
    ----------
    kind
        One of :data:`EVIDENCE_KINDS`.
    source
        The method the values came from: ``"decision_function"`` or
        ``"predict_proba"``.
    estimator
        Class name of the estimator that produced them.
    temperature
        The multiplicative factor applied, if any. ``1.0`` when untouched.
    scale
        Standard deviation of the mean-centred evidence, recorded so that two
        estimators can be compared on it rather than on vibes.

    Notes
    -----
    ``is_probabilistic`` is the question most callers actually want: may I treat
    ``softmax(logits)`` as this band's class distribution?
    """

    kind: str
    source: str
    estimator: str
    temperature: float = 1.0
    scale: float = float("nan")

    @property
    def is_probabilistic(self) -> bool:
        return self.kind != "uncalibrated_score"

    @property
    def is_calibrated(self) -> bool:
        return self.kind == "calibrated_log_proba"

    def warn_if_uncalibrated(self, context: str = "this analysis") -> None:
        """Emit :class:`UncalibratedEvidenceWarning` when the scale is arbitrary."""
        if self.is_probabilistic:
            return
        warnings.warn(
            f"{self.estimator}.{self.source} is not a log-odds scale, so the "
            f"softmax over it has an arbitrary temperature. Confidence weights, "
            f"stress, the gate threshold and DG are all computed on that scale, "
            f"so {context} is not comparable across estimators. Fit a "
            f"TemperatureCalibrator on held-out data, or pass "
            f"calibration='temperature' to BandNegotiationClassifier.",
            UncalibratedEvidenceWarning, stacklevel=3)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        t = "" if self.temperature == 1.0 else f", T={self.temperature:.3g}"
        return f"{self.kind} from {self.estimator}.{self.source}{t}"


def evidence_kind(estimator) -> tuple[str, str]:
    """Classify what an estimator's evidence means, without calling it.

    Returns
    -------
    tuple
        ``(kind, source)``, where ``kind`` is one of :data:`EVIDENCE_KINDS`
        excluding ``"calibrated_log_proba"`` (which only a calibrator can
        confer) and ``source`` names the method to call.
    """
    term = _terminal(estimator)
    name = type(term).__name__

    if hasattr(estimator, "decision_function"):
        if name == "SGDClassifier":
            # only the log-loss variant is a log-odds scale
            loss = getattr(term, "loss", None)
            kind = "exact_logit" if loss == "log_loss" else "uncalibrated_score"
            return kind, "decision_function"
        if name in _EXACT_LOGIT_TYPES:
            return "exact_logit", "decision_function"
        if name in _KNOWN_SCORE_TYPES:
            return "uncalibrated_score", "decision_function"
        # Unknown estimator exposing both: predict_proba is at least on a
        # probability scale, so prefer it over an unidentified margin.
        if hasattr(estimator, "predict_proba"):
            return "log_proba", "predict_proba"
        return "uncalibrated_score", "decision_function"

    if hasattr(estimator, "predict_proba"):
        return "log_proba", "predict_proba"

    raise TypeError(
        f"{name} exposes neither decision_function nor predict_proba, so it "
        "cannot provide class evidence")


def _raw_evidence(estimator, X: np.ndarray, source: str) -> np.ndarray:
    if source == "decision_function":
        d = np.asarray(estimator.decision_function(X), dtype=float)
        if d.ndim == 1:
            # Binary: [0, d] reproduces predict_proba exactly where the scale is
            # a log-odds. [-d, +d] would double it and inflate every divergence.
            return np.column_stack([np.zeros_like(d), d])
        if d.ndim == 2:
            return d
        raise ValueError(f"decision_function returned shape {d.shape}; expected 1-D or 2-D")

    p = np.asarray(estimator.predict_proba(X), dtype=float)
    if p.ndim != 2:
        raise ValueError(f"predict_proba returned shape {p.shape}; expected 2-D")
    return np.log(np.clip(p, _EPS, None))


def to_evidence(
    estimator,
    X: np.ndarray,
    n_classes: int | None = None,
    *,
    calibrator: "TemperatureCalibrator | None" = None,
    centre: bool = True,
) -> tuple[np.ndarray, EvidenceSpec]:
    """Extract class evidence together with a description of what it is.

    Parameters
    ----------
    estimator
        Any fitted scikit-learn-compatible classifier.
    X
        Design matrix for the band this estimator was fitted on.
    n_classes
        Expected number of classes; validated when given.
    calibrator
        A fitted :class:`TemperatureCalibrator`. When supplied, its temperature
        is applied and the resulting kind is ``"calibrated_log_proba"``.
    centre
        Mean-centre the evidence across classes. Logits are defined only up to
        an additive per-sample constant and the negotiation is invariant to it,
        but centring makes ``scale`` comparable between estimators.

    Returns
    -------
    tuple
        ``(logits, spec)`` with logits of shape ``(n_samples, n_classes)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    kind, source = evidence_kind(estimator)
    L = _raw_evidence(estimator, X, source)

    if not np.all(np.isfinite(L)):
        raise ValueError(
            f"{type(estimator).__name__} produced non-finite evidence; check the "
            "band for constant or degenerate features")
    if n_classes is not None and L.shape[1] != n_classes:
        raise ValueError(
            f"{type(estimator).__name__} produced {L.shape[1]} class scores but "
            f"{n_classes} classes were expected. A band-level model that never "
            "saw every class cannot contribute comparable evidence; consider "
            "wider bands or stratified folds.")

    if centre:
        L = L - L.mean(axis=-1, keepdims=True)

    temperature = 1.0
    if calibrator is not None:
        L = calibrator.transform(L)
        temperature = float(calibrator.temperature_)
        kind = "calibrated_log_proba"

    spec = EvidenceSpec(
        kind=kind,
        source=source,
        estimator=type(_terminal(estimator)).__name__,
        temperature=temperature,
        scale=float(np.std(L - L.mean(axis=-1, keepdims=True))),
    )
    return L, spec


class TemperatureCalibrator:
    """One scalar temperature, fitted by held-out negative log-likelihood.

    Multiplying mean-centred evidence by ``T`` and taking the softmax is the
    standard temperature-scaling correction. It is the mildest calibration that
    exists - a single parameter, monotone, order-preserving - which is what
    makes it the right one here: it repairs the *units* of the evidence without
    touching the *ranking*, so a band's predictions are unchanged and only the
    sharpness of its distribution moves.

    Parameters
    ----------
    mode : {"shared", "per_band"}
        ``"shared"`` fits one temperature across all bands, preserving how much
        more informative one band is than another. ``"per_band"`` fits one per
        band, which makes each band individually well-calibrated at the cost of
        flattening exactly that difference. Default ``"shared"``; see the module
        docstring for why.
    bounds : tuple
        Search range for ``log(T)``.

    Attributes
    ----------
    temperature_ : float or ndarray
        The fitted temperature: scalar for ``"shared"``, one per band otherwise.
    nll_before_, nll_after_ : float
        Mean negative log-likelihood of the band evidence on the calibration
        data, before and after. A large gap means the raw scale was not a
        log-odds; a small one means it already was.

    Examples
    --------
    >>> import numpy as np
    >>> from pypasi.evidence import TemperatureCalibrator
    >>> rng = np.random.default_rng(0)
    >>> L = rng.normal(size=(200, 3, 2)) * 0.2          # far too flat
    >>> y = (L.mean(axis=1)[:, 1] > 0).astype(int)
    >>> cal = TemperatureCalibrator().fit(L, y)
    >>> cal.nll_after_ <= cal.nll_before_ + 1e-9
    True
    """

    def __init__(self, mode: str = "shared", bounds: tuple[float, float] = (-5.0, 5.0)):
        if mode not in ("shared", "per_band"):
            raise ValueError(f"unknown mode {mode!r}; use 'shared' or 'per_band'")
        self.mode = mode
        self.bounds = bounds
        self.temperature_ = None
        self.nll_before_: float | None = None
        self.nll_after_: float | None = None

    # ------------------------------------------------------------------ core

    @staticmethod
    def _nll(L: np.ndarray, y_idx: np.ndarray, T: float) -> float:
        """Mean NLL over samples and bands of ``softmax(T * L)``."""
        Z = L * T
        Z = Z - Z.max(axis=-1, keepdims=True)
        logsum = np.log(np.exp(Z).sum(axis=-1))
        picked = np.take_along_axis(Z, y_idx[:, None, None], axis=-1)[..., 0]
        return float(np.mean(logsum - picked))

    def _fit_one(self, L: np.ndarray, y_idx: np.ndarray) -> float:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            lambda lt: self._nll(L, y_idx, float(np.exp(lt))),
            bounds=self.bounds, method="bounded")
        return float(np.exp(res.x))

    def fit(self, band_logits: np.ndarray, y_idx) -> "TemperatureCalibrator":
        """Fit on evidence the band models did not see.

        Parameters
        ----------
        band_logits
            Mean-centred evidence, shape ``(n_samples, n_bands, n_classes)``.
        y_idx
            Class *indices* (not labels) for those samples.

        Notes
        -----
        Calibrating on the data the band models were fitted on is the classic
        mistake: an over-confident model looks well calibrated on its own
        training set, and the temperature comes out near one. Use a held-out
        split. :class:`~pypasi.estimator.BandNegotiationClassifier` does this
        for you with an internal split when you do not supply one.
        """
        L = np.asarray(band_logits, dtype=float)
        if L.ndim != 3:
            raise ValueError(
                f"band_logits must be (n_samples, n_bands, n_classes), got {L.shape}")
        L = L - L.mean(axis=-1, keepdims=True)
        y_idx = np.asarray(y_idx, dtype=int).reshape(-1)
        if y_idx.size != L.shape[0]:
            raise ValueError(
                f"y_idx has {y_idx.size} entries for {L.shape[0]} samples")
        if y_idx.min() < 0 or y_idx.max() >= L.shape[-1]:
            raise ValueError("y_idx must contain class indices, not class labels")

        self.nll_before_ = self._nll(L, y_idx, 1.0)
        if self.mode == "shared":
            self.temperature_ = self._fit_one(L, y_idx)
        else:
            self.temperature_ = np.array(
                [self._fit_one(L[:, [k], :], y_idx) for k in range(L.shape[1])])
        self.nll_after_ = self._nll(L, y_idx, 1.0) if self.temperature_ is None else (
            self._nll(L, y_idx, self.temperature_) if self.mode == "shared"
            else float(np.mean([self._nll(L[:, [k], :], y_idx, self.temperature_[k])
                                for k in range(L.shape[1])])))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature.

        Accepts either a single band's ``(n_samples, n_classes)`` evidence or a
        full ``(n_samples, n_bands, n_classes)`` stack. In ``"per_band"`` mode a
        single band cannot be identified, so only the stacked form is allowed.
        """
        if self.temperature_ is None:
            raise RuntimeError("TemperatureCalibrator is not fitted")
        L = np.asarray(logits, dtype=float)
        if self.mode == "shared":
            return L * float(self.temperature_)
        if L.ndim != 3:
            raise ValueError(
                "per-band calibration needs the full (n_samples, n_bands, "
                "n_classes) stack; a single band cannot be identified")
        if L.shape[1] != len(self.temperature_):
            raise ValueError(
                f"calibrator was fitted for {len(self.temperature_)} bands, "
                f"got {L.shape[1]}")
        return L * np.asarray(self.temperature_, float)[None, :, None]

    @property
    def improvement(self) -> float:
        """NLL reduction on the calibration data. Large means the scale was wrong."""
        if self.nll_before_ is None:
            raise RuntimeError("TemperatureCalibrator is not fitted")
        return float(self.nll_before_ - self.nll_after_)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.temperature_ is None:
            return f"TemperatureCalibrator(mode={self.mode!r}, unfitted)"
        t = (f"{self.temperature_:.3g}" if self.mode == "shared"
             else f"{np.min(self.temperature_):.3g}-{np.max(self.temperature_):.3g}")
        return (f"TemperatureCalibrator(mode={self.mode!r}, T={t}, "
                f"NLL {self.nll_before_:.3f} -> {self.nll_after_:.3f})")
