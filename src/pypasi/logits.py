"""A uniform class-evidence interface for arbitrary scikit-learn estimators.

Band agents exchange *logits*: unnormalised, real-valued class scores whose
differences are meaningful and whose softmax is a class distribution. Getting
those out of an arbitrary estimator is less obvious than it looks, and the
details matter because the conflict gate compares divergences on a fixed scale.

Two traps this module exists to avoid.

**Not every estimator has ``decision_function``.** Random forests, naive Bayes
and k-NN expose only ``predict_proba``. We fall back to ``log(p)``, which is a
valid logit vector up to an additive constant - and the negotiation is invariant
to additive constants, because logits are mean-centred before any geometry is
computed.

**Binary ``decision_function`` returns one number, not two.** The natural-looking
``[-d, +d]`` doubles the scale of a proper two-class logit vector, making the
softmax twice as peaked and every divergence larger, so a threshold tuned on a
multi-class problem no longer transfers. We use ``[0, d]``, which reproduces the
model's own ``predict_proba`` exactly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["to_logits", "supports_logits", "LOGIT_EPS"]

LOGIT_EPS = 1e-12


def supports_logits(estimator) -> bool:
    """Whether ``estimator`` can produce class evidence at all."""
    return hasattr(estimator, "decision_function") or hasattr(estimator, "predict_proba")


def to_logits(estimator, X: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Extract an ``(n_samples, n_classes)`` logit matrix from a fitted estimator.

    Parameters
    ----------
    estimator
        Any fitted scikit-learn-compatible classifier exposing
        ``decision_function`` or ``predict_proba``.
    X
        Design matrix for the band this estimator was fitted on.
    n_classes
        Expected number of classes. When given, the result is validated against
        it, which catches the common case of a band whose training fold happened
        to contain only a subset of the classes.

    Returns
    -------
    numpy.ndarray
        Logits of shape ``(n_samples, n_classes)``.

    Notes
    -----
    Logits are defined only up to an additive per-sample constant. Callers that
    compare them geometrically should mean-centre first; :mod:`pypasi.geometry`
    does this for you.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    if hasattr(estimator, "decision_function"):
        d = np.asarray(estimator.decision_function(X), dtype=float)
        if d.ndim == 1:
            # Binary: decision_function is log(p1 / p0). The two-class logit
            # vector [0, d] reproduces predict_proba exactly; [-d, +d] would
            # double the scale and inflate every divergence.
            L = np.column_stack([np.zeros_like(d), d])
        elif d.ndim == 2:
            L = d
        else:
            raise ValueError(
                f"decision_function returned shape {d.shape}; expected 1-D or 2-D"
            )
    elif hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(X), dtype=float)
        if p.ndim != 2:
            raise ValueError(
                f"predict_proba returned shape {p.shape}; expected 2-D"
            )
        L = np.log(np.clip(p, LOGIT_EPS, None))
    else:
        raise TypeError(
            f"{type(estimator).__name__} exposes neither decision_function nor "
            "predict_proba, so it cannot provide class evidence"
        )

    if not np.all(np.isfinite(L)):
        raise ValueError(
            f"{type(estimator).__name__} produced non-finite logits; check the "
            "band for constant or degenerate features"
        )
    if n_classes is not None and L.shape[1] != n_classes:
        raise ValueError(
            f"{type(estimator).__name__} produced {L.shape[1]} class scores but "
            f"{n_classes} classes were expected. A band-level model that never "
            "saw every class cannot contribute comparable evidence; consider "
            "wider bands or stratified folds."
        )
    return L
