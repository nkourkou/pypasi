"""Batched softmax and divergences.

Everything here is vectorised over leading axes so that a whole cohort of
samples can be negotiated at once. Shapes are written as ``(..., C)`` where the
trailing axis indexes classes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["softmax", "kl_divergence", "js_divergence", "center", "DIV_EPS"]

DIV_EPS = 1e-12


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def center(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove the per-sample mean across classes.

    The softmax is invariant to adding a constant to every class score, so that
    direction carries no information about the decision. Removing it is what
    makes trajectory length a meaningful geometric quantity.
    """
    z = np.asarray(logits, dtype=float)
    return z - z.mean(axis=axis, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = DIV_EPS) -> np.ndarray:
    """Kullback-Leibler divergence ``KL(p || q)``, batched over leading axes."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = DIV_EPS) -> np.ndarray:
    """Jensen-Shannon divergence, batched over leading axes.

    Symmetric, bounded above by ``log 2``, and finite even when the two
    distributions have disjoint support - which is why it is the default
    measure of disagreement between band agents.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def get_divergence(name: str):
    """Look up a divergence by name (``"js"`` or ``"kl"``)."""
    key = str(name).lower()
    if key in ("js", "jensen-shannon", "jensenshannon"):
        return js_divergence
    if key in ("kl", "kullback-leibler"):
        return _symmetric_kl
    raise ValueError(f"unknown divergence {name!r}; use 'js' or 'kl'")


def _symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = DIV_EPS) -> np.ndarray:
    """Symmetrised KL, ``KL(p||q) + KL(q||p)``, for use as a distance."""
    return kl_divergence(p, q, eps) + kl_divergence(q, p, eps)
