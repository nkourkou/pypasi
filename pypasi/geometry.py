"""Trajectory descriptors: Decision Geometry and its early-phase variants.

Given the recorded logit trajectory of a negotiation, these functions summarise
*how* the decision was reached rather than what it was.

``DG``
    Total reconciliation effort - the cumulative Euclidean path length of all
    band agents in mean-centred logit space.
``REDG``
    Fraction of that effort spent in the first few iterations. High values mean
    the system settled quickly; low values mean prolonged renegotiation.
``eREDG``
    REDG with trivially inactive trajectories masked out, so that the ratio is
    only read where there was real reconciliation to measure.
"""

from __future__ import annotations

import numpy as np

from .divergence import center

__all__ = [
    "step_lengths",
    "band_step_lengths",
    "decision_geometry",
    "band_decision_geometry",
    "early_decision_geometry",
    "redg",
    "eredg",
    "activity_threshold",
]


def _as_trajectory(l_hist: np.ndarray) -> np.ndarray:
    """Validate a trajectory of shape ``(n_samples, n_steps + 1, n_bands, n_classes)``."""
    a = np.asarray(l_hist, dtype=float)
    if a.ndim == 3:  # single sample: (T+1, K, C)
        a = a[None, ...]
    if a.ndim != 4:
        raise ValueError(
            "trajectory must have shape (n_samples, n_steps+1, n_bands, n_classes) "
            f"or (n_steps+1, n_bands, n_classes); got {np.shape(l_hist)}"
        )
    return a


def band_step_lengths(l_hist: np.ndarray) -> np.ndarray:
    """Per-band, per-step displacement in centred logit space.

    Returns an array of shape ``(n_samples, n_steps, n_bands)``.
    """
    a = _as_trajectory(l_hist)
    centred = center(a, axis=-1)
    return np.linalg.norm(np.diff(centred, axis=1), axis=-1)


def step_lengths(l_hist: np.ndarray) -> np.ndarray:
    """Total displacement across bands at each step, shape ``(n_samples, n_steps)``."""
    return band_step_lengths(l_hist).sum(axis=-1)


def decision_geometry(l_hist: np.ndarray) -> np.ndarray:
    """Total reconciliation effort per sample, shape ``(n_samples,)``."""
    return step_lengths(l_hist).sum(axis=-1)


def band_decision_geometry(l_hist: np.ndarray) -> np.ndarray:
    """Each band's contribution to DG, shape ``(n_samples, n_bands)``.

    This is what makes DG attributable: a sample with high DG can be decomposed
    into the bands that did the moving.
    """
    return band_step_lengths(l_hist).sum(axis=1)


def early_decision_geometry(l_hist: np.ndarray, k_early: int = 3) -> np.ndarray:
    """Reconciliation effort in the first ``k_early`` steps, shape ``(n_samples,)``."""
    s = step_lengths(l_hist)
    k = min(int(k_early), s.shape[1])
    return s[:, :k].sum(axis=-1)


def redg(l_hist: np.ndarray, k_early: int = 3, eps: float = 1e-12) -> np.ndarray:
    """Regulated Early Decision Geometry: early effort as a fraction of total."""
    total = decision_geometry(l_hist)
    early = early_decision_geometry(l_hist, k_early)
    return early / (total + eps)


def activity_threshold(dg: np.ndarray, q: float = 0.2) -> float:
    """The ``q``-quantile of a DG distribution, used to mask inactive trajectories."""
    dg = np.asarray(dg, dtype=float)
    finite = dg[np.isfinite(dg)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def eredg(
    l_hist: np.ndarray,
    k_early: int = 3,
    *,
    gate_quantile: float = 0.2,
    threshold: float | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Effective REDG, with inactive trajectories set to NaN.

    A trajectory that barely moves has a REDG dominated by numerical noise, so
    the ratio is only defined where ``DG`` exceeds an activity threshold. That
    threshold is a quantile of the DG distribution over the samples supplied,
    unless one is given explicitly.

    Returns
    -------
    values : numpy.ndarray
        eREDG per sample, NaN where the trajectory was inactive.
    threshold : float
        The activity threshold actually used.
    """
    dg = decision_geometry(l_hist)
    early = early_decision_geometry(l_hist, k_early)
    delta = activity_threshold(dg, gate_quantile) if threshold is None else float(threshold)
    ratio = early / (dg + eps)
    return np.where(dg > delta, ratio, np.nan), delta
