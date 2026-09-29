"""Conflict gates: deciding which bands are too discordant to keep their say.

A gate maps per-band stress to a boolean mute mask. Three are provided.

:class:`AbsoluteGate`
    Mute where stress exceeds a fixed threshold. Faithful to the published
    definition, but stress is a divergence between softmax distributions, so its
    scale depends on the number of classes and the sharpness of the base
    learner. A threshold tuned on one dataset rarely transfers to another.
:class:`QuantileGate`
    Set that threshold from data: a quantile of the stress distribution over
    undamaged training signals. Scale-free across datasets, and the default.
:class:`RankGate`
    Mute a fixed proportion of each signal's own bands. Fully dimensionless and
    needs no calibration at all.

All gates accept ``protect_min``, the number of least-discordant bands that can
never be muted. It defaults to 1: if every band were muted the weight vector
would be all zeros, the aggregated evidence would be the zero vector, and the
predicted class would fall out of an arbitrary tie-break rather than out of
inference. Set ``protect_min=0`` to reproduce that unguarded behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Gate", "AbsoluteGate", "QuantileGate", "RankGate", "resolve_gate"]


class Gate(ABC):
    """Base class for conflict gates."""

    def __init__(self, protect_min: int = 1):
        if protect_min < 0:
            raise ValueError("protect_min must be >= 0")
        self.protect_min = int(protect_min)

    @abstractmethod
    def _raw_mask(self, stress: np.ndarray) -> np.ndarray:
        """Return the unguarded mute mask for ``stress`` of shape ``(n, k)``."""

    def fit(self, stress: np.ndarray) -> "Gate":
        """Calibrate on a reference stress distribution. No-op for most gates."""
        return self

    def mask(self, stress: np.ndarray) -> np.ndarray:
        """Boolean mute mask of shape ``(n_samples, n_bands)``."""
        stress = np.asarray(stress, dtype=float)
        if stress.ndim != 2:
            raise ValueError(f"stress must be 2-D (n_samples, n_bands), got {stress.shape}")
        m = self._raw_mask(stress)
        if self.protect_min > 0:
            m = _protect_quietest(m, stress, self.protect_min)
        return m

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self._describe()})"

    def _describe(self) -> str:
        return f"protect_min={self.protect_min}"


def _protect_quietest(mask: np.ndarray, stress: np.ndarray, keep: int) -> np.ndarray:
    """Un-mute the ``keep`` least-stressed bands of any fully-muted sample."""
    mask = mask.copy()
    n_bands = mask.shape[1]
    keep = min(keep, n_bands)
    fully = mask.all(axis=1)
    if not np.any(fully):
        return mask
    rows = np.nonzero(fully)[0]
    quietest = np.argsort(stress[rows], axis=1)[:, :keep]
    for r, cols in zip(rows, quietest):
        mask[r, cols] = False
    return mask


class AbsoluteGate(Gate):
    """Mute bands whose stress exceeds a fixed threshold ``tau``."""

    def __init__(self, tau: float, protect_min: int = 1):
        super().__init__(protect_min)
        self.tau = float(tau)

    def _raw_mask(self, stress: np.ndarray) -> np.ndarray:
        return stress > self.tau

    def _describe(self) -> str:
        return f"tau={self.tau:g}, protect_min={self.protect_min}"


class QuantileGate(Gate):
    """Threshold set to a quantile of a reference stress distribution.

    Call :meth:`fit` with stress computed on undamaged training signals; the
    resulting ``tau_`` is the ``q``-quantile of that distribution. With
    ``q=0.975`` roughly one band in forty is muted on clean input, and anything
    muted beyond that reflects genuine disagreement rather than the ambient
    level of disagreement the base learners always show.

    Attributes
    ----------
    tau_ : float
        The fitted threshold. ``None`` until :meth:`fit` is called.
    """

    def __init__(self, q: float = 0.975, protect_min: int = 1):
        super().__init__(protect_min)
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie strictly between 0 and 1")
        self.q = float(q)
        self.tau_: float | None = None

    def fit(self, stress: np.ndarray) -> "QuantileGate":
        s = np.asarray(stress, dtype=float).reshape(-1)
        s = s[np.isfinite(s)]
        if s.size == 0:
            raise ValueError("cannot calibrate a QuantileGate on an empty stress sample")
        self.tau_ = float(np.quantile(s, self.q))
        return self

    def _raw_mask(self, stress: np.ndarray) -> np.ndarray:
        if self.tau_ is None:
            raise RuntimeError(
                "QuantileGate must be fitted before use. The estimator does this "
                "for you during fit(); call gate.fit(reference_stress) if you are "
                "driving the negotiation directly."
            )
        return stress > self.tau_

    def _describe(self) -> str:
        tau = "unfitted" if self.tau_ is None else f"{self.tau_:.4g}"
        return f"q={self.q:g}, tau_={tau}, protect_min={self.protect_min}"


class RankGate(Gate):
    """Mute the most discordant ``fraction`` of each signal's own bands.

    Dimensionless and calibration-free: the threshold is implicit and adapts to
    every signal separately. Collapse is impossible as long as ``fraction < 1``.
    """

    def __init__(self, fraction: float = 0.25, protect_min: int = 1):
        super().__init__(protect_min)
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction must lie in [0, 1)")
        self.fraction = float(fraction)

    def _raw_mask(self, stress: np.ndarray) -> np.ndarray:
        n, k = stress.shape
        n_mute = int(np.floor(self.fraction * k))
        mask = np.zeros((n, k), dtype=bool)
        if n_mute == 0:
            return mask
        loudest = np.argsort(stress, axis=1)[:, -n_mute:]
        np.put_along_axis(mask, loudest, True, axis=1)
        return mask

    def _describe(self) -> str:
        return f"fraction={self.fraction:g}, protect_min={self.protect_min}"


def resolve_gate(spec) -> Gate:
    """Coerce a user-supplied gate specification into a :class:`Gate`.

    Accepts a :class:`Gate`, a float (read as an absolute threshold), or the
    strings ``"quantile"`` and ``"rank"``.
    """
    if isinstance(spec, Gate):
        return spec
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return AbsoluteGate(float(spec))
    if isinstance(spec, str):
        key = spec.lower()
        if key == "quantile":
            return QuantileGate()
        if key == "rank":
            return RankGate()
        if key == "absolute":
            raise ValueError("AbsoluteGate needs a threshold; pass AbsoluteGate(tau=...)")
    raise TypeError(
        f"cannot interpret {spec!r} as a gate; pass a Gate instance, a float "
        "threshold, or 'quantile'/'rank'"
    )
