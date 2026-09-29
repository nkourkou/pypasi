"""Regulatory regimes: how a band reacts to its neighbourhood.

Each regime proposes, for every band, an updated evidence vector and an updated
influence weight. All three published regimes share a diffusive move toward the
neighbourhood consensus; they differ in what happens to bands the gate has
flagged as discordant.

:class:`Plain`
    No regulation. Every band diffuses; weights never change.
:class:`H1`
    Gated suppression. A flagged band is silenced and adopts the consensus.
:class:`H2`
    Suppression plus reinforcement: unflagged bands gain influence.

Subclass :class:`Regime` to add your own; the engine only needs ``propose``.

A note on H2's reinforcement rule. It is **additive** - a flagged-clear band
gains a constant ``gamma``. A multiplicative rule scales every surviving band by
the same factor, which the sum-normalisation of the aggregation step cancels
exactly, leaving H2 indistinguishable from H1; it also makes suppression
permanent, since a silenced band at weight zero can never recover. Addition
changes the *ratios* between weights and lets a band that returns to agreement
regain its voice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Regime", "Plain", "H1", "H2", "resolve_regime", "REGIMES"]


class Regime(ABC):
    """Base class for regulatory regimes."""

    name = "regime"
    uses_gate = False

    def __init__(self, alpha: float = 0.3):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = float(alpha)

    @abstractmethod
    def propose(
        self,
        logits: np.ndarray,
        consensus: np.ndarray,
        weights: np.ndarray,
        muted: np.ndarray,
        frozen: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propose updated logits and weights.

        Parameters
        ----------
        logits
            Current band evidence, shape ``(n_samples, n_bands, n_classes)``.
        consensus
            Weighted neighbourhood consensus, same shape.
        weights
            Current influence weights, shape ``(n_samples, n_bands)``.
        muted
            Gate decision, shape ``(n_samples, n_bands)``.
        frozen
            Bands excluded from updating, shape ``(n_samples, n_bands)``.

        Returns
        -------
        tuple
            Proposed ``(logits, weights)``, same shapes as the inputs.
        """

    def _diffuse(self, logits: np.ndarray, consensus: np.ndarray) -> np.ndarray:
        return (1.0 - self.alpha) * logits + self.alpha * consensus

    @staticmethod
    def _hold_frozen(proposed, current, frozen):
        f = frozen[..., None] if proposed.ndim == 3 else frozen
        return np.where(f, current, proposed)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(alpha={self.alpha:g})"


class Plain(Regime):
    """Unregulated diffusion toward the neighbourhood consensus."""

    name = "plain"
    uses_gate = False

    def propose(self, logits, consensus, weights, muted, frozen):
        proposed = self._diffuse(logits, consensus)
        return self._hold_frozen(proposed, logits, frozen), weights.copy()


class H1(Regime):
    """Stress-gated suppression of persistently discordant bands."""

    name = "H1"
    uses_gate = True

    def __init__(self, alpha: float = 0.3, mute_weight: float = 0.0):
        super().__init__(alpha)
        self.mute_weight = float(mute_weight)

    def propose(self, logits, consensus, weights, muted, frozen):
        diffused = self._diffuse(logits, consensus)
        act = muted & ~frozen
        proposed = np.where(act[..., None], consensus, diffused)
        w = np.where(act, self.mute_weight, weights)
        return self._hold_frozen(proposed, logits, frozen), w

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"H1(alpha={self.alpha:g}, mute_weight={self.mute_weight:g})"


class H2(H1):
    """Suppression of discordant bands plus reinforcement of coherent ones."""

    name = "H2"
    uses_gate = True

    def __init__(
        self,
        alpha: float = 0.3,
        mute_weight: float = 0.0,
        gamma: float = 0.25,
        weight_clip: tuple[float, float] = (0.0, 5.0),
    ):
        super().__init__(alpha, mute_weight)
        self.gamma = float(gamma)
        self.weight_clip = (float(weight_clip[0]), float(weight_clip[1]))
        if self.weight_clip[0] >= self.weight_clip[1]:
            raise ValueError("weight_clip must be (min, max) with min < max")

    def propose(self, logits, consensus, weights, muted, frozen):
        proposed, w = super().propose(logits, consensus, weights, muted, frozen)
        reinforce = ~muted & ~frozen
        w = np.where(reinforce, w + self.gamma, w)
        w = np.clip(w, *self.weight_clip)
        return proposed, w

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"H2(alpha={self.alpha:g}, gamma={self.gamma:g}, "
            f"weight_clip={self.weight_clip})"
        )


REGIMES = {"plain": Plain, "h1": H1, "h2": H2}


def resolve_regime(spec, **kwargs) -> Regime:
    """Coerce a regime name or instance into a :class:`Regime`."""
    if isinstance(spec, Regime):
        return spec
    if isinstance(spec, str):
        cls = REGIMES.get(spec.lower())
        if cls is None:
            raise ValueError(
                f"unknown regime {spec!r}; use one of {sorted(REGIMES)} or a Regime instance"
            )
        valid = {}
        if issubclass(cls, H2):
            valid = kwargs
        elif issubclass(cls, H1):
            valid = {k: v for k, v in kwargs.items() if k in ("alpha", "mute_weight")}
        else:
            valid = {k: v for k, v in kwargs.items() if k == "alpha"}
        return cls(**valid)
    raise TypeError(f"cannot interpret {spec!r} as a regime")
