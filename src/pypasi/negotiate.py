"""The negotiation engine.

Band agents exchange class evidence over an interaction graph until the system
settles. Every sample in the batch is negotiated simultaneously - the samples are
independent, so the whole cohort advances one iteration at a time under the same
vectorised update, which is what makes cohort-scale auditing practical.

The one place samples differ is the Metropolis step, which accepts or rejects
each sample's proposal on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .divergence import js_divergence, softmax
from .energy import negotiation_energy
from .gates import Gate, QuantileGate, resolve_gate
from .geometry import (
    band_decision_geometry,
    decision_geometry,
    early_decision_geometry,
    eredg,
    redg,
)
from .regimes import Regime, resolve_regime
from .topology import Topology

__all__ = ["negotiate", "NegotiationResult", "initial_weights", "neighbourhood_consensus", "band_stress"]

_EPS = 1e-12


def initial_weights(logits: np.ndarray, scheme: str = "confidence") -> np.ndarray:
    """Starting influence weights, shape ``(n_samples, n_bands)``.

    ``"confidence"``
        The band's own maximum softmax probability - a band that is sure of
        itself starts out louder.
    ``"uniform"``
        Every band starts equal.
    """
    logits = np.asarray(logits, dtype=float)
    if scheme == "uniform":
        return np.ones(logits.shape[:2], dtype=float)
    if scheme == "confidence":
        return np.clip(softmax(logits).max(axis=-1), 0.0, 1.0)
    raise ValueError(f"unknown weight scheme {scheme!r}; use 'confidence' or 'uniform'")


def neighbourhood_consensus(
    logits: np.ndarray, weights: np.ndarray, topology: Topology
) -> np.ndarray:
    """Weighted mean of each band's neighbours, shape ``(n_samples, n_bands, n_classes)``.

    Bands with no surviving neighbours fall back to their own current evidence,
    so an isolated agent simply holds its position rather than collapsing.
    """
    A = topology.adjacency
    num = np.einsum("kj,nj,njc->nkc", A, weights, logits)
    den = np.einsum("kj,nj->nk", A, weights)
    out = num / (den[..., None] + _EPS)
    dead = den <= _EPS
    if np.any(dead):
        out = np.where(dead[..., None], logits, out)
    return out


def band_stress(
    logits: np.ndarray, consensus: np.ndarray, divergence=js_divergence
) -> np.ndarray:
    """Disagreement between each band and its neighbourhood, shape ``(n_samples, n_bands)``."""
    return divergence(softmax(logits), softmax(consensus))


@dataclass
class NegotiationResult:
    """Everything a negotiation produced, including how it got there.

    Attributes
    ----------
    y_pred
        Predicted class index per sample.
    proba
        Final class probabilities, shape ``(n_samples, n_classes)``.
    logits_final, weights_final
        End state of the band agents.
    logit_history
        Shape ``(n_samples, n_steps + 1, n_bands, n_classes)`` when recorded.
    stress_history, mute_history
        Shape ``(n_samples, n_steps, n_bands)``.
    weight_history
        Shape ``(n_samples, n_steps + 1, n_bands)``.
    energy_history
        Shape ``(n_samples, n_steps + 1)``.
    accepted
        Metropolis outcome per sample per step, shape ``(n_samples, n_steps)``.
    collapsed
        Samples whose bands were all silenced, leaving no evidence to aggregate.
        Empty unless the gate was built with ``protect_min=0``.
    """

    y_pred: np.ndarray
    proba: np.ndarray
    logits_final: np.ndarray
    weights_final: np.ndarray
    logits_initial: np.ndarray
    regime: str
    topology: Topology = field(repr=False)
    logit_history: np.ndarray | None = None
    stress_history: np.ndarray | None = None
    mute_history: np.ndarray | None = None
    weight_history: np.ndarray | None = None
    energy_history: np.ndarray | None = None
    accepted: np.ndarray | None = None
    collapsed: np.ndarray | None = None
    k_early: int = 3

    # --------------------------------------------------------- derived views

    def _require_history(self, what: str) -> np.ndarray:
        if self.logit_history is None:
            raise RuntimeError(
                f"{what} needs the logit trajectory; re-run negotiate() with "
                "record_trajectory=True"
            )
        return self.logit_history

    @property
    def n_samples(self) -> int:
        return int(self.logits_final.shape[0])

    @property
    def n_bands(self) -> int:
        return int(self.logits_final.shape[1])

    @property
    def dg(self) -> np.ndarray:
        """Decision Geometry per sample."""
        return decision_geometry(self._require_history("dg"))

    @property
    def band_dg(self) -> np.ndarray:
        """Per-band contribution to DG, shape ``(n_samples, n_bands)``."""
        return band_decision_geometry(self._require_history("band_dg"))

    @property
    def dg_early(self) -> np.ndarray:
        return early_decision_geometry(self._require_history("dg_early"), self.k_early)

    @property
    def redg(self) -> np.ndarray:
        return redg(self._require_history("redg"), self.k_early)

    def eredg(self, gate_quantile: float = 0.2, threshold: float | None = None):
        """eREDG and the activity threshold used, see :func:`pypasi.geometry.eredg`."""
        return eredg(
            self._require_history("eredg"),
            self.k_early,
            gate_quantile=gate_quantile,
            threshold=threshold,
        )

    @property
    def mute_frequency(self) -> np.ndarray:
        """Fraction of iterations each band spent silenced, ``(n_samples, n_bands)``."""
        if self.mute_history is None:
            raise RuntimeError("mute history was not recorded")
        return self.mute_history.mean(axis=1)

    @property
    def mean_stress(self) -> np.ndarray:
        """Mean stress per band over the negotiation, ``(n_samples, n_bands)``."""
        if self.stress_history is None:
            raise RuntimeError("stress history was not recorded")
        return self.stress_history.mean(axis=1)

    @property
    def cumulative_stress(self) -> np.ndarray:
        """Summed stress per band, ``(n_samples, n_bands)``."""
        if self.stress_history is None:
            raise RuntimeError("stress history was not recorded")
        return self.stress_history.sum(axis=1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NegotiationResult(regime={self.regime!r}, n_samples={self.n_samples}, "
            f"n_bands={self.n_bands}, trajectory="
            f"{'recorded' if self.logit_history is not None else 'discarded'})"
        )


def negotiate(
    band_logits: np.ndarray,
    *,
    topology: Topology | str = "chain",
    regime: Regime | str = "H1",
    gate: Gate | float | str | None = None,
    weights: np.ndarray | str = "confidence",
    frozen: np.ndarray | None = None,
    lam: float = 1.0,
    temperature: float = 0.01,
    max_iter: int = 30,
    divergence=js_divergence,
    record_trajectory: bool = True,
    k_early: int = 3,
    random_state: int | np.random.Generator | None = None,
) -> NegotiationResult:
    """Run the negotiation for a batch of samples.

    Parameters
    ----------
    band_logits
        Class evidence per band, shape ``(n_samples, n_bands, n_classes)``.
    topology
        Interaction graph, or ``"chain"`` / ``"complete"`` / ``"knn"``.
    regime
        ``"plain"``, ``"H1"``, ``"H2"`` or a :class:`~pypasi.regimes.Regime`.
    gate
        Conflict gate. Required for regimes that use one. A fitted
        :class:`~pypasi.gates.QuantileGate` is the usual choice.
    weights
        Initial influence weights, or ``"confidence"`` / ``"uniform"``.
    frozen
        Boolean mask of bands that must not update, shape ``(n_samples, n_bands)``
        or ``(n_bands,)``. Used to model a band whose evidence is unavailable.
    lam
        Weight of the consensus term in the energy.
    temperature
        Metropolis temperature. Zero makes the dynamics strictly downhill.
    max_iter
        Number of iterations. The negotiation always runs all of them; there is
        no convergence test, so trajectories are directly comparable.
    record_trajectory
        Keep the full history. Needed for every geometry descriptor.
    random_state
        Seed or generator for the Metropolis draws.

    Returns
    -------
    NegotiationResult
    """
    L0 = np.asarray(band_logits, dtype=float)
    if L0.ndim != 3:
        raise ValueError(
            f"band_logits must have shape (n_samples, n_bands, n_classes), got {L0.shape}"
        )
    n, k, c = L0.shape

    topo = Topology.resolve(topology, k)
    reg = resolve_regime(regime)
    if reg.uses_gate:
        if gate is None:
            raise ValueError(
                f"regime {reg.name!r} needs a gate; pass gate=QuantileGate(...) "
                "after fitting it on clean stress, or gate=AbsoluteGate(tau)"
            )
        g = resolve_gate(gate)
    else:
        g = None

    rng = np.random.default_rng(random_state)

    if isinstance(weights, str):
        w = initial_weights(L0, weights)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape == (k,):
            w = np.broadcast_to(w, (n, k)).copy()
        if w.shape != (n, k):
            raise ValueError(f"weights must have shape {(n, k)}, got {w.shape}")
        w = w.copy()

    if frozen is None:
        fz = np.zeros((n, k), dtype=bool)
    else:
        fz = np.asarray(frozen, dtype=bool)
        if fz.shape == (k,):
            fz = np.broadcast_to(fz, (n, k)).copy()
        if fz.shape != (n, k):
            raise ValueError(f"frozen must have shape {(n, k)}, got {fz.shape}")

    L = L0.copy()
    energy = negotiation_energy(L, L0, w, topo, lam=lam, divergence=divergence)

    l_hist = [L.copy()] if record_trajectory else None
    w_hist = [w.copy()] if record_trajectory else None
    e_hist = [energy.copy()] if record_trajectory else None
    s_hist: list[np.ndarray] = []
    m_hist: list[np.ndarray] = []
    acc_hist: list[np.ndarray] = []

    for _ in range(int(max_iter)):
        consensus = neighbourhood_consensus(L, w, topo)
        stress = band_stress(L, consensus, divergence)
        muted = g.mask(stress) if g is not None else np.zeros((n, k), dtype=bool)
        muted = muted & ~fz

        L_prop, w_prop = reg.propose(L, consensus, w, muted, fz)
        e_prop = negotiation_energy(L_prop, L0, w_prop, topo, lam=lam, divergence=divergence)

        accept = e_prop <= energy
        if temperature > 0.0:
            uphill = ~accept
            if np.any(uphill):
                p = np.exp(-(e_prop[uphill] - energy[uphill]) / float(temperature))
                accept[uphill] = rng.random(int(uphill.sum())) < p

        a3 = accept[:, None, None]
        a2 = accept[:, None]
        L = np.where(a3, L_prop, L)
        w = np.where(a2, w_prop, w)
        energy = np.where(accept, e_prop, energy)

        s_hist.append(stress)
        m_hist.append(muted & a2)
        acc_hist.append(accept.copy())
        if record_trajectory:
            l_hist.append(L.copy())
            w_hist.append(w.copy())
            e_hist.append(energy.copy())

    total_w = w.sum(axis=1)
    collapsed = total_w <= _EPS
    l_final = np.einsum("nk,nkc->nc", w, L) / (total_w[:, None] + _EPS)
    proba = softmax(l_final)
    y_pred = np.argmax(l_final, axis=1)
    if np.any(collapsed):
        proba[collapsed] = np.full(c, 1.0 / c)

    return NegotiationResult(
        y_pred=y_pred,
        proba=proba,
        logits_final=L,
        weights_final=w,
        logits_initial=L0,
        regime=reg.name,
        topology=topo,
        logit_history=np.stack(l_hist, axis=1) if record_trajectory else None,
        stress_history=np.stack(s_hist, axis=1) if s_hist else None,
        mute_history=np.stack(m_hist, axis=1) if m_hist else None,
        weight_history=np.stack(w_hist, axis=1) if record_trajectory else None,
        energy_history=np.stack(e_hist, axis=1) if record_trajectory else None,
        accepted=np.stack(acc_hist, axis=1) if acc_hist else None,
        collapsed=collapsed,
        k_early=int(k_early),
    )
