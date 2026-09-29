"""The energy functional that governs which proposed updates are accepted.

Two competing terms, both evaluated on softmax-normalised evidence.

*Data fidelity* anchors each band to the evidence it started with, weighted by
that band's current influence. A band that abandons its own reading pays for it.

*Consensus* penalises disagreement across every edge of the interaction graph.
A configuration in which coupled bands disagree pays for that too.

Proposals that lower the total are always accepted; proposals that raise it are
accepted with Metropolis probability ``exp(-dE / T)``, which lets the system
climb out of shallow local minima instead of freezing at the first configuration
it stumbles into.
"""

from __future__ import annotations

import numpy as np

from .divergence import js_divergence, softmax
from .topology import Topology

__all__ = ["negotiation_energy"]


def negotiation_energy(
    logits: np.ndarray,
    logits0: np.ndarray,
    weights: np.ndarray,
    topology: Topology,
    *,
    lam: float = 1.0,
    divergence=js_divergence,
) -> np.ndarray:
    """Total energy per sample.

    Parameters
    ----------
    logits, logits0
        Current and initial band evidence, shape ``(n_samples, n_bands, n_classes)``.
    weights
        Influence weights, shape ``(n_samples, n_bands)``.
    topology
        Interaction graph supplying the edges of the consensus term.
    lam
        Weight of the consensus term relative to data fidelity.
    divergence
        Batched divergence on probability vectors.

    Returns
    -------
    numpy.ndarray
        Energy per sample, shape ``(n_samples,)``.
    """
    P = softmax(logits)
    P0 = softmax(logits0)

    e_data = np.sum(weights * divergence(P, P0), axis=-1)

    edges = topology.edges
    if edges.shape[0] == 0:
        return e_data

    w_edge = topology.edge_weights
    i, j = edges[:, 0], edges[:, 1]
    e_cons = np.sum(w_edge * divergence(P[:, i, :], P[:, j, :]), axis=-1)
    return e_data + lam * e_cons
