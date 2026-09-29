"""Interaction graphs between bands.

The published method couples each band only to its immediate spectral
neighbours. That is one choice among several, and it cannot express coupling
between distant regions that share a biochemical origin. :class:`Topology`
generalises the neighbourhood to an arbitrary undirected graph over bands, of
which the chain and the complete graph are special cases.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["Topology"]


class Topology:
    """An undirected interaction graph over ``n_bands`` bands.

    Construct with one of the class methods: :meth:`chain`, :meth:`complete`,
    :meth:`knn`, :meth:`radius`, :meth:`from_edges` or :meth:`from_matrix`.

    Parameters
    ----------
    adjacency
        Square, symmetric, zero-diagonal matrix. Non-zero entries are edge
        weights; boolean input is promoted to 1.0.
    name
        Short description used in reports.
    """

    def __init__(self, adjacency: np.ndarray, name: str = "custom"):
        A = np.asarray(adjacency, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square, got shape {A.shape}")
        if not np.allclose(A, A.T):
            raise ValueError("adjacency must be symmetric")
        if np.any(np.diag(A) != 0):
            raise ValueError("adjacency must have a zero diagonal (no self-loops)")
        if np.any(A < 0):
            raise ValueError("adjacency weights must be non-negative")
        self.adjacency = A
        self.name = name

    # ------------------------------------------------------------ properties

    @property
    def n_bands(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def edges(self) -> np.ndarray:
        """Upper-triangular edge list, shape ``(n_edges, 2)``."""
        i, j = np.nonzero(np.triu(self.adjacency, k=1))
        return np.column_stack([i, j])

    @property
    def edge_weights(self) -> np.ndarray:
        e = self.edges
        return self.adjacency[e[:, 0], e[:, 1]]

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def degrees(self) -> np.ndarray:
        return (self.adjacency > 0).sum(axis=1).astype(int)

    @property
    def is_connected(self) -> bool:
        """Whether every band can reach every other through the graph."""
        n = self.n_bands
        if n <= 1:
            return True
        seen = {0}
        stack = [0]
        while stack:
            k = stack.pop()
            for j in np.nonzero(self.adjacency[k] > 0)[0]:
                if int(j) not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        return len(seen) == n

    def neighbours(self, k: int) -> np.ndarray:
        """Indices of the bands adjacent to band ``k``."""
        return np.nonzero(self.adjacency[k] > 0)[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Topology({self.name}, {self.n_bands} bands, {self.n_edges} edges)"

    # --------------------------------------------------------- constructors

    @classmethod
    def chain(cls, n_bands: int) -> "Topology":
        """Nearest-neighbour coupling along the axis. The published default."""
        A = np.zeros((n_bands, n_bands))
        for k in range(n_bands - 1):
            A[k, k + 1] = A[k + 1, k] = 1.0
        return cls(A, name="chain")

    @classmethod
    def complete(cls, n_bands: int) -> "Topology":
        """Every band couples to every other."""
        A = np.ones((n_bands, n_bands)) - np.eye(n_bands)
        return cls(A, name="complete")

    @classmethod
    def knn(cls, n_bands: int, k: int = 2) -> "Topology":
        """Couple each band to its ``k`` nearest neighbours in band index."""
        if k < 1:
            raise ValueError("k must be >= 1")
        A = np.zeros((n_bands, n_bands))
        for i in range(n_bands):
            for d in range(1, k + 1):
                for j in (i - d, i + d):
                    if 0 <= j < n_bands:
                        A[i, j] = A[j, i] = 1.0
        return cls(A, name=f"knn(k={k})")

    @classmethod
    def radius(
        cls, centers: Sequence[float], radius: float, *, decay: bool = False
    ) -> "Topology":
        """Couple bands whose centres lie within ``radius`` axis units.

        With ``decay=True`` the edge weight falls off as
        ``exp(-d / radius)`` instead of being binary, so nearby bands influence
        one another more strongly than distant ones.
        """
        c = np.asarray(centers, dtype=float).reshape(-1)
        d = np.abs(c[:, None] - c[None, :])
        A = (d <= radius).astype(float)
        np.fill_diagonal(A, 0.0)
        if decay:
            A = A * np.exp(-d / float(radius))
            np.fill_diagonal(A, 0.0)
        return cls(A, name=f"radius({radius:g}{', decay' if decay else ''})")

    @classmethod
    def from_edges(
        cls,
        n_bands: int,
        edges: Iterable[tuple[int, int] | tuple[int, int, float]],
        *,
        include_chain: bool = False,
        name: str = "custom",
    ) -> "Topology":
        """Build a graph from an explicit edge list.

        Each edge is ``(i, j)`` or ``(i, j, weight)``. Set ``include_chain`` to
        add nearest-neighbour coupling on top, which is the usual way to express
        "the spectral chain, plus these long-range links".
        """
        A = np.zeros((n_bands, n_bands))
        if include_chain:
            for k in range(n_bands - 1):
                A[k, k + 1] = A[k + 1, k] = 1.0
        for e in edges:
            if len(e) == 2:
                i, j = e
                w = 1.0
            else:
                i, j, w = e
            i, j = int(i), int(j)
            if i == j:
                raise ValueError(f"self-loop requested on band {i}")
            if not (0 <= i < n_bands and 0 <= j < n_bands):
                raise ValueError(f"edge ({i}, {j}) out of range for {n_bands} bands")
            A[i, j] = A[j, i] = float(w)
        return cls(A, name=name)

    @classmethod
    def from_matrix(cls, adjacency: np.ndarray, name: str = "custom") -> "Topology":
        """Wrap a precomputed adjacency matrix."""
        return cls(adjacency, name=name)

    @classmethod
    def resolve(cls, spec, n_bands: int, centers: np.ndarray | None = None) -> "Topology":
        """Coerce a user-supplied topology specification into a Topology.

        Accepts an existing :class:`Topology`, one of the strings ``"chain"``,
        ``"complete"``, ``"knn"``, or a raw adjacency array.
        """
        if isinstance(spec, Topology):
            if spec.n_bands != n_bands:
                raise ValueError(
                    f"topology covers {spec.n_bands} bands but {n_bands} were given"
                )
            return spec
        if isinstance(spec, str):
            key = spec.lower()
            if key == "chain":
                return cls.chain(n_bands)
            if key == "complete":
                return cls.complete(n_bands)
            if key == "knn":
                return cls.knn(n_bands, k=2)
            raise ValueError(
                f"unknown topology {spec!r}; use 'chain', 'complete', 'knn', "
                "or a Topology instance"
            )
        return cls.from_matrix(np.asarray(spec, dtype=float))

    def to_frame(self, labels: Sequence[str] | None = None):
        """Edge list as a :class:`pandas.DataFrame`."""
        import pandas as pd

        e = self.edges
        labels = list(labels) if labels is not None else [f"B{i + 1}" for i in range(self.n_bands)]
        return pd.DataFrame(
            {
                "band_i": e[:, 0],
                "band_j": e[:, 1],
                "label_i": [labels[i] for i in e[:, 0]],
                "label_j": [labels[j] for j in e[:, 1]],
                "weight": self.edge_weights,
            }
        )
