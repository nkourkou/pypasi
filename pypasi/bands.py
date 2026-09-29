"""Partitioning of a 1-D signal axis into contiguous bands.

A :class:`BandSet` is the bridge between the abstract negotiation machinery and
the physical signal. It remembers, for every band, both the feature columns it
covers and the axis interval (e.g. Raman shift in cm-1) those columns occupy, so
that a band implicated in conflict can always be traced back to the raw signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["Band", "BandSet"]


@dataclass(frozen=True)
class Band:
    """One contiguous region of the signal axis.

    Parameters
    ----------
    index
        Position of the band in the :class:`BandSet`, starting at 0.
    label
        Short human-readable name, e.g. ``"B3"``.
    lo, hi
        Inclusive lower and upper axis values spanned by the band.
    features
        Integer column indices of the design matrix belonging to this band.
    """

    index: int
    label: str
    lo: float
    hi: float
    features: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.features.size)

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Band({self.label}, {self.lo:.4g}-{self.hi:.4g}, "
            f"{self.n_features} features)"
        )


class BandSet:
    """An ordered collection of contiguous, non-overlapping bands.

    Construct with one of the class methods rather than directly:
    :meth:`equal_width`, :meth:`peak_informed` or :meth:`from_edges`.

    Parameters
    ----------
    axis
        Monotonically increasing axis values, one per column of the design
        matrix (for Raman data, the wavenumbers in cm-1).
    bands
        The bands themselves, in axis order.
    axis_name
        Name used in tables and figures.
    """

    def __init__(self, axis: np.ndarray, bands: Sequence[Band], axis_name: str = "axis"):
        self.axis = np.asarray(axis, dtype=float)
        self.bands = list(bands)
        self.axis_name = axis_name
        if self.axis.ndim != 1:
            raise ValueError(f"axis must be 1-D, got shape {self.axis.shape}")
        if not self.bands:
            raise ValueError("a BandSet must contain at least one band")
        seen: set[int] = set()
        for b in self.bands:
            if b.n_features == 0:
                raise ValueError(f"band {b.label} ({b.lo}-{b.hi}) contains no features")
            overlap = seen.intersection(b.features.tolist())
            if overlap:
                raise ValueError(
                    f"band {b.label} overlaps an earlier band on {len(overlap)} feature(s)"
                )
            seen.update(b.features.tolist())

    # ------------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return len(self.bands)

    def __iter__(self):
        return iter(self.bands)

    def __getitem__(self, key: int | str) -> Band:
        if isinstance(key, str):
            for b in self.bands:
                if b.label == key:
                    return b
            raise KeyError(f"no band labelled {key!r}; have {self.labels}")
        return self.bands[key]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BandSet({len(self)} bands, {self.lo:.4g}-{self.hi:.4g})"

    # ------------------------------------------------------------ properties

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def labels(self) -> list[str]:
        return [b.label for b in self.bands]

    @property
    def edges(self) -> np.ndarray:
        """Band boundaries as an array of length ``n_bands + 1``."""
        return np.array([self.bands[0].lo] + [b.hi for b in self.bands], dtype=float)

    @property
    def centers(self) -> np.ndarray:
        return np.array([0.5 * (b.lo + b.hi) for b in self.bands], dtype=float)

    @property
    def lo(self) -> float:
        return float(self.bands[0].lo)

    @property
    def hi(self) -> float:
        return float(self.bands[-1].hi)

    @property
    def n_features(self) -> int:
        """Total number of features covered by the band set."""
        return int(sum(b.n_features for b in self.bands))

    # --------------------------------------------------------- constructors

    @classmethod
    def from_edges(
        cls,
        axis: np.ndarray,
        edges: Sequence[float],
        *,
        axis_name: str = "axis",
        labels: Sequence[str] | None = None,
    ) -> "BandSet":
        """Build bands from explicit boundaries.

        Each band covers ``[edges[i], edges[i + 1])``; the last band includes its
        upper boundary so that the final axis point is not silently dropped.
        """
        axis = np.asarray(axis, dtype=float)
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1-D sequence of at least two values")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        k = edges.size - 1
        if labels is None:
            labels = [f"B{i + 1}" for i in range(k)]
        elif len(labels) != k:
            raise ValueError(f"expected {k} labels, got {len(labels)}")

        bands = []
        for i in range(k):
            lo, hi = float(edges[i]), float(edges[i + 1])
            if i < k - 1:
                idx = np.where((axis >= lo) & (axis < hi))[0]
            else:
                idx = np.where((axis >= lo) & (axis <= hi))[0]
            bands.append(Band(i, labels[i], lo, hi, idx))
        return cls(axis, bands, axis_name=axis_name)

    @classmethod
    def equal_width(
        cls,
        axis: np.ndarray,
        n_bands: int,
        *,
        lo: float | None = None,
        hi: float | None = None,
        axis_name: str = "axis",
    ) -> "BandSet":
        """Split ``[lo, hi]`` into ``n_bands`` intervals of equal axis width.

        ``lo`` and ``hi`` default to the extremes of ``axis``.
        """
        axis = np.asarray(axis, dtype=float)
        lo = float(axis.min()) if lo is None else float(lo)
        hi = float(axis.max()) if hi is None else float(hi)
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        if hi <= lo:
            raise ValueError(f"hi ({hi}) must exceed lo ({lo})")
        return cls.from_edges(axis, np.linspace(lo, hi, n_bands + 1), axis_name=axis_name)

    @classmethod
    def peak_informed(
        cls,
        axis: np.ndarray,
        X: np.ndarray,
        n_bands: int,
        *,
        lo: float | None = None,
        hi: float | None = None,
        smooth: int = 5,
        min_width: float | None = None,
        axis_name: str = "axis",
    ) -> "BandSet":
        """Place band boundaries in the troughs between the strongest peaks.

        The mean signal over ``X`` is smoothed, its peaks are ranked by
        prominence, and the ``n_bands - 1`` deepest minima separating the top
        peaks become the boundaries. This keeps prominent features away from
        band edges, where they would otherwise be split between two agents.

        Falls back to :meth:`equal_width` if too few peaks are found.

        Parameters
        ----------
        X
            Signal matrix of shape ``(n_samples, n_features)``.
        smooth
            Width of the moving-average window applied before peak finding.
        min_width
            Minimum permitted band width in axis units. Defaults to one tenth
            of the full range.
        """
        from scipy.signal import find_peaks, peak_prominences

        axis = np.asarray(axis, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != axis.size:
            raise ValueError(
                f"X must have shape (n_samples, {axis.size}), got {X.shape}"
            )
        lo = float(axis.min()) if lo is None else float(lo)
        hi = float(axis.max()) if hi is None else float(hi)
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        if n_bands == 1:
            return cls.from_edges(axis, [lo, hi], axis_name=axis_name)
        if min_width is None:
            min_width = (hi - lo) / 10.0

        window = (axis >= lo) & (axis <= hi)
        sub_axis = axis[window]
        profile = X[:, window].mean(axis=0)
        if smooth and smooth > 1:
            kernel = np.ones(int(smooth)) / float(smooth)
            profile = np.convolve(profile, kernel, mode="same")

        peaks, _ = find_peaks(profile)
        if peaks.size < n_bands:
            return cls.equal_width(axis, n_bands, lo=lo, hi=hi, axis_name=axis_name)
        prom = peak_prominences(profile, peaks)[0]
        top = peaks[np.argsort(prom)[::-1][:n_bands]]
        top = np.sort(top)

        cuts: list[float] = []
        for a, b in zip(top[:-1], top[1:]):
            trough = a + int(np.argmin(profile[a : b + 1]))
            cuts.append(float(sub_axis[trough]))

        edges = [lo] + cuts + [hi]
        # Enforce min_width by dropping boundaries that crowd their neighbour.
        pruned = [edges[0]]
        for e in edges[1:-1]:
            if e - pruned[-1] >= min_width and edges[-1] - e >= min_width:
                pruned.append(e)
        pruned.append(edges[-1])
        if len(pruned) - 1 < n_bands:
            # Top up with equal-width cuts in the widest remaining gaps.
            while len(pruned) - 1 < n_bands:
                widths = np.diff(pruned)
                j = int(np.argmax(widths))
                mid = 0.5 * (pruned[j] + pruned[j + 1])
                if widths[j] / 2 < min_width:
                    break
                pruned.insert(j + 1, mid)
        return cls.from_edges(axis, pruned, axis_name=axis_name)

    # ------------------------------------------------------------- utilities

    def split(self, X: np.ndarray) -> list[np.ndarray]:
        """Return the per-band column blocks of ``X``."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        return [X[:, b.features] for b in self.bands]

    def band_of_feature(self, j: int) -> Band | None:
        """Return the band containing column ``j``, or ``None`` if uncovered."""
        for b in self.bands:
            if j in b.features:
                return b
        return None

    def to_frame(self):
        """Describe the band set as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "band": [b.index for b in self.bands],
                "label": [b.label for b in self.bands],
                f"{self.axis_name}_lo": [b.lo for b in self.bands],
                f"{self.axis_name}_hi": [b.hi for b in self.bands],
                "width": [b.width for b in self.bands],
                "n_features": [b.n_features for b in self.bands],
            }
        )
