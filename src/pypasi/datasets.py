"""Data: a synthetic generator with known conflict, and the bacteria benchmark.

:func:`make_conflict_signals` is the one to reach for first. It plants
class-discriminative peaks in bands you choose and *contradictory* peaks in
others, so the band that should be flagged as conflicted is known in advance.
That makes it possible to test whether the audit finds what is actually there,
rather than only that it runs.

:func:`load_bacteria` reads the public Raman dataset of Ho et al. (2019) from a
local directory. Nothing is downloaded automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["make_conflict_signals", "ConflictSpec", "load_bacteria", "BacteriaData"]


@dataclass
class ConflictSpec:
    """Where the planted signal and the planted contradiction live.

    Attributes
    ----------
    informative
        Axis positions carrying genuine class information.
    conflicting
        Axis positions carrying evidence for the *wrong* class, which is what
        the audit should surface.
    conflicted
        Boolean mask of the samples that actually received contradictory
        evidence. This is the ground truth an audit should recover, and it is
        what the package's own tests check against.
    outlier
        Boolean mask of the samples given an out-of-distribution artefact. These
        are strange *spectra*, as opposed to conflicted ones, which are ordinary
        spectra carrying contradictory evidence. Keeping the two masks separate
        is what makes it possible to show that conflict detection and novelty
        detection are not the same thing.
    """

    informative: tuple[float, ...]
    conflicting: tuple[float, ...]
    conflicted: np.ndarray
    outlier: np.ndarray


def _gaussian(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / width) ** 2)


def make_conflict_signals(
    n_samples: int = 300,
    n_classes: int = 3,
    n_points: int = 400,
    *,
    axis_range: tuple[float, float] = (400.0, 1800.0),
    informative_centers: tuple[float, ...] = (700.0, 900.0, 1300.0, 1550.0),
    conflicting_centers: tuple[float, ...] = (1100.0,),
    peak_width: float = 40.0,
    signal_strength: float = 0.55,
    conflict_strength: float = 1.1,
    conflict_rate: float = 0.35,
    noise: float = 0.35,
    baseline: float = 0.3,
    outlier_rate: float = 0.0,
    outlier_strength: float = 8.0,
    random_state=None,
):
    """Generate 1-D signals with class structure and planted band conflict.

    The region at ``conflicting_centers`` always carries a class-coded peak, so
    a band model fitted there learns to read it confidently. For a fraction
    ``conflict_rate`` of samples that peak encodes the *wrong* class, making the
    band confidently mistaken rather than merely uninformative - which is the
    distinction that matters, because a band with no information produces
    near-uniform evidence that agrees with everything and registers no conflict
    at all.

    Setting ``outlier_rate`` additionally gives a share of signals a narrow
    artefact of the kind a cosmic ray leaves, placed in a region that carries no
    class information. That puts them off the training manifold in input space
    while leaving their band-level class evidence intact, so the two failure
    modes can be studied apart: an outlier is a strange spectrum, a conflicted
    sample is an ordinary spectrum whose regions disagree. Generate a clean
    training set and a contaminated evaluation set to exercise
    :class:`pypasi.triage.Triage`.

    The default layout puts informative peaks on both sides of the conflicting
    one. That is deliberate: stress measures disagreement with a band's
    *neighbours*, so under a chain topology a discordant band flanked by
    uninformative bands has nothing to disagree with and stays invisible.
    Surrounding it with confident, correct neighbours is what makes the planted
    conflict detectable - and comparing ``topology="chain"`` against
    ``topology="complete"`` on this data shows how much the interaction graph
    governs what an audit can see.

    Returns
    -------
    X : ndarray, shape (n_samples, n_points)
    y : ndarray, shape (n_samples,)
    axis : ndarray, shape (n_points,)
    spec : ConflictSpec
        Where the informative and conflicting features were placed, and which
        samples received the contradiction.

    Examples
    --------
    >>> X, y, axis, spec = make_conflict_signals(n_samples=12, random_state=0)
    >>> X.shape, y.shape, axis.shape, spec.conflicted.shape
    ((12, 400), (12,), (400,), (12,))
    """
    rng = np.random.default_rng(random_state)
    axis = np.linspace(axis_range[0], axis_range[1], n_points)
    y = rng.integers(0, n_classes, size=n_samples)
    conflicted = np.zeros(n_samples, dtype=bool)

    X = np.full((n_samples, n_points), baseline, dtype=float)
    X += 0.25 * _gaussian(axis, axis_range[0] + 0.15 * (axis_range[1] - axis_range[0]), 200.0)

    def _offset(cls: int) -> float:
        return (int(cls) - 0.5 * (n_classes - 1)) * peak_width * 1.6

    for i, cls in enumerate(y):
        for ctr in informative_centers:
            amp = signal_strength * (0.8 + 0.4 * rng.random())
            X[i] += amp * _gaussian(axis, ctr + _offset(int(cls)), peak_width)

        # The conflicting region always carries a readable, class-coded peak.
        # For the contaminated fraction it encodes a different class, so the
        # band model there is confident and wrong rather than merely unsure.
        if rng.random() < conflict_rate:
            conflicted[i] = True
            claimed = int((int(cls) + 1) % n_classes)
        else:
            claimed = int(cls)
        for ctr in conflicting_centers:
            amp = conflict_strength * (0.8 + 0.4 * rng.random())
            X[i] += amp * _gaussian(axis, ctr + _offset(claimed), peak_width)

    X += rng.normal(0.0, noise, size=X.shape)

    # Out-of-distribution artefact: a narrow spike, as a cosmic ray leaves, put
    # where no class information lives. Smooth backgrounds are a poor choice for
    # this - the training spectra already contain broad shapes, so a PCA model
    # reconstructs them happily and they are not off-manifold at all.
    outlier = np.zeros(n_samples, dtype=bool)
    if outlier_rate > 0:
        outlier = rng.random(n_samples) < float(outlier_rate)
        informative_span = np.zeros(n_points, dtype=bool)
        for ctr in tuple(informative_centers) + tuple(conflicting_centers):
            informative_span |= np.abs(axis - ctr) < 4 * peak_width
        quiet = np.nonzero(~informative_span)[0]
        if outlier.any() and quiet.size:
            for i in np.nonzero(outlier)[0]:
                where = axis[rng.choice(quiet)]
                X[i] += outlier_strength * (0.6 + 0.8 * rng.random()) * _gaussian(
                    axis, where, peak_width * 0.12
                )

    X = np.clip(X, 0.0, None)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
    spec = ConflictSpec(tuple(informative_centers), tuple(conflicting_centers),
                        conflicted, outlier)
    return X, y, axis, spec


@dataclass
class BacteriaData:
    """The Ho et al. clinical Raman cohorts.

    Attributes
    ----------
    X_dev, y_dev
        Development cohort (``clinical2018``), for fitting and validation.
    X_test, y_test
        Independent external cohort (``clinical2019``).
    axis
        Wavenumbers in cm-1.
    class_names
        Original label values, in encoded order.
    """

    X_dev: np.ndarray
    y_dev: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    axis: np.ndarray
    class_names: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BacteriaData(dev={self.X_dev.shape}, test={self.X_test.shape}, "
            f"{len(self.class_names)} classes, "
            f"{self.axis.min():.0f}-{self.axis.max():.0f} cm-1)"
        )


_BACTERIA_FILES = {
    "X_dev": "X_2018clinical.npy",
    "y_dev": "y_2018clinical.npy",
    "X_test": "X_2019clinical.npy",
    "y_test": "y_2019clinical.npy",
    "axis": "wavenumbers.npy",
}


def _find(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"could not find {name} anywhere under {root}. The bacteria dataset "
            "is not distributed with pypasi; download it from "
            "https://github.com/csho33/bacteria-ID and point data_dir at it."
        )
    return matches[0]


def load_bacteria(
    data_dir,
    *,
    classes: "list | None" = None,
    normalize: bool = True,
    subsample: int | None = None,
    random_state=None,
) -> BacteriaData:
    """Load the bacteria/yeast Raman cohorts from ``data_dir``.

    Parameters
    ----------
    data_dir
        Directory containing the ``.npy`` files, searched recursively.
    classes
        Restrict to these original label values. ``None`` keeps all of them.
    normalize
        Apply sample-wise L2 normalisation, as the published analysis does.
    subsample
        Keep at most this many samples per cohort, drawn at random. Useful when
        developing against the full 10,000-signal cohort.
    random_state
        Seed for subsampling.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    paths = {k: _find(root, v) for k, v in _BACTERIA_FILES.items()}

    X_dev = np.load(paths["X_dev"]).astype(float)
    y_dev = np.load(paths["y_dev"])
    X_test = np.load(paths["X_test"]).astype(float)
    y_test = np.load(paths["y_test"])
    axis = np.load(paths["axis"]).astype(float)

    for name, X in (("clinical2018", X_dev), ("clinical2019", X_test)):
        if X.shape[1] != axis.size:
            raise ValueError(
                f"{name} has {X.shape[1]} features but wavenumbers.npy has {axis.size}"
            )

    if classes is not None:
        keep = np.asarray(classes)
        m_dev = np.isin(y_dev, keep)
        m_test = np.isin(y_test, keep)
        X_dev, y_dev = X_dev[m_dev], y_dev[m_dev]
        X_test, y_test = X_test[m_test], y_test[m_test]

    if normalize:
        for X in (X_dev, X_test):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            np.divide(X, np.where(norms > 0, norms, 1.0), out=X)

    if subsample is not None:
        rng = np.random.default_rng(random_state)
        def _take(X, y):
            if X.shape[0] <= subsample:
                return X, y
            idx = rng.choice(X.shape[0], size=subsample, replace=False)
            return X[idx], y[idx]
        X_dev, y_dev = _take(X_dev, y_dev)
        X_test, y_test = _take(X_test, y_test)

    class_names = np.unique(np.concatenate([y_dev, y_test]))
    return BacteriaData(X_dev, y_dev, X_test, y_test, axis, class_names)
