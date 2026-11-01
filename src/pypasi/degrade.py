"""Controlled degradation of spectra, for stress-testing an audit on your own data.

:mod:`pypasi.perturb` corrupts the *evidence* a band contributes, which isolates
the negotiation from the spectroscopy. This module corrupts the *spectra*, the
way an instrument or a sample would, so that the whole pipeline, preprocessing
included, can be tested against faults whose type, position and size are known.

Each fault is a :class:`Family`: a function that degrades one spectrum at a
severity ``s``. The five families of the SoftwareX paper are built in, and so
are three more. A new fault is one small function (see :class:`Family`).

>>> planted = plant(X, axis, bands=bands, random_state=0)         # doctest: +SKIP
>>> planted.X, planted.degraded, planted.kind                     # doctest: +SKIP
>>> stress_test(triage, X, y, axis, bands=bands, preprocess=snv)  # doctest: +SKIP

**Plant before you preprocess.** A defect is something the instrument did, and
scaling is something the analyst does afterwards. Planting on SNV-scaled spectra
would let the normalisation partly undo the defect before the model sees it.

The magnitudes are written for spectra scaled to roughly [0, 1] per spectrum,
which is what the paper uses; each family documents what ``s`` means so that
the magnitudes can be translated to other units.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EXTRA_FAMILIES",
    "FAMILIES",
    "PAPER_FAMILIES",
    "Family",
    "Planted",
    "axis_shift",
    "cosmic_rays",
    "fluorescence_background",
    "intensity_loss",
    "local_contamination",
    "plant",
    "resolution_loss",
    "shot_noise",
    "stress_test",
    "throughput_dip",
]


# --------------------------------------------------------------------- helpers


def _sorted(axis, x):
    o = np.argsort(axis)
    return axis[o], x[..., o], o


def _step(axis) -> float:
    return abs(float(np.diff(np.sort(axis)).mean()))


# ------------------------------------------------------- transforms (any shape)


def axis_shift(X, axis, shift):
    """Translate every spectrum along the axis by ``shift`` axis units."""
    a, Xs, o = _sorted(np.asarray(axis, float), np.atleast_2d(np.asarray(X, float)))
    Y = np.array([np.interp(a, a + shift, r) for r in Xs])
    out = np.empty_like(Y)
    out[:, o] = Y
    return out


def resolution_loss(X, axis, fwhm):
    """Convolve with a Gaussian line shape of full width ``fwhm`` axis units."""
    from scipy.ndimage import gaussian_filter1d

    X = np.atleast_2d(np.asarray(X, float))
    if not np.isfinite(fwhm) or fwhm <= 0:
        return X
    a, Xs, o = _sorted(np.asarray(axis, float), X)
    Y = gaussian_filter1d(Xs, fwhm / 2.3548 / _step(axis), axis=1, mode="nearest")
    out = np.empty_like(Y)
    out[:, o] = Y
    return out


def throughput_dip(X, axis, depth, *, centre=None, width=None):
    """Multiply by a smooth envelope ``exp(-depth * g)``, with ``g`` a unit Gaussian.

    ``centre`` and ``width`` (one standard deviation) default to the middle of
    the axis and 22% of its span.
    """
    axis = np.asarray(axis, float)
    span = float(axis.max() - axis.min())
    c = axis.min() + 0.5 * span if centre is None else float(centre)
    w = 0.22 * span if width is None else float(width)
    env = np.exp(-depth * np.exp(-0.5 * ((axis - c) / w) ** 2))
    return np.atleast_2d(np.asarray(X, float)) * env[None, :]


def shot_noise(X, scale, rng=None, *, floor=0.05):
    """Add intensity-dependent noise, sd = ``scale * sqrt(max(x, 0) + floor)``."""
    rng = np.random.default_rng(rng)
    X = np.atleast_2d(np.asarray(X, float))
    return X + rng.normal(0, scale * np.sqrt(np.clip(X, 0, None) + floor), X.shape)


def local_contamination(X, axis, lo, hi, attenuation, noise, rng=None):
    """Attenuate ``[lo, hi)`` by the factor ``attenuation`` and roughen it."""
    rng = np.random.default_rng(rng)
    X = np.atleast_2d(np.asarray(X, float)).copy()
    axis = np.asarray(axis, float)
    sel = (axis >= lo) & (axis < hi)
    X[:, sel] = X[:, sel] * attenuation + rng.normal(
        0, noise, (X.shape[0], int(sel.sum()))
    )
    return X


def cosmic_rays(X, n_spikes, height, rng=None):
    """Add ``n_spikes`` one-channel spikes of ``height`` at random channels."""
    rng = np.random.default_rng(rng)
    X = np.atleast_2d(np.asarray(X, float)).copy()
    for row in X:
        idx = rng.choice(row.size, size=int(n_spikes), replace=False)
        row[idx] += height * rng.uniform(0.5, 1.5, idx.size)
    return X


def fluorescence_background(X, axis, amplitude, *, decay=0.5):
    """Add a smooth background ``amplitude * exp(-(axis - min) / (decay * span))``.

    The background falls monotonically across the axis, the usual shape of a
    fluorescence tail in the Stokes region.
    """
    axis = np.asarray(axis, float)
    span = float(axis.max() - axis.min())
    bg = amplitude * np.exp(-(axis - axis.min()) / (decay * span))
    return np.atleast_2d(np.asarray(X, float)) + bg[None, :]


def intensity_loss(X, factor):
    """Scale the whole spectrum by ``factor``. Any per-spectrum normalisation
    removes this exactly; it is included to make that point testable."""
    return np.atleast_2d(np.asarray(X, float)) * factor


# --------------------------------------------------------------------- families


@dataclass(frozen=True)
class Family:
    """One kind of fault.

    Parameters
    ----------
    name
        Label used in the ground truth and in reports.
    scope
        ``"global"`` (acts on the whole spectrum) or ``"local"`` (one interval).
        Only reported; the planting does not depend on it.
    apply
        ``apply(x, axis, s, rng, edges) -> (x_new, where)``. ``x`` has shape
        ``(1, n_channels)``; ``s`` is the severity; ``edges`` is the list of
        ``(lo, hi)`` intervals available to a local fault; ``where`` is the index
        of the interval hit, or ``-1``.
    magnitude
        Plain-language statement of what ``s`` means.
    """

    name: str
    scope: str
    apply: Callable
    magnitude: str = ""


def _f_shift(x, axis, s, rng, edges):
    return axis_shift(x, axis, 4.0 * s), -1


def _f_blur(x, axis, s, rng, edges):
    return resolution_loss(x, axis, 7.0 * s), -1


def _f_dip(x, axis, s, rng, edges):
    return throughput_dip(x, axis, 0.35 * s), -1


def _f_noise(x, axis, s, rng, edges):
    return shot_noise(x, 0.02 * s, rng), -1


def _f_local(x, axis, s, rng, edges):
    if not edges:
        raise ValueError(
            "local contamination needs band intervals: pass bands= or edges="
        )
    j = int(rng.integers(len(edges)))
    lo, hi = edges[j]
    return local_contamination(x, axis, lo, hi, np.exp(-0.45 * s), 0.02 * s, rng), j


def _f_cosmic(x, axis, s, rng, edges):
    rngx = float(x.max() - x.min()) or 1.0
    return cosmic_rays(x, 1 + rng.poisson(3.0 * s), 0.5 * s * rngx, rng), -1


def _f_fluor(x, axis, s, rng, edges):
    rngx = float(x.max() - x.min()) or 1.0
    return fluorescence_background(x, axis, 0.5 * s * rngx), -1


def _f_power(x, axis, s, rng, edges):
    return intensity_loss(x, np.exp(-0.7 * s)), -1


FAMILIES = {
    f.name: f
    for f in (
        Family("axis shift", "global", _f_shift, "shift of 4 s axis units"),
        Family(
            "resolution loss",
            "global",
            _f_blur,
            "Gaussian line shape of FWHM 7 s axis units",
        ),
        Family(
            "throughput dip",
            "global",
            _f_dip,
            "envelope exp(-0.35 s g), g a Gaussian at mid-axis of sd 22% of the span",
        ),
        Family("short exposure", "global", _f_noise, "noise sd 0.02 s sqrt(x + 0.05)"),
        Family(
            "local contamination",
            "local",
            _f_local,
            "one interval attenuated by exp(-0.45 s), plus noise of sd 0.02 s",
        ),
        Family(
            "cosmic rays",
            "global",
            _f_cosmic,
            "1 + Poisson(3 s) one-channel spikes of 0.25-0.75 s times the spectrum's range",
        ),
        Family(
            "fluorescence",
            "global",
            _f_fluor,
            "decaying background of amplitude 0.5 s times the spectrum's range",
        ),
        Family(
            "intensity loss", "global", _f_power, "whole spectrum scaled by exp(-0.7 s)"
        ),
    )
}
PAPER_FAMILIES = (
    "axis shift",
    "resolution loss",
    "throughput dip",
    "short exposure",
    "local contamination",
)
EXTRA_FAMILIES = ("cosmic rays", "fluorescence", "intensity loss")


@dataclass
class Planted:
    """Degraded spectra and the ground truth for each one."""

    X: np.ndarray  # the spectra, clean rows untouched
    degraded: np.ndarray  # 1 where a fault was planted
    kind: np.ndarray  # family name, or "none"
    severity: np.ndarray  # 0 for clean rows
    where: np.ndarray  # interval index for local faults, else -1

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "degraded": self.degraded,
                "kind": self.kind,
                "severity": self.severity,
                "where": self.where,
            }
        )


def _resolve(families) -> list[Family]:
    out = []
    for f in families:
        if isinstance(f, Family):
            out.append(f)
        elif f in FAMILIES:
            out.append(FAMILIES[f])
        else:
            raise ValueError(
                f"unknown family {f!r}; use one of {sorted(FAMILIES)} or a Family"
            )
    if not out:
        raise ValueError("families is empty")
    return out


def plant(
    X,
    axis,
    *,
    fraction: float = 0.5,
    severity: tuple[float, float] = (0.3, 1.0),
    families: Sequence = PAPER_FAMILIES,
    bands=None,
    edges=None,
    random_state=None,
) -> Planted:
    """Degrade a random ``fraction`` of the spectra, one fault each.

    Every degraded spectrum gets one family drawn uniformly from ``families``,
    at a severity drawn uniformly from ``severity``. With the defaults and the
    same seed, this is exactly the simulator used in the SoftwareX paper.

    Parameters
    ----------
    X, axis
        Spectra ``(n_samples, n_channels)`` and their axis, *before* scaling.
    bands, edges
        The intervals a local fault may hit: a :class:`~pypasi.BandSet`, or a
        list of ``(lo, hi)`` pairs. Needed only for local families.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    axis = np.asarray(axis, float)
    if X.ndim != 2 or X.shape[1] != axis.size:
        raise ValueError(f"X must have shape (n, {axis.size}), got {X.shape}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    if bands is not None:
        edges = [(b.lo, b.hi) for b in bands.bands]
    edges = None if edges is None else [tuple(map(float, e)) for e in edges]
    fams = _resolve(families)
    names = [f.name for f in fams]

    n = len(X)
    Xc = X.copy()
    n_bad = int(round(fraction * n))
    bad = rng.permutation(n)[:n_bad]
    kind = np.array(["none"] * n, dtype=object)
    sev = np.zeros(n)
    where = np.full(n, -1)
    sev[bad] = rng.uniform(*severity, n_bad)
    kind[bad] = rng.choice(names, n_bad)
    by_name = dict(zip(names, fams))
    for i in bad:
        row, where[i] = by_name[kind[i]].apply(X[i][None], axis, sev[i], rng, edges)
        Xc[i] = np.asarray(row, float).reshape(-1)
    return Planted(Xc, (sev > 0).astype(int), kind, sev, where)


def stress_test(
    triage,
    X,
    y,
    axis,
    *,
    preprocess=None,
    families: Sequence = PAPER_FAMILIES,
    fraction: float = 0.5,
    severity: tuple[float, float] = (0.3, 1.0),
    bands=None,
    edges=None,
    random_state=None,
):
    """Plant faults in held-out spectra and report how each audit score sees them.

    Parameters
    ----------
    triage
        A fitted :class:`~pypasi.Triage`. Its conflict sign should have been
        learned on a *different* defect-planted set (``orientation_data=``).
    X, y, axis
        Held-out spectra before preprocessing, their labels and axis.
    preprocess
        The scaling the model expects, applied after planting (e.g. SNV).

    Returns
    -------
    pandas.DataFrame, one row per family plus ``"all"``: the number of degraded
    spectra; the AUROC of conflict, 1 - confidence and novelty against the clean
    spectra; and the accuracy on the degraded spectra before and after the fault
    was planted, so that ``accuracy_cost_pp`` is a paired measure of the damage.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    if bands is None:
        bands = getattr(getattr(triage, "classifier_", None), "bands_", None)
    p = plant(
        X,
        axis,
        fraction=fraction,
        severity=severity,
        families=families,
        bands=bands,
        edges=edges,
        random_state=random_state,
    )
    Xp = p.X if preprocess is None else preprocess(p.X)
    T = triage.assess(Xp, y).table
    scores = {
        "conflict": T["conflict"].to_numpy(),
        "1 - confidence": 1.0 - T["confidence"].to_numpy(),
        "novelty": T["novelty"].to_numpy(),
    }
    correct = T["correct"].to_numpy().astype(bool)
    X0 = (
        np.asarray(X, float) if preprocess is None else preprocess(np.asarray(X, float))
    )
    correct0 = triage.assess(X0, y).table["correct"].to_numpy().astype(bool)
    clean = p.degraded == 0
    rows = []
    for name in list(dict.fromkeys(p.kind[~clean])) + ["all"]:
        m = ~clean if name == "all" else (p.kind == name)
        sel = m | clean
        row = {"family": name, "n": int(m.sum())}
        for k, v in scores.items():
            row[k] = (
                roc_auc_score(p.degraded[sel], v[sel])
                if m.any() and clean.any()
                else np.nan
            )
        row["accuracy_before"] = float(correct0[m].mean()) if m.any() else np.nan
        row["accuracy_after"] = float(correct[m].mean()) if m.any() else np.nan
        row["accuracy_cost_pp"] = 100 * (row["accuracy_before"] - row["accuracy_after"])
        rows.append(row)
    return pd.DataFrame(rows)
