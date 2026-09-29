"""Controlled corruption of band-level evidence.

Stress-testing the negotiation means injecting disagreement of a known size in a
known place. These functions perturb the *evidence* a band contributes, not the
raw signal, so the induced conflict is exactly localised and the base models
never need refitting.

Modes
-----
``"noise"``
    Affected bands are replaced by random evidence. Conflict is spread evenly
    across however many bands were selected.
``"spike"``
    As ``"noise"``, but a subset of the affected bands is corrupted far more
    violently, concentrating the conflict.
``"silence"``
    Affected bands are both randomised and frozen, so they never update. Models
    a band whose measurement is missing or unusable.
``"swap"``
    Affected bands are given another class's evidence, which is the sharpest
    form of disagreement: confident, coherent and wrong.
"""

from __future__ import annotations

import numpy as np

__all__ = ["perturb_bands", "MODES"]

MODES = ("noise", "spike", "silence", "swap")


def perturb_bands(
    band_logits: np.ndarray,
    fraction: float,
    mode: str = "noise",
    *,
    scale: float | None = None,
    spike_factor: float = 3.0,
    spike_share: float = 0.3,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrupt a fraction of each sample's bands.

    Parameters
    ----------
    band_logits
        Clean evidence, shape ``(n_samples, n_bands, n_classes)``.
    fraction
        Proportion of bands to affect, in ``[0, 1]``. The count is
        ``floor(fraction * n_bands)`` and is identical across modes, so modes are
        directly comparable at matched fraction.
    mode
        One of :data:`MODES`.
    scale
        Standard deviation of the injected evidence. Defaults to the standard
        deviation of ``band_logits``, so the corruption is calibrated to the
        problem rather than to an arbitrary constant.
    spike_factor, spike_share
        For ``"spike"``: how much louder the concentrated bands are, and what
        share of the affected bands get that treatment.
    random_state
        Seed or generator.

    Returns
    -------
    damaged : ndarray
        Perturbed evidence, same shape as the input.
    frozen : ndarray
        Boolean mask of bands that must not update, shape ``(n_samples, n_bands)``.
        All ``False`` except in ``"silence"`` mode.
    affected : ndarray
        Boolean mask of which bands were corrupted, shape ``(n_samples, n_bands)``.
    """
    L = np.asarray(band_logits, dtype=float)
    if L.ndim != 3:
        raise ValueError(f"band_logits must be 3-D, got shape {L.shape}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
    n, k, c = L.shape
    rng = np.random.default_rng(random_state)

    frac = float(np.clip(fraction, 0.0, 1.0))
    n_hit = int(np.floor(frac * k + 1e-9))
    damaged = L.copy()
    frozen = np.zeros((n, k), dtype=bool)
    affected = np.zeros((n, k), dtype=bool)
    if n_hit == 0:
        return damaged, frozen, affected

    if scale is None:
        s = float(np.std(L))
        scale = s if s > 1e-8 else 1.0

    order = np.argsort(rng.random((n, k)), axis=1)
    hit = order[:, :n_hit]
    np.put_along_axis(affected, hit, True, axis=1)

    if mode == "swap":
        shift = rng.integers(1, c, size=(n, n_hit)) if c > 1 else np.zeros((n, n_hit), int)
        for row in range(n):
            for slot, b in enumerate(hit[row]):
                damaged[row, b] = np.roll(L[row, b], int(shift[row, slot]))
        return damaged, frozen, affected

    noise = rng.normal(0.0, scale, size=(n, n_hit, c))
    if mode == "spike":
        n_spike = max(1, int(np.ceil(spike_share * n_hit)))
        noise[:, :n_spike, :] *= float(spike_factor)

    rows = np.repeat(np.arange(n), n_hit)
    damaged[rows, hit.ravel()] = noise.reshape(-1, c)
    if mode == "silence":
        frozen = affected.copy()
    return damaged, frozen, affected
