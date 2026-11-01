"""Acquisition-level checks that complement the band audit.

The band audit asks whether the regions of a spectrum agree about the specimen.
A degradation that acts identically on every channel, such as the extra shot
noise of a short exposure, gives the bands nothing to disagree about and is
invisible to it by construction. Those degradations are better caught where they
arise, in the signal itself. :func:`noise_level` is the simplest such check, and
:func:`noise_limit` turns it into a pass/fail limit learned from clean spectra.
"""

from __future__ import annotations

import numpy as np

__all__ = ["noise_level", "noise_limit"]


def noise_level(X, *, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Per-spectrum noise estimate: robust spread of the Savitzky-Golay residual.

    The spectrum is smoothed with a Savitzky-Golay filter and the noise is the
    median absolute deviation of what the filter removes, scaled by 1.4826 so
    that it estimates a standard deviation under Gaussian noise. The median makes
    the estimate insensitive to a few sharp Raman peaks or cosmic-ray spikes.

    Compute it on a representation in which intensity still means something:
    raw counts, or a per-spectrum min-max scaling. After SNV every spectrum has
    unit variance, which hides the loss of signal that makes a short exposure
    noisy.

    Parameters
    ----------
    X
        Spectra, shape ``(n_samples, n_channels)``.
    window, polyorder
        Savitzky-Golay window length (odd, in channels) and polynomial order.
        The window should be wider than the noise correlation and narrower than
        the narrowest Raman band.

    Returns
    -------
    ndarray of shape ``(n_samples,)``, in the units of ``X``.
    """
    from scipy.signal import savgol_filter

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and larger than polyorder")
    if X.shape[1] < window:
        raise ValueError(
            f"spectra have {X.shape[1]} channels, fewer than window={window}"
        )
    r = X - savgol_filter(X, window, polyorder, axis=1)
    med = np.median(r, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(r - med), axis=1)


def noise_limit(X_clean, *, q: float = 0.99, **kwargs) -> float:
    """Upper limit on :func:`noise_level`, the ``q`` quantile over clean spectra.

    >>> limit = noise_limit(X_reference)                     # doctest: +SKIP
    >>> too_noisy = noise_level(X_new) > limit               # doctest: +SKIP
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    return float(np.quantile(noise_level(X_clean, **kwargs), q))
