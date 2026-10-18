"""Spectral helpers for peak analysis."""

import numpy as np


def compute_peak_width_hz(freqs: np.ndarray, mag: np.ndarray, gi: int) -> float:
    """Return the -3 dB (half-power) bandwidth of a spectral peak, in Hz.

    Walks outward from bin ``gi`` on each side until the magnitude falls to
    ``mag[gi] / sqrt(2)``, then linearly interpolates between bins for
    sub-bin accuracy. If a side drops below half power immediately, that edge
    is taken as half a bin from the peak.

    Args:
        freqs: Uniformly spaced bin frequencies in Hz.
        mag: Linear (amplitude) magnitude spectrum, same length as ``freqs``.
        gi: Index of the peak bin.

    Returns:
        Peak width in Hz, floored at one tenth of the bin spacing. Returns
        ``inf`` if the peak magnitude is not positive.
    """
    peak_mag = mag[gi]
    if peak_mag <= 0:
        return float("inf")
    half_power = peak_mag / np.sqrt(2.0)
    freq_res = freqs[1] - freqs[0]

    i = gi
    while i > 0 and mag[i] > half_power:
        i -= 1
    if i == gi:
        left_freq = freqs[gi] - freq_res * 0.5
    else:
        y0, y1 = mag[i], mag[i + 1]
        frac = (half_power - y0) / (y1 - y0) if y1 != y0 else 0.0
        left_freq = freqs[i] + frac * freq_res

    j = gi
    n = len(mag)
    while j < n - 1 and mag[j] > half_power:
        j += 1
    if j == gi:
        right_freq = freqs[gi] + freq_res * 0.5
    else:
        y0, y1 = mag[j - 1], mag[j]
        frac = (half_power - y0) / (y1 - y0) if y1 != y0 else 0.0
        right_freq = freqs[j - 1] + frac * freq_res

    return float(max(right_freq - left_freq, freq_res * 0.1))
