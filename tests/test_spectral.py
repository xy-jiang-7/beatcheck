import numpy as np
import pytest

from beatcheck import compute_peak_width_hz


def test_peak_width_of_hann_windowed_stereo_sines():
    sr = 8000
    n = 8000  # 1 Hz bin spacing
    t = np.arange(n) / sr
    stereo = np.stack([np.sin(2 * np.pi * 200.0 * t), np.sin(2 * np.pi * 210.0 * t)])
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)

    for channel, f0 in zip(stereo, (200.0, 210.0)):
        mag = np.abs(np.fft.rfft(channel * window))
        gi = int(np.argmax(mag))
        assert freqs[gi] == f0
        width = compute_peak_width_hz(freqs, mag, gi)
        # Bin-centred Hann peak: neighbours sit at 0.5 of the peak, so linear
        # interpolation to 1/sqrt(2) gives 2 * (1 - (1/sqrt(2) - 0.5) / 0.5) bins.
        expected = 2 * (1 - (1 / np.sqrt(2) - 0.5) / 0.5)
        assert width == pytest.approx(expected, abs=1e-3)
