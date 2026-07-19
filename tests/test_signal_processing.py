import unittest

import numpy as np

from rppg.signal import SQI, bandpass_filter, get_hr, norm_bvp


class SignalProcessingTest(unittest.TestCase):
    def test_bandpass_short_signal_returns_fallback(self):
        signal = np.sin(np.linspace(0, 1, 10))
        filtered = bandpass_filter(signal, fs=30)
        self.assertEqual(filtered.shape, signal.shape)
        self.assertTrue(np.isfinite(filtered).all())

    def test_bandpass_low_sample_rate_returns_fallback(self):
        signal = np.sin(np.linspace(0, 4, 60))
        filtered = bandpass_filter(signal, fs=4)
        self.assertEqual(filtered.shape, signal.shape)
        self.assertTrue(np.isfinite(filtered).all())

    def test_norm_bvp_all_nan_is_finite(self):
        normalized = norm_bvp(np.full(60, np.nan), sr=30)
        self.assertEqual(normalized.shape, (60,))
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.allclose(normalized, 0))

    def test_norm_bvp_flat_signal_is_finite(self):
        normalized = norm_bvp(np.ones(60), sr=30)
        self.assertEqual(normalized.shape, (60,))
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.allclose(normalized, 0))

    def test_norm_bvp_handles_nan_segments(self):
        signal = np.sin(np.linspace(0, 12, 240))
        signal[20:30] = np.nan
        signal[120] = np.inf
        normalized = norm_bvp(signal, sr=30)
        self.assertEqual(normalized.shape, signal.shape)
        self.assertTrue(np.isfinite(normalized).all())

    def test_hr_and_sqi_for_sinusoid(self):
        sr = 30
        bpm = 72
        t = np.arange(sr * 20) / sr
        signal = np.sin(2 * np.pi * (bpm / 60) * t)
        self.assertAlmostEqual(get_hr(signal, sr=sr), bpm, delta=2)
        self.assertGreaterEqual(SQI(signal, sr=sr), 0)


if __name__ == "__main__":
    unittest.main()
