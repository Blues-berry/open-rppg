from __future__ import annotations

import builtins

import heartpy as hp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def _as_1d_float(values):
    try:
        return np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return np.array([], dtype=np.float64)


def _fill_nonfinite(values):
    arr = _as_1d_float(values)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float64)
    x = np.arange(arr.size)
    arr = arr.copy()
    arr[~finite] = np.interp(x[~finite], x[finite], arr[finite])
    return arr


def _demean(values):
    arr = _fill_nonfinite(values)
    if arr.size == 0:
        return arr
    return arr - float(np.mean(arr))


def _zscore(values):
    arr = _demean(values)
    if arr.size == 0:
        return arr
    std = float(np.std(arr))
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / std


def SQI(signal, sr=30, min_freq=0.5, max_freq=3.0, window_size=10):
    def _SQI(window):
        window = _zscore(window)
        n = len(window)
        if n < 2:
            return 0.0
        autocorr = np.correlate(window, window, mode="full")[n - 1:]
        scale = autocorr[0]
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            return 0.0
        autocorr = autocorr / scale
        min_lag = builtins.max(1, int(sr / max_freq))
        max_lag = builtins.min(len(autocorr) - 1, int(sr / min_freq))
        if min_lag >= max_lag:
            return 0.0
        target_autocorr = autocorr[min_lag : max_lag + 1]
        if target_autocorr.size == 0:
            return 0.0
        peak_value = float(np.max(target_autocorr))
        if not np.isfinite(peak_value):
            return 0.0
        return builtins.max(0.0, builtins.min(1.0, peak_value))

    arr = _fill_nonfinite(signal)
    sample_rate = float(sr or 0)
    if arr.size < 2 or sample_rate <= 0:
        return 0.0
    window_len = builtins.max(2, round(window_size * sample_rate))
    if arr.size <= window_len:
        return _SQI(arr)
    steps = int(arr.size / window_len) + 1
    stride = int((arr.size - window_len) / (steps - 1)) if steps > 1 else 0
    sqis = []
    for index in range(steps):
        start = index * stride
        sqis.append(_SQI(arr[start : start + window_len]))
    return float(np.mean(sqis)) if sqis else 0.0


def get_hr(y, sr=30, min=30, max=180):
    arr = _zscore(y)
    sample_rate = float(sr or 0)
    if arr.size < 4 or sample_rate <= 0:
        return None
    nperseg = int(builtins.min(arr.size, builtins.max(2, 256 / 30 * sample_rate)))
    try:
        freqs, power = welch(arr, sample_rate, nfft=20000, nperseg=nperseg)
    except Exception:
        return None
    mask = (freqs > min / 60) & (freqs < max / 60)
    if not np.any(mask):
        return None
    masked_power = power[mask]
    if masked_power.size == 0 or not np.isfinite(masked_power).any():
        return None
    index = int(np.nanargmax(masked_power))
    return float(freqs[mask][index] * 60)


def get_prv(y, ts=None, sr=30):
    arr = _fill_nonfinite(y)
    if arr.size < 4:
        return {}
    try:
        measures, stats = hp.process(arr, sr, high_precision=True, clean_rr=True)
        rr_intervals = measures["RR_list"][np.where(1 - np.array(measures["RR_masklist"]))] / 1000
        if rr_intervals.size < 3:
            return {}
        t = np.cumsum(rr_intervals)
        resampled_rate = 4
        if t[-1] <= 0:
            return {}
        resampled = CubicSpline(t, rr_intervals)(np.arange(0, t[-1], 1 / resampled_rate))
        freqs, power = welch(resampled, fs=resampled_rate, nperseg=builtins.min(len(resampled), 256), nfft=4096)
        vlf = power[(freqs >= 0.0033) & (freqs < 0.04)].sum()
        lf = power[(freqs >= 0.04) & (freqs < 0.15)].sum()
        hf = power[(freqs >= 0.15) & (freqs < 0.4)].sum()
        tp = vlf + lf + hf
        ratio = lf / hf if hf else None
        return {**stats, **{"VLF": vlf, "TP": tp, "HF": hf, "LF": lf, "LF/HF": ratio}}
    except Exception:
        return {}


def detrend(signal, min_freq=0.5, sr=30):
    arr = _fill_nonfinite(signal)
    if arr.size < 3:
        return _demean(arr)
    sample_rate = float(sr or 30)
    min_frequency = float(min_freq or 0.5)
    try:
        lam = 50 * (sample_rate / 30) ** 2 * (0.5 / min_frequency) ** 2
        signal_length = arr.shape[0]
        diags_data = [
            np.ones(signal_length - 2),
            -2 * np.ones(signal_length - 2),
            np.ones(signal_length - 2),
        ]
        offsets = [0, 1, 2]
        d_matrix = diags(diags_data, offsets, shape=(signal_length - 2, signal_length), format="csc")
        h_matrix = eye(signal_length, format="csc")
        trend = spsolve(h_matrix + (lam**2) * (d_matrix.T @ d_matrix), arr)
        filtered = arr - trend
    except Exception:
        filtered = _demean(arr)
    return _fill_nonfinite(filtered)


def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    arr = _fill_nonfinite(data)
    sample_rate = float(fs or 0)
    if arr.size == 0:
        return arr
    if arr.size < 2 or sample_rate <= 0:
        return _demean(arr)
    nyquist = sample_rate / 2
    low = float(lowcut or 0) / nyquist
    high = builtins.min(float(highcut or 0) / nyquist, 0.98)
    if not (0 < low < high < 1):
        return _demean(arr)
    try:
        b, a = butter(order, [low, high], btype="band")
        padlen = 3 * (builtins.max(len(a), len(b)) - 1)
        if arr.size <= padlen:
            return _demean(arr)
        return _fill_nonfinite(filtfilt(b, a, arr))
    except Exception:
        return _demean(arr)


def norm_bvp(bvp, sr=30):
    arr = _zscore(detrend(_fill_nonfinite(bvp), sr=sr))
    if arr.size < 3:
        return arr
    prominence = (1.5, None)
    distance = builtins.max(1, int(0.25 * float(sr or 30)))
    positive = find_peaks(arr, prominence=prominence, distance=distance)[0]
    negative = find_peaks(-arr, prominence=prominence, distance=distance)[0]
    peaks = np.sort(np.concatenate([positive, negative]))
    if peaks.size < 2:
        return arr

    segments = []
    for start, end in zip(peaks, peaks[1:] + 1):
        segment = arr[start:end]
        if segment.size == 0:
            continue
        span = float(np.max(segment) - np.min(segment))
        if not np.isfinite(span) or span < 1e-8:
            continue
        centered = (segment - (np.max(segment) + np.min(segment)) / 2) / span
        segments.append(centered)
    if not segments:
        return arr

    stitched = np.concatenate([segment[:-1] for segment in segments[:-1]] + segments[-1:])
    stitched = _zscore(stitched)
    normalized = arr.copy()
    usable = builtins.min(stitched.size, peaks[-1] + 1 - peaks[0])
    if usable > 0:
        normalized[peaks[0] : peaks[0] + usable] = stitched[:usable]
    upper = float(np.max(stitched)) if stitched.size else 2.0
    if not np.isfinite(upper) or upper <= -2:
        upper = 2.0
    return np.clip(normalized, -2, upper)
