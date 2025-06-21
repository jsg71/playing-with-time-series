import numpy as np


def bandpass_filter(
    sig: np.ndarray, fs: int, low: float = 300.0, high: float = 30000.0
) -> np.ndarray:
    """Simple FFT-based band-pass filter.

    Parameters
    ----------
    sig : np.ndarray
        1-D waveform array.
    fs : int
        Sample rate in Hz.
    low : float
        Low cut-off in Hz.
    high : float
        High cut-off in Hz.
    Returns
    -------
    np.ndarray
        Filtered signal of the same shape.
    """
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    mask = (freqs >= low) & (freqs <= high)
    spec *= mask
    return np.fft.irfft(spec, n=len(sig)).astype(sig.dtype)


def detect_events(waves: dict, fs: int, threshold: float = 0.02) -> list[int]:
    """Rudimentary event detector inspired by Tian et al. (2025).

    This computes the RMS across stations and returns indices where the
    amplitude exceeds ``threshold``. It mimics the cross-station matching step
    in the paper but is intentionally simple.
    """
    stack = np.stack(list(waves.values()))
    rms = np.sqrt(np.mean(stack**2, axis=0))
    mask = rms > threshold
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    events = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > int(0.0005 * fs):
            events.append((start + idx[i - 1]) // 2)
            start = idx[i]
    events.append((start + idx[-1]) // 2)
    return events
