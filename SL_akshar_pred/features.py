
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import scipy.fftpack

def load_wav_mono(path: str, target_sr: int = 16000):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return x, sr

def _hz_to_mel(hz): 
    return 2595 * np.log10(1 + hz / 700)

def _mel_to_hz(m): 
    return 700 * (10**(m / 2595) - 1)

def mfcc_energy_zcr(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    win_ms: int = 20,
    hop_ms: int = 5,
    n_fft: int = 512,
    n_mels: int = 26,
):
    """Return (T,15): 13 MFCC + energy + ZCR per frame."""
    sig = signal.astype(np.float32)
    if sig.size == 0:
        sig = np.zeros(int(sr * win_ms / 1000), dtype=np.float32)

    # Pre-emphasis
    sig = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])

    frame_len = int(sr * win_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    if sig.size < frame_len:
        sig = np.pad(sig, (0, frame_len - sig.size))

    num_frames = 1 + (sig.size - frame_len) // hop_len
    idx = (
        np.tile(np.arange(frame_len), (num_frames, 1))
        + np.tile(np.arange(num_frames) * hop_len, (frame_len, 1)).T
    )
    frames = sig[idx] * np.hamming(frame_len)

    # Energy + ZCR
    energy = np.mean(frames**2, axis=1, keepdims=True)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) / 2, axis=1, keepdims=True)

    # Power spectrum
    mag = np.abs(np.fft.rfft(frames, n=n_fft))
    pow_spec = (1.0 / n_fft) * (mag**2)

    # Mel filterbank
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        if f_m_minus == f_m:
            f_m += 1
        if f_m == f_m_plus:
            f_m_plus += 1
        for k in range(f_m_minus, f_m):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    fb = np.dot(pow_spec, fbank.T)
    fb = np.where(fb == 0, np.finfo(np.float32).eps, fb)
    log_fb = np.log(fb)

    mfcc = scipy.fftpack.dct(log_fb, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    feats = np.concatenate([mfcc, energy, zcr], axis=1).astype(np.float32)  # (T,15)
    return feats
