import typing as tp
import numpy as np
from scipy.signal import stft

EPS = np.finfo(float).eps
np.random.seed(0)


def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio


# ported from https://github.com/fgnt/sms_wsj/blob/master/sms_wsj/reverb/reverb_utils.py#L170
def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min(
            [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
        )

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample


def find_rir_onset_spectral(
    rir: np.ndarray,
    fs: int,
    window_ms: float = 10.0,
    hop_ms: float = 0.125,
    noise_scale: float = 1.0,
) -> tp.Tuple[int, float]:
    """
    Find the onset of a Room Impulse Response using the DS (Mean over Spectra) method.

    The DS method computes a spectral energy envelope via STFT, then finds the onset
    as the time step with the maximum successive frame ratio:
        Ẽ(n) = Σ_k |X(n·h, k)|
        t0 = h * argmax_n( Ẽ(n+1) / Ẽ(n) )

    Reference:
        Defrance et al., "Finding the onset of a room impulse response: Straightforward?"
        JASA Express Letters, 2008. DOI: 10.1121/1.2960935

    Parameters
    ----------
    rir : np.ndarray
        1-D array containing the room impulse response samples.
    fs : int
        Sampling frequency in Hz.
    window_ms : float
        STFT analysis window length in milliseconds (default: 10 ms).
    hop_ms : float
        STFT hop size in milliseconds (default: 1 ms).

    Returns
    -------
    onset_sample : int
        Sample index of the detected onset.
    onset_time_s : float
        Time in seconds of the detected onset.
    """
    win_len = int(round(window_ms * fs / 1000))
    hop_len = int(round(hop_ms * fs / 1000))

    # Add noise at the noise floor level.
    # This floors E_tilde in silent regions so ratios stay ~1 there,
    # while the true onset still produces a clean spike.
    # tail_rms = np.sqrt(np.mean(rir[-win_len:] ** 2))
    # global_rms = np.sqrt(np.mean(rir ** 2))
    # noise_rms = tail_rms if tail_rms > 0.0 else global_rms
    noise_rms = np.sqrt(np.mean(rir ** 2))
    rir_noisy = rir + noise_scale * noise_rms * np.random.normal(size=(len(rir),))

    prefix = rir_noisy[-win_len:]
    rir_padded = np.concatenate([prefix, rir_noisy])
    # Index mapping: rir_padded[i] == rir[i - win_len]

    _, _, Zxx = stft(
        rir_padded,
        fs=fs,
        window='boxcar',    # rectangular window → equal weight, no tapering
        nperseg=win_len,
        noverlap=win_len - hop_len,
        boundary=None,      # no extra padding — we handle prefix manually
        padded=False,
    )

    E_tilde = np.abs(Zxx).sum(axis=0)
    peak_frame = int(np.argmax(E_tilde))
    E_search = E_tilde[:peak_frame + 1]

    ratio = E_search[1:] / E_search[:-1]
    argmax_frame = int(np.argmax(ratio))

    # With rectangular window and no boundary centering:
    # Frame n covers rir_padded[n*hop_len : n*hop_len + win_len]
    # New samples entering frame n+1 (not in frame n):
    #   rir_padded[argmax_frame*hop_len + win_len : (argmax_frame+1)*hop_len + win_len]
    # Onset is at the start of these new samples in rir_padded:
    #   argmax_frame*hop_len + win_len
    # Subtract prefix (win_len) → rir index:
    #   argmax_frame*hop_len + win_len - win_len = argmax_frame*hop_len
    onset_sample = int(np.clip(argmax_frame * hop_len, 0, len(rir) - 1))
    onset_time_s = onset_sample / fs

    return onset_sample, onset_time_s


def active_rms_relative(
    wav: np.ndarray,
    fs: int = 16000,
    relative_threshold: float = -25.0,
    absolute_threshold: float = -50.0
):
    '''Returns the RMS calculated only in the active portions'''
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    
    wav_len = len(wav) // window_samples * window_samples
    if wav_len == 0:
        return 0.
    wav = wav[:wav_len].reshape((-1, window_samples))

    seg_rms = np.sqrt(np.square(wav).mean(1))
    threshold = seg_rms.max() * 10 ** (relative_threshold / 20)
    if absolute_threshold is not None:
        threshold = max(threshold, 10 ** (absolute_threshold / 20))
    active = seg_rms > threshold
    if not np.any(active):
        return 0.
    active_seg_rms = (seg_rms * active).sum() / active.sum()
    return active_seg_rms.item()
