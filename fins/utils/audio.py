import torch
import scipy.signal
import numpy as np
from typing import List
from fft_conv_pytorch import fft_conv
import librosa


def load_audio(path, target_sr: int = 48000, mono=False, offset=0.0, duration=None) -> np.ndarray:
    """
    return y : shape=(n_channels, n_samples)
    """
    y, orig_sr = librosa.load(path, sr=None, mono=mono, offset=offset, duration=duration)

    if target_sr:
        y = resample(y, orig_sr=orig_sr, target_sr=target_sr)

    return np.atleast_2d(y)


def resample(signal: np.ndarray, orig_sr: int, target_sr: int, **kwargs) -> np.ndarray:
    """signal: (N,) or (num_channel, N)"""
    return librosa.resample(y=signal, orig_sr=orig_sr, target_sr=target_sr, res_type="polyphase", **kwargs)


def crop_rir(rir, target_length):
    n_channels, num_samples = rir.shape

    # by default: all the test rirs will be aligned such that direct impulse starts with 90 sample delay@48kHz
    if num_samples < target_length:
        out_rir = np.zeros((n_channels, target_length))
        out_rir[:, :num_samples] = rir
    else:
        out_rir = rir[:, :target_length]

    return out_rir


def get_octave_filters():
    """10 octave bandpass filters, each with order 1023
    Return
        firs : shape = (10, 1, 1023)
    """
    f_bounds = []
    f_bounds.append([22.3, 44.5])
    f_bounds.append([44.5, 88.4])
    f_bounds.append([88.4, 176.8])
    f_bounds.append([176.8, 353.6])
    f_bounds.append([353.6, 707.1])
    f_bounds.append([707.1, 1414.2])
    f_bounds.append([1414.2, 2828.4])
    f_bounds.append([2828.4, 5656.8])
    f_bounds.append([5656.8, 11313.6])
    f_bounds.append([11313.6, 22627.2])

    firs: List = []
    for low, high in f_bounds:
        fir = scipy.signal.firwin(
            1023,
            np.array([low, high]),
            pass_zero='bandpass',
            window='hamming',
            fs=48000,
        )
        firs.append(fir)

    firs = np.array(firs)
    firs = np.expand_dims(firs, 1)
    return firs


def batch_convolution(signal, filter):
    """Performs batch convolution with pytorch fft convolution.
    Args
        signal : torch.FloatTensor (batch, n_channels, num_signal_samples)
        filter : torch.FloatTensor (batch, n_channels, num_filter_samples)
    Return
        filtered_signal : torch.FloatTensor (batch, n_channels, num_signal_samples)
    """
    batch_size, n_channels, signal_length = signal.size()
    _, _, filter_length = filter.size()

    # Pad signal in the beginning by the filter size
    padded_signal = torch.nn.functional.pad(signal, (filter_length, 0), 'constant', 0)

    # Transpose : move batch to channel dim for group convolution
    padded_signal = padded_signal.transpose(0, 1)

    filtered_signal = fft_conv(padded_signal.double(), filter.double(), padding=0, groups=batch_size).transpose(0, 1)[
        :, :, :signal_length
    ]

    filtered_signal = filtered_signal.type(signal.dtype)

    return filtered_signal


def add_noise_batch(batch_signal, noise, snr_db):
    """Add noise to signal with the given SNR
    Args
        batch_signal : torch.FloatTensor. shape=(batch, 1, signal_length)
        noise : torch.FloatTensor. shape=(batch, 1, signal_length)
        snr_db : torch.FloatTensor. shape=(batch, 1)
    Return
        noise_added_signal : torch.FloatTensor. shape=(batch, 1, signal_length)
    """
    b, n, l = batch_signal.size()

    mean_square_signal = torch.mean(batch_signal**2, dim=2)
    signal_level_db = 10 * torch.log10(mean_square_signal)
    noise_db = signal_level_db - snr_db
    mean_square_noise = torch.sqrt(10 ** (noise_db / 10))
    mean_square_noise = torch.unsqueeze(mean_square_noise, dim=2)
    mean_square_noise = mean_square_noise.repeat(1, 1, l)
    modified_noise = torch.mul(noise, mean_square_noise)

    return batch_signal + modified_noise


def rms_normalize(sig: np.ndarray, rms_level=0.1):
    """
    sig : shape=(channel, signal_length)
    rms_level : linear gain value
    """
    # linear rms level and scaling factor
    # r = 10 ** (rms_level / 10.0)
    a = np.sqrt((sig.shape[1] * rms_level**2) / (np.sum(sig**2) + 1e-7))

    # normalize
    y = sig * a
    return y


def rms_normalize_batch(sig: torch.tensor, rms_level=0.1):
    """
    sig : shape=(batch, channel, signal_length)
    """
    # linear rms level and scaling factor
    # r = 10 ** (rms_level / 10.0)
    a = torch.sqrt((sig.size(2) * rms_level**2) / (torch.sum(sig**2, dim=2, keepdims=True) + 1e-7))
    # normalize
    y = sig * a

    return y


def peak_normalize(sig: np.ndarray, peak_val):
    peak = np.max(np.abs(sig[:, :512]), axis=-1, keepdims=True)
    sig = np.divide(sig, peak + 1e-7)
    sig = sig * peak_val
    return sig


def peak_normalize_batch(sig, peak_val):
    """
    sig : shape=(batch, channel, signal_length)
    """
    peak = torch.max(torch.abs(sig), dim=2, keepdim=True).values
    sig = torch.div(sig, peak)
    sig *= peak_val
    return sig


def audio_normalize_batch(sig, type, rms_level=0.1, peak_val=1.0):
    if type == "peak":
        return peak_normalize_batch(sig, peak_val)
    else:
        return rms_normalize_batch(sig, rms_level)
