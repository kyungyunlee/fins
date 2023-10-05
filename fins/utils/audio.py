import torch
import scipy.signal
import numpy as np
from typing import List
from fft_conv_pytorch import fft_conv


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
