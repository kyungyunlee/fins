# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, fft_size // 2 + 1, #Frames).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8))

    return x_mag


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B,  #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B,  #freq_bins, #frames).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """

        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        log_mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[64, 512, 2048, 8192],
        hop_sizes=[32, 256, 1024, 4096],
        win_lengths=[64, 512, 2048, 8192],
        window="hann_window",
        sc_weight=1.0,
        mag_weight=1.0,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses = self.stft_losses + [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, 1, T).
            y (Tensor): Groundtruth signal (B, 1, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        x = x.squeeze(1)
        y = y.squeeze(1)

        for i, f in enumerate(self.stft_losses):
            sc_l, mag_l = f(x, y)
            sc_loss = sc_loss + sc_l
            mag_loss = mag_loss + mag_l

        return {
            "total": (sc_loss * self.sc_weight + mag_loss * self.mag_weight) / len(self.stft_losses),
            "sc_loss": sc_loss / len(self.stft_losses),
            "mag_loss": mag_loss / len(self.stft_losses),
        }
