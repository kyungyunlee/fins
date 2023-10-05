import random
import numpy as np
import colorednoise
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import pyloudnorm as pyln

from fins.utils.dsp import load_audio, crop_rir, apply_exponential_weighting, peak_normalize


class ReverbDataset(Dataset):
    """MONO RIR"""

    def __init__(
        self,
        rir_files,
        source_files,
        config,
        use_noise,
    ):
        """
        Args
            rir_file : list of rir audio files
            source_files : list of speech files
        """
        self.rir_files = rir_files
        self.source_files = source_files
        self.config = config
        self.use_noise = use_noise  # Add white noise to handle noisy environment

        self.rir_length = int(config.rir_duration * config.sr)  # 1 sec = 48000 samples
        self.input_signal_length = config.input_length  # 131070 samples

        # Group RIRs per room
        self.rir_room_list = list(self.rir_dict.keys())

    def __len__(self):
        return len(self.rir_dict)

    def __getitem__(self, idx):
        rir_file = self.rir_files[idx]

        rir = self._load_rir(rir_file)

        flipped_rir = np.flip(rir, 1).copy()
        rir = np.float32(rir)
        flipped_rir = np.float32(flipped_rir)

        source, source_f = self._get_source()

        if self.use_noise:
            if random.random() < 0.9:
                min_snr = 0.0
                max_snr = 30.0
                beta = random.random() + 1.0
                noise = colorednoise.powerlaw_psd_gaussian(beta, self.input_signal_length)
                noise = np.expand_dims(noise, 0)
                snr_db = np.array([random.random() * (max_snr - min_snr) + min_snr])
            else:
                noise = np.zeros_like(source)
                snr_db = np.array([0.0])

        else:
            noise = np.zeros_like(source)
            snr_db = np.array([0.0])

        noise = np.float32(noise)
        snr_db = np.float32(snr_db)

        return {"rir": rir, "flipped_rir": flipped_rir, "source": source, "noise": noise, "snr_db": snr_db}

    def _load_rir(self, rir_f):
        """Load RIR as 2D signal (channel, n_samples)"""

        rir = load_audio(rir_f, target_sr=self.config.sr)[0:1]  # Using only one channel for mono RIR experiment
        cropped_rir = crop_rir(rir, target_length=self.rir_length)

        # TODO : normalize rir

        return cropped_rir

    def _load_and_pad(self, audio_f, offset, duration, target_length):
        """
        audio_length : length of audio file in samples
        target_length : target length in samples
        """

        audio = load_audio(audio_f, target_sr=self.config.sr, mono=True, offset=offset, duration=duration)
        # audio = audio[0:1]  # take the first channel

        # remove DC components
        audio = audio - np.mean(audio)
        n_channels, audio_length = audio.shape

        if audio_length < target_length:
            # Pad
            target_audio = np.zeros((n_channels, target_length))
            target_audio[:, :audio_length] = audio
        else:
            # crop
            target_audio = audio[:, :target_length]

        return target_audio

    def _get_source(self):
        source_idx = random.randint(0, len(self.source_files) - 1)
        source_file, source_loudness, candidate_start_times = self.source_files[source_idx]
        offset = random.choice(candidate_start_times)

        source = self._load_and_pad(source_file, offset, duration=3.0, target_length=self.input_signal_length)

        # Remove empty
        rms = np.sqrt(np.mean(source**2))
        while rms < 0.001:
            # Re select segment
            source_idx = random.randint(0, len(self.source_files) - 1)
            source_file, source_loudness, candidate_start_times = self.source_files[source_idx]
            offset = random.choice(candidate_start_times)
            source = self._load_and_pad(source_file, offset, duration=3.0, target_length=self.input_signal_length)
            rms = np.sqrt(np.mean(source**2))

        # Normalize source at random loudness
        source *= 0.1

        source = np.float32(source)

        return source, source_file
