import os
from pathlib import Path
import random

random.seed(0)


def split(files, train_ratio, valid_ratio):
    assert train_ratio + valid_ratio < 1
    n_train = int(len(files) * train_ratio)
    n_valid = int(len(files) * valid_ratio)
    train_files = files[:n_train]
    valid_files = files[n_train : n_train + n_valid]
    test_files = files[n_train + n_valid :]
    return train_files, valid_files, test_files


def load_bird(path_to_dataset):
    rir_files = list(Path(path_to_dataset).rglob("*.flac"))
    random.shuffle(rir_files)

    train_files, valid_files, test_files = split(rir_files, 0.7, 0.1)
    return train_files, valid_files, test_files


def load_rir_dataset():
    train_files, valid_files, test_files = load_bird("./dataset/BIRD")
    return train_files, valid_files, test_files


def load_speech_dataset():
    speech_files = list(Path("./dataset/DAPS").rglob("*.wav"))
    random.shuffle(speech_files)

    train_files, valid_files, test_files = split(speech_files, 0.7, 0.1)
    return train_files, valid_files, test_files


if __name__ == "__main__":
    train_rir_files, valid_rir_files, test_rir_files = load_rir_dataset()
    train_speech_files, valid_speech_files, test_speech_files = load_speech_dataset()

    print(len(train_rir_files), len(valid_rir_files), len(test_rir_files))
    print(len(train_speech_files), len(valid_speech_files), len(test_speech_files))
