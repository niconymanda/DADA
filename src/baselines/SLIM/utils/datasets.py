"""

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import librosa

SUPPORTED_FORMATS = ["wav", "mp3", "flac"]


def load_audio(filename, sampling_rate=None):
    # Load audio file
    assert os.path.exists(filename), f"File {filename} does not exist"
    assert (
        filename.split(".")[-1] in SUPPORTED_FORMATS
    ), f"File {filename} is not supported"

    # Load audio file
    audio, sr = librosa.load(filename, sr=None)
    return audio, sr


def pad(x, max_len):
    """
    From src/baselines/asvspoof2021/DF/Baseline-RawNet2/data_utils.py
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def get_spoof_list(meta_dir, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(meta_dir, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


class ASVSpoof21Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        meta_dir,
        is_train=False,
        is_eval=False,
        sampling_rate=16000,
        max_duration=4,
    ):
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.cut = self.sampling_rate * self.max_duration  # padding
        self.list_IDs = get_spoof_list(meta_dir, is_train, is_eval)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def load_audio_tensor(self, key):
        filename = os.path.join(self.root_dir, f"flac/{key}.flac")
        audio_arr, _ = load_audio(filename, self.sampling_rate)
        audio_tensor = torch.tensor(pad(audio_arr, self.cut)).float()
        return audio_tensor

    def __getitem__(self, idx):
        y = self.list_IDs[idx]
        x = self.load_audio_tensor(y)
        return x, y


class InTheWildDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        metadata_file="meta.csv",
        include_spoofs=False,
        bonafide_label="bona-fide",
        filename_col="file",
        sampling_rate=16000,
        max_duration=4,
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            metadata_file (str): Name of the metadata file
            include_spoofs (bool): Whether to include spoofed data
            bonafide_label (str): Label for bonafide data
            sampling_rate (int): Sampling rate of the audio files
            max_duration (int): Maximum duration of the audio files in seconds
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, metadata_file))
        self.include_spoofs = include_spoofs

        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.cut = self.sampling_rate * self.max_duration  # padding

        # Filter out spoofed data if include_spoofs is False
        if not self.include_spoofs:
            self.df = self.df[self.df["label"] == bonafide_label]

        # Create id to filename mapping and id to label mapping
        ids = np.arange(len(self.df))
        filenames = self.df[filename_col].values
        labels = self.df["label"].values

        self.id_to_filename = dict(zip(ids, filenames))
        self.id_to_label = dict(zip(ids, labels))

    def load_audio_tensor(self, idx):
        # Load audio file
        filename = os.path.join(self.root_dir, self.id_to_filename[idx])
        audio_arr, _ = load_audio(filename, self.sampling_rate)
        audio_tensor = torch.tensor(pad(audio_arr, self.cut)).float()
        return audio_tensor

    def __len__(self):
        return len(self.id_to_filename)

    def __getitem__(self, idx):
        x = self.load_audio_tensor(idx)
        y = self.id_to_label[idx]
        return x, y
