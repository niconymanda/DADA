"""
Dataset classes for 
1. ASVspoof 2021 dataset : ASVSpoof21Dataset
2. In the Wild dataset : InTheWildDataset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import librosa
import yaml
from transformers import AutoTokenizer, DebertaV2Tokenizer

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
        transcription_col='content',
        sampling_rate=16000,
        max_duration=4,
        split="train",
        config=None,
        max_text_length = 64,
        mode="classification",
        text_tokenizer_name = None,
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            metadata_file (str): Name of the metadata file
            include_spoofs (bool): Whether to include spoofed data
            bonafide_label (str): Label for bonafide data
            sampling_rate (int): Sampling rate of the audio files
            max_duration (int): Maximum duration of the audio files in seconds
            split (str): Split of the dataset (train, val, test)
            config (dict): Configuration dictionary
            mode (str): Mode of the dataset (triplet, classification)
        """

        if text_tokenizer_name is None:
            raise NotImplementedError("Text tokenizer is required for this dataset")
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.max_text_length = max_text_length

        self.root_dir = root_dir
        self.metadata_file = metadata_file
        if self.metadata_file.endswith(".csv"):
            self.df = pd.read_csv(os.path.join(root_dir, metadata_file))
        elif self.metadata_file.endswith(".json"):
            self.df = pd.read_json(os.path.join(root_dir, metadata_file))
        self.include_spoofs = include_spoofs
        self.split = split
        self.mode = mode

        self.bonafide_label = bonafide_label

        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.cut = self.sampling_rate * self.max_duration  # padding

        # Filter out spoofed data if include_spoofs is False
        if not self.include_spoofs:
            self.df = self.df[self.df["label"] == bonafide_label]

        # Filter out data based on filename in config['split']
        if config is not None:
            self.config = yaml.safe_load(open(config, "r"))
            self.df = self.df[self.df[filename_col].isin(self.config[split])]

        # Create id to filename mapping and id to label mapping
        ids = np.arange(len(self.df))
        filenames = self.df[filename_col].values
        labels = self.df["label"].values
        authors = self.df["speaker"].values
        transcriptions = self.df[transcription_col].values

        self.id_to_filename = dict(zip(ids, filenames))
        self.id_to_label = dict(zip(ids, labels))
        self.id_to_author = dict(zip(ids, authors))
        self.id_to_transcription = dict(zip(ids, transcriptions))

    def text_to_input_dict(self, text):
        inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
        )

        inputs = {k: v.squeeze() for k, v in inputs.items()}

        return inputs

    def load_audio_tensor(self, idx):
        # Load audio file
        # print(idx)
        try:
            filename = os.path.join(self.root_dir, self.id_to_filename[idx])
        except Exception as e:
            print(e)
            print(idx)
        audio_arr, _ = load_audio(filename, self.sampling_rate)
        audio_tensor = torch.tensor(pad(audio_arr, self.cut)).float()
        return audio_tensor

    def get_triplets_from_anchor(self, anchor_idx):
        anchor_author = self.id_to_author[anchor_idx]
        positive_id = np.random.choice(
            np.where(self.df["speaker"].values == anchor_author)[0]
        )
        negative_ids = np.where(self.df["speaker"].values != anchor_author)[0]

        return positive_id, np.random.choice(negative_ids)

    def __len__(self):
        return len(self.id_to_filename)

    def __getitem__(self, idx):
        x = self.load_audio_tensor(idx)
        y = int(self.id_to_label[idx] == self.bonafide_label)
        author = self.id_to_author[idx]
        transcription = self.text_to_input_dict(self.id_to_transcription[idx])

        if self.mode == "classification":
            return {"x": x, "label": y, "author": author, "transcription": transcription}

        elif self.mode == "triplet":
            id_p, id_n = self.get_triplets_from_anchor(idx)

            x_p = self.load_audio_tensor(id_p)
            x_n = self.load_audio_tensor(id_n)

            return {"anchor": x, "positive": x_p, "negative": x_n}
        
        elif self.mode == "pair":
            a = x
            a_label = author

            # Choose another idx at random
            idx2 = np.random.choice(np.arange(len(self.df)))
            b = self.load_audio_tensor(idx2)
            b_label = self.id_to_author[idx2]

            return {"a": a, "b": b, "a_label": a_label, "b_label": b_label}

        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")