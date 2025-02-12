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
import json

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


def get_spoof_list19(meta_dir, is_train=False, is_eval=False):
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


def get_spoof_list(meta_dir, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(meta_dir, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            # _, key, _, _, label = line.strip().split(" ")

            key, label = line.split(" ")[1], line.split(" ")[5]
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return None, file_list
    else:
        for line in l_meta:
            key, label = line.split(" ")[1], line.split(" ")[5]
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
        split="train",
        config=None,
        mode="classification",
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

        self.id_to_filename = dict(zip(ids, filenames))
        self.id_to_label = dict(zip(ids, labels))
        self.id_to_author = dict(zip(ids, authors))

    def load_audio_tensor(self, idx):
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

        if self.mode == "classification":
            return {"x": x, "label": y, "author": author}

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


class VoxCeleb2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        sampling_rate=16000,
        max_duration=4,
        mode="classification",
        text_only=False,
        max_samples=20000,
    ):
        self.root_dir = root_dir
        self.split = split
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.cut = self.sampling_rate * self.max_duration
        self.mode = mode
        self.text_only = text_only

        self.metadata_file = os.path.join(root_dir, f"{split}/meta.csv")
        self.df = pd.read_csv(self.metadata_file)

        if max_samples is not None:
            self.df = self.df[:max_samples]

        ids = np.arange(len(self.df))
        self.id_to_filename = dict(zip(ids, self.df["file_path"]))
        self.id_to_author = dict(zip(ids, self.df["speaker_id"]))

    def __len__(self):
        return len(self.df)

    def get_sample_dict(self, idx):
        sample_dict = json.load(open(self.id_to_filename[idx], "r"))

        if "transcription" not in sample_dict:
            sample_dict["transcription"] = ""
            print(f"Transcription not found for {self.id_to_filename[idx]}")

        return {
            "speaker_id": sample_dict["speaker_id"],
            "transcription": sample_dict["transcription"],
            "array": np.array(sample_dict["audio_path"]["array"]),
            "sampling_rate": sample_dict["audio_path"]["sampling_rate"],
        }

    def __repr__(self):
        return f'VoxCeleb2Dataset_{self.split}'
    
    def load_audio_tensor(self, sample_dict):
        audio_arr = sample_dict["array"]
        sr = sample_dict["sampling_rate"]
        audio_arr = librosa.resample(
            audio_arr, orig_sr=sr, target_sr=self.sampling_rate
        )
        audio_tensor = torch.tensor(pad(audio_arr, self.cut)).float()
        return audio_tensor

    def get_triplets_from_anchor(self, anchor_idx):
        anchor_author = self.id_to_author[anchor_idx]
        positive_id = np.random.choice(
            np.where(self.df["speaker_id"].values == anchor_author)[0]
        )
        negative_ids = np.where(self.df["speaker_id"].values != anchor_author)[0]

        return positive_id, np.random.choice(negative_ids)

    def __getitem__(self, idx):

        x_dict = self.get_sample_dict(idx)

        if not self.text_only:
            x = self.load_audio_tensor(x_dict)
        author = self.id_to_author[idx]
        transcription = x_dict["transcription"]

        if self.mode == "classification":
            return {
                "x": x if not self.text_only else np.array([]),
                "author": author,
                # "transcription": transcription,
                "label": 1,  # VoxCeleb2 is not a spoof dataset
            }

        elif self.mode == "triplet":
            id_p, id_n = self.get_triplets_from_anchor(idx)

            if self.text_only:
                return {
                    "anchor": transcription,
                    "positive": self.get_sample_dict(id_p)["transcription"],
                    "negative": self.get_sample_dict(id_n)["transcription"],
                    "label": idx,
                    "negative_label": id_n,
                }

            x_p = self.load_audio_tensor(self.get_sample_dict(id_p))
            x_n = self.load_audio_tensor(self.get_sample_dict(id_n))

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


class CommonVoiceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        sampling_rate=16000,
        max_duration=4,
        return_transcription=False,
        mode="classification",
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            metadata_file (str): Name of the metadata file
            split (str): Split of the dataset (train, val, test)
            sampling_rate (int): Sampling rate of the audio files
            max_duration (int): Maximum duration of the audio files in seconds
        """
        self.root_dir = os.path.join(root_dir, "en")
        self.split = split
        self.mode = mode

        metadata_file = f"{split}.tsv"

        self.df = pd.read_csv(os.path.join(self.root_dir, metadata_file), sep="\t")

        self.clip_dir = os.path.join(self.root_dir, "clips")

        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.cut = self.sampling_rate * self.max_duration  # padding

        # Create id to filename mapping and id to label mapping
        ids = np.arange(len(self.df))
        filenames = self.df["path"].values
        transcriptions = self.df["sentence"].values
        speaker_ids = self.df["client_id"].values

        self.id_to_speaker_id = dict(zip(ids, speaker_ids))
        self.id_to_filename = dict(zip(ids, filenames))
        self.id_to_transcription = dict(zip(ids, transcriptions))

        self.return_transcription = return_transcription

    def get_triplets_from_anchor(self, anchor_idx):
        anchor_speaker_id = self.id_to_speaker_id[anchor_idx]
        positive_id = np.random.choice(
            np.where(self.df["client_id"].values == anchor_speaker_id)[0]
        )
        negative_ids = np.where(self.df["client_id"].values != anchor_speaker_id)[0]

        return positive_id, np.random.choice(negative_ids)

    def load_audio_tensor(self, idx):
        # Load audio file
        filename = os.path.join(self.clip_dir, self.id_to_filename[idx])
        audio_arr, sr = load_audio(filename, self.sampling_rate)

        audio_tensor = torch.tensor(pad(audio_arr, self.cut)).float()
        return audio_tensor

    def __len__(self):
        return len(self.id_to_filename)

    def __getitem__(self, idx):
        x = self.load_audio_tensor(idx)
        y = self.id_to_speaker_id[idx]

        if self.mode == "classification":
            return {
                "x": x,
                "label": 1,
                "author": y,
            }  # label is 1 as common voice is not a spoof dataset

        elif self.mode == "triplet":
            id_p, id_n = self.get_triplets_from_anchor(idx)

            x_p = self.load_audio_tensor(id_p)
            x_n = self.load_audio_tensor(id_n)

            return {"anchor": x, "positive": x_p, "negative": x_n}

        return x


class RAVDESSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        metadata_file="ravdess_config.yaml",
        split="train",
        sampling_rate=16000,
        max_duration=4,
        mode="classification",
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            metadata_file (str): Name of the metadata file
            sampling_rate (int): Sampling rate of the audio files
            max_duration (int): Maximum duration of the audio files in seconds
        """
        self.root_dir = root_dir
        self.metadata_path = os.path.join(root_dir, metadata_file)
        self.config = yaml.safe_load(open(self.metadata_path, "r"))
        self.mode = mode
        self.split = split

        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.cut = self.sampling_rate * self.max_duration  # padding

        self.audio_files = self.config[split]
        self.id_to_filename = dict(enumerate(self.audio_files))
        self.id_to_author = dict(
            enumerate([x.split("/")[-2] for x in self.audio_files])
        )

    def load_audio_tensor(self, idx):
        # Load audio file
        filename = self.id_to_filename[idx]
        audio_arr, sr = load_audio(filename)
        audio_tensor = torch.tensor(pad(audio_arr, self.cut)).float()
        return audio_tensor

    def get_triplets_from_anchor(self, anchor_idx):
        anchor_author = self.id_to_author[anchor_idx]
        positive_id = np.random.choice(
            np.where(np.array(list(self.id_to_author.values())) == anchor_author)[0]
        )
        negative_ids = np.where(
            np.array(list(self.id_to_author.values())) != anchor_author
        )[0]

        return positive_id, np.random.choice(negative_ids)

    def __len__(self):
        return len(self.id_to_filename)

    def __getitem__(self, idx):
        x = self.load_audio_tensor(idx)
        author = self.id_to_author[idx]

        if self.mode == "classification":
            return {
                "x": x,
                "label": 1,
                "author": author,
            }  # label is 1 as ravdess is not a spoof dataset

        elif self.mode == "triplet":
            id_p, id_n = self.get_triplets_from_anchor(idx)

            x_p = self.load_audio_tensor(id_p)
            x_n = self.load_audio_tensor(id_n)

            return {"anchor": x, "positive": x_p, "negative": x_n}
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
