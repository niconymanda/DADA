import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

os.environ["HF_HOME"] = "/data/amathur-23/DADA/"

from datasets import load_dataset
import random

# Load dataset in streaming mode
dataset = load_dataset("acul3/voxceleb2", split="train", streaming=True)
root_dir = "/data/amathur-23/DADA/VoxCeleb2"
# Define sample fraction
sample_fraction = 0.15  # 15%
val_fraction = 0.2  # 20%

# Reservoir sampling algorithm to randomly pick 20% of dataset
sampled_data_train = []
sampled_data_val = []

NUM_SAMPLES = 463000
TRAIN_SAMPLES = int(sample_fraction * NUM_SAMPLES * (1 - val_fraction))
VAL_SAMPLES = int(sample_fraction * NUM_SAMPLES * val_fraction)

all_ids = np.random.choice(NUM_SAMPLES, int(NUM_SAMPLES*sample_fraction), replace=False)
train_ids = np.random.choice(all_ids, TRAIN_SAMPLES, replace=False)
val_ids = np.setdiff1d(all_ids, train_ids)

train_filenames = []
train_speaker_ids = []

val_filenames = []
val_speaker_ids = []

train_dir = os.path.join(root_dir, "train")
if not os.path.exists(train_dir):
    os.makedirs(train_dir, exist_ok=True)

val_dir = os.path.join(root_dir, "val")
if not os.path.exists(val_dir):
    os.makedirs(val_dir, exist_ok=True)


for id, sample_dict in tqdm(enumerate(dataset)):
    # print(sample_dict)
    # exit()
    if id in train_ids:
        sample_dict['audio_path']['array'] = list(sample_dict['audio_path']['array'])
        file_path = os.path.join(train_dir, f"{id}.json")
        train_filenames.append(file_path)
        train_speaker_ids.append(sample_dict["speaker_id"])
        json.dump(sample_dict, open(file_path, "w"))    

    if id in val_ids:
        sample_dict['audio_path']['array'] = list(sample_dict['audio_path']['array'])
        file_path = os.path.join(val_dir, f"{id}.json")
        val_filenames.append(file_path)
        val_speaker_ids.append(sample_dict["speaker_id"])
        json.dump(sample_dict, open(file_path, "w"))

train_df = pd.DataFrame({"file_path": train_filenames, "speaker_id": train_speaker_ids})
train_df.to_csv(os.path.join(root_dir, "train/meta.csv"), index=False)

val_df = pd.DataFrame({"file_path": val_filenames, "speaker_id": val_speaker_ids})
val_df.to_csv(os.path.join(root_dir, "val/meta.csv"), index=False)



# for id in tqdm(val_ids):
#     sample_dict = dataset[id]



# for i, sample in tqdm(enumerate(dataset)):
#     # print(f"\rSample {i}/{NUM_SAMPLES}", end="")
#     if random.random() < sample_fraction:
#         if random.random() < val_fraction:
#             sampled_data_val.append(sample)
#         else:
#             sampled_data_train.append(sample)
#     if len(sampled_data_train) > sample_fraction * (1 - val_fraction) * NUM_SAMPLES:
#         break

# # print()

# # Convert sampled data to DataFrame
# df_train = pd.DataFrame(sampled_data_train)
# df_val = pd.DataFrame(sampled_data_val)

# # Ensure root_dir exists
# if not os.path.exists(root_dir):
#     os.makedirs(root_dir)

# # Save DataFrames as parquet files
# df_train.to_parquet(f"{root_dir}/train.parquet")
# df_val.to_parquet(f"{root_dir}/val.parquet")

# # Print some sampled entries
# print(sampled_data_train[:3])  # Print first 3 samples from train data
# print(f"Sampled {len(sampled_data_train)} training entries")
# print(f"Sampled {len(sampled_data_val)} validation entries")
