import os
from tqdm import tqdm
import pandas as pd

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

for i, sample in tqdm(enumerate(dataset)):
    # print(f"\rSample {i}/{NUM_SAMPLES}", end="")
    if random.random() < sample_fraction:
        if random.random() < val_fraction:
            sampled_data_val.append(sample)
        else:
            sampled_data_train.append(sample)
    if len(sampled_data_train) > sample_fraction * (1 - val_fraction) * NUM_SAMPLES:
        break

# print()

# Convert sampled data to DataFrame
df_train = pd.DataFrame(sampled_data_train)
df_val = pd.DataFrame(sampled_data_val)

# Ensure root_dir exists
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# Save DataFrames as parquet files
df_train.to_parquet(f"{root_dir}/train.parquet")
df_val.to_parquet(f"{root_dir}/val.parquet")

# Print some sampled entries
print(sampled_data_train[:3])  # Print first 3 samples from train data
print(f"Sampled {len(sampled_data_train)} training entries")
print(f"Sampled {len(sampled_data_val)} validation entries")
