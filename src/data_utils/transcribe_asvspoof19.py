# %%
import warnings
import numpy as np

warnings.simplefilter("ignore", category=FutureWarning)

import torch
import glob
from transformers import pipeline
import os

os.environ["HF_HOME"] = "/data/amathur-23/DADA"

whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:2")


def get_spoof_list19(meta_file):
    d_meta = {}
    file_list = []
    with open(meta_file, "r") as f:
        l_meta = f.readlines()

    for line in l_meta:
        _, key, _, _, label = line.strip().split(" ")
        file_list.append(key)
        d_meta[key] = 1 if label == "bonafide" else 0
    return d_meta

# %%
from tqdm import tqdm
import pandas as pd


batch_size = 32

TRAIN_SAMPLES = 8000
VAL_SAMPLES = 4000


train_dict = get_spoof_list19("/data/amathur-23/DADA/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
val_dict = get_spoof_list19("/data/amathur-23/DADA/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

train_dict = {k: v for k, v in train_dict.items() if v == 1}
val_dict = {k: v for k, v in val_dict.items() if v == 1}

VAL_SAMPLES = min(VAL_SAMPLES, len(val_dict))
TRAIN_SAMPLES = min(TRAIN_SAMPLES, len(train_dict))
print(f"Train samples : {TRAIN_SAMPLES}, Val samples : {VAL_SAMPLES}")

train_keys = np.random.choice(list(train_dict.keys()), TRAIN_SAMPLES, replace=False)
val_keys = np.random.choice(list(val_dict.keys()), VAL_SAMPLES, replace=False)

train_audio_files = [f"/data/amathur-23/DADA/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/{x}.flac" for x in train_keys]
val_audio_files = [f"/data/amathur-23/DADA/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/{x}.flac" for x in val_keys]

train_num_steps = len(train_audio_files) // batch_size
val_num_steps = len(val_audio_files) // batch_size

save_freq = 50

keys = []
transcriptions = []
labels = []

for i in tqdm(range(train_num_steps)):
    audio_files_batch = train_audio_files[i*batch_size:(i+1)*batch_size]
    keys.extend([x.split("/")[-1].split(".")[0] for x in audio_files_batch])
    transcription_list_dict = whisper(audio_files_batch)
    transcriptions.extend([x['text'] for x in transcription_list_dict])
    labels.extend([train_dict[k] for k in train_keys[i*batch_size:(i+1)*batch_size]])

    if i % save_freq == 0:
        df = pd.DataFrame({'key':keys, 'transcription':transcriptions, 'label':labels})
        df.to_csv(f"asvspoof19_la_train_meta2.csv", index=False)

df = pd.DataFrame({'key':keys, 'transcription':transcriptions, 'label':labels})
df.to_csv(f"asvspoof19_la_train_meta2.csv", index=False)



keys = []
transcriptions = []
labels = []

for i in tqdm(range(val_num_steps)):
    audio_files_batch = val_audio_files[i*batch_size:(i+1)*batch_size]
    keys.extend([x.split("/")[-1].split(".")[0] for x in audio_files_batch])
    transcription_list_dict = whisper(audio_files_batch)
    transcriptions.extend([x['text'] for x in transcription_list_dict])
    labels.extend([val_dict[k] for k in val_keys[i*batch_size:(i+1)*batch_size]])

    if i % save_freq == 0:
        df = pd.DataFrame({'key':keys, 'transcription':transcriptions, 'label':labels})
        df.to_csv(f"asvspoof19_la_val_meta2.csv", index=False)

df = pd.DataFrame({'key':keys, 'transcription':transcriptions, 'label':labels})
df.to_csv(f"asvspoof19_la_val_meta2.csv", index=False)


# for i in tqdm(range(num_steps)):
#     audio_files_batch = audio_files[i*batch_size:(i+1)*batch_size]
#     keys.extend([x.split("/")[-1].split(".")[0] for x in audio_files_batch])
#     transcription_list_dict = whisper(audio_files_batch)
#     transcriptions.extend([x['text'] for x in transcription_list_dict])

#     if i==0:
#         print(transcriptions)

#     df = pd.DataFrame({'key':keys, 'transcription':transcriptions})
#     df.to_csv("asvspoof21_df_eval_transcriptions.csv", index=False)

# %%
# %%
# df = pd.DataFrame({'key':keys, 'transcription':transcriptions})
# df.to_csv("asvspoof21_df_eval_transcriptions.csv", index=False)


# import multiprocessing
# from tqdm import tqdm
# import pandas as pd


# # Load Whisper model once in each worker process (avoid CUDA issues)
# def init_worker():
#     global model
#     model = pipeline(
#         "automatic-speech-recognition",
#         "openai/whisper-large-v3",
#         torch_dtype=torch.float16,
#         device="cuda:2",
#     )


# # Function to process a batch
# def transcribe_batch(audio_files_batch):
#     keys = [x.split("/")[-1].split(".")[0] for x in audio_files_batch]
#     transcription_list_dict = model(audio_files_batch)
#     transcriptions = [x["text"] for x in transcription_list_dict]
#     return list(zip(keys, transcriptions))


# # Process audio files in parallel
# def process_audio_files(audio_files, batch_size=32, num_workers=4):
#     num_steps = len(audio_files) // batch_size
#     batches = [
#         audio_files[i * batch_size : (i + 1) * batch_size] for i in range(num_steps)
#     ]

#     multiprocessing.set_start_method(
#         "spawn", force=True
#     )  # Set 'spawn' to avoid CUDA issues
#     pool = multiprocessing.Pool(num_workers, initializer=init_worker)  # Init workers

#     results = list(tqdm(pool.imap(transcribe_batch, batches), total=num_steps))

#     # Flatten results
#     keys, transcriptions = zip(*[item for sublist in results for item in sublist])

#     # Save results once after processing
#     df = pd.DataFrame({"key": keys, "transcription": transcriptions})
#     df.to_csv("asvspoof21_df_eval_transcriptions_parallel.csv", index=False)

#     pool.close()
#     pool.join()


# # Run function
# process_audio_files(audio_files, batch_size=32, num_workers=4)
