

# %%
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import torch
import glob
from transformers import pipeline

whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:2")

# %%
from tqdm import tqdm

keys = []
transcriptions = []


import pandas as pd

audio_files = glob.glob("/data/amathur-23/DADA/ASVspoof2021_DF_eval/flac/*.flac")
batch_size = 32

# num_steps = len(audio_files) // batch_size

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




import multiprocessing
from tqdm import tqdm
import pandas as pd

def transcribe_batch(audio_files_batch):
    """Helper function to transcribe a batch of audio files."""
    transcription_list_dict = whisper(audio_files_batch)
    keys = [x.split("/")[-1].split(".")[0] for x in audio_files_batch]
    transcriptions = [x['text'] for x in transcription_list_dict]
    return list(zip(keys, transcriptions))

# Define batch processing function
def process_audio_files(audio_files, batch_size=8, num_workers=4):
    num_steps = len(audio_files) // batch_size
    pool = multiprocessing.Pool(num_workers)

    # Split into batches
    batches = [audio_files[i * batch_size: (i + 1) * batch_size] for i in range(num_steps)]

    # Process in parallel
    results = list(tqdm(pool.imap(transcribe_batch, batches), total=num_steps))
    
    # Flatten results
    keys, transcriptions = zip(*[item for sublist in results for item in sublist])
    
    # Save once after all processing
    df = pd.DataFrame({'key': keys, 'transcription': transcriptions})
    df.to_csv("asvspoof21_df_eval_transcriptions_parallel.csv", index=False)

    pool.close()
    pool.join()

# Run function
process_audio_files(audio_files, batch_size=32, num_workers=4)
