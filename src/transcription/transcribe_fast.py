import warnings

warnings.simplefilter("ignore", category=FutureWarning)

import torch
import glob
from transformers import pipeline
from tqdm import tqdm
import multiprocessing
from tqdm import tqdm
import pandas as pd

keys = []
transcriptions = []

audio_files = glob.glob("/data/amathur-23/DADA/InTheWild/release_in_the_wild/*.wav")
batch_size = 32


# Load Whisper model once in each worker process (avoid CUDA issues)
def init_worker():
    global model
    model = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda:2",
    )


# Function to process a batch
def transcribe_batch(audio_files_batch):
    keys = [x.split("/")[-1].split(".")[0] for x in audio_files_batch]
    # print(keys)
    transcription_list_dict = model(audio_files_batch)
    transcriptions = [x["text"] for x in transcription_list_dict]
    return list(zip(keys, transcriptions))


# Process audio files in parallel
def process_audio_files(audio_files, batch_size=32, num_workers=4):
    num_steps = len(audio_files) // batch_size
    batches = [
        audio_files[i * batch_size : (i + 1) * batch_size] for i in range(num_steps)
    ]

    multiprocessing.set_start_method(
        "spawn", force=True
    )  # Set 'spawn' to avoid CUDA issues
    pool = multiprocessing.Pool(num_workers, initializer=init_worker)  # Init workers

    results = list(tqdm(pool.imap(transcribe_batch, batches), total=num_steps))

    # Flatten results
    keys, transcriptions = zip(*[item for sublist in results for item in sublist])

    # Save results once after processing
    df = pd.DataFrame({"key": keys, "transcription": transcriptions})
    df.to_csv("/data/iivanova-23/data/inthewild/in_the_wild_transcriptions_parallel.csv", index=False)

    pool.close()
    pool.join()
    
def merge_metadata_with_transcriptions(metadata_file, transcription_file, output_file):
    metadata_df = pd.read_csv(metadata_file)
    transcriptions_df = pd.read_csv(transcription_file)
    
    transcriptions_df['key'] = transcriptions_df['key'].astype(str).apply(lambda x: x + '.wav')
    merged_df = pd.merge(metadata_df, transcriptions_df, left_on='file', right_on='key')
    
    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Run function
    # process_audio_files(audio_files, batch_size=32, num_workers=4)
    merge_metadata_with_transcriptions(
        "/data/amathur-23/DADA/InTheWild/release_in_the_wild/meta.csv",
        "/data/iivanova-23/data/inthewild/in_the_wild_transcriptions_parallel.csv",
        "/data/iivanova-23/data/inthewild/in_the_wild_transcriptions.csv")