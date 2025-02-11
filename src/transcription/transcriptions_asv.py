import os
import csv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from datasets import Dataset, Audio
from tqdm import tqdm
import librosa

def load_model(model_id, device):
    print("Loading model")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
        generate_kwargs={"language": "english"}
    )
    print("Model loaded")
    return pipe

def load_dataset(meta_data_path, folder):
    metadata = {}
    with open(meta_data_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                file_name = parts[1]
                file_type = parts[5]
                metadata[file_name] = file_type
    
    audio_files = []
    if os.path.exists(folder):
        for file_name in os.listdir(folder):
            if file_name.endswith(".flac"):
                file_path = os.path.join(folder, file_name)
                base_name = file_name.replace(".flac", "")
                file_type = metadata.get(base_name, "unknown")
                audio_files.append({"path": file_path, "file_name": file_name, "type": file_type})

    dataset = Dataset.from_list(audio_files)
    dataset = dataset.cast_column("path", Audio(sampling_rate=12000))
    # dataset_audio = dataset.map(lambda x: {"audio": load_audio_librosa(x["path"])})
    return dataset, metadata

# def load_audio_librosa(file_path, sampling_rate=16000):
#     waveform, sr = librosa.load(file_path, sr=sampling_rate)
#     return waveform

def transcribe_audio(pipe, dataset, output_csv, metadata):
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "type", "transcription"])

        for example in tqdm(dataset):
            file_name = example["file_name"]
            file_path = example["path"]

            try:
                result = pipe(file_path)
                transcription = result["text"]
            except Exception as e:
                print(f"Error transcribing {file_name}: {e}")
                transcription = ""

            base_name = file_name.replace(".flac", "")
            file_type = metadata.get(base_name, "unknown")

            writer.writerow([file_name, file_type, transcription])

    print(f"Transcriptions saved to {output_csv}")      

    
def main():
    os.environ['HF_HOME'] = '/data/iivanova-23/cache/'
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    model_id = "openai/whisper-large-v3"
    meta_data_path_asv = "/data/amathur-23/DADA/ASVspoof2021_DF_eval/keys/DF/CM/trial_metadata.txt"
    output_csv = "/data/iivanova-23/data/asvspoof2021/transcriptions_new.csv"
    folder = "/data/amathur-23/DADA/ASVspoof2021_DF_eval/flac/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipline = load_model(model_id, device)
    dataset, metadata = load_dataset(meta_data_path_asv, folder)
    print(len(dataset))
    transcribe_audio(pipline, dataset, output_csv, metadata)
    
if __name__ == "__main__":
    main()