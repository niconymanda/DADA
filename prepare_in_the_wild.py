import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import read
import os
import json
import csv

from inputs import wild_audios, transcription_model_id, transcriptions_path, meta_data_path_csv, in_the_wild


# Aux Functions
def update_json_file(json_file_path, new_data):

    if not os.path.exists(json_file_path):
        with open(json_file_path, "w") as file:
            json.dump({}, file)

    with open(json_file_path, "r") as file:
        try:
            existing_data = json.load(file)
        except json.JSONDecodeError:
            existing_data = {}

    existing_data.update(new_data)

    with open(json_file_path, "w") as file:
        json.dump(existing_data, file, indent=4)


def run_asr_pipeline():
    """automatic speech recognition pipeline with Whisper"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        transcription_model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(transcription_model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"}
    )


    # load to transcription file in batches (= 15)
    batches = [i for i in range(len(wild_audios)) if i%15 == 0]
    shortened_wild_audios = [os.path.basename(audio) for audio in wild_audios]

    for batch in batches:
        audios = [read(file_name) for file_name in wild_audios[batch:min(batch+15, len(wild_audios))]]
        all_generated_texts = [pipe(audio_array[1]) for audio_array in audios]
        dict_output = {shortened_wild_audios[batch+i]: list(all_generated_texts[i].values())[0] for i in range(len(all_generated_texts))}
        update_json_file(transcriptions_path, dict_output)
    
    return dict_output


def join_meta_transcriptions(transcriptions):
    """Connect with meta-data"""
    with open(meta_data_path_csv, "r") as file:
        csv_reader = csv.DictReader(file, delimiter=";")
        data = [row for row in csv_reader]
    
    joined_data = []

    for item in data:
        file_name = item["file"]
        if file_name in transcriptions:
            combined_entry = {
                "file": file_name,
                "speaker": item["speaker"],
                "label": item["label"],
                "transcription": transcriptions[file_name]
            }
            joined_data.append(combined_entry)
    
    if not os.path.exists(in_the_wild):
        with open(in_the_wild, "w") as file:
            json.dump({}, file)

    with open(in_the_wild, "w") as file:
        json.dump(joined_data, file, indent=4)
    
    return joined_data
