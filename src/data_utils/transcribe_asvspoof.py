from SpeechCLR.utils.datasets import load_audio
import glob

import pandas as pd

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import argparse


def get_args():
    def get_args():
        parser = argparse.ArgumentParser(description="Transcribe ASVspoof dataset")
        parser.add_argument(
            "--root_dir", type=str, required=True, help="Root directory of the dataset"
        )
        parser.add_argument(
            "--gpu_id", type=int, default=0, help="GPU ID to use for computation"
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="openai/whisper-large-v3",
            help="Name of the model to use for transcription",
        )
        args = parser.parse_args()
        return args


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

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
        torch_dtype=torch_dtype,
        device=device,
    )
