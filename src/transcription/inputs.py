from glob import glob

wild_audios = glob("data/release_in_the_wild/*.wav")
transcription_model_id = "openai/whisper-large-v3"
transcriptions_path = "data/transcriptions/test_in_the_wild_transcriptions.json"
meta_data_path_csv = "data/release_in_the_wild/meta.csv"
in_the_wild = "data/in_the_wild.json"