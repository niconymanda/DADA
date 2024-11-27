# Mid-Fusion

## Usage :

**Note : Ensure you are in this folder's parent directory (`src`)**

```bash
usage: python3 -m mid_fusion.main [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--learning_rate LEARNING_RATE] [--data_path DATA_PATH]
               [--model_save_path MODEL_SAVE_PATH]
               [--log_interval LOG_INTERVAL] [--dataset_config DATASET_CONFIG]
               [--log_dir LOG_DIR] [--max_duration MAX_DURATION]
               [--sampling_rate SAMPLING_RATE]
               [--early_stopping_patience EARLY_STOPPING_PATIENCE]
               [--early_stopping_threshold EARLY_STOPPING_THRESHOLD]
               [--gpu_id GPU_ID] [--seed SEED]
               [--save_visualisations SAVE_VISUALISATIONS]
               [--text_model_name TEXT_MODEL_NAME]
               [--text_model_path TEXT_MODEL_PATH]
               [--audio_model_name AUDIO_MODEL_NAME]
               [--audio_model_path AUDIO_MODEL_PATH]
               [--load_checkpoint LOAD_CHECKPOINT]

Train Mid Fusion model

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size for training
  --epochs EPOCHS       Number of epochs to train
  --learning_rate LEARNING_RATE
                        Learning rate for optimizer
  --data_path DATA_PATH
                        Path to the training data
  --model_save_path MODEL_SAVE_PATH
                        Path to save the trained model
  --log_interval LOG_INTERVAL
                        Interval for logging training status
  --dataset_config DATASET_CONFIG
                        Path to the dataset configuration file
  --log_dir LOG_DIR     Path to save logs
  --max_duration MAX_DURATION
                        Maximum duration of audio files
  --sampling_rate SAMPLING_RATE
                        Sampling rate of audio files
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs to wait before early stopping
  --early_stopping_threshold EARLY_STOPPING_THRESHOLD
                        Threshold for early stopping
  --gpu_id GPU_ID       ID of the GPU to use for training
  --seed SEED           Seed for reproducibility
  --save_visualisations SAVE_VISUALISATIONS
                        Save visualizations of embeddings
  --text_model_name TEXT_MODEL_NAME
                        Name of the text model to use
  --text_model_path TEXT_MODEL_PATH
                        Path to the text model
  --audio_model_name AUDIO_MODEL_NAME
                        Name of the audio model to use
  --audio_model_path AUDIO_MODEL_PATH
                        Path to the audio model
  --load_checkpoint LOAD_CHECKPOINT
                        Path to the checkpoint to load
```