# SpeechCLR : Contrastive Learning for Speaker Representations

## Usage
```txt
usage: main.py [-h] [--model_name MODEL_NAME] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
               [--data_path DATA_PATH] [--model_save_path MODEL_SAVE_PATH]
               [--log_interval LOG_INTERVAL] [--dataset_config DATASET_CONFIG]
               [--log_dir LOG_DIR] [--max_duration MAX_DURATION]
               [--sampling_rate SAMPLING_RATE]
               [--early_stopping_patience EARLY_STOPPING_PATIENCE]
               [--early_stopping_threshold EARLY_STOPPING_THRESHOLD]
               [--triplet_margin TRIPLET_MARGIN]
               [--loss_fn {triplet,triplet_cosine,ada_triplet}]
               [--gpu_id GPU_ID] [--margin MARGIN] [--at_lambda AT_LAMBDA]
               [--seed SEED] [--save_visualisations SAVE_VISUALISATIONS]
               [--lr_scheduler {plateau,step,cosine}]
               [--early_stopping_metric {loss,accuracy}]
               [--load_checkpoint LOAD_CHECKPOINT]

Train SpeechCLR model

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of the model to train
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
  --triplet_margin TRIPLET_MARGIN
                        Margin for triplet loss
  --loss_fn {triplet,triplet_cosine,ada_triplet}
                        Loss function to use for training
  --gpu_id GPU_ID       ID of the GPU to use for training
  --margin MARGIN       Margin for triplet loss
  --at_lambda AT_LAMBDA
                        Lambda for AdaTriplet loss
  --seed SEED           Seed for reproducibility
  --save_visualisations SAVE_VISUALISATIONS
                        Save visualizations of embeddings
  --lr_scheduler {plateau,step,cosine}
                        Learning rate scheduler to use
  --early_stopping_metric {loss,accuracy}
                        Metric to use for early stopping
  --load_checkpoint LOAD_CHECKPOINT
                        Path to the checkpoint to load
```
