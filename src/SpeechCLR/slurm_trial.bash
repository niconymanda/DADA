#!/bin/bash

#SBATCH --output=slurm_trial%j.out   # Output file
#SBATCH --error=slurm_trial%j.err    # Error file
#SBATCH --time=00:10:00              # Maximum allocated time #SBATCH --partition=A100           # Partition to submit to
#SBATCH --nodes=1                    # No. of nodes to reserve
#SBATCH --gpus=1                     # No. of GPUs
#SBATCH --cpus-per-task=1            # No. of CPUs

set -x # Activate command verbosity

source /home/infres/amathur-23/DADA/dada/bin/activate # Activate virtual environment
srun python3 -u main.py --data_path /home/infres/amathur-23/DADA/datasets/InTheWild --model_save_path /home/infres/amathur-23/DADA/src/models --dataset_config configs/data/inthewild_10v8.yaml --loss_fn triplet_cosine --gpu_id 0 --epochs=1 --log_dir=runs/tricos_p35_10v8_plateau_ids --batch_size=16 --at_lambda=0 --margin=0.35 --lr_scheduler plateau --early_stopping_metric=accuracy
deactivate
