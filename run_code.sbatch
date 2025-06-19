#!/bin/bash
#SBATCH -t 7-40:01:00                    # Max runtime: 7 days, 40 hours
#SBATCH --mem=60G                        # Total memory
#SBATCH -J PCL_Training                  # Job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT # Email notifications
#SBATCH -n 1                             # Number of tasks
#SBATCH --cpus-per-task=6               # CPUs for data loading
#SBATCH -N 1                             # One node
#SBATCH --gres=gpu:a30:1                # GPU: A30 (1 unit)

#SBATCH --output=PCL_%j_out.txt         # STDOUT
#SBATCH --error=PCL_%j_err.txt          # STDERR

# Load Anaconda and activate environment
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate pcl_env

# Run training
python train.py \
  --dataset_root /path/to/MPI_INF_3DHP_dataset \
  --use_dataset 3DHP \
  --exp_type 3dFromImage \
  --use_pcl \
  --use_slant_compensation \
  --use_2d_scale \
  --use_canonical \
  --use_resnet50 \
  --use_pretrain \
  --total_epochs 30 \
  --batch_size 8 \
  --num_workers 6 \
  --lr 0.0005 \
  --run_name pcl_run1 \
  --model_path ./model \
  --denormalize_during_training

conda deactivate
