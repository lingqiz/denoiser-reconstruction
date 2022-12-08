#!/bin/bash
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --constraint=a100
#SBATCH -c 32
#SBATCH -t 6-00:00:00

cd ~/denoiser_recon/
module load cuda
source recon/bin/activate

python3 linear.py --n_sample 128 --batch_size 128 --n_epoch 1 --dataset islvrc64 --loss_type MSE
