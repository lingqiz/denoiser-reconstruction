#!/bin/bash
#SBATCH --job-name=lnopt
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --constraint=a100
#SBATCH -c 48
#SBATCH -t 7-00:00:00

cd ~/denoiser_recon/
module load python/3.10.8 cuda cudnn nccl
source recon/bin/activate

python3 linear.py --n_sample $NSP --dataset $DST --model_path $MDP --loss_type $LST --recon_method $MTD --batch_size $BSZ --n_epoch $NEP
