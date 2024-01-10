#!/bin/bash
#SBATCH --job-name=lnopt
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --constraint=a100
#SBATCH -c 48
#SBATCH -t 7-00:00:00

cd ~/denoiser_recon/
module load python/3.10 cuda cudnn nccl
source recon/bin/activate

# set parameter values
MTD=Denoiser
LST=MSE
NLY=26
KNS=3
PAD=1
LNR=0.0001
BSZ=64
NEP=16

python3 linear.py --recon_method $MTD --loss_type $LST \
                  --num_layers $NLY --kernel_size $KNS --padding $PAD \
                  --n_sample $NSP --data_path $DST --model_path $MDP \
                  --lr $LNR --batch_size $BSZ --n_epoch $NEP
