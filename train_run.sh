#!/bin/bash
#SBATCH --job-name=denoiser
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --constraint=a100
#SBATCH --mem=768G
#SBATCH -c 48
#SBATCH -t 7-00:00:00

cd ~/denoiser_recon/
module load cuda
module load python/3.10.8
source recon/bin/activate

echo Training on CelebA
python3 main.py --mode train --batch_size $BSZ --n_epoch $NEP --opt_index $OPT --ddp True --bias_sd True --scale_image Flase --save_path ./assets/conv3_celeba_tiny.pt --data_path npy_celeba_tiny

echo Training on CIFAR
python3 main.py --mode train --batch_size $BSZ --n_epoch $NEP --opt_index $OPT --ddp True --bias_sd True --scale_image Flase --save_path ./assets/conv3_cifar_10.pt --data_path npy_cifar_10

echo Training on Pink Noise
python3 main.py --mode train --batch_size $BSZ --n_epoch $NEP --opt_index $OPT --ddp True --bias_sd True --scale_image Flase --save_path ./assets/conv3_pink_noise.pt --data_path npy_pink_noise

echo Training on MNIST
python3 main.py --mode train --batch_size $BSZ --n_epoch $NEP --opt_index $OPT --ddp True --bias_sd True --scale_image Flase --save_path ./assets/conv3_mnist.pt --data_path mnist

