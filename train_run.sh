#!/bin/bash
#SBATCH --job-name=denoiser
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --constraint=a100
#SBATCH --mem=1000G
#SBATCH -c 48
#SBATCH -t 2-00:00:00

cd ~/denoiser_recon/
module load cuda
module load python/3.10.8
source recon/bin/activate

# batch size = 32
# AdamW (with weight decay)

echo Training on CelebA
python3 main.py --mode train --batch_size 32 --opt_index 2 --n_epoch 368 --save_path ./assets/conv3_celeba_tiny.pt --data_path npy_celeba_tiny --ddp True --bias_sd True --scale_image True --save_model True

echo Training on CIFAR
python3 main.py --mode train --batch_size 32 --opt_index 2 --n_epoch 512 --save_path ./assets/conv3_cifar_10.pt --data_path npy_cifar_10 --ddp True --bias_sd True --scale_image True --save_model True

echo Training on Pink Noise
python3 main.py --mode train --batch_size 32 --opt_index 2 --n_epoch 512 --save_path ./assets/conv3_pink_noise.pt --data_path npy_pink_noise --ddp True --bias_sd True --scale_image True --save_model True

echo Training on MNIST
python3 main.py --mode train --batch_size 32 --opt_index 2 --n_epoch 512 --save_path ./assets/conv3_mnist.pt --data_path mnist --ddp True --bias_sd True --scale_image True --save_model True

