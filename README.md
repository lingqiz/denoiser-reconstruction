## Image Reconstruction with Denoiser Implicit Prior

Image reconstruction from cone excitations using the prior
implicit in a convolutional neural network denoiser.  
See [Kadkhodaie & Simoncelli, 2022](https://arxiv.org/abs/2007.13640) for the details of the method.

### Installation
```
python -m venv denoiser-recon
source denoiser-recon/bin/activate
pip install -r requirements.txt
```

### SLURM command
```
sbatch train_run.sh
sbatch --export=NSP=16,DST=dataset lnopt_run.sh 
```