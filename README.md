## Optimized Linear Measurements for Image Reconstruction

**Optimized compressed sensing for image reconstruction with diffusion probabilistic models**    
Ling-Qi Zhang, Zahra Kadkhodaie, Eero P. Simoncelli, and David H. Brainard   
https://arxiv.org/abs/2405.17456

### Installation
```
python -m venv denoiser-recon
source denoiser-recon/bin/activate
pip install -r requirements.txt
```

### Training (Cluster)
```
sbatch train_run.sh
sbatch --export=NSP=16,DST=dataset lnopt_run.sh
```
