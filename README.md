## Optimized Linear Measurements for Inverse Problems

README under construction.

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
