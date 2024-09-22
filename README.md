## Introduction
This is the first knowledge graph recommendation system that integrates variational inference. It is implemented using PyTorch and runs on Linux.

## Environment Building via Anaconda
1. Donwload [Ananconda](https://www.anaconda.com/download).
2. Follow the commands below:
```
conda create -n ramvae python=3.10.12 -y
conda activate ramvae 
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
bash requirements.sh
```


## Reproduce

**LastFM**

`python run_RAMVAE.py --log --log_fn test --cl_coef 0`


**Alibaba-iFashion**

`python run_RAMVAE.py --log --log_fn test --cl_coef 0 --dataset alibaba-fashion`



**Movielens-20M**

`python run_RAMVAE.py --log --log_fn test --cl_coef 0 --dataset movie-lens`

