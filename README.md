# MSAmba: Exploring Multimodal Sentiment Analysis with State Space Models

Pytorch implementation of paper: 

> **MSAmba: Exploring Multimodal Sentiment Analysis with State Space Models**



## Content

- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Test](#Test)
- [Training](#Training)
- [Citation](#Citation)
  
## Data Preparation
MOSI/MOSEI/CH-SIMS Download: See [MMSA](https://github.com/thuiar/MMSA)

## Environment
The basic training environment for the results in the paper is Pytorch 2.1.1, CUDA 11.8, with NVIDIA A100. It should be noted that different hardware and software environments can cause the results to fluctuate.

Please install the Mamba packages from [here](https://github.com/state-spaces/mamba)



## Training
You can quickly run the code with train.py (you can refer to opts.py to modify more hyperparameters)