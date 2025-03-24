#!/bin/bash

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
srun env | grep CUDA_VISIBLE_DEVICES

pip install -r requirements.txt
python finetune.py -c "./experiments/ft_llama32-1b_ev9sz62l-2.yaml"

