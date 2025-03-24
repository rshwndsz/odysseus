#!/bin/bash

module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
srun env | grep CUDA_VISIBLE_DEVICES
pip install -r requirements.txt
python zero_shot.py -c "./experiments/configs/zs_phi35-mini.yaml"
