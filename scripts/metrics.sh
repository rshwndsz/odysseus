#!/bin/bash

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
srun env | grep CUDA_VISIBLE_DEVICES

pip install -r requirements.txt

python metrics.py \
--Config.data_dir data \
--Config.device "cuda:0" \
--Config.model_id results/finetunes/20250205-121158__microsoft_Phi-3.5-mini-instruct__ft/checkpoint-1792 \
--Config.num_labels 2 \
--Config.dataset_dir data/dataset \
--Config.dynamic_padding \
--Config.max_length 8192 \
--Config.quantization_bits 4 \
--Config.batch_size 8 \
--Config.save_dir results/metrics
