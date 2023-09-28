#!/bin/bash

# script
python train.py \
    --project_name Ego3DPose \
    --experiment_name egoglass \
    --model egoglass \
\
    --use_amp \
    --init_ImageNet \
    --gpu_ids 0 \
\
    --lambda_mpjpe 0.1 \
    --lambda_heatmap 1.0 \
    --lambda_cos_sim -0.01 \
    --lambda_heatmap_rec 0.001 \
    --lambda_segmentation 1 \
\
    --niter 5 \
    --niter_decay 5 \
    --lr 1e-3 \
    --batch_size 16 \
    --num_heatmap 15 \
    --num_rot_heatmap 0 \
    --data_dir /ssd_data1/UnrealEgoData \
\
