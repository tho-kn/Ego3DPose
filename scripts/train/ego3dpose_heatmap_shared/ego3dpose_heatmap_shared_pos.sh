#!/bin/bash

# script
python train.py \
    --project_name Ego3DPose \
    --experiment_name ego3dpose_heatmap_shared_pos \
    --model ego3dpose_heatmap_shared \
\
\
    --use_amp \
    --init_ImageNet \
\
    --lambda_mpjpe 0.1 \
    --lambda_heatmap 1.0 \
    --lambda_rot_heatmap 1.0 \
    --lambda_cos_sim -0.01 \
    --lambda_heatmap_rec 0.001 \
    --lambda_rot_heatmap_rec 0.001 \
\
    --niter 5 \
    --niter_decay 5 \
    --lr 1e-3 \
    --batch_size 16 \
    --num_rot_heatmap 0 \
    --num_heatmap 15 \
    --data_dir /ssd_data1/UnrealEgoData \
\
