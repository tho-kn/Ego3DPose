#!/bin/bash

# script
python train.py \
    --project_name Ego3DPose \
    --experiment_name ego3dpose_autoencoder \
    --model ego3dpose_autoencoder \
\
\
    --use_amp \
    --init_ImageNet \
\
    --lambda_mpjpe 0.1 \
    --lambda_rot 1.0 \
    --lambda_heatmap 1.0 \
    --lambda_rot_heatmap 1.0 \
    --lambda_cos_sim -0.01 \
    --lambda_heatmap_rec 0.001 \
    --lambda_rot_heatmap_rec 0.001 \
\
    --epoch_count 1 \
    --niter 5 \
    --niter_decay 10 \
    --lr 1e-3 \
    --batch_size 16 \
    --num_rot_heatmap 14 \
    --num_heatmap 15 \
    --heatmap_type sin \
    --path_to_trained_heatmap ./log/ego3dpose_heatmap_shared_pretrained/best_net_HeatMap.pth \
    --data_dir /ssd_data1/UnrealEgoData \

    # In paper the epoch is set to niter 5 / niter_decay 5
    # Later we found that niter 5 / niter_decay 10 is found to be more stable
