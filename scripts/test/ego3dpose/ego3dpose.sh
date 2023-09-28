#!/bin/bash

# script
python test.py  \
    --experiment_name ego3dpose_autoencoder \
    --model ego3dpose_autoencoder \
    --use_amp \
    --batch_size 1 \
    --num_rot_heatmap 14 \
    --num_heatmap 15 \
    --heatmap_type sin \
    --data_dir /ssd_data1/UnrealEgoData \
