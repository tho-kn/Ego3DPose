#!/bin/bash

# script
python test.py  \
    --experiment_name egoglass \
    --model egoglass  \
    --use_amp   \
    --batch_size 1 \
    --data_dir /ssd_data1/UnrealEgoData \
