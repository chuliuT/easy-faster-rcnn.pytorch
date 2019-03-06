#!/usr/bin/env bash
BACKBONE=$1
OUTPUTS_DIR=$2
if ! [[ -n "${OUTPUTS_DIR}" ]]; then
    echo "Argument OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=coco2017 -b=${BACKBONE} -o=${OUTPUTS_DIR} --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 256, 512]" --anchor_smooth_l1_loss_beta=0.1111 --batch_size=1 --learning_rate=0.00125 --weight_decay=0.0001 --step_lr_sizes="[960000, 1280000]" --num_steps_to_snapshot=320000 --num_steps_to_finish=1440000