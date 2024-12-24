#!/bin/bash

device=$1
nproc_per_node=$2
checkpoint=$1

echo "Evaluating VOC CAM Results>>>>>>>>>>>"
file=./tools/infer_cam.py
infer_set=train
cam2seg=false
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$nproc_per_node  $file --model_path $checkpoint --infer_set $infer_set --refine_with_multiscale $refine_with_multiscale  --crf_post $crf_post --cam2seg $cam2seg


echo "Evaluating VOC LAM Results>>>>>>>>>>>"
file=./tools/infer_lam.py
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$nproc_per_node  $file --model_path $checkpoint --infer_set $infer_set --refine_with_multiscale $refine_with_multiscale  --crf_post $crf_post --cam2seg $cam2seg


echo "Evaluating VOC seg Results>>>>>>>>>>>"
file=./tools/infer_seg_voc.py
infer_set=val
CUDA_VISIBLE_DEVICES=$device python $file --model_path $checkpoint --infer_set $infer_set
