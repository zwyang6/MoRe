#!/bin/bash

echo "Evaluating VOC seg Results>>>>>>>>>>>"
refine_with_multiscale=true
crf_post=true
cam2seg=true

file=$1
device=$2
nproc_per_node=$3
infer_set=$4
checkpoint=$5


CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$nproc_per_node  $file --model_path $checkpoint --infer_set $infer_set --refine_with_multiscale $refine_with_multiscale  --crf_post $crf_post --cam2seg $cam2seg

echo "Evaluating VOC LAM Results>>>>>>>>>>>"
cam2seg=false
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$nproc_per_node  $file --model_path $checkpoint --infer_set $infer_set --refine_with_multiscale $refine_with_multiscale  --crf_post $crf_post --cam2seg $cam2seg

