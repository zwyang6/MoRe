#!/bin/bash

file=$1
device_gpu=$2
nproc_per_node=$3
master_port=$4
exp_des=$5

CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --log_tag=$exp_des