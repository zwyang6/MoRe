#!/bin/bash

#SBATCH -J coco_eval
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 6
#SBATCH --gpus=8
#SBATCH -t 3-0:00:00
#SBATCH -e coco_eval-%j.out

file=/HOME/scz0658/run/Jaye_Files/MoRe/tools/infer_seg_coco.py
device_gpu=0,1,2,3,4,5,6,7
nproc_per_node=8
infer_set=val
model_path=MoRe/coco_sota/checkpoints/model_iter_80000.pth
CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node  $file --model_path $model_path --infer_set $infer_set