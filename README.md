# [AAAI2025] MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation

We proposed MoRe to effectively tackle the artifact issue when generating Localization Attention Map (LAM) from class-patch attention in WSSS. 

## News

* **If you find this work helpful, please give us a :star2: to receive the updation !**
* **` Dec. 10th, 2024`:** MoRe is accepted by AAAI2025.
* **Code is available on VOC. We are sorting code for COCO** 🔥🔥🔥

## Overview

<p align="middle">
<img src="/sources/main_fig.png" alt="MoRe pipeline" width="1200px">
</p>

Weakly Supervised Semantic Segmentation (WSSS) with image-level labels typically uses Class Activation Maps (CAM) to achieve dense predictions. Recently, Vision Transformer (ViT) has provided an alternative to generate localization maps from class-patch attention. However, due to insufficient constraints on modeling such attention, we observe that the Localization Attention Maps (LAM) often struggle with the artifact issue, i.e., patch regions with minimal semantic relevance are falsely activated by class tokens. In this work, we propose MoRe to address this issue and further explore the potential of LAM. Our findings suggest that imposing additional regularization on class-patch attention is necessary. To this end, we first view the attention as a novel directed graph and propose the Graph Category Representation module to implicitly regularize the interaction among class-patch entities. It ensures that class tokens dynamically condense the related patch information and suppress unrelated artifacts at a graph level. Second, motivated by the observation that CAM from classification weights maintains smooth localization of objects, we devise the Localization-informed Regularization module to explicitly regularize the class-patch attention. It directly mines the token relations from CAM and further supervises the consistency between class and patch tokens in a learnable manner. Extensive experiments are conducted on PASCAL VOC and MS COCO, validating that MoRe effectively addresses the artifact issue and achieves state-of-the-art performance, surpassing recent single-stage and even multi-stage methods.

## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). The download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

To generate VOC style segmentation labels for COCO, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc), or just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```

## Requirement

Please refer to the requirements.txt. 

We incorporate a regularization loss for segmentation. Please refer to the instruction for this [python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).

## Train MoRe
``` bash
### train voc
bash run_train.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port]  train_voc

### train coco
bash run_train.sh scripts/train_coco.py [gpu_devices] [gpu_numbers] [master_port] train_coco
```

## Evaluate MoRe
``` bash
### eval voc seg and LAM
bash run_evaluate_voc.sh tools/infer_lam.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]

### eval coco seg
bash run_evaluate_seg_coco.sh tools/infer_seg_coco.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]
```

## Main Results

#### 1. Artifact Issue

<p align="middle">
<img src="/sources/artifact_issue.png" alt="artifact issue" width="1200px">
</p>

#### 2. Semantic Results
Semantic performance on VOC and COCO. Logs and weights are available now.
| Dataset | Backbone |  Val  | Test | Log |
|:-------:|:--------:|:-----:|:----:|:---:|
|   PASCAL VOC   |   ViT-B  | 76.4  | [75.0](http://host.robots.ox.ac.uk/anonymous/9QW1IM.html) | [log](logs/voc_train.log) |
|   MS COCO  |   ViT-B  |  47.4 |   -  | [log](logs/coco_train.log) |

## Citation 
Please cite our work if you find it helpful to your reseach. :two_hearts:
```bash
@article{yang2024more,
  title={MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation},
  author={Yang, Zhiwei and Neng, Yucong and Fu, Kexue and Wang, Shuo and Song, Zhijian},
  journal={arXiv preprint arXiv:2402.18467},
  year={2024}
}
```
If you have any questions, please feel free to contact the author by zwyang21@m.fudan.edu.cn.

## Acknowledgement
This repo is built upon [MCTformer Series](https://github.com/xulianuwa/MCTformer.git) and [SeCo](https://github.com/zwyang6/SeCo.git). Many thanks to their brilliant works!!!
