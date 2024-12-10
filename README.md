# [AAAI2025] MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation

We proposed MoRe to effectively tackle the artifact issue when generating Localization Attention Map (LAM) from class-patch attention in WSSS. 

## News

* **If you find this work helpful, please give us a :star2: to receive the updation !**
* **` Dec. 10th, 2024`:** MoRe is accepted by AAAI2025.
* **Code will be available very soon** 🔥🔥🔥

<!-- * **` Aug. 12th, 2024`:** We released our paper on Arxiv. Further details can be found in the updated [arXiv](http://arxiv.org/abs/2402.18467).
  
* **` Mar. 1st, 2024`:**  Code is available now.
* **` Mar. 2st, 2024`:**  Logs and weights are available now. -->

## Overview

<p align="middle">
<img src="/sources/main_fig.png" alt="SeCo pipeline" width="1200px">
</p>

Weakly Supervised Semantic Segmentation (WSSS) with image-level labels typically uses Class Activation Maps (CAM) to achieve dense predictions. Recently, Vision Transformer (ViT) has provided an alternative to generate localization maps from class-patch attention. However, due to the little constraint to model such attention, we observe that the Localization Attention Maps (LAM) usually struggle with the artifact issue, i.e., patch regions with minimal semantic relevance are falsely activated by class tokens. In this work, we propose MoRe to address this issue and further explore the potential of LAM. Our findings suggest that additional regularization on class-patch attention is necessary. To this end, we first view the attention as a novel directed graph and propose the Graph Category Representation module to implicitly regularize the interaction among class-patch entities. It ensures that class tokens dynamically condense the related patch information and suppress unrelated artifacts at a graph level. Second, motivated by the observation that CAM from classification weights maintains smooth localization of objects, we devise the Localization-informed Regularization module to explicitly regularize the class-patch attention. It directly mines the token relations from CAM and further supervises the consistency between class and patch tokens in a learnable manner. Extensive experiments are conducted on PASCAL VOC and MS COCO, validating the efficiency of MoRe in addressing the artifact issue and thereby achieving state-of-the-art performance over recent single-stage and even multi-stage methods.
