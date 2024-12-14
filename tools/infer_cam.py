import argparse
import os
import sys
sys.path.append(".")
from collections import OrderedDict
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.model_mal import network
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cam_to_label, get_valid_cam, multi_scale_lam2, multi_scale_cam2
from utils.pyutils import format_tabs, setup_logger, format_tabs_multi_metircs,convert_test_seg2RGB
from utils.dcrf import DenseCRF
import joblib
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='matvit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--finetune", default="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth", type=str, help="pretrain url")
parser.add_argument("--model", default="MATformer", type=str, help="MATformer")
parser.add_argument("--num_attri", default=24, type=int, help="number of attribution tokens")
parser.add_argument('--sim', default=1.0, type=float, help='momentum for computing the momving average of prototypes')
parser.add_argument("--text_attri_pt_path", default="./attributes_text/attribute_embedding/pascal_voc_desc_bge-base-en-v1.5_gpt4.0_cluster_32_embedding_bank.pth", type=str, help="attri_path")
parser.add_argument("--update_prototype", default=0, type=int, help="begin to update prototypes")

#! TO DO
####refine_raw_CAM 2 masks with multiscales, if False output cam_seeds, else output refined_masks
parser.add_argument("--infer_set", default="train", type=str, help="infer_set")
parser.add_argument("--model_path", default=".pth", type=str, help="model_path")

parser.add_argument("--refine_with_multiscale", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="refine_cam_with_multiscale")
parser.add_argument("--cam2seg", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--crf_post", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--save_cam", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="save the cam figs")
parser.add_argument("--save_cls_specific_cam", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="save the cam figs")
parser.add_argument("--clspreds_from_mct", default='xxx',type=str, help="pretrain url")


parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="use imagenet pretrained weights")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")

parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="work_dir")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")

parser.add_argument("--nproc_per_node", default=8, type=int, help="nproc_per_node")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def _validate(pid,model=None, dataset=None, args=None):

    model.eval()
    color_map = plt.get_cmap("jet")
    data_loader = DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)


    with torch.no_grad(), torch.cuda.device(0):

        if os.path.exists(args.clspreds_from_mct):
            cls_preds = np.load(args.clspreds_from_mct, allow_pickle=True).item()
            
        model.cuda()

        gts, cams, aux_cams = [], [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            img = imutils.denormalize_img(inputs)[0].permute(1,2,0).cpu().numpy()

            inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            labels = labels.cuda()
            cls_label_gt = cls_label.cuda()

            ###
            if args.cam2seg:
                cls, segs, _x4, cls_aux, attri_tokens_, score_maps,= model(inputs,)
                cls_pred = (cls>0).type(torch.int16)
                if os.path.exists(args.clspreds_from_mct):
                    cls_pred = cls_preds[name[0]].cuda()
                cls_idx = torch.argmax(segs, dim=1).unique()
                non_zero_cls = cls_idx[cls_idx != 0] - 1
                cls_pred_from_seg = torch.eye(args.num_classes-1).cuda()[non_zero_cls].sum(dim=0,keepdim=True)
                
                for id in cls_idx[1:]:
                    if (torch.argmax(segs, dim=1)==id).sum() < 5:
                        cls_pred_from_seg[0,id-1] = 0

                cls_label = cls_pred_from_seg if cls_pred.sum() <= 0 else cls_pred
            else:
                cls_label = cls_label_gt

            if args.refine_with_multiscale:
                _cams, _cams_aux = multi_scale_cam2(model, inputs, [1.0,0.5,0.75,1.25,1.5,1.75,2.0])
            else:
                _cams, _cams_aux = multi_scale_cam2(model, inputs, [1.0])

            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)

            _,cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre)

            resized_valid_cam = get_valid_cam(resized_cam, cls_label)

            cam_np = torch.max(resized_valid_cam[0], dim=0)[0].cpu().numpy()

            cam_rgb = color_map(cam_np)[:,:,:3] * 255

            alpha = 0.5
            cam_rgb = alpha*cam_rgb + (1-alpha)*img
            imageio.imsave(os.path.join(args.cam_dir, name[0] + ".jpg"), cam_rgb.astype(np.uint8))
            if args.save_cam:
                if not args.save_cls_specific_cam:
                    imageio.imsave(os.path.join(args.cam_dir, name[0] + ".jpg"), cam_rgb.astype(np.uint8))
                else:
                    cls_idx = torch.where(cls_label != 0)[1]
                    for _,idx in enumerate(cls_idx):
                        cam_np = resized_cam[0,idx,...].cpu().numpy()
                        cam_rgb = color_map(cam_np)[:,:,:3] * 255
                        alpha = 0.6
                        cam_rgb = alpha*cam_rgb + (1-alpha)*img
                        imageio.imsave(os.path.join(args.cs_cam_dir, name[0] + f"_{voc.class_list[idx+1]}.jpg"), cam_rgb.astype(np.uint8))

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            if args.crf_post:
                keys = torch.where(cls_label[0] != 0)[0]
                keys_gt = torch.where(cls_label_gt[0] != 0)[0]
                valid_lam = resized_cam[0][keys,...]
                np.save(args.logits_dir + "/" + name[0] + '.npy', {"valid_cam":valid_lam.cpu().numpy(),"keys":keys.cpu().numpy(),"keys_gt":keys_gt.cpu().numpy()})

    logging.info('MASK/CAM_score:')
    cam_score = evaluate.scores(gts, cams)
    metrics_tab = format_tabs([cam_score], ["cam"], cat_list=voc.class_list)
    logging.info("\n"+metrics_tab)

    return metrics_tab


def validate(args=None):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    model = network(args,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer=-3,
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    model.to(torch.device(args.local_rank))
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[args.local_rank],)
    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]

    results = _validate(pid=args.local_rank,model=model, dataset=split_dataset, args=args)

    torch.cuda.empty_cache()

    if args.crf_post:
        crf_score = crf_proc()

    return True


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    def _job(i):

        name = name_list[i]
        logit_name = args.logits_dir + "/" + name + ".npy"
        logit_ = np.load(logit_name, allow_pickle=True).item()
        lams = logit_['valid_cam']
        keys = logit_['keys']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        H, W, _ = image.shape

        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)
    
        if keys.shape[0] <= 0:
            pred_crf = np.zeros((H,W)).astype(np.uint8)
            imageio.imsave(args.segs_crf_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred_crf)).astype(np.uint8))
            print(f'{name} not exist')
            if args.infer_set == 'test':
                convert_test_seg2RGB(np.squeeze(pred_crf).astype(np.uint8),args.test_segs_dir + "/" + name + ".png")
            return pred_crf,label

        bg_score = np.power(1 - np.max(lams, axis=0, keepdims=True), 1)
        lams = np.concatenate((bg_score, lams), axis=0)
        prob = lams

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)
        keys = np.pad(keys + 1, (1, 0), mode='constant')
        pred_crf = keys[pred].astype(np.uint8)
        imageio.imsave(args.segs_crf_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred_crf)).astype(np.uint8))
    
        if args.infer_set == 'test':
            convert_test_seg2RGB(np.squeeze(pred_crf).astype(np.uint8),args.test_segs_dir + "/" + name + ".png")
        return pred_crf,label
    
    n_jobs = int(os.cpu_count() * 0.6)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score

if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints/")[0] + f'/{args.infer_set}/'
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')

   
    if args.cam2seg:
        tag = 'multiscale_cam2seg/mscam' if args.refine_with_multiscale else 'cam2seg_seeds/cam'
    else:
        tag = 'multiscale_cams/mscam' if args.refine_with_multiscale else 'cam_seeds/cam'
    
    if args.crf_post:
        args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_logits")
        args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds")
        args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds_rgb")
        args.segs_crf_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/segcrf_preds_rgb")
        os.makedirs(args.logits_dir, exist_ok=True)
        os.makedirs(args.segs_dir, exist_ok=True)
        os.makedirs(args.segs_rgb_dir, exist_ok=True)
        os.makedirs(args.segs_crf_rgb_dir, exist_ok=True)
    
    if args.infer_set == 'test':
        args.test_segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/results/VOC2012/Segmentation/comp6_test_cls/")
        os.makedirs(args.test_segs_dir, exist_ok=True)

    if args.save_cls_specific_cam:
        args.cs_cam_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_class_specific_img")
        os.makedirs(args.cs_cam_dir, exist_ok=True)

    args.cam_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_img")
    args.cam_aux_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_aux_img")
    args.log_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_results.log")

    os.makedirs(args.cam_dir, exist_ok=True)
    os.makedirs(args.cam_aux_dir, exist_ok=True)

    setup_logger(filename=args.log_dir)

    validate(args=args)