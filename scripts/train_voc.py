import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append("/data/PROJECTS/MoRe_2024/MoRe/")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc
from datasets.transforms import MultiviewTransform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.losses import (get_masked_ptc_loss, get_seg_loss, get_energy_loss, CRELoss_neg, URELoss_neg,
                        multi_cls_loss,  DenseEnergyLoss)
from torch.nn.parallel import DistributedDataParallel
from model.PAR import PAR
from utils import imutils,evaluate
from utils.camutils import (cam_to_label, multi_scale_cam2, label_to_aff_mask, score_map_cam, 
                            refine_cams_with_bkg_v2,)
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from engine import build_network, build_optimizer, build_validation
parser = argparse.ArgumentParser()
torch.hub.set_dir("./pretrained/")

##### Parameter settings
parser.add_argument("--backbone", default='matvit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--finetune", default="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth", type=str, help="pretrain url")
parser.add_argument("--model", default="MATformer", type=str, help="MATformer")
parser.add_argument("--num_attri", default=24, type=int, help="number of attribution tokens")
parser.add_argument('--sim', default=1.0, type=float, help='momentum for computing the momving average of prototypes')
parser.add_argument("--text_attri_pt_path", default="./attributes_text/attribute_embedding/pascal_voc_desc_bge-base-en-v1.5_gpt4.0_cluster_32_embedding_bank.pth", type=str, help="attri_path")
parser.add_argument("--update_prototype", default=0, type=int, help="begin to update prototypes")
parser.add_argument("--with_gcr", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")

### loss weight
parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_seg", default=0.12, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_var", default=0.5, type=float, help="w_var")
parser.add_argument("--w_une", default=0.1, type=float, help="w_ctc")
parser.add_argument("--w_cle", default=0.2, type=float, help="w_ptc")


### training utils
parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.5), help="multi_scales for cam")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")

### knowledge extraction
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save_ckpt")
parser.add_argument("--tensorboard", default=True, type=bool, help="log tb")
parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")

### dataset utils
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--use_aa", type=bool, default=False)
parser.add_argument("--use_gauss", type=bool, default=False)
parser.add_argument("--use_solar", type=bool, default=False)
parser.add_argument("--global_crops_number", type=int, default=2)
parser.add_argument("--local_crops_number", type=int, default=0)

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    device = torch.device(args.local_rank)

    ### build model 
    model, param_groups = build_network(args)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    ### build dataloader 
    train_transform = MultiviewTransform(
        size1=args.crop_size,
        num1=args.global_crops_number,
        num2=args.local_crops_number,
        use_aa=args.use_aa,
        use_gauss=args.use_gauss,
        use_solar=args.use_solar,
    )

    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        num_classes=args.num_classes,
        transform=train_transform,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)
    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    ### build optimizer 
    optim = build_optimizer(args,param_groups)
    logging.info('\nOptimizer: \n%s' % optim)

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):
        global_step = n_iter + 1
        try:
            img_name, inputs, cls_label, img_lst = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_lst = next(train_loader_iter)
        img_box = None

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales,)
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )

        cls, segs, fmap, cls_aux, attr_tokens, _ = model(inputs, with_gcr=args.with_gcr)

        # cle loss
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_cle = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=0.5, low_thre=args.low_thre, ignore_index=args.ignore_index)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        resized_cams = F.interpolate(cams, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label = cam_to_label(resized_cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=0.6, low_thre=0.35, ignore_index=args.ignore_index)

        mask_cle = pseudo_label_cle if n_iter <= 10000 else pseudo_label
        mask_une = pseudo_label_aux if n_iter <= 10000 else pseudo_label
        CLE_loss = CRELoss_neg(temp=0.5).cuda()
        UNE_loss = URELoss_neg(temp=0.5).cuda()

        cle_loss = CLE_loss(attr_tokens,fmap,mask_cle,cls_label)
        une_loss = UNE_loss(attr_tokens,fmap,mask_une,cls_label)

        ### cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)
        attri_loss = multi_cls_loss(attr_tokens, cls_label)

        ### seg_loss & reg_loss
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        ### aff loss from ToCo, https://github.com/rulixiang/ToCo
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=0.65, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # warmup
        if n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 1.0 * attri_loss + args.w_cle * cle_loss + args.w_une * une_loss + args.w_ptc * ptc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 1.0 * attri_loss  + args.w_ptc * ptc_loss + args.w_cle * cle_loss + args.w_une * une_loss + args.w_seg * seg_loss + args.w_reg * reg_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'attri_loss': attri_loss.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
            'cle_loss': cle_loss.item(),
            'une_loss': une_loss.item(),
        })

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, cls_loss_mct: %.4f, ptc_loss: %.4f,  cle_loss: %.4f, une_loss: %.4f, seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('attri_loss'), avg_meter.pop('ptc_loss'), avg_meter.pop('cle_loss'), avg_meter.pop('une_loss'), avg_meter.pop('seg_loss')))
        
        if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = build_validation(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)

    return True

if __name__ == "__main__":

    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    dataset = os.path.basename(args.list_folder)
    exp_tag = f'{dataset}_{args.log_tag}_{timestamp_2}'
    args.work_dir = os.path.join(args.work_dir, timestamp_1, exp_tag)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")


    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
