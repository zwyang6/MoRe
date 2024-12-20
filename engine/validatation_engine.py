from utils.pyutils import AverageMeter, format_tabs
import torch
from tqdm import tqdm
from utils import evaluate
from utils.camutils import cam_to_label, multi_scale_cam2, norm_map_cam
import numpy as np
import torch.nn.functional as F
from datasets import voc

def build_validation(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux, sms, fused = [], [], [], [],[],[]
    seg_cam = []

    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _x4, cls_aux, attri_tokens_, score_maps,= model(inputs,)
            score_maps = torch.einsum('bchw,bkc->bkhw', _x4, attri_tokens_).clone().detach()

            _cams, _ = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            _, cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            # index_map = torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16)
            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_sm = F.interpolate(score_maps, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_sm = norm_map_cam(resized_sm)
            _, sm_label = cam_to_label(resized_sm, cls_label, bkg_thre=args.bkg_thre, high_thre=0.5, low_thre=0.25, ignore_index=args.ignore_index)

            fused_cam = resized_sm * resized_cam
            fused_cam = norm_map_cam(fused_cam)
            _, f_label = cam_to_label(fused_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            sms += list(sm_label.cpu().numpy().astype(np.int16))
            fused += list(f_label.cpu().numpy().astype(np.int16))


    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    sms_score = evaluate.scores(gts, sms)
    f_score = evaluate.scores(gts, fused)

    model.train()

    tab_results = format_tabs([cam_score, sms_score, f_score, seg_score], name_list=["CAM", "Score Map", "Fused", "Seg_Pred"], cat_list=voc.class_list)

    return cls_score, tab_results
