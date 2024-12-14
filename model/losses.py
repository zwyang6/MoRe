import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
from bilateralfilter import bilateralfilter, bilateralfilter_batch

def multi_cls_loss(cls_tokens,cls_label,only_regu=True):

    B, num_classes, _ = cls_tokens.shape
    cls_tokens = F.normalize(cls_tokens, dim=-1)
    regular_logits = torch.matmul(cls_tokens,cls_tokens.permute(0,2,1)) / 0.07  # b, 20 

    regu_labels = torch.arange(num_classes, device=cls_tokens.device).unsqueeze(0).repeat(B,1)
    regular_loss = F.cross_entropy(regular_logits, regu_labels)

    if only_regu:
        loss = regular_loss
    else:
        mcls_logits = cls_tokens.mean(-1) # b, 20 
        mcls_loss = F.multilabel_soft_margin_loss(mcls_logits, cls_label)
        loss = 0.5 * mcls_loss + 0.5 * regular_loss

    return loss


def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss

def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)

    if img_box is not None:
        crop_mask = torch.zeros_like(pred_prob[:, 0, ...])
        for idx, coord in enumerate(img_box):
            crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    else:
        crop_mask = torch.ones_like(pred_prob[:, 0, ...])

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DRELoss_neg(nn.Module):
    def __init__(self, temp=1.0,):
        super().__init__()
        self.temp = temp

    def forward(self, cls_tokens_visual_, patch_tokens_, masks,cls_labels):
        #cal_logit
        b = masks.shape[0]
        # cls_tokens_visual = cls_tokens_visual_
        cls_tokens_visual =  F.normalize(cls_tokens_visual_, p=2, dim=2, eps=1e-8)
        # cls_tokens_visual =  F.normalize(cls_tokens_visual_, p=2, dim=2, eps=1e-8).detach().clone()   
        # patch_tokens =    F.normalize(patch_tokens_, p=2, dim=1, eps=1e-8)
        patch_tokens =    patch_tokens_.detach().clone()    
        logits = torch.matmul(cls_tokens_visual, (patch_tokens.reshape(b,patch_tokens.shape[1],-1)))
        logits = torch.exp(logits / self.temp)

        total_loss = 0
        for i in range(b):
            cer_loss = 0
            ## fore tokens contrast
            pos_rows = torch.where(cls_labels[i] == 1)[0].tolist()
            for idx in pos_rows:
                logit = logits[i,idx,:]
                if logit.sum() != 0:
                    loss = -torch.log(logit / (logit.sum() + 1e-4))
                else:
                    loss = -torch.log((1) / (1 + logit.sum() + 1e-4))
                flag_cer, _ = self.mask2flag(masks[i],idx+1)       
                cer_loss += ((flag_cer * loss).sum() / (flag_cer.sum() + 1e-4))
    
            total_loss = total_loss + cer_loss

        total_loss = total_loss / b

        return total_loss

    def mask2flag(self,mask, cls_idx):
        mask_un = mask.clone()
        mask_un = (mask_un == 255) 

        mask_cer = mask.clone()
        mask_cer = (mask_cer == cls_idx) 

        flags_cer = mask_cer.flatten()
        flags_un = mask_un.flatten()

        return flags_cer,flags_un

## token
class SRELoss_neg(nn.Module):
    def __init__(self, temp=1.0,):
        super().__init__()
        self.temp = temp

    def forward(self, cls_tokens_visual_, patch_tokens, masks, cls_labels):
        #cal_logit
        patch_tokens =    F.normalize(patch_tokens, p=2, dim=1, eps=1e-8)
        cls_tokens_visual_ =  F.normalize(cls_tokens_visual_, p=2, dim=2, eps=1e-8)
        roi_mask = self.mask2roi(masks)
        un_embeddings = self.crop_from_roi_neg(patch_tokens, roi_mask)
        cls_tokens_visual =  cls_tokens_visual_.detach()
        patch_tokens_cer = cls_tokens_visual
        total_loss = 0
        n = 0

        bkg_cls_label = (cls_labels == 0)

        for idx, item_un_embeds in enumerate(un_embeddings):
            for un in item_un_embeds:
                logits = self.cos_sim(un, patch_tokens_cer[idx].transpose(1,0))
                total_loss += (0.5*(1 - torch.sum(cls_labels[idx] * logits) / (cls_labels[idx].sum()+1)) + 0.5 * torch.sum(bkg_cls_label[idx] * logits) / (bkg_cls_label[idx].sum()+1))
                n += 1
        total_loss = total_loss / n

        return total_loss

    def crop_from_roi_neg(self, patch_tokens, roi_mask=None, crop_num=8, kernel_size=6):

        un_tokens = []
        b, c, h, w =  patch_tokens.shape
        margin = kernel_size//2

        for i1 in range(b):    
            roi_un_tokens = []
            roi_index_ = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] == 1).nonzero()
            # select all un tokens
            roi_index = roi_index_
            for i2 in range(roi_index.shape[0]):
                h0, w0 = roi_index[i2, 0], roi_index[i2, 1] # centered at (h0, w0)
                temp_mask = roi_mask[i1, h0:(h0+kernel_size), w0:(w0+kernel_size)]
                if temp_mask.sum() / (kernel_size*kernel_size) >= 1.2:
                    ## if ratio of uncertain regions < 0.2 then negative
                    roi_un_tokens.append(patch_tokens[i1,:, h0,w0])

            if len(roi_un_tokens) > crop_num:
                rand_index = torch.randperm(len(roi_un_tokens))
                roi_un_tokens = torch.stack(roi_un_tokens,dim=0)
                roi_un_tokens_ = roi_un_tokens[rand_index[:crop_num]]

            else:
                # add_idx = crop_num - len(roi_un_tokens)
                add_idx = (crop_num - len(roi_un_tokens)) // 4
                pos_roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] == 2).nonzero()
                if pos_roi_index.shape[0] < add_idx:
                    pos_roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= 0).nonzero() ## if NULL then random crop
                rand_index = torch.randperm(pos_roi_index.shape[0])
                pos_index = pos_roi_index[rand_index[:add_idx], :]
                for ip in range(pos_index.shape[0]):
                    h0, w0 = pos_index[ip, 0], pos_index[ip, 1] # centered at (h0, w0)
                    roi_un_tokens.append(patch_tokens[i1,:, h0,w0])

                roi_un_tokens_ = torch.stack(roi_un_tokens,dim=0)
            un_tokens.append(roi_un_tokens_) 

        return un_tokens

    def mask2roi(self,mask):
        mask_ = mask.clone()
        roi_mask = torch.zeros_like(mask, dtype=torch.int16)
        mask_[mask_==255] = 0
        roi_mask[mask == 255] = 1
        roi_mask[mask_ >= 1 ] = 2

        return roi_mask
    
    def mask2flag(self,mask):

        mask_un = mask.clone()
        mask_un = mask_un == 255
        flag_un = mask_un.flatten(1)

        mask_cer = mask.clone()
        mask_cer[mask==255] = 0
        mask_cer[mask_cer>=1] = 1
        flags_cer = mask_cer.flatten(1)

        # bkg + fore
        mask_mix = mask.clone()
        mask_mix[mask==0] = 1
        mask_mix[mask==255] = 0
        mask_mix[mask_mix>=1] = 1
        flag_mix = mask_mix.flatten(1)

        flag_bkg = flag_mix - flags_cer

        return flag_mix, flags_cer, flag_un, flag_bkg

    def cos_sim(self,x, y):
        cos_sim = torch.matmul(x,y)
        return torch.abs(cos_sim)
