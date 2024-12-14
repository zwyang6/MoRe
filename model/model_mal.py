import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
from . GCA import graphic_cls_aggregation

class network(nn.Module):
    def __init__(self, args,backbone, num_classes=None, pretrained=None, aux_layer=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, args=args, if_reload=True, aux_layer=aux_layer, num_classes=(args.num_attri)) # extra token for cls token
        self.num_heads = self.encoder.num_heads
    
        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 
        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes,)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.graph_layer = graphic_cls_aggregation(topk=392,agg_type='bi-interaction')
        
    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups


    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, cam_only=False, with_gcr=False):

        mctokens, _x, x_aux, score_maps = self.encoder.forward_features(x)

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size
        _x4 = self.to_2D(_x, h, w)
        _x_aux = self.to_2D(x_aux, h, w)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
            return cam_aux, cam
                
        cls_aux = self.pooling(_x_aux, (1,1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1,1))
        cls_x4 = self.classifier(cls_x4)

        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        cls_aux = cls_aux.view(-1, self.num_classes-1)

        seg = self.decoder(_x4)
        if with_gcr:
            mctokens = self.graph_layer(mctokens,_x)

        score_maps_ = torch.sum(score_maps,dim=1)[:,:mctokens.shape[1],mctokens.shape[1]:]
        score_maps = score_maps_.reshape(score_maps.shape[0],mctokens.shape[1],h,w).detach()

        return cls_x4, seg, _x4, cls_aux, mctokens, score_maps