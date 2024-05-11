import os
import torch
import timm
import torch.nn as nn
import numpy as np
from torchvision import transforms 
import torch.nn.functional as F
import math
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
# from torch.utils.tensorboard.writer import SummaryWriter

class DinoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.model =  timm.create_model('vit_base_patch14_reg4_dinov2',
                        in_chans=3,
                        dynamic_img_size=True,
                        pretrained = True,
                        pretrained_cfg_overlay=dict(file='/bd_byta6000i0/users/dataset/feat_visualize_models/vit_base_patch14_reg4_dinov2.lvd142m.bin')
                    )
        
        self.num_ch_enc = np.array([96,192,384,768])

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=768,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in self.num_ch_enc
        ])

        

    def forward(self, image):
        height, width = image.shape[-2:]
        target_size = []
        for i in [2,4,8,16]:
            target_size.append((math.ceil(height / i), math.ceil(width / i)))
        patch_h, patch_w = height // 14, width // 14

        image = F.interpolate(image,(patch_h * 14, patch_w * 14))

        out_features = self.model.get_intermediate_layers(image,4)
        out = []

        for feature in out_features:
            out.append(feature.permute(0, 2, 1).reshape((feature.shape[0], feature.shape[-1], patch_h, patch_w)))

        for i in range(4):
            out[i] = self.projects[i](out[i])
            out[i] = F.interpolate(out[i], target_size[i],mode='bilinear',align_corners=False)
            

        return out
    
    def temporal_forward(self, features):
        features = torch.chunk(features, features.shape[-3], 2)
        frame_features = []
        for feature in features:
            frame_features.append(self.forward(feature.squeeze(2)))

        merged_features = []

        for tensors in zip(*frame_features):
            merged_tensor = torch.stack(tensors).permute(1,2,0,3,4)
            merged_features.append(merged_tensor)
        

        return merged_features
                
    
    

