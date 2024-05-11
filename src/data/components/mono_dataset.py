from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import h5py



class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False):
        super().__init__()

        for i in range(len(filenames) - 1, -1, -1):
            line = filenames[i].split()
            folder = line[0]

            if len(line) == 3:
                frame_index = int(line[1])
            else:
                frame_index = 0

            if len(line) == 3:
                side = line[2]
            else:
                side = None
            
            if frame_index + frame_idxs[0] < 1:
                del filenames[i]


        self.data = h5py.File(data_path, 'r')
        self.filenames = filenames

        self.num_scales = num_scales
        self.frame_idxs = frame_idxs

        self.is_train = is_train


        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = A.Compose([
                A.Resize(
                    height // s, 
                    width // s,
                    interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(),
                ToTensorV2(),
            ])

        self.depth_to_tensor = ToTensorV2()


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        inputs["color_clip"] = []
        inputs["depth_gt_clip"] = []

        # do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        if len(line) == 3:
            side = line[2]
        else:
            side = None


        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                color = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                color = self.get_color(folder, frame_index + i, side, do_flip)
            color = self.resize[0](image=color)['image']
            inputs["color_clip"].append(color)


        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                depth_gt = self.get_depth(folder, frame_index, other_side, do_flip)
            else:
                depth_gt = self.get_depth(folder, frame_index + i, side, do_flip)
            
            depth_gt = self.depth_to_tensor(image=depth_gt)['image']
            inputs['depth_gt_clip'].append(depth_gt)
            # print(i)


        inputs['depth_gt_clip'] = torch.stack(inputs['depth_gt_clip']).permute(1,0,2,3)
        inputs['color_clip'] = torch.stack(inputs['color_clip']).permute(1,0,2,3)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}".format(frame_index)
        group_path = os.path.join(folder, f_str)
        # print(group_path)
        color = self.data[group_path]['image'][:]
        

        if do_flip:
            color = np.fliplr(color)
            color = color.copy()

        return color


    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}".format(frame_index)
        group_path = os.path.join(folder, f_str)

        depth_gt = self.data[group_path]['gt'][:]
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
            depth_gt = depth_gt.copy()

        return depth_gt
