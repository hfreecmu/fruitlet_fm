import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet_2023/scripts/nbv/feature_association')

import os
import cv2
import numpy as np
import torch

from data.dataloader import get_data_loader
from utils.nbv_utils import vis_lines, vis_segs

images_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/images_1400'
segmentations_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/scripts/nbv/feature_association/datasets/train'
output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_homography'
min_dim = 24
batch_size = 1
shuffle = True
num_images = 20#5
num_lines = 10
dataloader = get_data_loader(images_dir, segmentations_dir, min_dim,
                             batch_size, shuffle)

im_count = 0
for i, data in enumerate(dataloader):
    im_0, im_1, seg_inds_pad_0, seg_inds_pad_1, num_points_0, num_points_1, matches_pad_0, matches_pad_1, _ = data
    
    cv_im_0 = im_0[0].permute(1, 2, 0).numpy().copy()
    cv_im_1 = im_1[0].permute(1, 2, 0).numpy().copy()
    
    seg_inds_0 = seg_inds_pad_0[0, 0:num_points_0[0]].numpy().copy()
    seg_inds_1 = seg_inds_pad_1[0, 0:num_points_1[0]].numpy().copy()

    matches_0 = matches_pad_0[0, 0:num_points_0[0]].numpy().copy()
    matches_1 = matches_pad_1[0, 0:num_points_1[0]].numpy().copy()

    has_match_0 = (matches_0 != -1)
    has_match_1 = (matches_1 != -1)

    assert has_match_0.sum() == has_match_1.sum()

    matched_inds_0 = np.arange(matches_0.shape[0])[has_match_0]
    matched_inds_1 = matches_0[has_match_0]

    sg_0 = seg_inds_0[matched_inds_0]
    sg_1 = seg_inds_1[matched_inds_1]

    if sg_0.shape[0] > num_lines:
        rand_inds = np.random.choice(sg_0.shape[0], size=(num_lines), replace=False)
        sg_0 = sg_0[rand_inds]
        sg_1 = sg_1[rand_inds]

    output_line_filename = 'debug_dataloader_' + str(im_count) + '_lines.png'
    output_line_path = os.path.join(output_dir, output_line_filename)
    vis_lines(im_0[0], im_1[0], torch.from_numpy(sg_0), torch.from_numpy(sg_1), output_line_path)

    output_seg_filename = 'debug_dataloader_' + str(im_count) + '_seg.png'
    output_seg_path = os.path.join(output_dir, output_seg_filename)
    vis_segs(im_0[0], im_1[0], torch.from_numpy(seg_inds_0), torch.from_numpy(seg_inds_1), output_seg_path)

    im_count += 1
    if (im_count >= num_images):
        break