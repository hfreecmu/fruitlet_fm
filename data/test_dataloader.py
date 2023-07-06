import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet_2023/scripts/nbv/feature_association')

import os
import cv2
import numpy as np
import torch

from data.dataloader import get_data_loader
from utils.nbv_utils import vis

images_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/images_1400'
segmentations_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/scripts/nbv/feature_association/datasets/train'
output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_homography'
window_length = 16
num_points = 2000
batch_size = 1
shuffle = True
num_images = 5
num_lines = 5
dataloader = get_data_loader(images_dir, segmentations_dir,
                             window_length,
                             num_points,
                             batch_size, shuffle)

im_count = 0
for i, data in enumerate(dataloader):
    im_0, im_1, _, _, seg_inds_0, seg_inds_1, _, _, _, _, matches_0, matches_1 = data
    
    cv_im_0 = im_0[0].permute(1, 2, 0).numpy().copy()
    cv_im_1 = im_1[0].permute(1, 2, 0).numpy().copy()

    seg_inds_0 = seg_inds_0[0].numpy().copy()
    seg_inds_1 = seg_inds_1[0].numpy().copy()

    matches_0 = matches_0[0].numpy().copy()
    matches_1 = matches_1[0].numpy().copy()

    has_match_0 = (matches_0 != -1)
    has_match_1 = (matches_1 != -1)

    assert has_match_0.sum() == has_match_1.sum()

    matched_inds_0 = np.arange(matches_0.shape[0])[has_match_0]
    matched_inds_1 = matches_0[has_match_0]

    sg_0 = seg_inds_0[matched_inds_0]
    sg_1 = seg_inds_1[matched_inds_1]

    output_filename = 'debug_dataloader_' + str(im_count) + '.png'
    output_path = os.path.join(output_dir, output_filename)

    rand_inds = np.random.choice(sg_0.shape[0], size=(num_lines), replace=False)
    sg_0 = sg_0[rand_inds]
    sg_1 = sg_1[rand_inds]

    vis(im_0[0], im_1[0], torch.from_numpy(sg_0), torch.from_numpy(sg_1), output_path)

    im_count += 1
    if (im_count >= num_images):
        break