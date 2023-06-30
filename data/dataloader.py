import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch

from utils.nbv_utils import read_pickle, warp_points
from utils.torch_utils import load_torch_image

class FeatureDataset(Dataset):
    def __init__(self, images_dir, segmentations_dir, window_length, max_segs,
                 rand_flip=True, affine_thresh=0.3, aug_orig_thresh=0.3):
        self.paths = self.get_paths(images_dir, segmentations_dir)
        self.window_length = window_length
        self.max_segs = max_segs

        self.rand_flip = rand_flip
        self.affine_thresh = affine_thresh
        self.aug_orig_thresh = aug_orig_thresh

        self.random_affine = T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.75, 0.9))
        self.random_brightness = T.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))
        self.perspective_distortion_scale = 0.6

    def get_paths(self, images_dir, segmentations_dir):
        paths = []
        for fileneame in os.listdir(segmentations_dir):
            if not fileneame.endswith('.pkl'):
                continue

            seg_path = os.path.join(segmentations_dir, fileneame)
            im_path = os.path.join(images_dir, fileneame.replace('.pkl', '.png'))

            if not os.path.exists(im_path):
                continue

            segmentations = read_pickle(seg_path)

            for seg_ind in range(len(segmentations)):
                paths.append((im_path, seg_path, seg_ind))

        return paths
    
    def __len__(self):
        return len(self.paths)
    
    def augment_affine(self, torch_im, seg_inds):
        angle, translations, scale, shear = T.RandomAffine.get_params(self.random_affine.degrees, 
                                                                      self.random_affine.translate,
                                                                      self.random_affine.scale,
                                                                      self.random_affine.shear,
                                                                      (torch_im.shape[-1], torch_im.shape[-2]))
    
        center = [torch_im.shape[-1] * 0.5, torch_im.shape[-2] * 0.5]
        translations = list(translations)
        shear = list(shear)
        M = F._get_inverse_affine_matrix(center, angle, translations, scale, shear)
        M = np.array([[M[0], M[1], M[2]],
                      [M[3], M[4], M[5]],
                      [0, 0, 1.0]])
        
        #have to invert not sure why torch does this
        M = np.linalg.inv(M)

        affine_seg_inds = warp_points(seg_inds, M)
        affine_seg_inds = np.round(affine_seg_inds).astype(int)

        torch_affine_img = F.affine(torch_im, angle, translations, scale, shear)

        #alternatively:

        # width = torch_im.shape[-1]
        # height = torch_im.shape[-2]

        # start_points_a = np.array([[0, 0],
        #                            [width - 1, 0],
        #                            [width - 1, height - 1],
        #                            [0, height - 1]])

        #end_points_a = warp_points(np.array(start_points_a), M)
        #end_points_a = np.round(end_points_a).astype(int)

        # M = F._get_perspective_coeffs(end_points_a.tolist(), start_points_a.tolist())
        # M = np.array([[M[0], M[1], M[2]],
        #               [M[3], M[4], M[5]],
        #               [M[6], M[7], 1.0]])

        #torch_affine_img = F.perspective(torch_im, start_points_a.tolist(), 
        #                                 end_points_a.tolist())

        return torch_affine_img, affine_seg_inds
    
    def augment_perspective(self, torch_im, seg_inds):
        start_points, end_points = T.RandomPerspective.get_params(torch_im.shape[-1], 
                                                                  torch_im.shape[-2], 
                                                                  self.perspective_distortion_scale)

        #not sure why torch does opposite direction when documentation
        #says otherwise but it does
        H = F._get_perspective_coeffs(end_points, start_points)

        H = np.array([[H[0], H[1], H[2]],
                      [H[3], H[4], H[5]],
                      [H[6], H[7], 1.0]])
        
        perspective_seg_inds = warp_points(seg_inds, H)
        perspective_seg_inds = np.round(perspective_seg_inds).astype(int)

        torch_perspective_img = F.perspective(torch_im, start_points, 
                                              end_points)
        
        return torch_perspective_img, perspective_seg_inds

    def augment(self, torch_im, seg_inds, affine_thresh):
        affine = np.random.random() < affine_thresh

        if affine:
            aug_torch_im, aug_seg_inds = self.augment_affine(torch_im, seg_inds)
        else:
            aug_torch_im, aug_seg_inds = self.augment_perspective(torch_im, seg_inds)

        aug_torch_im = self.random_brightness(aug_torch_im)

        return aug_torch_im, aug_seg_inds
    
    def __getitem__(self, idx):
        img_path_locs, seg_path_locs, seg_ind_locs = self.paths[idx]

        torch_im, _, flip = load_torch_image(img_path_locs, rand_flip=self.rand_flip)

        segmentations = read_pickle(seg_path_locs)
        seg_inds = segmentations[seg_ind_locs]
        seg_inds = np.stack((seg_inds[1], seg_inds[0]), axis=1)
        if flip:
            seg_inds[:, 0] = torch_im.shape[-1] - seg_inds[:, 0]

        torch_im_1, seg_inds_1 = self.augment(torch_im, seg_inds, self.affine_thresh)

        augment_im_0 = np.random.random() < self.aug_orig_thresh
        if augment_im_0:
            torch_im_0, seg_inds_0 = self.augment(torch_im, seg_inds, self.affine_thresh)
        else:
            should_rand_bright = np.random.random() < 0.5
            if should_rand_bright:
                torch_im_0 = self.random_brightness(torch_im)
            else:
                torch_im_0 = torch_im
            seg_inds_0 = seg_inds

        good_inds_0 = np.where((seg_inds_0[:, 0] - self.window_length >= 0) & 
                               (seg_inds_0[:, 0] + self.window_length < torch_im_0.shape[-1]) & 
                               (seg_inds_0[:, 1] - self.window_length >= 0) & 
                               (seg_inds_0[:, 1] + self.window_length < torch_im_0.shape[-2]))
        seg_inds_0 = seg_inds_0[good_inds_0]
        seg_inds_1 = seg_inds_1[good_inds_0]

        good_inds_1 = np.where((seg_inds_1[:, 0] - self.window_length >= 0) & 
                               (seg_inds_1[:, 0] + self.window_length < torch_im_1.shape[-1]) & 
                               (seg_inds_1[:, 1] - self.window_length >= 0) & 
                               (seg_inds_1[:, 1] + self.window_length < torch_im_1.shape[-2]))
        seg_inds_0 = seg_inds_0[good_inds_1]
        seg_inds_1 = seg_inds_1[good_inds_1]

        if not (seg_inds_0.shape[0] == seg_inds_1.shape[0]):
            raise RuntimeError("Something wrong, seg inds should be equal size")
        
        is_val = np.zeros((self.max_segs), dtype=int)
        num_seg_inds_orig = np.min((seg_inds_0.shape[0], self.max_segs))
        if (seg_inds_0.shape[0] > self.max_segs):
            selected_inds = np.random.choice(seg_inds_0.shape[0], size=(self.max_segs,), replace=False)
            seg_inds_0 = seg_inds_0[selected_inds]
            seg_inds_1 = seg_inds_1[selected_inds]
            is_val[:] = 1.0
        else:
            full_seg_inds_0 = np.zeros((self.max_segs, 2), dtype=seg_inds_0.dtype)
            full_seg_inds_1 = np.zeros((self.max_segs, 2), dtype=seg_inds_1.dtype)

            full_seg_inds_0[0:seg_inds_0.shape[0]] = seg_inds_0
            full_seg_inds_1[0:seg_inds_1.shape[0]] = seg_inds_1

            is_val[0:seg_inds_0.shape[0]] = 1.0

            seg_inds_0 = full_seg_inds_0
            seg_inds_1 = full_seg_inds_1

        #torch_im_0 = torch_im_0.float() / 255
        #torch_im_1 = torch_im_1.float() / 255

        window_size = 2*self.window_length + 1
        bgrs_0 = torch.zeros((self.max_segs, 3, window_size, window_size), dtype=torch_im_0.dtype)
        bgrs_1 = torch.zeros((self.max_segs, 3, window_size, window_size), dtype=torch_im_1.dtype)
        for ind in range(num_seg_inds_orig):
            x_0, y_0 = seg_inds_0[ind]
            x_1, y_1 = seg_inds_1[ind]

            window_0 = torch_im_0[0, :, y_0-self.window_length:y_0+self.window_length+1, x_0-self.window_length:x_0+self.window_length+1]
            window_1 = torch_im_1[0, :, y_1-self.window_length:y_1+self.window_length+1, x_1-self.window_length:x_1+self.window_length+1]

            bgrs_0[ind] = window_0
            bgrs_1[ind] = window_1

        return torch_im_0.squeeze(0), torch_im_1.squeeze(0), bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, is_val

def get_data_loader(images_dir, segmentations_dir, window_length,
                    max_segs, batch_size, shuffle):
    dataset = FeatureDataset(images_dir, segmentations_dir, window_length, max_segs)
    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dloader

# images_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_bag/raw_images/left'
# seg_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_bag/segmentations'
# dataloader = get_data_loader(images_dir, seg_dir, 5000, 1, True)

# data_size = len(dataloader)
# output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_homography'
# import cv2
# for i, data in enumerate(dataloader):
#     im_0, im_1, seg_inds_0, seg_inds_1, is_val = data
    
#     cv_im_0 = im_0[0].permute(1, 2, 0).numpy().copy()
#     cv_im_1 = im_1[0].permute(1, 2, 0).numpy().copy()

#     valid_inds = is_val[0].numpy().copy()

#     sg_0 = seg_inds_0[0, valid_inds > 0]
#     sg_1 = seg_inds_1[0, valid_inds > 0]

#     cv_im_0[sg_0[:, 1], sg_0[:, 0]] = [255, 0, 0]
#     cv_im_1[sg_1[:, 1], sg_1[:, 0]] = [255, 0, 0]

#     cv2.imwrite(os.path.join(output_dir, str(i) + '_0.png'), cv_im_0)
#     cv2.imwrite(os.path.join(output_dir, str(i) + '_1.png'), cv_im_1)

#     if (i >= 3):
#         break
