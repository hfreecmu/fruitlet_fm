import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch

from utils.nbv_utils import read_pickle, warp_points
from utils.torch_utils import load_torch_image

class FeatureDataset(Dataset):
    def __init__(self, images_dir, segmentations_dir, window_length, num_good,
                 num_bad, shuffle_inds=True,
                 rand_flip=True, affine_thresh=0.3, aug_orig_thresh=0.3):
        
        self.paths = self.get_paths(images_dir, segmentations_dir)
        self.window_length = window_length
        self.num_good = num_good
        self.num_bad = num_bad
        self.shuffle_inds = shuffle_inds

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
    
    def get_bad_samples(self, seg_inds, num_pad, num_bad_sample):
        x_min, y_min = np.min(seg_inds, axis=0)
        x_max, y_max = np.max(seg_inds, axis=0)

        x_sample = np.arange(x_min - num_pad, x_max + num_pad + 1)
        y_sample = np.arange(y_min - num_pad, y_max + num_pad + 1)
        xv, yv = np.meshgrid(x_sample, y_sample, indexing='ij')
        bad_sample = np.stack((xv.flatten(), yv.flatten()), axis=1)
        bad_sample_inds = np.invert(np.bitwise_and.reduce(np.in1d(bad_sample, seg_inds).reshape((-1, 2)), axis=1))
        bad_sample = bad_sample[bad_sample_inds]

        if bad_sample.shape[0] >= num_bad_sample:
            sample_inds = np.random.choice(bad_sample.shape[0], size=(num_bad_sample,), replace=False)
        else:
            sample_inds = np.random.choice(bad_sample.shape[0], size=(num_bad_sample,), replace=True)

        bad_sample = bad_sample[sample_inds]

        return bad_sample   
    
    def __getitem__(self, idx):
        while True:
            try:
                return self.get_aug_item(idx)
            except:
                pass

    def get_aug_item(self, idx):
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
                               (seg_inds_0[:, 0] + self.window_length < torch_im_0.shape[-1] - 1) & 
                               (seg_inds_0[:, 1] - self.window_length >= 0) & 
                               (seg_inds_0[:, 1] + self.window_length < torch_im_0.shape[-2] - 1))
        seg_inds_0 = seg_inds_0[good_inds_0]
        seg_inds_1 = seg_inds_1[good_inds_0]

        good_inds_1 = np.where((seg_inds_1[:, 0] - self.window_length >= 0) & 
                               (seg_inds_1[:, 0] + self.window_length < torch_im_1.shape[-1] - 1) & 
                               (seg_inds_1[:, 1] - self.window_length >= 0) & 
                               (seg_inds_1[:, 1] + self.window_length < torch_im_1.shape[-2] - 1))
        seg_inds_0 = seg_inds_0[good_inds_1]
        seg_inds_1 = seg_inds_1[good_inds_1]

        if not (seg_inds_0.shape[0] == seg_inds_1.shape[0]):
            raise RuntimeError("Something wrong, seg inds should be equal size")
        
        num_segs = self.num_good + self.num_bad
        if seg_inds_0.shape[0] > self.num_good:
            selected_inds = np.random.choice(seg_inds_0.shape[0], size=(self.num_good,), replace=False)
            seg_inds_0 = seg_inds_0[selected_inds]
            seg_inds_1 = seg_inds_1[selected_inds]

        num_bad_sample = num_segs - seg_inds_0.shape[0]
        num_pad = np.ceil(np.sqrt(num_bad_sample) / 2).astype(int)

        bad_sample_0 = self.get_bad_samples(seg_inds_0, num_pad, num_bad_sample)
        bad_sample_1 = self.get_bad_samples(seg_inds_1, num_pad, num_bad_sample)

        ###filter bad samples
        good_inds_0 = np.where((bad_sample_0[:, 0] - self.window_length >= 0) & 
                               (bad_sample_0[:, 0] + self.window_length < torch_im_0.shape[-1] - 1) & 
                               (bad_sample_0[:, 1] - self.window_length >= 0) & 
                               (bad_sample_0[:, 1] + self.window_length < torch_im_0.shape[-2] - 1))
        bad_sample_0 = bad_sample_0[good_inds_0]
        bad_sample_1 = bad_sample_1[good_inds_0]

        good_inds_1 = np.where((bad_sample_1[:, 0] - self.window_length >= 0) & 
                               (bad_sample_1[:, 0] + self.window_length < torch_im_1.shape[-1] - 1) & 
                               (bad_sample_1[:, 1] - self.window_length >= 0) & 
                               (bad_sample_1[:, 1] + self.window_length < torch_im_1.shape[-2] - 1))
        bad_sample_0 = bad_sample_0[good_inds_1]
        bad_sample_1 = bad_sample_1[good_inds_1]
        ###

        #matches_0 is the index in 1 that maps to that index in 0
        #matches_1 is the index in 0 that maps to that index 1
        matches_0 = np.concatenate((np.arange(seg_inds_0.shape[0]), np.zeros((bad_sample_0.shape[0]), dtype=int) - 1))
        matches_1 = np.concatenate((np.arange(seg_inds_1.shape[0]), np.zeros((bad_sample_1.shape[0]), dtype=int) - 1))
         
        seg_inds_0 = np.concatenate((seg_inds_0, bad_sample_0), axis=0)
        seg_inds_1 = np.concatenate((seg_inds_1, bad_sample_1), axis=0)

        ###if not enough add more
        if seg_inds_0.shape[0] < num_segs:
            num_to_add = num_segs - seg_inds_0.shape[0]
            new_inds = np.random.choice(seg_inds_0.shape[0], size=(num_to_add,), replace=True)

            seg_inds_0 = np.concatenate((seg_inds_0, seg_inds_0[new_inds]), axis=0)
            seg_inds_1 = np.concatenate((seg_inds_1, seg_inds_1[new_inds]), axis=0)
            matches_0 = np.concatenate((matches_0, matches_0[new_inds]), axis=0)
            matches_1 = np.concatenate((matches_1, matches_1[new_inds]), axis=0)
        ###

        if self.shuffle_inds:
            #shuffle one at a time            
            p_0 = np.random.permutation(seg_inds_0.shape[0])
            seg_inds_0 = seg_inds_0[p_0]
            matches_0 = matches_0[p_0]

            matched_inds = np.argwhere(matches_0 != -1)[:, 0]
            matches_1[matches_0[matched_inds]] = matched_inds

            #shuffle one at a time            
            p_1 = np.random.permutation(seg_inds_1.shape[0])
            seg_inds_1 = seg_inds_1[p_1]
            matches_1 = matches_1[p_1]

            matched_inds = np.argwhere(matches_1 != -1)[:, 0]
            matches_0[matches_1[matched_inds]] = matched_inds

        pair_inds_0 = matches_0 > -1
        pair_inds_1 = matches_1 > -1

        if not (matches_1[matches_0[pair_inds_0]] == np.arange(matches_0.shape[0])[pair_inds_0]).all():
            raise RuntimeError('Failed match check 0')
        
        if not (matches_0[matches_1[pair_inds_1]] == np.arange(matches_1.shape[0])[pair_inds_1]).all():
            raise RuntimeError('Failed match check 1')

        if not ((seg_inds_0.shape[0] == seg_inds_1.shape[0]) and (seg_inds_0.shape[0] == num_segs)):
            raise RuntimeError("Something wrong #2, seg inds should be equal size")

        window_size = 2*self.window_length
        bgrs_0 = torch.zeros((num_segs, 3, window_size, window_size), dtype=torch_im_0.dtype)
        bgrs_1 = torch.zeros((num_segs, 3, window_size, window_size), dtype=torch_im_1.dtype)
        for ind in range(num_segs):
            x_0, y_0 = seg_inds_0[ind]
            x_1, y_1 = seg_inds_1[ind]

            window_0 = torch_im_0[0, :, y_0-self.window_length:y_0+self.window_length, x_0-self.window_length:x_0+self.window_length]
            window_1 = torch_im_1[0, :, y_1-self.window_length:y_1+self.window_length, x_1-self.window_length:x_1+self.window_length]

            bgrs_0[ind] = window_0
            bgrs_1[ind] = window_1 

        return torch_im_0.squeeze(0), torch_im_1.squeeze(0), bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, matches_0, matches_1

def get_data_loader(images_dir, segmentations_dir, window_length,
                    num_good, num_bad, batch_size, shuffle):
    dataset = FeatureDataset(images_dir, segmentations_dir, window_length, num_good, num_bad)
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
