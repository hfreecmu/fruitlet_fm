import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch

from utils.nbv_utils import read_pickle, warp_points, drop_points
from utils.nbv_utils import select_valid_points, select_points
from utils.torch_utils import load_torch_image, load_feature_inputs
from utils.torch_utils import create_mask

#TODO add bad points back in?
#TODO add centroid back in?
class FeatureDataset(Dataset):
    def __init__(self, images_dir, segmentations_dir, window_length,
                 num_points, #max number of transformer points (if smaller will be masked)
                 shuffle_inds=True, #whether to shuffle order of points
                 rand_flip=True, #whether to flip image
                 drop_point_thresh=0.1, #percentage to drop random point from both masks 
                 affine_thresh=0.3, #percentage of using affine thresh
                 aug_orig_thresh=0.3, #percentage of augmenting original image
                 aug_bright_orig_thresh=0.4, #if not augment original image, whether or not to randomly brighten
                 other_match_thresh=0.2, #percentage of matching against two different fruitlets
                 other_attempts=5, #number of attempts to try this if there is a bug
                 cluster_match_thresh=0.5, #if above, percentage of matching with fruitlet from same cluster
                 swap_thresh=0.5, #whether to swap two images
                 ):
        
        self.paths = self.get_paths(images_dir, segmentations_dir)
        self.window_length = window_length
        self.num_points = num_points
        self.shuffle_inds = shuffle_inds

        self.rand_flip = rand_flip
        self.drop_point_thresh = drop_point_thresh
        self.affine_thresh = affine_thresh
        self.aug_orig_thresh = aug_orig_thresh
        self.aug_bright_orig_thresh = aug_bright_orig_thresh
        self.other_match_thresh = other_match_thresh
        self.other_attempts = other_attempts
        self.cluster_match_thresh = cluster_match_thresh
        self.swap_thresh = swap_thresh

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

    def augment(self, torch_im, seg_inds):
        affine = np.random.random() < self.affine_thresh

        if affine:
            aug_torch_im, aug_seg_inds = self.augment_affine(torch_im, seg_inds)
        else:
            aug_torch_im, aug_seg_inds = self.augment_perspective(torch_im, seg_inds)

        aug_torch_im = self.random_brightness(aug_torch_im)

        return aug_torch_im, aug_seg_inds
    
    def __getitem__(self, idx):
        use_other_match = np.random.random() < self.other_match_thresh
        if False and use_other_match:
            func = self.get_other_item
        else:
            func = self.get_aug_item

        num_attempts = 0
        while True:
            try:
                if num_attempts < self.other_attempts:
                    return func(idx)
                else:
                    #revert to what we know
                    return self.get_aug_item(idx)
            except Exception as e:
                print('Error in data loader')
                num_attempts += 1
                if ((num_attempts == self.other_attempts) and (use_other_match)):
                    print('Num attempts exceeded other attempts')
    
    def get_other_item(self, idx):
        img_path_locs_0, seg_path_locs_0, seg_ind_locs_0 = self.paths[idx]

        #determine if use non-matche fruitlet from same cluster or not
        cluster_match = np.random.random() < self.cluster_match_thresh
        if cluster_match:
            #if yes, read segmentation from same path but different idx
            rand_segmentations = read_pickle(seg_path_locs_0)
            rand_ind_idx = seg_ind_locs_0
            while rand_ind_idx == seg_ind_locs_0:
                rand_ind_idx = np.random.randint(0, len(rand_segmentations))
            img_path_locs_1, seg_path_locs_1, seg_ind_locs_1 = img_path_locs_0, seg_path_locs_0, rand_ind_idx
        else:
            #if not, read segmentation from different path
            rand_idx = idx
            while rand_idx == idx:
                rand_idx = np.random.randint(0, len(self.paths))
            img_path_locs_1, seg_path_locs_1, seg_ind_locs_1 = self.paths[rand_idx]

        #load torch image and whether it was randomly flippped
        torch_im_0, _, flip_0 = load_torch_image(img_path_locs_0, rand_flip=self.rand_flip)

        #load second torch image
        #if cluster matching, want to also random flip other fruitlet in cluster
        if cluster_match:
            torch_im_1, _, flip_1 = load_torch_image(img_path_locs_1, rand_flip=self.rand_flip, force_flip=flip_0)
            if flip_0 != flip_1:
                raise RuntimeError('flip_0 and flip_1 must match use_self_match')
        else:
            torch_im_1, _, flip_1 = load_torch_image(img_path_locs_1, rand_flip=self.rand_flip)

        #get segmnetations and flip ir originals where flipped
        segmentations_0 = read_pickle(seg_path_locs_0)
        segmentations_1 = read_pickle(seg_path_locs_1)
        seg_inds_0 = segmentations_0[seg_ind_locs_0]
        seg_inds_1 = segmentations_1[seg_ind_locs_1]
        seg_inds_0 = np.stack((seg_inds_0[:, 1], seg_inds_0[:, 0]), axis=1)
        seg_inds_1 = np.stack((seg_inds_1[:, 1], seg_inds_1[:, 0]), axis=1)
        if flip_0:
            seg_inds_0[:, 0] = torch_im_0.shape[-1] - seg_inds_0[:, 0]
        if flip_1:
            seg_inds_1[:, 0] = torch_im_1.shape[-1] - seg_inds_1[:, 0]

        #determine whether we should augment images
        #this will use aug_orig_thresh because we may not want to 
        #augment both all the time
        #if not augment, still add random brightness
        should_augment_im_0 = np.random.random() < self.aug_orig_thresh
        should_augment_im_1 = np.random.random() < self.aug_orig_thresh
        if should_augment_im_0:
            torch_im_0, seg_inds_0 = self.augment(torch_im_0, seg_inds_0)
        else:
            should_rand_bright_0 = np.random.random() < self.aug_bright_orig_thresh
            if should_rand_bright_0:
                torch_im_0 = self.random_brightness(torch_im_0)
        if should_augment_im_1:
            torch_im_1, seg_inds_1 = self.augment(torch_im_1, seg_inds_1)
        else:
            should_rand_bright_1 = np.random.random() < self.aug_bright_orig_thresh
            if should_rand_bright_1:
                torch_im_1 = self.random_brightness(torch_im_1)

        #determine to swap inputs
        should_swap = np.random.random() < self.swap_thresh
        if should_swap:
            torch_im_0, torch_im_1 = torch_im_1, torch_im_0
            seg_inds_0, seg_inds_1 = seg_inds_1, seg_inds_0

        #get bgr features and which seg_inds to use
        bgrs_0, used_seg_inds_0 = load_feature_inputs(torch_im_0, seg_inds_0, self.window_length) 
        bgrs_1, used_seg_inds_1 = load_feature_inputs(torch_im_1, seg_inds_1, self.window_length) 

        #randomly drop points independantly
        drop_points(used_seg_inds_0, self.drop_point_thresh)
        drop_points(used_seg_inds_1, self.drop_point_thresh)

        #if more than num_points need to drop
        #output of this does not mean array is length num_points,
        #only the sum is less than
        used_seg_inds_0 = select_valid_points(used_seg_inds_0, self.num_points)
        used_seg_inds_1 = select_valid_points(used_seg_inds_1, self.num_points)

        #we can now shuffle both points
        #because there is no 1:1 matching,
        #this is done differently than in get_aug_item
        if self.shuffle_inds:
            perm_0 = np.random.permutation(seg_inds_0.shape[0])
            seg_inds_0 = seg_inds_0[perm_0]
            bgrs_0 = bgrs_0[perm_0]
            used_seg_inds_0 = used_seg_inds_0[perm_0]

            perm_1 = np.random.permutation(seg_inds_1.shape[0])
            seg_inds_1 = seg_inds_1[perm_1]
            bgrs_1 = bgrs_1[perm_1]
            used_seg_inds_1 = used_seg_inds_1[perm_1]

        #select our segmentation indices and our matches
        #used_seg_inds_0 and used_seg_inds_1 are done after this
        seg_inds_0, bgrs_0 = select_points(used_seg_inds_0, seg_inds_0, bgrs_0)
        seg_inds_1, bgrs_1 = select_points(used_seg_inds_1, seg_inds_1, bgrs_1)

        #set matches to -1
        matches_0 = np.zeros((seg_inds_0.shape[0]), dtype=int) - 1
        matches_1 = np.zeros((seg_inds_1.shape[0]), dtype=int) - 1

        if seg_inds_0.shape[0] > self.num_points:
            raise RuntimeError('seg_inds_0 too large, something is wrong')

        if seg_inds_1.shape[0] > self.num_points:
            raise RuntimeError('seg_inds_1 too large, something is wrong')
        
        #now we can add masked points
        seg_inds_0, bgrs_0, matches_0, is_mask_0, num_mask_0 = create_mask(seg_inds_0, bgrs_0, matches_0, self.num_points)
        seg_inds_1, bgrs_1, matches_1, is_mask_1, num_mask_1 = create_mask(seg_inds_1, bgrs_1, matches_1, self.num_points)

        #return
        return torch_im_0.squeeze(0), torch_im_1.squeeze(0), bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, is_mask_0, is_mask_1, num_mask_0, num_mask_1, matches_0, matches_1

    def get_aug_item(self, idx):
        img_path_locs, seg_path_locs, seg_ind_locs = self.paths[idx]

        #load torch image and whether it was randomly flipped
        torch_im, _, flip = load_torch_image(img_path_locs, rand_flip=self.rand_flip)

        #get segmentations and flip if original image was fipped
        segmentations = read_pickle(seg_path_locs)
        seg_inds = segmentations[seg_ind_locs]
        seg_inds = np.stack((seg_inds[:, 1], seg_inds[:, 0]), axis=1)
        if flip:
            seg_inds[:, 0] = torch_im.shape[-1] - seg_inds[:, 0]

        #get augmented with perspective or affine transform and random brightntess
        torch_im_1, seg_inds_1 = self.augment(torch_im, seg_inds)

        #determine whether or not to augment original image
        should_augment_im_0 = np.random.random() < self.aug_orig_thresh
        if should_augment_im_0:
            #augment same way
            torch_im_0, seg_inds_0 = self.augment(torch_im, seg_inds)
        else:
            should_rand_bright = np.random.random() < self.aug_bright_orig_thresh
            if should_rand_bright:
                torch_im_0 = self.random_brightness(torch_im)
            else:
                torch_im_0 = torch_im
            seg_inds_0 = seg_inds

        #determine to swap inputs
        should_swap = np.random.random() < self.swap_thresh
        if should_swap:
            torch_im_0, torch_im_1 = torch_im_1, torch_im_0
            seg_inds_0, seg_inds_1 = seg_inds_1, seg_inds_0

        #get bgr features and which seg_inds to use
        bgrs_0, used_seg_inds_0 = load_feature_inputs(torch_im_0, seg_inds_0, self.window_length) 
        bgrs_1, used_seg_inds_1 = load_feature_inputs(torch_im_1, seg_inds_1, self.window_length) 
        
        #randomly drop points independantly
        drop_points(used_seg_inds_0, self.drop_point_thresh)
        drop_points(used_seg_inds_1, self.drop_point_thresh)

        #if more than num_points need to drop
        #output of this does not mean array is length num_points,
        #only the sum is less than
        used_seg_inds_0 = select_valid_points(used_seg_inds_0, self.num_points)
        used_seg_inds_1 = select_valid_points(used_seg_inds_1, self.num_points)

        #shuffle both
        if self.shuffle_inds:
            perm = np.random.permutation(seg_inds_0.shape[0])
            
            seg_inds_0 = seg_inds_0[perm]
            bgrs_0 = bgrs_0[perm]
            used_seg_inds_0 = used_seg_inds_0[perm]

            seg_inds_1 = seg_inds_1[perm]
            bgrs_1 = bgrs_1[perm]
            used_seg_inds_1 = used_seg_inds_1[perm]

        #select our segmentation indices and our matches
        #used_seg_inds_0 and used_seg_inds_1 are done after this
        seg_inds_0, bgrs_0 = select_points(used_seg_inds_0, seg_inds_0, bgrs_0)
        seg_inds_1, bgrs_1 = select_points(used_seg_inds_1, seg_inds_1, bgrs_1)

        #use these for matching
        inds_0 = np.arange(used_seg_inds_0.shape[0])[used_seg_inds_0]
        inds_1 = np.arange(used_seg_inds_1.shape[0])[used_seg_inds_1]

        #shuffle one now
        #don't have to do other because we shuffled both before
        if self.shuffle_inds:
            perm = np.random.permutation(seg_inds_0.shape[0])
            seg_inds_0 = seg_inds_0[perm]
            bgrs_0 = bgrs_0[perm]
            inds_0 = inds_0[perm]

        #and do our matches
        matches_0 = np.zeros((seg_inds_0.shape[0]), dtype=int) - 1
        matches_1 = np.zeros((seg_inds_1.shape[0]), dtype=int) - 1

        dual_matches = np.where(inds_1.reshape(inds_1.size, 1) == inds_0)

        matches_0[dual_matches[1]] = dual_matches[0]
        matches_1[dual_matches[0]] = dual_matches[1]

        if seg_inds_0.shape[0] > self.num_points:
            raise RuntimeError('seg_inds_0 too large, something is wrong')

        if seg_inds_1.shape[0] > self.num_points:
            raise RuntimeError('seg_inds_1 too large, something is wrong')
        
        #now we can add masked points
        seg_inds_0, bgrs_0, matches_0, is_mask_0, num_mask_0 = create_mask(seg_inds_0, bgrs_0, matches_0, self.num_points)
        seg_inds_1, bgrs_1, matches_1, is_mask_1, num_mask_1 = create_mask(seg_inds_1, bgrs_1, matches_1, self.num_points)

        #return
        return torch_im_0.squeeze(0), torch_im_1.squeeze(0), bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, is_mask_0, is_mask_1, num_mask_0, num_mask_1, matches_0, matches_1
   
def get_data_loader(images_dir, segmentations_dir, window_length,
                    num_points, batch_size, shuffle):
    dataset = FeatureDataset(images_dir, segmentations_dir, window_length, num_points)
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
