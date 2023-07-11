import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

from utils.nbv_utils import read_pickle, warp_points, drop_points
from utils.nbv_utils import select_valid_points, select_points
from utils.torch_utils import load_torch_image, load_feature_inputs
from utils.torch_utils import create_mask

def throwRuntimeError(msg):
    print(msg)
    raise RuntimeError(msg)

#TODO add bad points back in?
#TODO add centroid back in?
class FeatureDataset(Dataset):
    def __init__(self, images_dir, segmentations_dir, min_dim,
                 shuffle_inds=True, #whether to shuffle order of points
                 rand_flip=True, #whether to flip image
                 should_gauss_blur=False,
                 should_rand_bright=False,
                 erase_thresh=0.4,
                 affine_thresh=0.3, #percentage of using affine thresh
                 aug_orig_thresh=-1, #percentage of augmenting original image
                 aug_bright_orig_thresh=-1, #if not augment original image, whether or not to randomly brighten
                 other_match_thresh=0.5, #percentage of matching against two different fruitlets
                 other_attempts=5, #number of attempts to try this if there is a bug
                 cluster_match_thresh=0.8, #if above, percentage of matching with fruitlet from same cluster
                 swap_thresh=0.5, #whether to swap two images
                 ):
        
        self.paths = self.get_paths(images_dir, segmentations_dir, min_dim)
        self.shuffle_inds = shuffle_inds

        self.rand_flip = rand_flip
        self.should_gauss_blur = should_gauss_blur
        self.should_rand_bright = should_rand_bright
        self.erase_thresh = erase_thresh
        self.affine_thresh = affine_thresh
        self.aug_orig_thresh = aug_orig_thresh
        self.aug_bright_orig_thresh = aug_bright_orig_thresh
        self.other_match_thresh = other_match_thresh
        self.other_attempts = other_attempts
        self.cluster_match_thresh = cluster_match_thresh
        self.swap_thresh = swap_thresh
        
        self.random_affine = T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.75, 0.9))
        # self.random_brightness = T.ColorJitter(brightness=0.2,contrast=0.4,saturation=0.2,hue=0.1)
        self.random_brightness = T.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05)
        self.perspective_distortion_scale = 0.4
        self.gauss_blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))

        self.width = 1440
        self.height = 1080

    def get_paths(self, images_dir, segmentations_dir, min_dim):
        max_num_points = -1
        max_dim = -1
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
                seg_points = segmentations[seg_ind]
                
                min_y, min_x = seg_points.min(axis=0)
                max_y, max_x = seg_points.max(axis=0)

                height = max_y + 1 - min_y
                width = max_x + 1 - min_x

                if height < min_dim:
                    continue

                if width < min_dim:
                    continue

                if height > max_dim:
                    max_dim = height

                if width > max_dim:
                    max_dim = width

                num_points = seg_points.shape[0]
                if num_points > max_num_points:
                    max_num_points = num_points

                paths.append((im_path, seg_path, seg_ind))

        self.min_dim = min_dim
        self.max_dim = max_dim
        self.max_num_points = max_num_points
        return paths
    
    def __len__(self):
        return len(self.paths)

    def erase(self, seg_inds, inds):
        min_x, min_y = seg_inds.min(axis=0)
        max_x, max_y = seg_inds.max(axis=0)

        height = max_y + 1 - min_y
        width = max_x + 1 - min_x

        #these are areas to erase
        x0 = 0
        x1 = width
        y0 = 0
        y1 = height

        erase_type = np.random.randint(0, 8)

        #erase left half
        if erase_type == 0:
            x1 = width // 2
        #erase right half
        elif erase_type == 1:
            x0 = width // 2
        #erase top half
        elif erase_type == 2:
            y1 = height // 2
        #erase bottom half
        elif erase_type == 3:
            y0 = height // 2
        #erase top left
        elif erase_type == 4:
            x1 = width // 2
            y1 = height // 2
        #erase top right
        elif erase_type == 5:
            x0 = width // 2
            y1 = height // 2
        #bottom left
        elif erase_type == 6:
            x1 = width // 2
            y0 = height // 2
        #bottom right
        elif erase_type == 7:
            x0 = width // 2
            y0 = height // 2
        else:
            throwRuntimeError('Illegal value')
        
        x0 = x0 + min_x
        x1 = x1 + min_x

        y0 = y0 + min_y
        y1 = y1 + min_y
        
        erase_inds = ((seg_inds[:, 0] >= x0) & 
                             (seg_inds[:, 0] <= x1) & 
                             (seg_inds[:, 1] >= y0) & 
                             (seg_inds[:, 1] <= y1))
        
        return seg_inds[~erase_inds], inds[~erase_inds]
    
    def filter_inds(self, seg_inds, inds=None):
        good_inds = ((seg_inds[:, 0] >= 0) & 
                             (seg_inds[:, 0] < self.width) & 
                             (seg_inds[:, 1] >= 0) & 
                             (seg_inds[:, 1] < self.height))
        
        if inds is not None:
            return seg_inds[good_inds], inds[good_inds]
        else:
            return seg_inds[good_inds]

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

        if self.should_rand_bright:
            aug_torch_im = self.random_brightness(aug_torch_im)

        if self.should_gauss_blur:
            aug_torch_im = self.gauss_blur(aug_torch_im)

        return aug_torch_im, aug_seg_inds
    
    def validate_return(self, res):
        _, _, seg_inds_pad_0, seg_inds_pad_1, np_0, np_1, _, _, _ = res

        if seg_inds_pad_0[0:np_0].shape[0] == 0:
            throwRuntimeError('no seg inds 0')

        if seg_inds_pad_0[0:np_1].shape[0] == 0:
            throwRuntimeError('no seg inds 1')

        valid = True
        for seg_inds in [seg_inds_pad_0, seg_inds_pad_1]:
            if seg_inds[:, 0].max() < 0:
                valid = False
        
            if seg_inds[:, 1].max() < 0:
                valid = False

            if seg_inds[:, 0].min() >= self.width:
                valid = False

            if seg_inds[:, 1].min() >= self.height:
                valid = False

            width = seg_inds[:, 0].max() - seg_inds[:, 0].min() + 1
            height = seg_inds[:, 1].max() - seg_inds[:, 1].min() + 1

            if width < 20:
                valid = False

            if height < 20:
                valid = False

        if not valid:
            throwRuntimeError('Invalid res')

        return res

    def __getitem__(self, idx):
        use_other_match = np.random.random() < self.other_match_thresh
        if use_other_match:
            func = self.get_other_item
        else:
            func = self.get_aug_item

        num_attempts = 0
        while True:
            try:
                if num_attempts < self.other_attempts:
                    return self.validate_return(func(idx))
                else:
                    #revert to what we know
                    return self.validate_return(self.get_aug_item(idx))
            except Exception as e:
                print('Error in data loader: ', use_other_match)
                if hasattr(e, 'message'):
                    print(e.message)
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
            for i in range(5):
                rand_ind_idx = np.random.randint(0, len(rand_segmentations))
                if rand_ind_idx != seg_ind_locs_0:
                    break
            if rand_ind_idx == seg_ind_locs_0:
                throwRuntimeError('Could not find other segmentation in cluster match')
            img_path_locs_1, seg_path_locs_1, seg_ind_locs_1 = img_path_locs_0, seg_path_locs_0, rand_ind_idx
        else:
            #if not, read segmentation from different path
            rand_idx = idx
            for i in range(5):
                rand_idx = np.random.randint(0, len(self.paths))
                if rand_idx != idx:
                    break
            if rand_idx == idx:
                throwRuntimeError('Could not find rand segmentation')
            img_path_locs_1, seg_path_locs_1, seg_ind_locs_1 = self.paths[rand_idx]

        #load torch image and whether it was randomly flippped
        torch_im_0, _, flip_0 = load_torch_image(img_path_locs_0, rand_flip=self.rand_flip)

        #load second torch image
        #if cluster matching, want to also random flip other fruitlet in cluster
        if cluster_match:
            torch_im_1, _, flip_1 = load_torch_image(img_path_locs_1, rand_flip=self.rand_flip, force_flip=flip_0)
            if flip_0 != flip_1:
                throwRuntimeError('flip_0 and flip_1 must match use_self_match')
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
            if should_rand_bright_0 and self.should_rand_bright:
                torch_im_0 = self.random_brightness(torch_im_0)
        if should_augment_im_1:
            torch_im_1, seg_inds_1 = self.augment(torch_im_1, seg_inds_1)
        else:
            should_rand_bright_1 = np.random.random() < self.aug_bright_orig_thresh
            if should_rand_bright_1 and self.should_rand_bright:
                torch_im_1 = self.random_brightness(torch_im_1)

        #determine to swap inputs
        should_swap = np.random.random() < self.swap_thresh
        if should_swap:
            torch_im_0, torch_im_1 = torch_im_1, torch_im_0
            seg_inds_0, seg_inds_1 = seg_inds_1, seg_inds_0

        #we can now shuffle both points
        #because there is no 1:1 matching,
        #this is done differently than in get_aug_item
        if self.shuffle_inds:
            perm_0 = np.random.permutation(seg_inds_0.shape[0])
            seg_inds_0 = seg_inds_0[perm_0]

            perm_1 = np.random.permutation(seg_inds_1.shape[0])
            seg_inds_1 = seg_inds_1[perm_1]

        #filter bad inds
        seg_inds_0 = self.filter_inds(seg_inds_0)
        seg_inds_1 = self.filter_inds(seg_inds_1)

        #set matches to -1
        matches_0 = np.zeros((seg_inds_0.shape[0]), dtype=int) - 1
        matches_1 = np.zeros((seg_inds_1.shape[0]), dtype=int) - 1
        
        ###pad
        num_points_0 = seg_inds_0.shape[0]
        num_points_1 = seg_inds_1.shape[0]

        seg_inds_pad_0 = np.zeros((self.max_num_points, 2), dtype=seg_inds_0.dtype)
        seg_inds_pad_1 = np.zeros((self.max_num_points, 2), dtype=seg_inds_1.dtype)

        seg_inds_pad_0[0:num_points_0] = seg_inds_0
        seg_inds_pad_1[0:num_points_1] = seg_inds_1
        
        matches_pad_0 = np.zeros((self.max_num_points,), dtype=matches_0.dtype) - 1
        matches_pad_1 = np.zeros((self.max_num_points,), dtype=matches_1.dtype) - 1

        matches_pad_0[0:num_points_0] = matches_0
        matches_pad_1[0:num_points_1] = matches_1
        ###

        #return
        return torch_im_0.squeeze(0), torch_im_1.squeeze(0), seg_inds_pad_0, seg_inds_pad_1, num_points_0, num_points_1, matches_pad_0, matches_pad_1, False
   

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
            if should_rand_bright and self.should_rand_bright:
                torch_im_0 = self.random_brightness(torch_im)
            else:
                torch_im_0 = torch_im
            seg_inds_0 = seg_inds

        #determine to swap inputs
        should_swap = np.random.random() < self.swap_thresh
        if should_swap:
            torch_im_0, torch_im_1 = torch_im_1, torch_im_0
            seg_inds_0, seg_inds_1 = seg_inds_1, seg_inds_0

        #use these for matching
        inds_0 = np.arange(seg_inds_0.shape[0])
        inds_1 = np.arange(seg_inds_0.shape[0])

        #shuffle one now
        if self.shuffle_inds:
            perm_0 = np.random.permutation(seg_inds_0.shape[0])
            seg_inds_0 = seg_inds_0[perm_0]
            inds_0 = inds_0[perm_0]

            perm_1 = np.random.permutation(seg_inds_1.shape[0])
            seg_inds_1 = seg_inds_1[perm_1]
            inds_1 = inds_1[perm_1]

        #filter bad inds
        seg_inds_0, inds_0 = self.filter_inds(seg_inds_0, inds_0)
        seg_inds_1, inds_1 = self.filter_inds(seg_inds_1, inds_1)

        #erase
        should_erase_0 = np.random.random() < self.erase_thresh
        should_erase_1 = np.random.random() < self.erase_thresh
        if should_erase_0:
            seg_inds_0, inds_0 = self.erase(seg_inds_0, inds_0)
        if should_erase_1:
            seg_inds_1, inds_1 = self.erase(seg_inds_1, inds_1)

        #and do our matches
        matches_0 = np.zeros((seg_inds_0.shape[0]), dtype=int) - 1
        matches_1 = np.zeros((seg_inds_1.shape[0]), dtype=int) - 1

        dual_matches = np.where(inds_1.reshape(inds_1.size, 1) == inds_0)

        matches_0[dual_matches[1]] = dual_matches[0]
        matches_1[dual_matches[0]] = dual_matches[1]

        ###pad
        num_points_0 = seg_inds_0.shape[0]
        num_points_1 = seg_inds_1.shape[0]

        seg_inds_pad_0 = np.zeros((self.max_num_points, 2), dtype=seg_inds_0.dtype)
        seg_inds_pad_1 = np.zeros((self.max_num_points, 2), dtype=seg_inds_1.dtype)

        seg_inds_pad_0[0:num_points_0] = seg_inds_0
        seg_inds_pad_1[0:num_points_1] = seg_inds_1
        
        matches_pad_0 = np.zeros((self.max_num_points,), dtype=matches_0.dtype) - 1
        matches_pad_1 = np.zeros((self.max_num_points,), dtype=matches_1.dtype) - 1

        matches_pad_0[0:num_points_0] = matches_0
        matches_pad_1[0:num_points_1] = matches_1
        ###

        #return
        return torch_im_0.squeeze(0), torch_im_1.squeeze(0), seg_inds_pad_0, seg_inds_pad_1, num_points_0, num_points_1, matches_pad_0, matches_pad_1, True
   
def get_data_loader(images_dir, segmentations_dir,
                    min_dim,
                    batch_size, shuffle):
    dataset = FeatureDataset(images_dir, segmentations_dir, min_dim)
    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dloader
