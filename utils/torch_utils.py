import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

#img is returned unsqueezed
def load_torch_image(imfile, rand_flip, force_flip=None):
    im = cv2.imread(imfile)
    flip=False
    if rand_flip:
        if force_flip is None:
            hflip = np.random.random() < 0.5
        else:
            hflip = force_flip
        if hflip:
            im = cv2.flip(im, 2)
            flip=True

    img = torch.from_numpy(im).permute(2, 0, 1)
    return img[None], im, flip

def save_checkpoint(epoch, checkpoint_dir, transformer_model):
    transformer_path = os.path.join(checkpoint_dir, 'epoch_%d_transformer.pth' % epoch)
    torch.save(transformer_model.state_dict(), transformer_path)

def load_checkpoint(epoch, checkpoint_dir, transformer_model):
    transformer_path = os.path.join(checkpoint_dir, 'epoch_%d_transformer.pth' % epoch)
    transformer_model.load_state_dict(torch.load(transformer_path))

def plot_loss(loss_array, checkpoint_dir):
    loss_plot_path = os.path.join(checkpoint_dir, 'loss.png')
    loss_np_path = os.path.join(checkpoint_dir, 'loss.npy')

    loss_array = np.array(loss_array)

    np.save(loss_np_path, loss_array)

    plt.plot(loss_array[:, 0], loss_array[:, 1], 'b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(loss_plot_path)

    plt.clf()

def load_feature_inputs(torch_im, seg_inds, window_length):
    valid_window_inds = np.where((seg_inds[:, 0] - window_length >= 0) & 
                                 (seg_inds[:, 0] + window_length <= torch_im.shape[-1] - 1) & 
                                 (seg_inds[:, 1] - window_length >= 0) & 
                                 (seg_inds[:, 1] + window_length <= torch_im.shape[-2] - 1))
    
    used_seg_inds = np.zeros(seg_inds.shape[0], dtype=bool)
    used_seg_inds[valid_window_inds] = True

    window_size = 2*window_length
    bgrs = torch.zeros((seg_inds.shape[0], 3, window_size, window_size), dtype=torch_im.dtype)
    for ind in range(seg_inds.shape[0]):
        if not used_seg_inds[ind]:
            continue

        x, y = seg_inds[ind]
        window = torch_im[0, :, y-window_length:y+window_length, x-window_length:x+window_length]
        bgrs[ind] = window

    #seg_inds is numpy, bgrs is torch, used_seg_inds is numpy
    return bgrs, used_seg_inds

def create_mask(seg_inds, bgr_vals, matches, num_pts):
    if seg_inds.shape[0] > num_pts:
        raise RuntimeError('Cannot create mask with size greater than num_pts')

    num_to_add = num_pts - seg_inds.shape[0]

    is_mask = np.zeros((num_pts), dtype=bool)
    if (num_to_add == 0):
        return seg_inds, bgr_vals, matches, is_mask, num_to_add

    seg_inds = np.concatenate((seg_inds, np.zeros((num_to_add, 2), dtype=int)), axis=0)
    bgr_vals = torch.concatenate((bgr_vals, 
                                  torch.zeros((num_to_add, *bgr_vals.shape[1:]), dtype=bgr_vals.dtype)),
                                  axis=0)
    matches = np.concatenate((matches, np.zeros((num_to_add), dtype=int) - 1))
    
    is_mask[-num_to_add:] = True

    return seg_inds, bgr_vals, matches, is_mask, num_to_add

#taken out, but keeping for reference
# def get_bad_samples(seg_inds, num_pad, num_bad_sample):
#     x_min, y_min = np.min(seg_inds, axis=0)
#     x_max, y_max = np.max(seg_inds, axis=0)

#     x_sample = np.arange(x_min - num_pad, x_max + num_pad + 1)
#     y_sample = np.arange(y_min - num_pad, y_max + num_pad + 1)
#     xv, yv = np.meshgrid(x_sample, y_sample, indexing='ij')
#     bad_sample = np.stack((xv.flatten(), yv.flatten()), axis=1)
#     bad_sample_inds = np.invert(np.bitwise_and.reduce(np.in1d(bad_sample, seg_inds).reshape((-1, 2)), axis=1))
#     bad_sample = bad_sample[bad_sample_inds]

#     if bad_sample.shape[0] >= num_bad_sample:
#         sample_inds = np.random.choice(bad_sample.shape[0], size=(num_bad_sample,), replace=False)
#     else:
#         sample_inds = np.random.choice(bad_sample.shape[0], size=(num_bad_sample,), replace=True)

#     bad_sample = bad_sample[sample_inds]

#     return bad_sample 

#old but saving for reference
# def load_feature_inputs(torch_im, seg_inds, window_length, num_good, num_bad, use_centroid):
#     good_inds = np.where((seg_inds[:, 0] - window_length >= 0) & 
#                             (seg_inds[:, 0] + window_length < torch_im.shape[-1] - 1) & 
#                             (seg_inds[:, 1] - window_length >= 0) & 
#                             (seg_inds[:, 1] + window_length < torch_im.shape[-2] - 1))
#     seg_inds = seg_inds[good_inds]

#     num_segs = num_good + num_bad
#     if seg_inds.shape[0] > num_good:
#         if not use_centroid:
#             selected_inds = np.random.choice(seg_inds.shape[0], size=(num_good,), replace=False)
#             seg_inds = seg_inds[selected_inds]
#         else:
#             med_point = np.median(seg_inds, axis=0)
#             dists = np.linalg.norm(seg_inds - med_point, axis=1)
#             min_dists_inds = np.argsort(dists)
#             seg_inds = seg_inds[min_dists_inds[0:num_good]]

#     num_bad_sample = num_segs - seg_inds.shape[0]
#     num_pad = np.ceil(np.sqrt(num_bad_sample) / 2).astype(int)

#     bad_sample = get_bad_samples(seg_inds, num_pad, num_bad_sample)

#     ###filter bad samples
#     good_inds = np.where((bad_sample[:, 0] - window_length >= 0) & 
#                          (bad_sample[:, 0] + window_length < torch_im.shape[-1] - 1) & 
#                          (bad_sample[:, 1] - window_length >= 0) & 
#                          (bad_sample[:, 1] + window_length < torch_im.shape[-2] - 1))
#     bad_sample = bad_sample[good_inds]

#     is_val = np.concatenate((np.zeros((seg_inds.shape[0]), dtype=int) + 1, np.zeros((bad_sample.shape[0]), dtype=int)))
#     seg_inds = np.concatenate((seg_inds, bad_sample), axis=0)

#     if seg_inds.shape[0] < num_segs:
#         num_to_add = num_segs - seg_inds.shape[0]
#         new_inds = np.random.choice(seg_inds.shape[0], size=(num_to_add,), replace=True)
#         seg_inds = np.concatenate((seg_inds, seg_inds[new_inds]), axis=0)
#         is_val = np.concatenate((is_val, is_val[new_inds]), axis=0)

#     window_size = 2*window_length
#     bgrs = torch.zeros((seg_inds.shape[0], 3, window_size, window_size), dtype=torch_im.dtype)
#     for ind in range(seg_inds.shape[0]):
#         x, y = seg_inds[ind]
#         window = torch_im[0, :, y-window_length:y+window_length, x-window_length:x+window_length]
#         bgrs[ind] = window

#     return torch.from_numpy(seg_inds), bgrs, torch.from_numpy(is_val)
