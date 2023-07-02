import os
import torch
import cv2
import numpy as np

#img is returned unsqueezed
def load_torch_image(imfile, rand_flip):
    im = cv2.imread(imfile)
    flip=False
    if rand_flip:
        hflip = np.random.random() < 0.5
        if hflip:
            im = cv2.flip(im, 2)
            flip=True

    img = torch.from_numpy(im).permute(2, 0, 1)
    return img[None], im, flip

def save_checkpoint(epoch, checkpoint_dir, feature_model, transformer_model):
    feature_path = os.path.join(checkpoint_dir, 'epoch_%d_feature.pth' % epoch)
    transformer_path = os.path.join(checkpoint_dir, 'epoch_%d_transformer.pth' % epoch)

    torch.save(feature_model.state_dict(), feature_path)
    torch.save(transformer_model.state_dict(), transformer_path)

def load_checkpoint(epoch, checkpoint_dir, feature_model, transformer_model):
    feature_path = os.path.join(checkpoint_dir, 'epoch_%d_feature.pth' % epoch)
    transformer_path = os.path.join(checkpoint_dir, 'epoch_%d_transformer.pth' % epoch)

    feature_model.load_state_dict(torch.load(feature_path))
    transformer_model.load_state_dict(torch.load(transformer_path))

#TODO will be called twice in dataloader, how speed up?
def load_feature_inputs(torch_im, seg_inds, window_length):
    good_inds_0 = np.where((seg_inds[:, 0] - window_length >= 0) & 
                            (seg_inds[:, 0] + window_length < torch_im.shape[-1] - 1) & 
                            (seg_inds[:, 1] - window_length >= 0) & 
                            (seg_inds[:, 1] + window_length < torch_im.shape[-2] - 1))
    seg_inds = seg_inds[good_inds_0]

    window_size = 2*window_length
    bgrs = torch.zeros((seg_inds.shape[0], 3, window_size, window_size), dtype=torch_im.dtype)
    for ind in range(seg_inds.shape[0]):
        x, y = seg_inds[ind]
        window = torch_im[0, :, y-window_length:y+window_length, x-window_length:x+window_length]
        bgrs[ind] = window

    return torch.from_numpy(seg_inds), bgrs
