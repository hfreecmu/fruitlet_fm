import os
import torch
import cv2
import numpy as np

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
