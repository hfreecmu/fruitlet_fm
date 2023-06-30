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