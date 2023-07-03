import argparse
import os
import torch
import cv2
import numpy as np
import distinctipy

from utils.nbv_utils import read_json, write_pickle

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

def load_seg_model(model_file, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_file 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    cfg.MODEL.LINE_ON = False #lines_on

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_file)['model'])
    model.eval()

    return model

def segment_image(model, image_path, seg_bounds):
    im = cv2.imread(image_path)
    segmentations = []

    x_max = int((1-seg_bounds)*im.shape[1])
    x_min = int(seg_bounds*im.shape[1])

    y_max = int((1-seg_bounds)*im.shape[0])
    y_min = int(seg_bounds*im.shape[0])

    image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image}
    with torch.no_grad():
        outputs = model([inputs])[0]

    masks = outputs['instances'].get('pred_masks').to('cpu').numpy()
    boxes = outputs['instances'].get('pred_boxes').to('cpu')

    num = len(boxes)
    #for segmentation
    assert num < 254
    colors = distinctipy.get_colors(num)
    for i in range(num):
        color = ([int(255*colors[i][0]), int(255*colors[i][1]), int(255*colors[i][2])])

        seg_inds = np.argwhere(masks[i, :, :] > 0)
        
        if (seg_inds[:, 0] <= y_min).any():
            continue

        if (seg_inds[:, 0] >= y_max).any():
            continue

        if (seg_inds[:, 1] <= x_min).any():
            continue

        if (seg_inds[:, 1] >= x_max).any():
            continue

        im[seg_inds[:, 0], seg_inds[:, 1]] = color
        segmentations.append(seg_inds)

    return im, segmentations

def get_image_paths(image_dir):

    image_paths = []

    for filename in os.listdir(image_dir):
        if not filename.endswith('.png'):
            continue

        image_path = os.path.join(image_dir, filename)
        image_paths.append(image_path)

    return image_paths

def extract(image_dir, output_dir, model_path, score_thresh, train_split, seg_bounds):
    train_dir = os.path.join(output_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    test_dir = os.path.join(output_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    image_paths = get_image_paths(image_dir)
    seg_model = load_seg_model(model_path, score_thresh)

    num_train = int(train_split*len(image_paths))
    train_inds = np.random.choice(len(image_paths), size=(num_train,), replace=False)

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        
        vis_im, segmentations = segment_image(seg_model, image_path, seg_bounds)

        if len(segmentations) == 0:
            continue

        basename = os.path.basename(image_path).replace('.png', '.pkl')

        if i in train_inds:
            seg_output_path = os.path.join(train_dir, basename)
        else:
            seg_output_path = os.path.join(test_dir, basename)

        write_pickle(seg_output_path, segmentations)
        cv2.imwrite(seg_output_path.replace('.pkl', '_vis.png'), vis_im)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--score_thresh', type=float, default=0.5)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--seg_bounds', type=float, default=0.1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    model_path = args.model_path
    output_dir = args.output_dir
    score_thresh = args.score_thresh
    train_split = args.train_split
    seg_bounds = args.seg_bounds

    if not os.path.exists(image_dir):
        raise RuntimeError('image_dir does not exist: ' + image_dir)
    
    if not os.path.exists(model_path):
        raise RuntimeError('model_path does not exist: ' + model_path)
    
    if not os.path.exists(output_dir):
        raise RuntimeError('output_dir does not exist: ' + output_dir)
    
    extract(image_dir, output_dir, model_path, score_thresh, train_split, seg_bounds)

