import argparse
import os
import torch
import torch.nn.functional as F

from data.dataloader import get_data_loader
from models.nbv_models import TransformerAssociator, prep_feature_data, extract_matches
from utils.torch_utils import load_checkpoint
from utils.nbv_utils import vis_segs
import numpy as np

def infer(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.min_dim,
                                 1, True)

    ###manually chnage things
    dims = [32, 64, 128, 256]
    strides = [1, 1, 1, 1]
    pools = [False, True, False, False]
    mlp_layers = [128, 8, 1]
    max_dim = 170
    torch.backends.cudnn.enabled = False
    ####

    transformer = TransformerAssociator(dims, strides,
                                        opt.transformer_layers, dims[-1], opt.dim_feedforward,
                                        mlp_layers, pools,
                                        opt.dual_softmax,
                                        opt.sinkhorn_iterations,
                                        opt.device).to(opt.device)
    
    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, transformer)

    transformer.eval()

    image_num = 0    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            torch_im_0, torch_im_1, seg_inds_pad_0, seg_inds_pad_1, num_points_0, num_points_1, matches_pad_0, matches_pad_1, is_aug = data

            num_images = torch_im_0.shape[0]
            if not (num_images == 1):
                raise RuntimeError('must have 1 image batch size for infer')
            image_ind = 0

            sub_ims_0 = []
            positional_encodings_0 = []
            is_keypoints_0 = []

            sub_ims_1 = []
            positional_encodings_1 = []
            is_keypoints_1 = []

            for _ in range(1):
                np_0 = num_points_0[image_ind]
                np_1 = num_points_1[image_ind]

                sub_im_0, pe_0, ik_0, x0_0, y0_0 = prep_feature_data(torch_im_0[image_ind], seg_inds_pad_0[image_ind, 0:np_0], 
                                                   matches_pad_0[image_ind, 0:np_0],
                                             dims[-1], max_dim, max_dim, opt.device)
                sub_im_1, pe_1, ik_1, x0_1, y0_1 = prep_feature_data(torch_im_1[image_ind], seg_inds_pad_1[image_ind, 0:np_1], 
                                                   matches_pad_1[image_ind, 0:np_1],
                                             dims[-1], max_dim, max_dim, opt.device)

                sub_ims_0.append(sub_im_0.unsqueeze(0))
                sub_ims_1.append(sub_im_1.unsqueeze(0))
                positional_encodings_0.append(pe_0.unsqueeze(0))
                positional_encodings_1.append(pe_1.unsqueeze(0))
                is_keypoints_0.append(ik_0)
                is_keypoints_1.append(ik_1)


            x_0 = (sub_ims_0, positional_encodings_0)
            x_1 = (sub_ims_1, positional_encodings_1)

            _, is_features_0, is_features_1 = transformer(x_0, x_1)
            
            if_0 = torch.sigmoid(is_features_0[image_ind]).cpu()
            if_1 = torch.sigmoid(is_features_1[image_ind]).cpu()
            
            ik_0 = is_keypoints_0[image_ind].cpu()
            ik_1 = is_keypoints_1[image_ind].cpu()

            h_0, w_0 = if_0.shape
            h_1, w_1 = if_1.shape

            h0_rat = sub_ims_0[image_ind].shape[2] / h_0
            w0_rat = sub_ims_0[image_ind].shape[3] / w_0
            h1_rat = sub_ims_1[image_ind].shape[2] / h_1
            w1_rat = sub_ims_1[image_ind].shape[3] / w_1

            kpts_0 = torch.argwhere(if_0 > opt.kpts_thresh)
            kpts_1 = torch.argwhere(if_1 > opt.kpts_thresh)

            kpts_0[:, 0] = torch.round(kpts_0[:, 0] * h0_rat)
            kpts_0[:, 1] = torch.round(kpts_0[:, 1] * w0_rat)
            kpts_1[:, 0] = torch.round(kpts_1[:, 0] * h1_rat)
            kpts_1[:, 1] = torch.round(kpts_1[:, 1] * w1_rat)

            gt_kpts_0 = torch.argwhere(ik_0 > opt.kpts_thresh)
            gt_kpts_1 = torch.argwhere(ik_1 > opt.kpts_thresh)

            kpts_0[:, 0], kpts_0[:, 1] = kpts_0[:, 1] + x0_0, kpts_0[:, 0] + y0_0
            kpts_1[:, 0], kpts_1[:, 1] = kpts_1[:, 1] + x0_1, kpts_1[:, 0] + y0_1

            gt_kpts_0[:, 0], gt_kpts_0[:, 1] = gt_kpts_0[:, 1] + x0_0, gt_kpts_0[:, 0] + y0_0
            gt_kpts_1[:, 0], gt_kpts_1[:, 1] = gt_kpts_1[:, 1] + x0_1, gt_kpts_1[:, 0] + y0_1
            
            ###
            kpts_output_filename = str(image_num) + '_infer_kpts.png'
            kpts_output_path = os.path.join(opt.vis_dir, kpts_output_filename)
            vis_segs(torch_im_0[image_ind], torch_im_1[image_ind], kpts_0, kpts_1, kpts_output_path)

            gt_kpts_output_filename = str(image_num) + '_gt_kpts.png'
            gt_kpts_output_path = os.path.join(opt.vis_dir, gt_kpts_output_filename)
            vis_segs(torch_im_0[image_ind], torch_im_1[image_ind], gt_kpts_0, gt_kpts_1, gt_kpts_output_path)

            image_num += 1
            if (image_num >= opt.num_images):
                return

            ###

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--segmentations_dir', required=True)
    parser.add_argument('--checkpoint_epoch', type=int, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--vis_dir', type=str, default='./vis/infer')
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=50)

    parser.add_argument('--match_threshold', type=float, default=0.1)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--use_dustbin', action='store_false')
    parser.add_argument('--kpts_thresh', type=float, default=0.6)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--min_dim', type=int, default=32)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)