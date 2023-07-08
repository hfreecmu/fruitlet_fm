import argparse
import os
import torch
import torch.nn.functional as F

from data.dataloader import get_data_loader
from models.nbv_models import TransformerAssociator, prep_feature_data, extract_matches
from utils.torch_utils import load_checkpoint
from utils.nbv_utils import vis
import numpy as np

def infer(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.window_length,
                                 opt.num_points,
                                 1, True)

    width, height = opt.width, opt.height
    dims = [32, 32, 64, 64, 128]
    strides = [2, 2, 2, 2, 2]

    transformer = TransformerAssociator(dims, strides,
                                        opt.transformer_layers, dims[-1], opt.dim_feedforward,
                                        opt.dual_softmax,
                                        opt.sinkhorn_iterations,
                                        opt.device).to(opt.device)
    
    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, transformer)

    transformer.eval()

    image_num = 0    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            torch_im_0, torch_im_1, bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, is_mask_0, is_mask_1, _, _, matches_0, matches_1, is_aug = data

            num_images = bgrs_0.shape[0]
            if not (num_images == 1):
                raise RuntimeError('must have 1 image batch size for infer')
            image_ind = 0

            bgrs_0, positional_encodings_0 = prep_feature_data(seg_inds_0, bgrs_0, dims[-1], 
                                                               width, height, opt.device)
            bgrs_1, positional_encodings_1 = prep_feature_data(seg_inds_1, bgrs_1, dims[-1], 
                                                               width, height, opt.device)


            x_0 = (bgrs_0, positional_encodings_0, is_mask_0)
            x_1 = (bgrs_1, positional_encodings_1, is_mask_1)

            scores = transformer(x_0, x_1)
            
            indices_0, indices_1, mscores_0, _ = extract_matches(scores[image_ind].unsqueeze(0), opt.match_threshold, opt.use_dustbin)
            indices_0, indices_1 = indices_0.cpu(), indices_1.cpu()
            mscores_0 = mscores_0.cpu()
            
            ###
            has_match_i = (indices_0[image_ind] != -1)

            matched_inds_i = torch.arange(indices_0[image_ind].shape[0])[has_match_i]
            matched_inds_j = indices_0[image_ind, has_match_i]

            im_0_inds =  seg_inds_0[image_ind, matched_inds_i]
            im_1_inds = seg_inds_1[image_ind, matched_inds_j] 
            matching_scores = mscores_0[image_ind, matched_inds_i]

            if im_0_inds.shape[0] > opt.top_n:
                sorted_score_inds = torch.argsort(-matching_scores)[0:opt.top_n]
                im_0_inds = im_0_inds[sorted_score_inds]
                im_1_inds = im_1_inds[sorted_score_inds]
                matching_scores = matching_scores[sorted_score_inds]
            
            matches_output_filename = str(image_num) + '_infer_matches.png'
            matches_output_path = os.path.join(opt.vis_dir, matches_output_filename)
            vis(torch_im_0[image_ind], torch_im_1[image_ind], im_0_inds, im_1_inds, matches_output_path)

            gt_is_match = (matches_0[image_ind] != -1)
            gt_inds_0 = np.arange(seg_inds_0.shape[1])[gt_is_match]
            gt_inds_1 = matches_0[image_ind, gt_is_match].numpy()
            if gt_inds_0.shape[0] > 0:
                rand_inds = np.random.choice(gt_inds_0.shape[0], size=(opt.top_n,), replace=False)
                gt_inds_0 = gt_inds_0[rand_inds]
                gt_inds_1 = gt_inds_1[rand_inds]

            gt_output_filename = str(image_num) + '_gt_matches.png'
            gt_output_path = os.path.join(opt.vis_dir, gt_output_filename)
            vis(torch_im_0[0], torch_im_1[0], seg_inds_0[image_ind, gt_inds_0], seg_inds_1[image_ind, gt_inds_1], gt_output_path)
            
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
    parser.add_argument('--window_length', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=2000)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=10)

    parser.add_argument('--match_threshold', type=float, default=0.1)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--use_dustbin', action='store_false')

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)