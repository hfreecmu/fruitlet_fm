import argparse
import os
import torch
import torch.nn.functional as F

from data.dataloader import get_data_loader
from models.nbv_models import TransformerAssociator, prep_feature_data, extract_matches
from utils.torch_utils import load_checkpoint
from utils.nbv_utils import vis_segs, vis_lines
import numpy as np

def infer(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.min_dim,
                                 1, True)

    ###manually change things
    max_dim = 200
    #pos_weight = torch.ones((1)).to(opt.device)
    torch.backends.cudnn.enabled = False
    ###

    transformer = TransformerAssociator(opt.transformer_layers, opt.d_model, opt.dim_feedforward,
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

            sub_ims_1 = []
            positional_encodings_1 = []

            for _ in range(1):
                np_0 = num_points_0[image_ind]
                np_1 = num_points_1[image_ind]

                sub_im_0, pe_0, x0_0, y0_0, _ = prep_feature_data(torch_im_0[image_ind], seg_inds_pad_0[image_ind, 0:np_0], 
                                                   matches_pad_0[image_ind, 0:np_0],
                                             max_dim, opt.device)
                sub_im_1, pe_1, x0_1, y0_1, _ = prep_feature_data(torch_im_1[image_ind], seg_inds_pad_1[image_ind, 0:np_1], 
                                                   matches_pad_1[image_ind, 0:np_1],
                                             max_dim, opt.device)

                sub_ims_0.append(sub_im_0.unsqueeze(0))
                sub_ims_1.append(sub_im_1.unsqueeze(0))
                positional_encodings_0.append(pe_0.unsqueeze(0))
                positional_encodings_1.append(pe_1.unsqueeze(0))

            x_0 = (sub_ims_0, positional_encodings_0)
            x_1 = (sub_ims_1, positional_encodings_1)

            scores_i, scores_j = transformer(x_0, x_1)
            
            ###fm stuff
            indices_0, indices_1, mscores_0, _ = extract_matches(scores_i[image_ind].unsqueeze(0), scores_j[image_ind].unsqueeze(0),
                                                                 opt.match_threshold)
            indices_0, indices_1 = indices_0.cpu(), indices_1.cpu()
            mscores_0 = mscores_0.cpu()

            has_match_i = (indices_0[image_ind] != -1)

            matched_inds_i = torch.arange(indices_0[image_ind].shape[0])[has_match_i]
            matched_inds_j = indices_0[image_ind, has_match_i]
            matching_scores = mscores_0[image_ind, matched_inds_i]

            if matched_inds_i.shape[0] > opt.top_n:
                sorted_score_inds = torch.argsort(-matching_scores)[0:opt.top_n]
                matched_inds_i = matched_inds_i[sorted_score_inds]
                matched_inds_j = matched_inds_j[sorted_score_inds]
                matching_scores = matching_scores[sorted_score_inds]


            _, width_0 = sub_ims_0[image_ind].shape[-2:]
            _, width_1 = sub_ims_1[image_ind].shape[-2:]

            x0s = matched_inds_i % width_0 + x0_0
            y0s = matched_inds_i // width_0 + y0_0
            x1s = matched_inds_j % width_1 + x0_1
            y1s = matched_inds_j // width_1 + y0_1

            im_0_inds = torch.stack((x0s, y0s), dim=1)
            im_1_inds = torch.stack((x1s, y1s), dim=1)

            matches_output_filename = str(image_num) + '_infer_matches.png'
            matches_output_path = os.path.join(opt.vis_dir, matches_output_filename)
            vis_lines(torch_im_0[image_ind], torch_im_1[image_ind], im_0_inds, im_1_inds, matches_output_path)

            matches_0 = matches_pad_0[image_ind, 0:num_points_0[image_ind]]

            # gt_is_match = (matches_0 != -1)
            # gt_inds_0 = np.arange(seg_inds_0.shape[0])[gt_is_match]
            # gt_inds_1 = matches_0[gt_is_match].numpy()

            # gt_output_filename = str(image_num) + '_gt_matches.png'
            # gt_output_path = os.path.join(opt.vis_dir, gt_output_filename)
            # vis_lines(torch_im_0[0], torch_im_1[0], seg_inds_0[image_ind, gt_inds_0], seg_inds_1[image_ind, gt_inds_1], gt_output_path)
            
            ###
            

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

    parser.add_argument('--transformer_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=1024)

    parser.add_argument('--match_threshold', type=float, default=0.000)
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--num_images', type=int, default=20)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--min_dim', type=int, default=32)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)