import argparse
import os
import numpy as np
import torch
import cv2

from utils.nbv_utils import get_paths, read_pickle, write_pickle, vis
from utils.torch_utils import load_torch_image, load_feature_inputs, load_checkpoint
from models.nbv_models import prep_feature_data, load_feature_encoder, Transformer, extract_matches

from train import log_optimal_transport

def load_input_data(data_dir, image_ind, fruitlet_id, args):
    paths = get_paths(data_dir, [image_ind], single=True)

    _, left_path, _, _, _, segmentations_path = paths

    segmentations = read_pickle(segmentations_path)
    seg_inds = segmentations[fruitlet_id]
    seg_inds = np.stack((seg_inds[:, 1], seg_inds[:, 0]), axis=1)

    torch_im, _, _ = load_torch_image(left_path, rand_flip=False)

    sub_seg_inds, bgrs, is_val = load_feature_inputs(torch_im, seg_inds, args.window_length,
                                                 args.num_good, args.num_bad, 
                                                 use_centroid=args.use_centroid)

    #TODO randomize?

    return torch_im, bgrs.unsqueeze(0), sub_seg_inds.unsqueeze(0), is_val.unsqueeze(0), seg_inds

def get_homography(src_inds, dest_inds):
    M, mask = cv2.findHomography(src_inds.astype(float), dest_inds.astype(float), cv2.RANSAC, ransacReprojThreshold=5)
    return M, mask[:, 0]

def run(data_dir, output_dir, fruitlet_pairs, args):
    #image_inds = list(fruitlet_dict.keys())
    #paths = get_paths(data_dir, image_inds)


    dims = [32, 32, 64, 64, 128, 128]
    strides = [2, 1, 2, 1, 2, 2]
    feature_encoder = load_feature_encoder(dims, strides, device=args.device)
    transformer = Transformer(n_layers=args.transformer_layers, d_model=dims[-1]).to(args.device)
    load_checkpoint(args.checkpoint_epoch, args.checkpoint_dir, feature_encoder, transformer)
    feature_encoder.eval()
    transformer.eval()

    for pair in fruitlet_pairs:
        image_ind_0, fruitlet_id_0 = pair[0]
        image_ind_1, fruitlet_id_1 = pair[1]

        basename = '_'.join([str(image_ind_0), str(fruitlet_id_0), str(image_ind_1), str(fruitlet_id_1)])

        torch_im_0, bgrs_0, seg_inds_0, is_val_0, full_seg_inds_0 = load_input_data(data_dir, image_ind_0, fruitlet_id_0, args)
        torch_im_1, bgrs_1, seg_inds_1, is_val_1, full_seg_inds_1 = load_input_data(data_dir, image_ind_1, fruitlet_id_1, args)

        bgrs_0, positional_encodings_0 = prep_feature_data(seg_inds_0, bgrs_0,
                                                           dims[-1], args.width,
                                                           args.height, args.device)
        bgrs_1, positional_encodings_1 = prep_feature_data(seg_inds_1, bgrs_1,
                                                               dims[-1], args.width,
                                                               args.height, args.device)

        seg_inds_0 = seg_inds_0.squeeze(0)
        is_val_0 = (is_val_0.squeeze(0) == 1)

        seg_inds_1 = seg_inds_1.squeeze(0)
        is_val_1 = (is_val_1.squeeze(0) == 1)

        with torch.no_grad():
            features_0 = feature_encoder(bgrs_0.squeeze(0)).reshape((-1, dims[-1])).unsqueeze(0)
            src_0 = features_0 + positional_encodings_0

            features_1 = feature_encoder(bgrs_1.squeeze(0)).reshape((-1, dims[-1])).unsqueeze(0)
            src_1 = features_1 + positional_encodings_1

            desc_0, desc_1 = transformer(src_0, src_1)

            desc_0 = torch.permute(desc_0, (0, 2, 1))
            desc_1 = torch.permute(desc_1, (0, 2, 1))

            #TODO assuming log transfport
            scores = torch.einsum('bdn,bdm->bnm', desc_0, desc_1)
            scores = scores / dims[-1]**.5
            scores = log_optimal_transport(scores, transformer.bin_score, iters=args.sinkhorn_iterations)

        indices_0, indices_1, mscores_0, _ = extract_matches(scores, args.match_threshold, 
                                                             use_dustbin=args.use_dustbin)
        indices_0, indices_1 = indices_0.cpu().squeeze(0), indices_1.cpu().squeeze(0)
        mscores_0 = mscores_0.cpu().squeeze(0)

        has_match_i = (indices_0 != -1)
        matched_inds_i = torch.arange(indices_0.shape[0])[has_match_i]
        matched_inds_j = indices_0[has_match_i]

        matched_inds_i, matched_inds_j = matched_inds_i[is_val_0[matched_inds_i]], matched_inds_j[is_val_0[matched_inds_i]]
        matched_inds_i, matched_inds_j = matched_inds_i[is_val_1[matched_inds_j]], matched_inds_j[is_val_1[matched_inds_j]]

        im_0_inds =  seg_inds_0[matched_inds_i]
        im_1_inds = seg_inds_1[matched_inds_j] 
        matching_scores = mscores_0[matched_inds_i]

        #TODO this before or after top_n?
        if args.use_homography and im_0_inds.shape[0] >= 4:
            _, mask = get_homography(im_0_inds.numpy().copy(), im_1_inds.numpy().copy())
            ransac_inds = (mask > 0) 
            im_0_inds = im_0_inds[ransac_inds]
            im_1_inds = im_1_inds[ransac_inds]
            matching_scores = matching_scores[ransac_inds]

        ###save keypoint inds for bundle
        #do this after homography before vis
        keypoints_0 = im_0_inds.numpy().copy()
        keypoints_1 = im_1_inds.numpy().copy()
        keypoints_0 = np.stack((keypoints_0[:, 0], keypoints_0[:, 1]), axis=1)
        keypoints_1 = np.stack((keypoints_1[:, 0], keypoints_1[:, 1]), axis=1)
            
        output_match_path = os.path.join(args.vis_dir, basename + '.pkl')
        print('matched num keypoints: ', keypoints_0.shape)
        write_pickle(output_match_path, [keypoints_0, keypoints_1])
        ###

        if im_0_inds.shape[0] > args.top_n:
            if not args.use_rand_n:
                sorted_score_inds = torch.argsort(-matching_scores)[0:args.top_n]
            else:
                sorted_score_inds = np.random.choice(im_0_inds.shape[0], size=(args.top_n,), replace=False)
            im_0_inds = im_0_inds[sorted_score_inds]
            im_1_inds = im_1_inds[sorted_score_inds]
            matching_scores = matching_scores[sorted_score_inds]

        output_vis_path = os.path.join(args.vis_dir, basename + '.png')
        vis(torch_im_0[0], torch_im_1[0], im_0_inds, im_1_inds, output_vis_path)

        fm = cv2.SIFT_create()
        cv_im_0 = torch_im_0[0].permute(1, 2, 0).numpy().copy()
        cv_im_1 = torch_im_1[0].permute(1, 2, 0).numpy().copy()

        #cv_im_0 = cv2.imread('/home/frc-ag-3/Downloads/box_1.png')
        #cv_im_1 = cv2.imread('/home/frc-ag-3/Downloads/box_0.png')

        gray_im_0 = cv2.cvtColor(cv_im_0, cv2.COLOR_BGR2GRAY)
        gray_im_1 = cv2.cvtColor(cv_im_1, cv2.COLOR_BGR2GRAY)

        mask_0 = np.zeros((cv_im_0.shape[0], cv_im_0.shape[1]), dtype=np.uint8)
        mask_1 = np.zeros((cv_im_1.shape[0], cv_im_1.shape[1]), dtype=np.uint8)

        mask_0[full_seg_inds_0[:, 1], full_seg_inds_0[:, 0]] = 255
        mask_1[full_seg_inds_1[:, 1], full_seg_inds_1[:, 0]] = 255

        kp1, desc1 = fm.detectAndCompute(gray_im_0, mask_0)
        kp2, desc2 = fm.detectAndCompute(gray_im_1, mask_1)
        #kp1, desc1 = fm.detectAndCompute(gray_im_0, None)
        #kp2, desc2 = fm.detectAndCompute(gray_im_1, None)

        fm_matcher = cv2.BFMatcher(crossCheck=True)
        fm_matches = fm_matcher.match(desc1, desc2)

        fm_matches = sorted(fm_matches, key = lambda x:x.distance)

        img3 = cv2.drawMatches(cv_im_0,kp1,cv_im_1,kp2,fm_matches[0:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        output_fm_path = os.path.join(args.vis_dir, basename + '_fm.png')
        cv2.imwrite(output_fm_path, img3)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint_epoch', type=int, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--vis_dir', type=str, default='./vis')
    parser.add_argument('--num_good', type=int, default=1000)
    parser.add_argument('--num_bad', type=int, default=1000)
    parser.add_argument('--window_length', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--sinkhorn_iterations', type=int, default=100)

    #0.5 no centroid, 0.01 with centroid for 33000 model?
    parser.add_argument('--use_centroid', action='store_true')
    parser.add_argument('--use_dustbin', action='store_true')
    parser.add_argument('--match_threshold', type=float, default=0.5)
    parser.add_argument('--use_homography', action='store_true')
    parser.add_argument('--top_n', type=int, default=20)
    parser.add_argument('--use_rand_n', action='store_true')

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)

    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    return args

data_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_bag'
output_dir = None
# fruitlet_dict = {2: 3, 
#                  3: 6,
#                  5: 3,
#                  6: 7,
#                  7: 3}
fruitlet_pairs = [((2, 3), (3, 6)),
                  ((5, 3), (3, 6))]

fruitlet_pairs = [((2, 3), (3, 6)),
                  ((5, 3), (3, 6)),
                  ((2, 3), (6, 7))]

# fruitlet_pairs = [((2, 3), (6, 7))]

if __name__ == "__main__":
    args = parse_args()

    run(data_dir, output_dir, fruitlet_pairs, args)