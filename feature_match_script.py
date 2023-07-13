import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from utils.nbv_utils import get_paths, read_pickle, write_pickle, vis_segs, vis_lines
from utils.torch_utils import load_torch_image, load_checkpoint
from models.nbv_models import prep_feature_data, TransformerAssociator, extract_matches

def load_input_data(data_dir, image_ind, fruitlet_id, args):
    paths = get_paths(data_dir, [image_ind], single=True)

    _, left_path, _, _, _, segmentations_path = paths

    segmentations = read_pickle(segmentations_path)
    seg_inds = segmentations[fruitlet_id]
    seg_inds = np.stack((seg_inds[:, 1], seg_inds[:, 0]), axis=1)

    torch_im, _, _ = load_torch_image(left_path, rand_flip=False)

    return torch_im, torch.from_numpy(seg_inds).unsqueeze(0)

def get_homography(src_inds, dest_inds):
    M, mask = cv2.findHomography(src_inds.astype(float), dest_inds.astype(float), cv2.RANSAC, ransacReprojThreshold=5)
    return M, mask[:, 0]

def run(data_dir, fruitlet_pairs, args):

    ###manually change things
    kpts_dims = [64, 64, 128, 128, 128]
    kpts_strides = [1, 1, 1, 1, 1]
    kpts_pools = [False, False, True, False, False]

    dims = [32, 32, 64, 64, 128]
    strides = [1, 1, 1, 1, 1]
    pools = [False, False, True, False, False]
    max_dim = 200
    #pos_weight = torch.ones((1)).to(opt.device)
    torch.backends.cudnn.enabled = False
    ###

    transformer = TransformerAssociator((dims, kpts_dims), (strides, kpts_strides),
                                        args.transformer_layers, dims[-1], args.dim_feedforward,
                                        (pools, kpts_pools),
                                        args.dual_softmax,
                                        args.sinkhorn_iterations,
                                        args.device).to(args.device)
    
    load_checkpoint(args.checkpoint_epoch, args.checkpoint_dir, transformer)

    transformer.eval()

    for pair in fruitlet_pairs:
        image_ind_0, fruitlet_id_0 = pair[0]
        image_ind_1, fruitlet_id_1 = pair[1]

        basename = '_'.join([str(image_ind_0), str(fruitlet_id_0), str(image_ind_1), str(fruitlet_id_1)])

        torch_im_0, seg_inds_0 = load_input_data(data_dir, image_ind_0, fruitlet_id_0, args)
        torch_im_1, seg_inds_1 = load_input_data(data_dir, image_ind_1, fruitlet_id_1, args)

        sub_im_0, pe_0, _, x0_0, y0_0, _ = prep_feature_data(torch_im_0[0], seg_inds_0[0],
                                                           None, max_dim, args.device)
        sub_im_1, pe_1, _, x0_1, y0_1, _ = prep_feature_data(torch_im_1[0], seg_inds_1[0],
                                                           None, max_dim, args.device)
                
        x_0 = ([sub_im_0.unsqueeze(0)], [pe_0.unsqueeze(0)])
        x_1 = ([sub_im_1.unsqueeze(0)], [pe_1.unsqueeze(0)])
        with torch.no_grad():
            scores = transformer(x_0, x_1)

        indices_0, indices_1, mscores_0, _ = extract_matches(scores[0].unsqueeze(0), args.match_threshold, args.use_dustbin)
        
        indices_0, indices_1 = indices_0.cpu().squeeze(0), indices_1.cpu().squeeze(0)
        mscores_0 = mscores_0.cpu().squeeze(0)

        seg_inds_0 = seg_inds_0.squeeze()
        seg_inds_1 = seg_inds_1.squeeze()

        has_match_i = (indices_0 != -1)
        matched_inds_i = torch.arange(indices_0.shape[0])[has_match_i]
        matched_inds_j = indices_0[has_match_i]

        wi_0 = sub_im_0.shape[-1]
        wi_1 = sub_im_1.shape[-1]

        x0s = matched_inds_i % wi_0 + x0_0
        y0s = matched_inds_i // wi_0 + y0_0
        x1s = matched_inds_j % wi_1 + x0_1
        y1s = matched_inds_j // wi_1 + y0_1

        im_0_inds = torch.stack((x0s, y0s), dim=1)
        im_1_inds = torch.stack((x1s, y1s), dim=1)
        matching_scores = mscores_0[matched_inds_i]

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
        print('matched num keypoints: ', basename, keypoints_0.shape)
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
        vis_lines(torch_im_0[0], torch_im_1[0], im_0_inds, im_1_inds, output_vis_path)

        # fm = cv2.SIFT_create()
        # cv_im_0 = torch_im_0[0].permute(1, 2, 0).numpy().copy()
        # cv_im_1 = torch_im_1[0].permute(1, 2, 0).numpy().copy()

        # #cv_im_0 = cv2.imread('/home/frc-ag-3/Downloads/box_1.png')
        # #cv_im_1 = cv2.imread('/home/frc-ag-3/Downloads/box_0.png')

        # gray_im_0 = cv2.cvtColor(cv_im_0, cv2.COLOR_BGR2GRAY)
        # gray_im_1 = cv2.cvtColor(cv_im_1, cv2.COLOR_BGR2GRAY)

        # mask_0 = np.zeros((cv_im_0.shape[0], cv_im_0.shape[1]), dtype=np.uint8)
        # mask_1 = np.zeros((cv_im_1.shape[0], cv_im_1.shape[1]), dtype=np.uint8)

        # mask_0[full_seg_inds_0[:, 1], full_seg_inds_0[:, 0]] = 255
        # mask_1[full_seg_inds_1[:, 1], full_seg_inds_1[:, 0]] = 255

        # kp1, desc1 = fm.detectAndCompute(gray_im_0, mask_0)
        # kp2, desc2 = fm.detectAndCompute(gray_im_1, mask_1)
        # #kp1, desc1 = fm.detectAndCompute(gray_im_0, None)
        # #kp2, desc2 = fm.detectAndCompute(gray_im_1, None)

        # fm_matcher = cv2.BFMatcher(crossCheck=True)
        # fm_matches = fm_matcher.match(desc1, desc2)

        # fm_matches = sorted(fm_matches, key = lambda x:x.distance)

        # # img3 = cv2.drawMatches(cv_im_0,kp1,cv_im_1,kp2,fm_matches[0:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # img3 = cv2.drawMatches(cv_im_0,kp1,cv_im_1,kp2,fm_matches[0:10],None)

        # output_fm_path = os.path.join(args.vis_dir, basename + '_fm.png')
        # cv2.imwrite(output_fm_path, img3)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint_epoch', type=int, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--vis_dir', type=str, default='./vis/feature_match')
    parser.add_argument('--transformer_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=10)

    parser.add_argument('--match_threshold', type=float, default=0.01)
    parser.add_argument('--top_n', type=int, default=50)
    parser.add_argument('--use_dustbin', action='store_false')
    parser.add_argument('--use_homography', action='store_true')
    parser.add_argument('--use_rand_n', action='store_true')

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)

    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    return args

data_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/nbv/debug_bag'
fruitlet_dict = {2: 3, 
                 3: 6,
                 5: 3,
                 6: 7,
                 7: 3}
# fruitlet_pairs = [((2, 3), (3, 6)),
#                   ((5, 3), (3, 6))]

# fruitlet_pairs = [((2, 3), (3, 6)),
#                   ((5, 3), (3, 6)),
#                   ((2, 3), (6, 7))]

# fruitlet_pairs = [((2, 3), (6, 7))]

fruitlet_pairs = []
keys = list(fruitlet_dict.keys())
for i in range(len(keys)):
    pair_i = (keys[i], fruitlet_dict[keys[i]])
    for j in range(i+1, len(keys)):
        pair_j = (keys[j], fruitlet_dict[keys[j]])
        fruitlet_pairs.append((pair_i, pair_j))

if __name__ == "__main__":
    args = parse_args()

    run(data_dir, fruitlet_pairs, args)