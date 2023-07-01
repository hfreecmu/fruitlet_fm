import argparse
import os
import torch
import torch.nn.functional as F

from data.dataloader import get_data_loader
from models.nbv_models import load_feature_encoder, positional_encoder, extract_matches, Transformer
from utils.torch_utils import load_checkpoint
from utils.nbv_utils import vis

from train import log_optimal_transport

def infer(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.window_length,
                                 opt.num_good, opt.num_bad,
                                 1, False)

    width, height = opt.width, opt.height
    dims = [32, 32, 64, 64, 128, 128]
    strides = [2, 1, 2, 1, 2, 2]

    feature_encoder = load_feature_encoder(dims, strides, device=opt.device)
    transformer = Transformer(n_layers=opt.transformer_layers, d_model=dims[-1]).to(opt.device)
    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, feature_encoder, transformer)

    feature_encoder.eval()
    transformer.eval()

    with torch.no_grad():
        for batch_num, data in enumerate(dataloader):
            torch_im_0, torch_im_1, bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, matches_0, matches_1 = data

            num_images = bgrs_0.shape[0]

            seg_inds_0_orig = seg_inds_0
            seg_inds_1_orig = seg_inds_1

            bgrs_0 = bgrs_0.float() / 255
            bgrs_0 = bgrs_0.to(opt.device)
            bgrs_1 = bgrs_1.float() / 255
            bgrs_1 = bgrs_1.to(opt.device)

            seg_inds_0 = seg_inds_0.float()
            seg_inds_1 = seg_inds_1.float()
            seg_inds_0[:, :, 0] = (2*seg_inds_0[:, :, 0] - width) / width
            seg_inds_0[:, :, 1] = (2*seg_inds_0[:, :, 1] - height) / height
            seg_inds_1[:, :, 0] = (2*seg_inds_1[:, :, 0] - width) / width
            seg_inds_1[:, :, 1] = (2*seg_inds_1[:, :, 1] - height) / height

            positional_encodings_0 = positional_encoder(seg_inds_0, dims[-1]//4)
            positional_encodings_1 = positional_encoder(seg_inds_1, dims[-1]//4)
            positional_encodings_0 = positional_encodings_0.to(opt.device)
            positional_encodings_1 = positional_encodings_1.to(opt.device)

            num_segs = opt.num_good + opt.num_bad
            features_0 = torch.zeros((num_images, num_segs, dims[-1])).float().to(opt.device)
            features_1 = torch.zeros((num_images, num_segs, dims[-1]), dtype=float).float().to(opt.device)

            #TODO make one cnn call if speed is an issue?
            for image_ind in range(num_images):
                features_0[image_ind] = feature_encoder(bgrs_0[image_ind]).reshape((-1, dims[-1]))
                features_1[image_ind] = feature_encoder(bgrs_1[image_ind]).reshape((-1, dims[-1]))

            src_0 = features_0 + positional_encodings_0
            src_1 = features_1 + positional_encodings_1

            desc_0, desc_1 = transformer(src_0, src_1)

            desc_0 = torch.permute(desc_0, (0, 2, 1))
            desc_1 = torch.permute(desc_1, (0, 2, 1))

            #TODO can use other method as described in 
            #https://arxiv.org/pdf/2104.00680.pdf?
            if opt.dual_softmax:
                #dustbin_i = transformer.bin_i_score.expand(desc_0.shape[0], dims[-1]).reshape((-1, dims[-1], 1))
                #desc_0 = torch.cat((desc_0, dustbin_i), axis=2)

                #dustbin_j = transformer.bin_j_score.expand(desc_1.shape[0], dims[-1]).reshape((-1, dims[-1], 1))
                #desc_1 = torch.cat((desc_1, dustbin_j), axis=2)

                scores = torch.einsum('bdn,bdm->bnm', desc_0, desc_1)
                scores = scores / dims[-1]

                #TODO are these axis right?
                S_i = F.softmax(scores, dim=1)
                S_j = F.softmax(scores, dim=2)
                scores = torch.log(S_i*S_j)
            else:
                scores = torch.einsum('bdn,bdm->bnm', desc_0, desc_1)
                scores = scores / dims[-1]**.5
                ot_scores = []
                for image_ind in range(num_images):
                     ot_scores.append(log_optimal_transport(scores[image_ind:image_ind+1], transformer.bin_score, iters=opt.sinkhorn_iterations))
                scores = torch.concatenate(ot_scores, dim=0)
            
            #part where I stopped copying
            #TODO this will dustbin match so maybe modify extract matches
            indices_0, indices_1 = extract_matches(scores, opt.match_threshold)
            indices_0, indices_1 = indices_0.cpu(), indices_1.cpu()
            
            for image_ind in range(num_images):
                has_match_i = (indices_0[image_ind] != -1)

                matched_inds_i = torch.arange(indices_0[image_ind].shape[0])[has_match_i]
                matched_inds_j = indices_0[image_ind, has_match_i]

                im_0_inds =  seg_inds_0_orig[image_ind, matched_inds_i]
                im_1_inds = seg_inds_1_orig[image_ind, matched_inds_j]              
                
                output_path = os.path.join(opt.vis_dir, 'debug_im.png')
                vis(torch_im_0[0], torch_im_1[0], im_0_inds, im_1_inds, output_path)
                print('Done')

                return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--segmentations_dir', required=True)
    parser.add_argument('--checkpoint_epoch', type=int, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--vis_dir', type=str, default='./vis')
    parser.add_argument('--window_length', type=int, default=8)
    parser.add_argument('--num_good', type=int, default=20)
    parser.add_argument('--num_bad', type=int, default=0)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=100)

    parser.add_argument('--match_threshold', type=float, default=0.4)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)