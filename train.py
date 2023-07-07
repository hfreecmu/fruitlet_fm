import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from data.dataloader import get_data_loader
from models.nbv_models import TransformerAssociator, prep_feature_data
from utils.torch_utils import save_checkpoint, plot_loss

def train(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.min_dim,
                                 opt.batch_size, opt.shuffle)
    
    width, height = opt.width, opt.height

    ###manually change things
    dims = [32, 32, 64, 64, 128]
    strides = [1, 1, 2, 1, 2]
    mlp_layers = [128, 64, 1]
    pos_weight = torch.ones((1)).to(opt.device)
    ###

    transformer = TransformerAssociator(dims, strides,
                                        opt.transformer_layers, dims[-1], opt.dim_feedforward,
                                        mlp_layers,
                                        opt.dual_softmax,
                                        opt.sinkhorn_iterations,
                                        opt.device).to(opt.device)

    ###optimizers
    feature_optimizer = optim.Adam(transformer.encoder.parameters(), opt.conv_lr)
    bin_optimizer = optim.Adam([transformer.bin_score], opt.bin_lr)
    transformer_optimizer = optim.Adam(transformer.transformer.parameters(), opt.trans_lr)

    milestones = [30000, 50000, 55000, 57500, 60000]
    feature_scheduler = optim.lr_scheduler.MultiStepLR(feature_optimizer, milestones=milestones, gamma=0.5)
    bin_scheduler = optim.lr_scheduler.MultiStepLR(bin_optimizer, milestones=milestones, gamma=0.5)
    transform_scheduler = optim.lr_scheduler.MultiStepLR(transformer_optimizer, milestones=milestones, gamma=0.5)
    ###

    ###bce loss
    bce_loss_fn = loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ###

    loss_array = []
    step_num = 0
    while step_num < opt.num_steps:
        for _, data in enumerate(dataloader):
            if (step_num % opt.log_steps) == 0:
                losses = 0
                num_losses = 0

            torch_im_0, torch_im_1, seg_inds_pad_0, seg_inds_pad_1, num_points_0, num_points_1, matches_pad_0, matches_pad_1, is_aug = data

            num_images = torch_im_0.shape[0]

            sub_ims_0 = []
            positional_encodings_0 = []
            is_keypoints_0 = []

            sub_ims_1 = []
            positional_encodings_1 = []
            is_keypoints_1 = []

            for image_ind in range(num_images):
                np_0 = num_points_0[image_ind]
                np_1 = num_points_1[image_ind]

                sub_im_0, pe_0, ik_0 = prep_feature_data(torch_im_0[image_ind], seg_inds_pad_0[image_ind, 0:np_0], 
                                                   matches_pad_0[image_ind, 0:np_0],
                                             dims[-1], width, height, opt.device)
                sub_im_1, pe_1, ik_1 = prep_feature_data(torch_im_1[image_ind], seg_inds_pad_1[image_ind, 0:np_1], 
                                                   matches_pad_1[image_ind, 0:np_1],
                                             dims[-1], width, height, opt.device)

                sub_ims_0.append(sub_im_0)
                sub_ims_1.append(sub_im_1)
                positional_encodings_0.append(pe_0)
                positional_encodings_1.append(pe_1)
                is_keypoints_0.append(ik_0)
                is_keypoints_1.append(ik_1)

            x_0 = (sub_ims_0, positional_encodings_0)
            x_1 = (sub_ims_1, positional_encodings_1)

            scores, is_features_0, is_features_1 = transformer(x_0, x_1)

            loss = []
            #this works because masks were appended
            for image_ind in range(num_images):
                ind_scores = scores[image_ind]
                if_0 = is_features_0[image_ind]
                if_1 = is_features_1[image_ind]

                ik_0 = is_keypoints_0[image_ind]
                ik_1 = is_keypoints_1[image_ind]

                h_0, w_0 = if_0.shape
                ik_0 = F.interpolate(ik_0.unsqueeze(0).unsqueeze(0), size=(h_0, w_0), mode='bilinear').squeeze().squeeze(0)

                h_1, w_1 = if_1.shape
                ik_1 = F.interpolate(ik_1.unsqueeze(0).unsqueeze(0), size=(h_1, w_1), mode='bilinear').squeeze().squeeze(0)


                ik_0 = torch.round(ik_0)
                ik_1 = torch.round(ik_1)

                bce_loss_0 = bce_loss_fn(if_0, ik_0)
                bce_loss_1 = bce_loss_fn(if_1, ik_1)

                loss.append(bce_loss_0 + bce_loss_1)
                
                # #this accounts for mask
                # has_match_i = (matches_0[image_ind] != -1)
                # has_match_j = (matches_1[image_ind] != -1)

                # #only doing once as two way
                # matched_inds_i = torch.arange(matches_0[image_ind].shape[0])[has_match_i]
                # matched_inds_j = matches_0[image_ind, has_match_i]

                # is_mask_i = is_mask_0[image_ind]
                # is_mask_j = is_mask_1[image_ind]

                # has_unmatched_i = ((~has_match_i) & (~is_mask_i))
                # has_unmatched_j = ((~has_match_j) & (~is_mask_j))

                # unmatched_inds_i = torch.arange(matches_0[image_ind].shape[0])[has_unmatched_i]
                # unmatched_inds_j = torch.arange(matches_1[image_ind].shape[0])[has_unmatched_j]

                # matched_scores = ind_scores[matched_inds_i, matched_inds_j]
                # unmatched_i_scores = opt.unmatch_scale*ind_scores[unmatched_inds_i, ind_scores.shape[1] - 1]
                # unmatched_j_scores = opt.unmatch_scale*ind_scores[ind_scores.shape[0] - 1, unmatched_inds_j]

                # loss_scores = torch.concatenate((matched_scores, unmatched_i_scores, unmatched_j_scores))

                # ind_loss = torch.mean(-loss_scores)
                # loss.append(ind_loss)

            loss = torch.mean(torch.stack((loss)))
            feature_optimizer.zero_grad()
            bin_optimizer.zero_grad()
            transformer_optimizer.zero_grad()
            loss.backward()
            feature_optimizer.step()
            bin_optimizer.step()
            transformer_optimizer.step()

            for step_sched in range(num_images):
                feature_scheduler.step()
                bin_scheduler.step()
                transform_scheduler.step()

            losses += loss.item()
            num_losses += 1

            if ((step_num + 1) % opt.log_steps) == 0:
                if num_losses == 0:
                    epoch_loss = -1
                else:
                    epoch_loss = losses/num_losses
                print('loss for step: ', step_num + 1, ' is: ', epoch_loss)

                loss_array.append([step_num + 1, epoch_loss])
                plot_loss(loss_array, opt.checkpoint_dir)


            if ((step_num + 1) % opt.log_checkpoints) == 0:
                print('saving checkpoint for step: ', step_num + 1)
                save_checkpoint(step_num + 1, opt.checkpoint_dir, transformer) 

            step_num += 1

    print('Done')         
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--segmentations_dir', required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=50)
    parser.add_argument('--unmatch_scale', type=float, default=0.5)

    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--trans_lr', type=float, default=1e-5)
    parser.add_argument('--conv_lr', type=float, default=1e-3)
    parser.add_argument('--bin_lr', type=float, default=1e-3)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=15000)
    parser.add_argument('--log_steps', type=int, default=25)
    parser.add_argument('--log_checkpoints', type=int, default=25)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--min_dim', type=int, default=24)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)