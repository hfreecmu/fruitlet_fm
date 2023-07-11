import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from data.dataloader import get_data_loader
from models.nbv_models import TransformerAssociator, prep_feature_data
from utils.torch_utils import save_checkpoint, plot_loss

from utils.torch_utils import load_checkpoint

def train(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.min_dim,
                                 opt.batch_size, opt.shuffle)
    
    ###manually change things
    max_dim = 200
    #pos_weight = torch.ones((1)).to(opt.device)
    torch.backends.cudnn.enabled = False
    ###

    transformer = TransformerAssociator(opt.transformer_layers, opt.d_model, opt.dim_feedforward,
                                        opt.device).to(opt.device)
    
    #load_checkpoint(2300, './checkpoints', transformer)

    ###optimizers
    feature_optimizer = optim.Adam(list(transformer.encoder.parameters()) + list(transformer.kpts_encoder.parameters()), opt.conv_lr)
    bin_optimizer = optim.Adam([transformer.bin_score], opt.bin_lr)
    transformer_optimizer = optim.Adam(transformer.transformer.parameters(), opt.trans_lr)

    milestones = [10000, 20000, 30000, 50000, 75000]
    feature_scheduler = optim.lr_scheduler.MultiStepLR(feature_optimizer, milestones=milestones, gamma=0.5)
    bin_scheduler = optim.lr_scheduler.MultiStepLR(bin_optimizer, milestones=milestones, gamma=0.5)
    transform_scheduler = optim.lr_scheduler.MultiStepLR(transformer_optimizer, milestones=milestones, gamma=0.5)
    ###

    ###bce loss
    #bce_loss_fn = loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #bce_loss_fn = loss = nn.BCEWithLogitsLoss()
    ###

    loss_array = []
    step_num = 0
    while step_num < opt.num_steps:
        for _, data in enumerate(dataloader):
            if (step_num % opt.log_steps) == 0:
                losses = 0
                num_losses = 0

            torch_im_0, torch_im_1, seg_inds_pad_0, seg_inds_pad_1, num_points_0, num_points_1, matches_pad_0, matches_pad_1, _ = data

            num_images = torch_im_0.shape[0]

            sub_ims_0 = []
            positional_encodings_0 = []
            new_seg_inds_0 = []

            sub_ims_1 = []
            positional_encodings_1 = []
            new_seg_inds_1 = []

            for image_ind in range(num_images):
                np_0 = num_points_0[image_ind]
                np_1 = num_points_1[image_ind]

                sub_im_0, pe_0, _, _, nsi_0 = prep_feature_data(torch_im_0[image_ind], seg_inds_pad_0[image_ind, 0:np_0], 
                                                   matches_pad_0[image_ind, 0:np_0],
                                             max_dim, opt.device)
                sub_im_1, pe_1, _, _, nsi_1 = prep_feature_data(torch_im_1[image_ind], seg_inds_pad_1[image_ind, 0:np_1], 
                                                   matches_pad_1[image_ind, 0:np_1],
                                             max_dim, opt.device)
                
                sub_ims_0.append(sub_im_0.unsqueeze(0))
                sub_ims_1.append(sub_im_1.unsqueeze(0))
                positional_encodings_0.append(pe_0.unsqueeze(0))
                positional_encodings_1.append(pe_1.unsqueeze(0))
                new_seg_inds_0.append(nsi_0)
                new_seg_inds_1.append(nsi_1)

            x_0 = (sub_ims_0, positional_encodings_0)
            x_1 = (sub_ims_1, positional_encodings_1)

            scores_i, scores_j = transformer(x_0, x_1)

            loss = []
            #this works because masks were appended
            for image_ind in range(num_images):   
                #WARNING WARNING WARNING
                #did I switch i and j?             
                ind_scores_i = scores_i[image_ind]
                ind_scores_j = scores_j[image_ind]

                matches_0 = matches_pad_0[image_ind, 0:num_points_0[image_ind]]
                matches_1 = matches_pad_1[image_ind, 0:num_points_1[image_ind]]

                ###match score
                has_match_i = (matches_0 != -1)
                has_match_j = (matches_1 != -1)

                #doing both way
                matched_inds_i = torch.arange(matches_0.shape[0])[has_match_i]
                matched_inds_j = matches_0[has_match_i]

                nsi_0 = new_seg_inds_0[image_ind]
                nsi_1 = new_seg_inds_1[image_ind]

                _, w_0 = sub_ims_0[image_ind].shape[-2:]
                _, w_1 = sub_ims_1[image_ind].shape[-2:]
                
                mnsi_0 = (nsi_0[matched_inds_i, 1] * w_0) + nsi_0[matched_inds_i, 0]
                mnsi_1 = (nsi_1[matched_inds_j, 1] * w_1) + nsi_1[matched_inds_j, 0]

                match_matrix = torch.zeros_like(ind_scores_i)
                match_matrix[mnsi_0, mnsi_1] = 1.0
                matched_scores = ind_scores_i * match_matrix + ind_scores_j * match_matrix
                ###

                ###unmatched score
                has_unmatched_i = ((~has_match_i))
                has_unmatched_j = ((~has_match_j))

                unmatched_inds_i = torch.arange(matches_0.shape[0])[has_unmatched_i]
                unmatched_inds_j = torch.arange(matches_1.shape[0])[has_unmatched_j]

                umnsi_0 = (nsi_0[unmatched_inds_i, 1] * w_0) + nsi_0[unmatched_inds_i, 0]
                umnsi_1 = (nsi_1[unmatched_inds_j, 1] * w_1) + nsi_1[unmatched_inds_j, 0]

                unmatched_i_matrix = torch.zeros_like(ind_scores_i)
                unmatched_i_matrix[umnsi_0, ind_scores_i.shape[-1] - 1] = 1.0
                unmatched_i_scores = ind_scores_i * unmatched_i_matrix

                unmatched_j_matrix = torch.zeros_like(ind_scores_j)
                unmatched_j_matrix[ind_scores_j.shape[-2] - 1, umnsi_1] = 1.0
                unmatched_j_scores = ind_scores_j * unmatched_j_matrix
                ###
 
                num_matches = matched_inds_i.shape[0]
                num_unmatched_i = unmatched_inds_i.shape[0]
                num_unmatched_j = unmatched_inds_j.shape[0]

                if num_matches > 0:
                    match_loss = torch.sum(matched_scores) / num_matches
                else:
                    match_loss = 0

                if num_unmatched_i > 0:
                    unmatch_i_loss = torch.sum(unmatched_i_scores) / num_unmatched_i
                else:
                    unmatch_i_loss = 0
                
                if num_unmatched_j > 0:
                    unmatch_j_loss = torch.sum(unmatched_j_scores) / num_unmatched_j
                else:
                    unmatch_j_loss = 0

                total_match_loss = -0.5*(match_loss + opt.unmatch_scale*unmatch_i_loss + opt.unmatch_scale*unmatch_j_loss)

                loss.append(total_match_loss)

            loss = torch.mean(torch.stack((loss)))
            feature_optimizer.zero_grad()
            bin_optimizer.zero_grad()
            transformer_optimizer.zero_grad()
            loss.backward()
            feature_optimizer.step()
            bin_optimizer.step()
            transformer_optimizer.step()

            #for step_sched in range(num_images):
            if True:
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

    parser.add_argument('--transformer_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    
    parser.add_argument('--unmatch_scale', type=float, default=1.0)

    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--trans_lr', type=float, default=3.5e-6)
    parser.add_argument('--conv_lr', type=float, default=1e-4)
    parser.add_argument('--bin_lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--log_checkpoints', type=int, default=50)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--min_dim', type=int, default=32)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)