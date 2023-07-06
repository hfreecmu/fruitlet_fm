import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

from data.dataloader import get_data_loader
from models.nbv_models import TransformerAssociator, prep_feature_data
from utils.torch_utils import save_checkpoint, plot_loss

def train(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.window_length,
                                 opt.num_points,
                                 opt.batch_size, opt.shuffle)
    
    width, height = opt.width, opt.height
    dims = [32, 32, 64, 64, 128]
    strides = [2, 2, 2, 2, 2]

    transformer = TransformerAssociator(dims, strides,
                                        opt.transformer_layers, dims[-1], opt.dim_feedforward,
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

    loss_array = []
    step_num = 0
    while step_num < opt.num_steps:
        for _, data in enumerate(dataloader):
            if (step_num % opt.log_steps) == 0:
                losses = 0
                num_losses = 0

            _, _, bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, is_mask_0, is_mask_1, _, _, matches_0, matches_1, is_aug = data

            num_images = bgrs_0.shape[0]

            bgrs_0, positional_encodings_0 = prep_feature_data(seg_inds_0, bgrs_0, dims[-1], 
                                                               width, height, opt.device)
            bgrs_1, positional_encodings_1 = prep_feature_data(seg_inds_1, bgrs_1, dims[-1], 
                                                               width, height, opt.device)

            x_0 = (bgrs_0, positional_encodings_0, is_mask_0)
            x_1 = (bgrs_1, positional_encodings_1, is_mask_1)

            scores = transformer(x_0, x_1)

            loss = []
            #this works because masks were appended
            for image_ind in range(num_images):
                ind_scores = scores[image_ind]
                
                #this accounts for mask
                has_match_i = (matches_0[image_ind] != -1)
                has_match_j = (matches_1[image_ind] != -1)

                #only doing once as two way
                matched_inds_i = torch.arange(matches_0[image_ind].shape[0])[has_match_i]
                matched_inds_j = matches_0[image_ind, has_match_i]

                is_mask_i = is_mask_0[image_ind]
                is_mask_j = is_mask_1[image_ind]

                has_unmatched_i = ((~has_match_i) & (~is_mask_i))
                has_unmatched_j = ((~has_match_j) & (~is_mask_j))

                unmatched_inds_i = torch.arange(matches_0[image_ind].shape[0])[has_unmatched_i]
                unmatched_inds_j = torch.arange(matches_1[image_ind].shape[0])[has_unmatched_j]

                matched_scores = ind_scores[matched_inds_i, matched_inds_j]
                unmatched_i_scores = opt.unmatch_scale*ind_scores[unmatched_inds_i, ind_scores.shape[1] - 1]
                unmatched_j_scores = opt.unmatch_scale*ind_scores[ind_scores.shape[0] - 1, unmatched_inds_j]

                loss_scores = torch.concatenate((matched_scores, unmatched_i_scores, unmatched_j_scores))

                ind_loss = torch.mean(-loss_scores)
                loss.append(ind_loss)

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
    parser.add_argument('--window_length', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=2000)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=10)
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

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)