import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

from data.dataloader import get_data_loader
from models.nbv_models import load_feature_encoder, positional_encoder, Transformer
from utils.torch_utils import save_checkpoint

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)


    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def train(opt):
    dataloader = get_data_loader(opt.images_dir, opt.segmentations_dir,
                                 opt.window_length,
                                 opt.num_good, opt.num_bad,
                                 opt.batch_size, opt.shuffle)
    
    width, height = opt.width, opt.height
    dims = [32, 32, 64, 64, 128, 128]
    strides = [2, 1, 2, 1, 2, 2]
    feature_encoder = load_feature_encoder(dims, strides, device=opt.device)

    transformer = Transformer(n_layers=opt.transformer_layers, d_model=dims[-1]).to(opt.device)

    ###optimizers
    feature_optimizer = optim.Adam(feature_encoder.parameters(), opt.conv_lr)
    transformer_optimizer = optim.Adam(transformer.parameters(), opt.trans_lr)
    ###

    for epoch in range(opt.num_epochs):
        losses = 0
        num_losses = 0
        for batch_num, data in enumerate(dataloader):
            _, _, bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, matches_0, matches_1 = data
            
            num_images = bgrs_0.shape[0]

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

            loss = []
            for image_ind in range(num_images):
                ind_scores = scores[image_ind]

                has_match_i = (matches_0[image_ind] != -1)
                has_match_j = (matches_1[image_ind] != -1)

                #only doing once as two way
                #TODO could double weigth of proper match and do two way?
                matched_inds_i = torch.arange(matches_0[image_ind].shape[0])[has_match_i]
                matched_inds_j = matches_0[image_ind, has_match_i]

                unmatched_inds_i = torch.arange(matches_0[image_ind].shape[0])[~has_match_i]
                unmatched_inds_j = torch.arange(matches_1[image_ind].shape[0])[~has_match_j]

                matched_scores = 2*ind_scores[matched_inds_i, matched_inds_j]
                unmatched_i_scores = 0.05*ind_scores[unmatched_inds_i, matches_0[image_ind].shape[0]]
                unmatched_j_scores = 0.05*ind_scores[matches_1[image_ind].shape[0], unmatched_inds_j]

                loss_scores = torch.concatenate((matched_scores, unmatched_i_scores, unmatched_j_scores))
                #loss_scores = matched_scores

                ind_loss = torch.mean(-loss_scores)
                loss.append(ind_loss)

            loss = torch.mean(torch.stack((loss)))
            feature_optimizer.zero_grad()
            transformer_optimizer.zero_grad()
            loss.backward()
            feature_optimizer.step()
            transformer_optimizer.step()

            losses += loss.item()
            num_losses += 1

        if num_losses == 0:
            epoch_loss = 'NA'
        else:
            epoch_loss = str(losses/num_losses)
        print('loss for epoch: ', epoch, ' is: ', epoch_loss)

        if (epoch + 1) % 20 == 0:
            save_checkpoint(epoch, opt.checkpoint_dir, feature_encoder, transformer) 

    save_checkpoint(epoch, opt.checkpoint_dir, feature_encoder, transformer)   
    print('Done')         
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--segmentations_dir', required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--window_length', type=int, default=8)
    parser.add_argument('--num_good', type=int, default=1000)
    parser.add_argument('--num_bad', type=int, default=1000)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dual_softmax', action='store_true')
    parser.add_argument('--sinkhorn_iterations', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--trans_lr', type=float, default=1e-4)
    parser.add_argument('--conv_lr', type=float, default=1e-3)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)