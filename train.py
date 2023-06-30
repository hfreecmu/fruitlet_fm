import argparse
import torch
import torch.optim as optim

from data.dataloader import get_data_loader
from models.nbv_models import load_feature_encoder, positional_encoder, Transformer

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
                                 opt.max_segs, opt.batch_size, 
                                 opt.shuffle)
    
    width, height = opt.width, opt.height
    dims = [16, 32, 64, 128]
    feature_encoder = load_feature_encoder(dims, device=opt.device)

    transformer = Transformer(n_layers=opt.transformer_layers, d_model=dims[-1]).to(opt.device)

    ###optimizers
    feature_optimizer = optim.Adam(feature_encoder.parameters(), opt.lr)
    transformer_optimizer = optim.Adam(transformer.parameters(), opt.lr)
    ###

    for epoch in range(opt.num_epochs):
        for batch_num, data in enumerate(dataloader):
            _, _, bgrs_0, bgrs_1, seg_inds_0, seg_inds_1, is_val = data

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

            features_0 = torch.zeros((num_images, opt.max_segs, dims[-1])).float().to(opt.device)
            features_1 = torch.zeros((num_images, opt.max_segs, dims[-1]), dtype=float).float().to(opt.device)
            val_indices_array = []
            #TODO make one cnn call if speed is an issue?
            for image_ind in range(num_images):
                val_indices = torch.nonzero(is_val[image_ind])[:, 0]
                features_0[image_ind, val_indices] = feature_encoder(bgrs_0[image_ind, val_indices]).reshape((val_indices.shape[0], dims[-1]))
                features_1[image_ind, val_indices] = feature_encoder(bgrs_1[image_ind, val_indices]).reshape((val_indices.shape[0], dims[-1]))
                val_indices_array.append(val_indices)

            src_0 = features_0 + positional_encodings_0
            src_1 = features_1 + positional_encodings_1

            desc_0, desc_1 = transformer(src_0, src_1)

            desc_0 = torch.permute(desc_0, (0, 2, 1))
            desc_1 = torch.permute(desc_1, (0, 2, 1))

            #TODO can use other method as described in 
            #https://arxiv.org/pdf/2104.00680.pdf?
            scores = torch.einsum('bdn,bdm->bnm', desc_0, desc_1)
            scores = scores / dims[-1]**.5
            scores = log_optimal_transport(scores, transformer.bin_score, iters=opt.sinkhorn_iterations)

            diag_scores = torch.diagonal(scores[:, 0:-1, 0:-1], dim1=-2, dim2=-1)

            losses = []
            for image_ind in range(num_images):
                val_indices = val_indices_array[image_ind]
                diag_score = diag_scores[image_ind]
                loss = torch.sum(-diag_score)
                losses.append(loss)


            loss = torch.mean(torch.stack((losses)))
            feature_optimizer.zero_grad()
            transformer_optimizer.zero_grad()
            loss.backward()
            feature_optimizer.step()
            transformer_optimizer.step()

            print('loss is: ', loss.item())            
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--segmentations_dir', required=True)
    parser.add_argument('--window_length', type=int, default=16)
    parser.add_argument('--max_segs', type=int, default=2000)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--sinkhorn_iterations', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=1080)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)