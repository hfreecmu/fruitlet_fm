import torch.nn as nn
import torch
import numpy as np
from models.MyTransformerEncoder import TransformerEncoderLayer

### OT
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
###

##### Encoder
#TODO bias true or false?
#TODO relu or leaky relu?
def conv(dims, strides, kernel_size, padding):
    layers = []
    
    prev_dim = 3
    for i in range(len(dims) - 1):
        conv = nn.Conv2d(prev_dim, dims[i], kernel_size=kernel_size, stride=strides[i], padding=padding)
        relu = nn.ReLU()
        layers.append(conv)
        layers.append(relu)

        prev_dim = dims[i]

    final_conv = nn.Conv2d(prev_dim, dims[-1], kernel_size=kernel_size, stride=strides[-1], padding=padding)
    layers.append(final_conv)

    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, dims, strides, kernel_size=3, padding=1):
        super(Encoder, self).__init__()
    
        self.network = conv(dims, strides, kernel_size, padding)
    
    def forward(self, x):
        x = self.network(x)
        return x
#####

##### Transformer
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
def alt_transformer(n_layers, d_model, dim_feedforward, nhead=8, batch_first=True):
    layers = []
    for i in range(n_layers):
        self_layer = TransformerEncoderLayer(name='self', d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                             batch_first=batch_first)
        cross_layer = TransformerEncoderLayer(name='cross', d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                              batch_first=batch_first)

        layers.append(self_layer)
        layers.append(cross_layer)

    return mySequential(*layers)
    
class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, dim_feedforward, batch_first=True):
        super(Transformer, self).__init__()

        self.network = alt_transformer(n_layers, d_model, dim_feedforward, batch_first)
        
    def forward(self, src0, src1):
        src0, src1 = self.network(src0, src1)
        return src0, src1
#####

##### Main module
class TransformerAssociator(nn.Module):
    def __init__(self, dims, strides, 
                 n_layers, d_model, dim_feedforward,
                 dual_softmax,
                 sinkhorn_iterations,
                 device):
        super(TransformerAssociator, self).__init__()

        self.encoder = Encoder(dims, strides)
        self.transformer = Transformer(n_layers, d_model, dim_feedforward)
        self.dual_softmax = dual_softmax
        self.sinkhorn_iterations = sinkhorn_iterations
        self.device = device

        if not self.dual_softmax:
            bin_score = torch.nn.Parameter(torch.tensor(1.))
            self.register_parameter('bin_score', bin_score)
        else:
            raise RuntimeError('dual softmax not supported yet')
            dustbin_scores = torch.nn.Parameter(torch.tensor([1.]*d_model))
            self.register_parameter('bin_score', dustbin_scores)

    def forward(self, x_0, x_1):
        bgrs_0, positional_encodings_0, is_mask_0 = x_0
        bgrs_1, positional_encodings_1, is_mask_1 = x_1

        num_images, num_points, dim = positional_encodings_0.shape

        num_images = bgrs_0.shape[0]
        scores = []
        for image_ind in range(num_images):
            mask_inds_0 = is_mask_0[image_ind]
            mask_inds_1 = is_mask_1[image_ind]

            #indexing these by mask_inds caused memory leak
            features_0 = self.encoder(bgrs_0[image_ind]).reshape((-1, dim)).unsqueeze(0)
            features_1 = self.encoder(bgrs_1[image_ind]).reshape((-1, dim)).unsqueeze(0)

            pe_0 = positional_encodings_0[image_ind:image_ind+1]
            pe_1 = positional_encodings_1[image_ind:image_ind+1]

            src_0 = features_0 + pe_0
            src_1 = features_1 + pe_1

            desc_0, desc_1 = self.transformer(src_0[:, ~mask_inds_0], src_1[:, ~mask_inds_1])

            #used to be torch einsum thing
            ot_score = torch.matmul(desc_0[0], desc_1[0].T).unsqueeze(0)
            ot_score = ot_score / dim**.5
            ot_score = log_optimal_transport(ot_score, self.bin_score, iters=self.sinkhorn_iterations)

            scores.append(ot_score.squeeze(0))

        return scores
    
# def create_mask(num_points, query_mask_inds, key_mask_inds):
#     mask = torch.zeros((num_points, num_points), dtype=torch.bool)
#     mask[query_mask_inds, :] = True
#     mask[:, key_mask_inds] = True

#     return mask

#####

##### Utils
#assumed positions are normalized between -1 and 1'
#R^2 to R^(2 + 4*L)
#actuatlly,to R^(4*L) as  I will remove x and y
#paper read uses L = 10
def positional_encoder(p, L, include_orig=False):
    x = p[:, :, 0]
    y = p[:, :, 1]

    x_out = torch.zeros((x.shape[0], x.shape[1], 1 + 2*L), dtype=p.dtype)
    y_out = torch.zeros((y.shape[0], y.shape[1], 1 + 2*L), dtype=p.dtype)

    x_out[:, :, 0] = x
    y_out[:, :, 0] = y

    for k in range(L):
        ykcos = torch.cos(2**k * np.pi * y)
        yksin = torch.sin(2**k * np.pi * y)

        xkcos = torch.cos(2**k * np.pi * x)
        xksin = torch.sin(2**k * np.pi * x)

        y_out[:, :, 2*k + 1] = ykcos
        y_out[:, :, 2*k + 2] = yksin

        x_out[:, :, 2*k + 1] = xkcos
        x_out[:, :, 2*k + 2] = xksin

    enc_out = torch.concatenate((x_out, y_out), axis=2)

    if not include_orig:
        enc_out = enc_out[:, :, 2:]

    return enc_out  

#assumes inputs are unsqueezed
def prep_feature_data(seg_inds, bgrs, dim, width, height, device):
    bgrs = bgrs.float() / 255
    bgrs = bgrs.to(device)

    seg_inds_float = seg_inds.float()
    seg_inds_float[:, :, 0] = (2*seg_inds_float[:, :, 0] - width) / width
    seg_inds_float[:, :, 1] = (2*seg_inds_float[:, :, 1] - height) / height

    positional_encodings = positional_encoder(seg_inds_float, dim//4)
    positional_encodings = positional_encodings.to(device)

    return bgrs, positional_encodings

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def extract_matches(scores, match_threshold, use_dustbin):
    if use_dustbin:
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    else:
        max0, max1 = scores[:, :, :].max(2), scores[:, :, :].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    if use_dustbin:
        indices_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    else:
        indices_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))[:, 0:-1]
        indices_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))[:, 0:-1]
        indices_0[indices_0 == (scores.shape[2] - 1)] = -1
        indices_1[indices_1 == (scores.shape[1] - 1)] = -1

        mscores0 = mscores0[:, 0:-1]
        mscores1 = mscores1[:, 0:-1]

    return indices_0, indices_1, mscores0, mscores1
###