import torch.nn as nn
import torch
import numpy as np
from models.MyTransformerEncoder import TransformerEncoderLayer
import torch.nn.functional as F

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
def conv(dims, strides, pools, kernel_size, padding, orig_dim):
    layers = []
    
    prev_dim = orig_dim
    for i in range(len(dims) - 1):
        conv = nn.Conv2d(prev_dim, dims[i], kernel_size=kernel_size, stride=strides[i], padding=padding)
        #relu = nn.ReLU()
        relu = nn.LeakyReLU(0.01)
        layers.append(conv)

        if pools[i]:
            max_pool = nn.MaxPool2d(2, stride=2)
            layers.append(max_pool)

        layers.append(relu)


        prev_dim = dims[i]

    final_conv = nn.Conv2d(prev_dim, dims[-1], kernel_size=kernel_size, stride=strides[-1], padding=padding)
    layers.append(final_conv)

    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, dims, strides, pools, orig_dim, kernel_size=3, padding=1):
        super(Encoder, self).__init__()
    
        self.network = conv(dims, strides, pools, kernel_size, padding, orig_dim)
    
    def forward(self, x):
        x = self.network(x)
        return x
#####

### mlp
def fc(in_dim, out_dims):
    layers = []
    prev_dim = in_dim
    for i in range(len(out_dims) - 1):
        fc = nn.Linear(prev_dim, out_dims[i])
        #relu = nn.ReLU()
        relu = nn.LeakyReLU(0.01)

        layers.append(fc)
        layers.append(relu)

        prev_dim = out_dims[i]

    final_fc = nn.Linear(prev_dim, out_dims[-1])
    layers.append(final_fc)

    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(MLP, self).__init__()

        self.network = fc(in_dim, out_dims)

    def forward(self, x):
        x = self.network(x)
        return x


###

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
    def __init__(self, n_layers, d_model, dim_feedforward, mlp_layers, batch_first=True):
        super(Transformer, self).__init__()

        if not mlp_layers[-1] == 1:
            raise RuntimeError('mlp layers must end with -1')

        self.network = alt_transformer(n_layers, d_model, dim_feedforward, batch_first)
        self.mlp = MLP(d_model, mlp_layers)

    def forward(self, src0, src1):
        src0, src1 = self.network(src0, src1)

        is_feature_0 = self.mlp(src0)
        is_feature_1 = self.mlp(src1)
        
        return src0, src1, is_feature_0, is_feature_1
#####

##### Main module
class TransformerAssociator(nn.Module):
    def __init__(self, dims, strides, 
                 n_layers, d_model, dim_feedforward,
                 mlp_layers, pools, 
                 dual_softmax,
                 sinkhorn_iterations,
                 device):
        super(TransformerAssociator, self).__init__()

        dims, kpts_dims = dims
        strides, kpts_strides = strides
        pools, kpts_pools = pools

        self.kpts_encoder = Encoder(kpts_dims, kpts_strides, kpts_pools, 42)
        self.encoder = Encoder(dims, strides, pools, 3)
        self.transformer = Transformer(n_layers, d_model, dim_feedforward, mlp_layers)
        self.dual_softmax = dual_softmax
        self.sinkhorn_iterations = sinkhorn_iterations
        self.device = device

        if not self.dual_softmax:
            bin_score = torch.nn.Parameter(torch.tensor(1.))
            self.register_parameter('bin_score', bin_score)
        else:
            #raise RuntimeError('dual softmax not supported yet')
            dustbin_scores = torch.nn.Parameter(torch.tensor([1.]*d_model))
            self.register_parameter('bin_score', dustbin_scores)

    def forward(self, x_0, x_1):
        bgrs_0, positional_encodings_0 = x_0
        bgrs_1, positional_encodings_1 = x_1

        num_images = len(bgrs_0)

        scores = []
        is_features_0 = []
        is_features_1 = []
        for image_ind in range(num_images):

            #indexing these by mask_inds caused memory leak
            features_0 = self.encoder(bgrs_0[image_ind])
            features_1 = self.encoder(bgrs_1[image_ind])

            h_0, w_0 = features_0.shape[-2:]
            h_1, w_1 = features_1.shape[-2:]

            pe_0 = self.kpts_encoder(positional_encodings_0[image_ind])
            pe_1 = self.kpts_encoder(positional_encodings_1[image_ind])

            src_0 = features_0 + pe_0
            src_1 = features_1 + pe_1

            src_0 = torch.permute(src_0, (0, 2, 3, 1)).reshape((1, -1, src_0.shape[1]))
            src_1 = torch.permute(src_1, (0, 2, 3, 1)).reshape((1, -1, src_1.shape[1]))

            desc_0, desc_1, is_feature_0, is_feature_1 = self.transformer(src_0, src_1)

            is_feature_0 = is_feature_0.reshape((h_0, w_0))
            is_feature_1 = is_feature_1.reshape((h_1, w_1))

            size_0_orig = (bgrs_0[image_ind].shape[-2], bgrs_0[image_ind].shape[-1])
            size_1_orig = (bgrs_1[image_ind].shape[-2], bgrs_1[image_ind].shape[-1])

            desc_0 = desc_0.reshape((1, h_0, w_0, -1))
            desc_1 = desc_1.reshape((1, h_1, w_1, -1))

            desc_0 = torch.permute(F.interpolate(torch.permute(desc_0, (0, 3, 1, 2)), size=(size_0_orig), mode='bilinear'), (0, 2, 3, 1))
            desc_1 = torch.permute(F.interpolate(torch.permute(desc_1, (0, 3, 1, 2)), size=(size_1_orig), mode='bilinear'), (0, 2, 3, 1))

            desc_0 = desc_0.reshape((1, -1, desc_0.shape[-1]))
            desc_1 = desc_1.reshape((1, -1, desc_1.shape[-1]))

            if not self.dual_softmax:
                #used to be torch einsum thing
                ot_score = torch.matmul(desc_0[0], desc_1[0].T).unsqueeze(0)
                ot_score = ot_score / desc_0.shape[-1]**.5
                ot_score = log_optimal_transport(ot_score, self.bin_score, iters=self.sinkhorn_iterations)
            else:
                desc_0 = torch.cat((desc_0, self.bin_score.unsqueeze(0).unsqueeze(0)), dim=1)
                desc_1 = torch.cat((desc_1, self.bin_score.unsqueeze(0).unsqueeze(0)), dim=1)
                ot_score = torch.matmul(desc_0[0], desc_1[0].T)
                #
                #ot_score = ot_score / 0.1
                #ot_score = ot_score / desc_0.shape[-1]**.5
                ot_score = ot_score / desc_0.shape[-1]
                ot_score = F.log_softmax(ot_score, dim=0) + F.log_softmax(ot_score, dim=1)
                ot_score = ot_score.unsqueeze(0)

            scores.append(ot_score.squeeze(0))
            is_features_0.append(is_feature_0)
            is_features_1.append(is_feature_1)


        return scores, is_features_0, is_features_1
    
# def create_mask(num_points, query_mask_inds, key_mask_inds):
#     mask = torch.zeros((num_points, num_points), dtype=torch.bool)
#     mask[query_mask_inds, :] = True
#     mask[:, key_mask_inds] = True

#     return mask

#####

##### Utils
#assumes squeezed inputs
#assumed positions are normalized between -1 and 1'
#R^2 to R^(2 + 4*L)
#actuatlly,to R^(4*L) as  I will remove x and y
#paper read (nerf) uses L = 10
def positional_encoder(p, L=10, include_orig=False):
    x = p[0]
    y = p[1]

    x_out = torch.zeros((1 + 2*L, x.shape[0], x.shape[1]), dtype=p.dtype)
    y_out = torch.zeros((1 + 2*L, y.shape[0], y.shape[1]), dtype=p.dtype)

    x_out[0] = x
    y_out[0] = y

    for k in range(L):
        ykcos = torch.cos(2**k * np.pi * y)
        yksin = torch.sin(2**k * np.pi * y)

        xkcos = torch.cos(2**k * np.pi * x)
        xksin = torch.sin(2**k * np.pi * x)

        y_out[2*k + 1] = ykcos
        y_out[2*k + 2] = yksin

        x_out[2*k + 1] = xkcos
        x_out[2*k + 2] = xksin

    enc_out = torch.concatenate((x_out, y_out), axis=0)

    if not include_orig:
        enc_out = enc_out[2:]

    return enc_out  

#assumes SQUEEZED
def prep_feature_data(torch_im, seg_inds, matches, norm_dim, device):
    x0 = torch.min(seg_inds[:, 0])
    x1 = torch.max(seg_inds[:, 0])
    y0 = torch.min(seg_inds[:, 1])
    y1 = torch.max(seg_inds[:, 1])

    width = x1 + 1 - x0
    height = y1 + 1 - y0

    bgrs = 2*(torch_im[:, y0:y1+1, x0:x1+1].float() / 255) - 1
    bgrs = bgrs.to(device)

    x_pts = torch.arange(bgrs.shape[-1]).repeat(bgrs.shape[-2], 1) - width//2 #+ x0
    #x_pts = (x_pts - width) / width
    x_pts = 2*x_pts / norm_dim

    y_pts = torch.arange(bgrs.shape[-2]).repeat(bgrs.shape[-1], 1).T - height//2 #+ y0
    #y_pts = (y_pts - height)
    y_pts = 2*y_pts / norm_dim
    
    kpts = torch.stack((x_pts, y_pts), dim=0)

    positional_encodings = positional_encoder(kpts, include_orig=True)
    positional_encodings = positional_encodings.to(device)

    #feature keypoint, not same as kpts above
    if matches is not None:
        has_match = (matches != -1)
        matched_seg_inds = seg_inds[has_match]
        try:
            is_keypoint = torch.zeros((bgrs.shape[-2], bgrs.shape[-1])).float()
            if (matched_seg_inds.shape[0] > 0):  
                is_keypoint[matched_seg_inds[:, 1] - y0, matched_seg_inds[:, 0] - x0] = 1.0
            is_keypoint = is_keypoint.to(device)
        except Exception as e:
            breakpoint()
    else:
        is_keypoint = None

    new_seg_inds = torch.clone(seg_inds)
    new_seg_inds[:, 0] -= x0
    new_seg_inds[:, 1] -= y0

    return bgrs, positional_encodings, is_keypoint, x0, y0, new_seg_inds

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