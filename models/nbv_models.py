import torch.nn as nn
import torch
import numpy as np
from models.MyTransformerEncoder import TransformerEncoderLayer

#bias true or false?
#relu or leaky relu?
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

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

def alt_transformer(n_layers, d_model, nhead=8):
    layers = []
    for i in range(n_layers):
        self_layer = TransformerEncoderLayer(name='self', d_model=d_model, nhead=nhead, dim_feedforward=1024)
        cross_layer = TransformerEncoderLayer(name='cross', d_model=d_model, nhead=nhead, dim_feedforward=1024)
        # self_layer = TransformerEncoderLayer(name='self', d_model=d_model, nhead=nhead)
        # cross_layer = TransformerEncoderLayer(name='cross', d_model=d_model, nhead=nhead)

        layers.append(self_layer)
        layers.append(cross_layer)

    return mySequential(*layers)

class Transformer(nn.Module):
    def __init__(self, n_layers, d_model):
        super(Transformer, self).__init__()

        self.network = alt_transformer(n_layers, d_model)
        
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # dustbin_i_score = torch.nn.Parameter(torch.tensor([1.]*d_model))
        # self.register_parameter('bin_i_score', dustbin_i_score)

        # dustbin_j_score = torch.nn.Parameter(torch.tensor([1.]*d_model))
        # self.register_parameter('bin_j_score', dustbin_j_score)

    def forward(self, src0, src1):
        src0, src1 = self.network(src0, src1)
        return src0, src1

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

def load_feature_encoder(dims, strides, device, eval=False):
    encoder = Encoder(dims, strides)

    if eval:
        encoder.eval()

    encoder.to(device)

    return encoder

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def extract_matches(scores, match_threshold):
    # max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    max0, max1 = scores[:, :, :].max(2), scores[:, :, :].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    # indices_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    # indices_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    indices_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))[:, 0:-1]
    indices_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))[:, 0:-1]
    indices_0[indices_0 == (scores.shape[2] - 1)] = -1
    indices_1[indices_1 == (scores.shape[1] - 1)] = -1

    return indices_0, indices_1, mscores0[:, 0:-1], mscores1[:, 0:-1]

#assumes inputs are unsqueezed
def prep_feature_data(seg_inds, bgrs, dim, width, height, device):
    bgrs = bgrs.float() / 255
    bgrs = bgrs.to(device)

    seg_inds = seg_inds.float()
    seg_inds[:, :, 0] = (2*seg_inds[:, :, 0] - width) / width
    seg_inds[:, :, 1] = (2*seg_inds[:, :, 1] - height) / height

    positional_encodings = positional_encoder(seg_inds, dim//4)
    positional_encodings = positional_encodings.to(device)

    return bgrs, positional_encodings
