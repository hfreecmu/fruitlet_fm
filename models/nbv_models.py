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
