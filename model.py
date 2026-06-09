import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     spectral_norm(torch.nn.Linear(_in, _hidden)),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2, inplace=True)
                                 ))
        self.tail = torch.nn.Sequential(torch.nn.Linear(_hidden, 1, bias=False),
                                        torch.nn.Sigmoid())
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            self.layers.add_module(f"{i}bn", torch.nn.BatchNorm1d(_out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        return x


class PatchMaker:
    def __init__(self, patchsize, stride=1):
        self.patchsize = patchsize
        self.stride = stride
        self.unfolder = nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=(self.patchsize - 1) // 2)

    def patchify(self, features):
        b, c, h, w = features.shape
        unfolded_features = self.unfolder(features)
        patches_flat = unfolded_features.transpose(1, 2)
        h_p = (h + 2 * ((self.patchsize - 1) // 2) - self.patchsize) // self.stride + 1
        w_p = (w + 2 * ((self.patchsize - 1) // 2) - self.patchsize) // self.stride + 1
        return patches_flat, [h_p, w_p]

    def score(self, x):
        x = x.squeeze(-1)
        num_patches = x.shape[1]
        k_ratio = 0.03
        k = int(num_patches * k_ratio)
        if k < 1: k = 1
        topk_values, _ = torch.topk(x, k, dim=1)
        image_scores = torch.mean(topk_values, dim=1)
        return image_scores


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        mid_channel = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return scale


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w_pool = torch.mean(x, dim=2, keepdim=True)
        x_w = x_w_pool.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return a_w * a_h


class SelectiveHybridAttention(nn.Module):
    def __init__(self, dim, num_patches, reduction=16, mode='sha'):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.side = int(math.sqrt(num_patches))
        self.mode = mode

        self.se = SEBlock(dim, reduction=reduction)
        self.ca = CoordAtt(inp=dim, oup=dim, reduction=reduction)

        if self.mode == 'sha':
            self.selector = nn.Sequential(
                nn.Linear(dim, dim // reduction),
                nn.ReLU(),
                nn.Linear(dim // reduction, 2 * dim)
            )
            self.softmax = nn.Softmax(dim=1)
        else:
            self.selector = None

        self.last_avg_alpha = 0.5

    def forward(self, x_flat):
        total_len, D = x_flat.shape
        B = total_len // self.num_patches

        if self.mode == 'se_only':
            scale_final = self.se(x_flat)
            if self.training: self.last_avg_alpha = 0.0

        elif self.mode == 'ca_only':
            x_spatial = x_flat.view(B, self.num_patches, D).transpose(1, 2).view(B, D, self.side, self.side)
            scale_ca_spatial = self.ca(x_spatial)
            scale_final = scale_ca_spatial.view(B, D, -1).transpose(1, 2).reshape(-1, D)
            if self.training: self.last_avg_alpha = 1.0

        else:
            scale_se = self.se(x_flat)
            x_spatial = x_flat.view(B, self.num_patches, D).transpose(1, 2).view(B, D, self.side, self.side)
            scale_ca_spatial = self.ca(x_spatial)
            scale_ca = scale_ca_spatial.view(B, D, -1).transpose(1, 2).reshape(-1, D)

            x_reshaped = x_flat.view(B, self.num_patches, D)
            global_context = x_reshaped.mean(dim=1)
            selection_vector = self.selector(global_context).view(B, 2, D)
            selection_vector = self.softmax(selection_vector)

            alpha = selection_vector[:, 0, :]
            beta = selection_vector[:, 1, :]

            if self.training:
                self.last_avg_alpha = alpha.mean().detach().item()

            alpha = alpha.repeat_interleave(self.num_patches, dim=0)
            beta = beta.repeat_interleave(self.num_patches, dim=0)
            scale_final = alpha * scale_ca + beta * scale_se

        out = x_flat + (x_flat * scale_final)
        return out


class BN(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.bn = bottleneck

    def train_eval(self, type='train'):
        if type == 'train':
            self.bn.train()
        else:
            self.bn.eval()
        return self

    def forward(self, x):
        return self.bn(x)


class Adapter(nn.Module):
    def __init__(self, student, num_patches=0, embed_dim=0, use_se=True, no_pe=False, attention_mode='sha'):
        super().__init__()
        self.s1 = student
        self.num_patches = num_patches
        self.use_pe = (num_patches > 0) and (embed_dim > 0) and (not no_pe)

        if self.use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        if embed_dim > 0:
            print(f"Adapter: Initializing Attention with mode: {attention_mode.upper()}")
            self.attention = SelectiveHybridAttention(dim=embed_dim, num_patches=num_patches, mode=attention_mode)
        else:
            self.attention = nn.Identity()

    def train_eval(self, type='train'):
        if type == 'train':
            self.s1.train()
            if hasattr(self, 'attention') and isinstance(self.attention, nn.Module):
                self.attention.train()
        else:
            self.s1.eval()
            if hasattr(self, 'attention') and isinstance(self.attention, nn.Module):
                self.attention.eval()
        return self

    def forward(self, bn_outs):
        if self.use_pe:
            total_len, C = bn_outs.shape
            L = self.pos_embed.shape[1]
            if total_len % L == 0:
                B = total_len // L
                x = bn_outs.view(B, L, C) + self.pos_embed
                bn_outs = x.view(total_len, C)

        features = self.s1(bn_outs)
        features = self.attention(features)
        return features
