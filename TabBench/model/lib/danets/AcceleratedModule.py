import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://github.com/WhatAShot/DANet/blob/main/model/AcceleratedModule.py
class AcceleratedCreator(object):
    def __init__(self, input_dim, base_out_dim, k):
        super(AcceleratedCreator, self).__init__()
        self.input_dim = input_dim
        self.base_out_dim = base_out_dim
        self.computer = Extractor(k)

    def __call__(self, network):
        network.init_layer = self.extract_module(network.init_layer, self.input_dim, self.input_dim)
        for i in range(len(network.layer)):
            network.layer[i] = self.extract_module(network.layer[i], self.base_out_dim, self.input_dim)
        return network

    def extract_module(self, basicblock, base_input_dim, fix_input_dim):
        basicblock.conv1 = self.computer(basicblock.conv1, base_input_dim, self.base_out_dim // 2)
        basicblock.conv2 = self.computer(basicblock.conv2, self.base_out_dim // 2, self.base_out_dim)
        basicblock.downsample = self.computer(basicblock.downsample._modules['1'], fix_input_dim, self.base_out_dim)
        return basicblock


class Extractor(object):
    def __init__(self, k):
        super(Extractor, self).__init__()
        self.k = k

    @staticmethod
    def get_parameter(abs_layer):
        bn = abs_layer.bn.bn
        alpha, beta, eps = bn.weight.data, bn.bias.data, bn.eps  # [240]
        mu, var = bn.running_mean.data, bn.running_var.data
        locality = abs_layer.masker
        sparse_weight = locality.smax(locality.weight.data)  # 6, 10

        feat_pro = abs_layer.fc
        process_weight = feat_pro.weight.data  # ([240, 10, 1])  [240]
        process_bias = feat_pro.bias.data if feat_pro.bias is not None else None
        return alpha, beta, eps, mu, var, sparse_weight, process_weight, process_bias

    @staticmethod
    def compute_weights(a, b, eps, mu, var, sw, pw, pb, base_input_dim, base_output_dim, k):
        """
        standard shape: [path, output_shape, input_shape, branch]
        """
        sw_ = sw[:, None, :, None]
        pw_ = pw.view(k, 2, base_output_dim, base_input_dim).permute(0, 2, 3, 1)
        if pb is not None:
            pb_ = pb.view(k, 2, base_output_dim).permute(0, 2, 1)[:, :, None, :]
        a_ = a.view(k, 2, base_output_dim).permute(0, 2, 1)[:, :, None, :]
        b_ = b.view(k, 2, base_output_dim).permute(0, 2, 1)[:, :, None, :]
        mu_ = mu.view(k, 2, base_output_dim).permute(0, 2, 1)[:, :, None, :]
        var_ = var.view(k, 2, base_output_dim).permute(0, 2, 1)[:, :, None, :]

        W = sw_ * pw_
        if pb is not None:
            mu_ = mu_ - pb_
        W = a_ / (var_ + eps).sqrt() * W
        B = b_ - a_ / (var_ + eps).sqrt() * mu_

        W_att = W[..., 0]
        B_att = B[..., 0]

        W_fc = W[..., 1]
        B_fc = B[..., 1]

        return W_att, W_fc, B_att.squeeze(), B_fc.squeeze()

    def __call__(self, abslayer, input_dim, base_out_dim):
        (a, b, e, m, v, s, pw, pb) = self.get_parameter(abslayer)
        wa, wf, ba, bf = self.compute_weights(a, b, e, m, v, s, pw, pb, input_dim, base_out_dim, self.k)
        return CompressAbstractLayer(wa, wf, ba, bf)


class CompressAbstractLayer(nn.Module):
    def __init__(self, att_w, f_w, att_b, f_b):
        super(CompressAbstractLayer, self).__init__()
        self.att_w = nn.Parameter(att_w)
        self.f_w = nn.Parameter(f_w)
        self.att_bias = nn.Parameter(att_b[None, :, :])
        self.f_bias = nn.Parameter(f_b[None, :, :])

    def forward(self, x):
        att = torch.sigmoid(torch.einsum('poi,bi->bpo', self.att_w, x) + self.att_bias) # (2 * i + 2) * p * o
        y = torch.einsum('poi,bi->bpo', self.f_w, x) + self.f_bias # (2 * i + 1) * p * o
        return torch.sum(F.relu(att * y), dim=-2, keepdim=False) # 3 * p * o


# if __name__ == '__main__':
#     import torch.optim as optim
#     from DANet import AbstractLayer

#     input_feat = torch.rand((8, 10), requires_grad=False)
#     loss_function = nn.L1Loss()
#     target = torch.rand((8, 20), requires_grad=False)
#     abs_layer = AbstractLayer(base_input_dim=10, base_output_dim=20, k=6, virtual_batch_size=4, bias=False)
#     y_ = abs_layer(input_feat)
#     optimizer = optim.SGD(abs_layer.parameters(), lr=0.3)
#     abs_layer.zero_grad()
#     loss_function(y_, target).backward()
#     optimizer.step()

#     abs_layer = abs_layer.eval()
#     y = abs_layer(input_feat)
#     computer = Extractor(k=6)
#     (a, b, e, m, v, s, pw, pb) = computer.get_parameter(abs_layer)
#     wa, wf, ba, bf = computer.compute_weights(a, b, e, m, v, s, pw, pb, 10, 20, 6)
#     acc_abs = CompressAbstractLayer(wa, wf, ba, bf)
#     y2 = acc_abs(input_feat)