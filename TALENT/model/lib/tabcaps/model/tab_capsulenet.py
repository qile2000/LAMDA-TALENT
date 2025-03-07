import torch
from torch import nn
import torch.nn.functional as F
from TALENT.model.lib.tabcaps.model.AbstractLayer import AbstractLayer
import TALENT.model.lib.tabcaps.model.sparsemax as sparsemax

class PrimaryCapsuleGenerator(nn.Module):
    def __init__(self, num_feature, capsule_dim):
        super(PrimaryCapsuleGenerator, self).__init__()
        self.num_feature = num_feature
        self.capsule_dim = capsule_dim
        self.fc = nn.Parameter(torch.randn(num_feature, capsule_dim))

    def forward(self, x):
        out = torch.einsum('bm,md->bmd', x, self.fc)
        out = torch.cat([out, x[:, :, None]], dim=-1)

        return out.transpose(-1, -2)  # 最后的输出维度为（batch_size，capsule_dim+1，num_feature），将x和capsule之后的结果cat起来

class InferCapsule(nn.Module):
    def __init__(self, in_capsule_num, out_capsule_num, in_capsule_size, out_capsule_size, n_leaves):
        super(InferCapsule, self).__init__()
        self.in_capsule_num = in_capsule_num
        self.out_capsule_num = out_capsule_num
        self.routing_dim = out_capsule_size
        self.route_weights = nn.Parameter(torch.randn(in_capsule_num, out_capsule_num, in_capsule_size, out_capsule_size)) # (np,ns,m,n)
        self.smax = sparsemax.Entmax15(dim=-2)
        self.thread = nn.Parameter(torch.rand(1, in_capsule_num, out_capsule_num), requires_grad=True)
        self.routing_leaves = nn.Parameter(torch.rand(n_leaves, out_capsule_size))
        self.ln = nn.LayerNorm(out_capsule_size)

    @staticmethod
    def js_similarity(x, x_m):
        # x1, x2 : B, M, N, L
        dis = torch.mean((x - x_m) ** 2, dim=-1)  # B, M, N
        return dis

    def new_routing(self, priors):
        leave_hash = F.normalize(self.routing_leaves, dim=-1)
        votes = torch.sigmoid(torch.einsum('ld, bmnd->bmnl', leave_hash, priors))
        mean_cap = votes.mean(dim=1, keepdim=True)  # B, 1, N, L
        dis = self.js_similarity(votes, mean_cap)
        weight = F.relu(self.thread ** 2 - dis)
        prob = torch.softmax(weight, dim=-2)  # B, M, N

        next_caps = torch.sum(prob[:, :, :, None] * priors, dim=1)
        return self.ln(next_caps)

    def forward(self, x):
        weights = self.smax(self.route_weights)
        priors = torch.einsum('bmd,mndt->bmnt', x, weights)
        outputs = self.new_routing(priors)

        return outputs

class CapsuleEncoder(nn.Module):
    def __init__(self, input_dim, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super(CapsuleEncoder, self).__init__()
        self.input_dim = input_dim
        self.init_fc = nn.Linear(input_dim, init_dim)
        digit_input_dim = init_dim + input_dim
        self.guass_primary_capsules = PrimaryCapsuleGenerator(digit_input_dim, primary_capsule_dim)
        self.digit_capsules = InferCapsule(in_capsule_num=primary_capsule_dim + 1, out_capsule_num=out_capsule_num,
                                           in_capsule_size=digit_input_dim, out_capsule_size=digit_capsule_dim,
                                           n_leaves=n_leaves)
        self.ln = nn.LayerNorm(digit_input_dim)

    def forward(self, x):
        init_x = self.init_fc(x)  # x: B, D'
        x = self.guass_primary_capsules(torch.cat([x, init_x], dim=1))
        x = self.ln(x)
        x = self.digit_capsules(x)  # x: B, N, T
        return x

class CapsuleClassifier(nn.Module):
    def __init__(self, input_dim, num_class, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super(CapsuleClassifier, self).__init__()
        self.net = CapsuleEncoder(input_dim, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves)
        self.head = head(num_class)

    def forward(self, x):
        x = self.net(x)
        out = self.head(x)
        return out


class ReconstructCapsNet(nn.Module):
    def __init__(self, input_dim, num_class, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super(ReconstructCapsNet, self).__init__()
        self.encoder = CapsuleEncoder(input_dim, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves)
        self.num_class = num_class
        self.sub_class = out_capsule_num // num_class
        self.digit_capsule_dim = digit_capsule_dim
        self.head = head(num_class)

        self.decoder = nn.Sequential(
            CapsuleDecoder_BasicBlock(digit_capsule_dim * self.sub_class, 32, 3),
            nn.Linear(32, input_dim)
        )

    def forward(self, x, y=None):
        hidden = self.encoder(x)
        pred = self.head(hidden)
        y = y.repeat(1, (hidden.shape[1] // self.num_class)).view(y.shape[0], -1, self.num_class)
        hidden = hidden.view(hidden.shape[0], -1, self.num_class, self.digit_capsule_dim) # [B, out_capsule_num, num_class]
        hidden = (hidden * y[:, :, :, None]).sum(dim=2)
        hidden = hidden.view(hidden.shape[0], -1)
        rec = self.decoder(hidden)
        return pred, rec

class head(nn.Module):
    def __init__(self, num_class):
        super(head, self).__init__()
        self.num_class = num_class

    def forward(self, x):
        x = (x ** 2).sum(dim=-1) ** 0.5
        x = x.view(x.shape[0], self.num_class, -1)
        if self.training == True:
            x = F.dropout(x, p=0.2)
        out = torch.sum(x, dim=-1)
        return out

class CapsuleDecoder_BasicBlock(nn.Module):
    def __init__(self, input_dim, base_outdim, n_path):
        super(CapsuleDecoder_BasicBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim // 2, n_path)
        self.conv2 = AbstractLayer(input_dim + base_outdim // 2, base_outdim, n_path)
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = torch.cat([x, out1], dim=-1)
        out = self.conv2(out1)
        return out