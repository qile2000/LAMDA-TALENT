import math
import typing as ty
from enum import Enum
import torch
import torch.nn as nn
# Source: https://github.com/sbadirli/GrowNet/
# %%
class ForwardType(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3


class DynamicNet:
    def __init__(
        self,
        lr,
        categories: ty.Optional[ty.List[int]],
        d_embedding: ty.Optional[int],
    ):
        self.models = []
        self.lr = lr
        self.boost_rate = nn.Parameter(
            torch.tensor(lr, requires_grad=True, device="cuda")
        )
        if categories is not None:
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.category_offsets = category_offsets
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')
        else:
            self.category_embeddings = None
            self.category_offsets = None

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        if self.category_embeddings is not None:
            params.extend(self.category_embeddings.parameters())
            params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()
        if self.category_embeddings is not None:
            self.category_embeddings = self.category_embeddings.cuda()
            self.category_offsets = self.category_offsets.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_double(self):
        for m in self.models:
            m.double()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def embed_input(self, x_num, x_cat):
        if x_cat is not None:
            x_cat = self.category_embeddings(x_cat + self.category_offsets[None])
            if x_num is not None:
                x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)
            else:
                x = x_cat.view(x_cat.size(0), -1).double()
        else:
            x = x_num
        return x

    def forward(self, x_num, x_cat):
        if len(self.models) == 0:
            return None, 0
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(
                        self.embed_input(x_num, x_cat), middle_feat_cum
                    )
                else:
                    middle_feat_cum, pred = m(
                        self.embed_input(x_num, x_cat), middle_feat_cum
                    )
                    prediction += pred
        return middle_feat_cum, self.boost_rate * prediction

    def forward_grad(self, x_num, x_cat):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(
                    self.embed_input(x_num, x_cat), middle_feat_cum
                )
            else:
                middle_feat_cum, pred = m(
                    self.embed_input(x_num, x_cat), middle_feat_cum
                )
                prediction += pred
        # return middle_feat_cum, self.c0 + self.boost_rate * prediction
        return middle_feat_cum, self.boost_rate * prediction

    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet( d['lr'], categories=None, d_embedding=None)
        net.boost_rate = d['boost_rate']
        if 'category_embeddings' in d:
            net.category_embeddings = d['category_embeddings']
            net.category_offsets = d['category_offsets']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {
            'models': models,
            'lr': self.lr,
            'boost_rate': self.boost_rate,
            'category_embeddings': self.category_embeddings,
            'category_offsets': self.category_offsets,
        }
        torch.save(d, path)


class SpLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (input.t().mm(grad_output)).t()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


splinear = SpLinearFunc.apply


class SpLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(SpLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        # TODO write a default initialization
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return splinear(input, self.weight, self.bias)


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2,dim_out,sparse=False, bn=True):
        super(MLP_2HL, self).__init__()
        self.in_layer = (
            SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        )
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, dim_out)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.lrelu(self.in_layer(x))
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()
        # return out, self.out_layer(self.relu(out))
    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        dim_out = opt.dim_out
        # dim_out = 1
        model = MLP_2HL(dim_in, opt.hidden_d, opt.hidden_d, dim_out, opt.sparse)
        return model