import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
# use GPU if available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

# Source: https://github.com/SilenceX12138/ProtoGate/blob/1a2ecdcbe6a6954f3f322a39bec4a081b78acf52/src/custom_models/protogate/ProtoGate.py
class GatingNet(nn.Module):

    def __init__(self, input_dim: int, a: float, sigma: float,  hidden_layer_list: list,activation: str = 'tanh') -> None:
        """Gating Network for feature selection

        Args:
            input_dim (int): input dimension of the gating network
            a (float): coefficient in hard relu activation function
            sigma (float): std of the gaussion reparameterization noise
            activation (str): activation function of the gating net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            hidden_layer_list (list): number of nodes for each hidden layer of the gating net, example: [200,200]
        """
        super().__init__()

        self.a = a
        self.sigma = sigma
        self.act = get_activation(activation)
        full_layer_list = [input_dim, *hidden_layer_list]

        self.embed = nn.Sequential()
        for i in range(len(full_layer_list) - 1):
            self.embed.add_module('fn{}'.format(i), nn.Linear(full_layer_list[i], full_layer_list[i + 1]))
            self.embed.add_module('act{}'.format(i), self.act)

        self.gate = nn.Sequential()
        self.gate.add_module('fn', nn.Linear(full_layer_list[-1], input_dim))
        self.gate.add_module('act', self.act)

    def forward(self, x):
        x_all = x
        x_emb = self.embed(x)
        alpha = self.gate(x_emb)
        stochastic_gate = self.get_stochastic_gate(alpha)
        x_selected = x_all * stochastic_gate

        return x_selected, alpha, stochastic_gate

    def get_stochastic_gate(self, alpha):
        """
        This function replaced the feature_selector function in order to save Z
        """
        # gaussian reparametrization
        noise = self.sigma * torch.randn(alpha.shape, device=alpha.device) \
                if self.training == True else torch.zeros(1, device=alpha.device)
        z = alpha + noise
        stochastic_gate = self.hard_sigmoid(z)

        return stochastic_gate

    def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = self.a * x
        x = torch.clamp(x, 0, 1)

        return x

class KNNNet(torch.nn.Module):

    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1, similarity='euclidean'):
        super(KNNNet, self).__init__()
        self.k = k
        self.soft_sort = HybridSort(tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples
        self.similarity = similarity

    # query: M x p
    # neighbors: N x p
    #
    # returns:
    def forward(self, query, neighbors, tau=1.0):
        if self.similarity == 'euclidean':
            diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
            squared_diffs = diffs**2
            l2_norms = squared_diffs.sum(2)
            norms = l2_norms
            scores = -norms  # B * N
        elif self.similarity == 'cosine':
            scores = F.cosine_similarity(query.unsqueeze(1), neighbors.unsqueeze(0), dim=2) - 1
        else:
            raise ValueError('Unknown similarity for KNNNet: {}'.format(self.similarity))

        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores)  # B*N*N
            top_k = P_hat[:, :self.k, :].sum(1)  # B*N
            return top_k
        if self.method == 'stochastic':
            pl_s = PL(scores, tau, hard=False)
            P_hat = pl_s.sample((self.num_samples, ))
            top_k = P_hat[:, :, :self.k, :].sum(2)
            return top_k

class HybridSort(torch.nn.Module):

    def __init__(self, tau=1.0, hard=False):
        super(HybridSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        device = scores.device
        one = torch.DoubleTensor(dim, 1).fill_(1).to(device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.DoubleTensor).to(device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_max = P_max.to('cpu')
        P_hat = sm(P_max / self.tau)
        P_hat = P_hat.to(device)
        if self.hard:
            P = torch.zeros_like(P_hat, device=device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(dim0=1, dim1=0).flatten().type(
                torch.LongTensor).to(device)
            r_idx = torch.arange(dim).repeat([bsize, 1]).flatten().type(torch.LongTensor).to(device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat

class PL(Distribution):

    arg_constraints = {'scores': constraints.positive, 'tau': constraints.positive}
    has_rsample = True

    @property
    def mean(self):
        # mode of the PL distribution
        return self.relaxed_sort(self.scores)

    def __init__(self, scores, tau, hard=True, validate_args=None):
        """
        scores. Shape: (batch_size x) n 
        tau: temperature for the relaxation. Scalar.
        hard: use straight-through estimation if True
        """
        self.scores = scores.unsqueeze(-1)
        self.tau = tau
        self.hard = hard
        self.n = self.scores.size()[1]

        try:
            batch_shape = torch.Size()
        except:
            batch_shape = self.scores.size()
        # if isinstance(scores, Number):
        #     batch_shape = torch.Size()
        # else:
        #     batch_shape = self.scores.size()
        super(PL, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args:
            if not torch.gt(self.scores, torch.zeros_like(self.scores)).all():
                raise ValueError("PL is not defined when scores <= 0")

    def relaxed_sort(self, inp):
        """
        inp: elements to be sorted. Typical shape: batch_size x n x 1
        """
        bsize = inp.size()[0]
        dim = inp.size()[1]
        one = FloatTensor(dim, 1).fill_(1)

        A_inp = torch.abs(inp - inp.permute(0, 2, 1))
        B = torch.matmul(A_inp, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(FloatTensor)
        C = torch.matmul(inp, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(dim0=1,
                                                                                    dim1=0).flatten().type(LongTensor)
            r_idx = torch.arange(dim).repeat([bsize, 1]).flatten().type(LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat

    def rsample(self, sample_shape, log_score=True):
        """
        sample_shape: number of samples from the PL distribution. Scalar.
        """
        with torch.enable_grad():  # torch.distributions turns off autograd
            n_samples = sample_shape[0]

            def sample_gumbel(samples_shape, eps=1e-20):
                U = torch.zeros(samples_shape, device='cuda').uniform_()
                return -torch.log(-torch.log(U + eps) + eps)

            if not log_score:
                log_s_perturb = torch.log(self.scores.unsqueeze(0)) + sample_gumbel([n_samples, 1, self.n, 1])
            else:
                log_s_perturb = self.scores.unsqueeze(0) + sample_gumbel([n_samples, 1, self.n, 1])
            log_s_perturb = log_s_perturb.view(-1, self.n, 1)
            P_hat = self.relaxed_sort(log_s_perturb)
            P_hat = P_hat.view(n_samples, -1, self.n, self.n)

            return P_hat.squeeze()

    def log_prob(self, value):
        """
        value: permutation matrix. shape: batch_size x n x n
        """
        permuted_scores = torch.squeeze(torch.matmul(value, self.scores))
        log_numerator = torch.sum(torch.log(permuted_scores), dim=-1)
        idx = LongTensor([i for i in range(self.n - 1, -1, -1)])
        invert_permuted_scores = permuted_scores.index_select(-1, idx)
        denominators = torch.cumsum(invert_permuted_scores, dim=-1)
        log_denominator = torch.sum(torch.log(denominators), dim=-1)
        return (log_numerator - log_denominator)
    
class DeactFunc(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
      
def get_activation(value):
    if value == 'relu':
        return torch.nn.ReLU()
    elif value == 'l_relu':
        # set the slope to align tensorflow
        return torch.nn.LeakyReLU(negative_slope=0.2)
    elif value == 'sigmoid':
        return torch.nn.Sigmoid()
    elif value == 'tanh':
        return torch.nn.Tanh()
    elif value == 'none':
        return DeactFunc()
    else:
        raise NotImplementedError('activation for the gating network not recognized')