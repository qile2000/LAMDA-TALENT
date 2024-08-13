import torch
import numpy as np


"""Feature-wise Mixup"""
def batch_feat_shuffle(Xs: torch.Tensor, beta=0.5):
    b, f = Xs.shape[0], Xs.shape[1]
    shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
    feat_masks = np.random.random(size=(b, f)) > shuffle_rates # b f
    feat_masks = torch.from_numpy(feat_masks).to(Xs.device)

    shuffled_sample_ids = np.random.permutation(b)

    Xs_shuffled = Xs[shuffled_sample_ids]
    feat_masks = feat_masks.unsqueeze(-1) if Xs.ndim == 3 else feat_masks
    Xs_mixup = feat_masks * Xs + ~feat_masks * Xs_shuffled

    return Xs_mixup, feat_masks.squeeze(-1), shuffled_sample_ids

"""Dim-wise Mixup"""
def batch_dim_shuffle(Xs: torch.Tensor, beta=0.5):
    b, f, d = Xs.shape
    shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
    dim_masks = np.random.random(size=(b, d)) < shuffle_rates # b d
    dim_masks = torch.from_numpy(dim_masks).to(Xs.device)

    shuffled_sample_ids = np.random.permutation(b)
    
    Xs_shuffled = Xs[shuffled_sample_ids]
    dim_masks = dim_masks.unsqueeze(1) # b 1 d
    Xs_mixup = dim_masks * Xs + ~dim_masks * Xs_shuffled

    return Xs_mixup, torch.from_numpy(shuffle_rates[:,0]).float().to(Xs.device), shuffled_sample_ids

"""Naive Mixup"""
def mixup_data(Xs: torch.Tensor, beta=0.5):
    b, f = Xs.shape
    lam = np.random.beta(beta, beta)
    shuffle_sample_ids = np.random.permutation(b)
    mixed_X = lam * Xs + (1 - lam) * Xs[shuffle_sample_ids]
    # shuffle_sample_ids = torch.randperm(b).to(Xs.device)
    # mixed_X = lam * Xs + (1 - lam) * Xs[shuffle_sample_ids, :]
    return mixed_X, lam, shuffle_sample_ids