# @Github: https://github.com/RElbers/info-nce-pytorch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples
    to be close and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or
    more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating
        the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See F.cross_entropy for more details about each option.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples
        (e.g. embeddings of augmented input).
        negative_keys (optional): (M, D) Tensor with negative samples
        (e.g. embeddings of other inputs). If None, then the negative keys for a
        sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> batch_size, embedding_size = 32, 128
        >>> loss = InfoNCE()
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(4 * batch_size, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """
    
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature, reduction=self.reduction)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1,
             reduction='mean'):
    # Inputs all have 2 dimensions.
    if query.dim() != 2 or positive_key.dim() != 2 or (
            negative_keys is not None and negative_keys.dim() != 2):
        raise ValueError(
            'query, positive_key and negative_keys should all have 2 dimensions.')
    
    # Each query sample is paired with exactly one positive key sample.
    if len(query) != len(positive_key):
        raise ValueError(
            'query and positive_key must have the same number of samples.')
    
    # Embedding vectors should have same number of components.
    if query.shape[1] != positive_key.shape[1] != (
            positive_key.shape[1] if negative_keys is None
            else negative_keys.shape[1]):
        raise ValueError(
            'query, positive_key and negative_keys should have '
            'the same number of components.')
    
    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(
        query, positive_key, negative_keys)
    
    if negative_keys is not None:
        # Explicit negative keys
        
        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        
        # Cosine between all query-negative combinations (@ same as *)
        negative_logits = query @ transpose(negative_keys)
        
        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        
        # Cosine between all combinations
        logits = query @ transpose(positive_key)
        
        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def contrastive_nce(anchor, pos, neg):
    # Normalize to unit vectors
    anchor, pos, neg = normalize(anchor, pos, neg)
    pos_cosine = anchor @ pos.transpose(0, 1)  # [num_pos]
    neg_cosine = anchor @ neg.transpose(0, 1)  # [num_neg]
    
    lb = torch.log(pos_cosine.exp().sum() / (
            pos_cosine.exp().sum() + neg_cosine.exp().sum()))
    return -lb


def contrastive_nce_v2(anchor, pos, pos_neg, pos_neg_mask=None):
    # mask positive samples
    pos, pos_mask = pad_seq_1d(pos)
    
    pos_cosine = anchor.unsqueeze(1) @ pos.transpose(1, 2)  # [bs, max_pos]
    pos_cosine = pos_cosine.squeeze(1).exp()
    pos_cosine = (pos_cosine * pos_mask).sum(1) / pos_mask.sum(1)  # [bs]
    
    sum_cosine = anchor.unsqueeze(1) @ pos_neg.transpose(1, 2)
    sum_cosine = sum_cosine.squeeze(1).exp()  # [bs, max_seq_length]
    # mask sum
    if pos_neg_mask is not None:
        # [bs]
        sum_cosine = (sum_cosine * pos_neg_mask).sum(1) / pos_neg_mask.sum(1)
        c = torch.log(pos_neg_mask.sum(1))
    else:
        sum_cosine = sum_cosine.mean(1)
        c = np.log(pos_neg.shape[1])
    
    lb = torch.mean(torch.log(pos_cosine / sum_cosine) - c)
    
    return -lb


def pad_seq_1d(seqs):
    extra_dims = seqs[0].shape[1:]  # tuple
    lengths = [len(seq) for seq in seqs]
    padded_seqs = torch.zeros((len(seqs), max(lengths)) + extra_dims,
                              dtype=torch.float32, device=seqs[0].device)
    mask = torch.zeros((len(seqs), max(lengths)),
                       dtype=torch.int32, device=seqs[0].device)
    for idx, seq in enumerate(seqs):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask
