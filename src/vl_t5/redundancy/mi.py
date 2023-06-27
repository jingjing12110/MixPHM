import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0, dim=-1)
    x1 = F.normalize(x1, dim=-1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


def compute_mean_skl_loss(x, y):
    px = F.softmax(x, dim=-1)
    log_px = F.log_softmax(x, dim=-1)
    py = F.softmax(y, dim=-1)
    log_py = F.log_softmax(y, dim=-1)
    
    kl_loss = 0.5 * (F.kl_div(log_px, py, reduction='none') + F.kl_div(
        log_py, px, reduction='none'))
    
    return kl_loss.mean()


def compute_skl_loss(x, y):
    px = F.softmax(x, dim=-1)
    log_px = F.log_softmax(x, dim=-1)
    py = F.softmax(y, dim=-1)
    log_py = F.log_softmax(y, dim=-1)
    
    kl_loss = 0.5 * (F.kl_div(log_px, py, reduction='none') + F.kl_div(
        log_py, px, reduction='none'))
    
    return kl_loss.sum(-1).mean()


def compute_skl_masked_loss(x, y, mask, temp_factor=1):
    x = x / temp_factor
    y = y / temp_factor
    
    px = F.softmax(x, dim=-1)
    log_px = F.log_softmax(x, dim=-1)
    py = F.softmax(y, dim=-1)
    log_py = F.log_softmax(y, dim=-1)
    
    kl_loss = 0.5 * (F.kl_div(log_px, py, reduction='none') + F.kl_div(
        log_py, px, reduction='none'))
    kl_loss = (kl_loss.sum(-1) * mask).sum(-1) / mask.sum(-1)
    return kl_loss.mean() * temp_factor ** 2


def compute_jsd_masked_loss(x, y, mask):
    m = 0.5 * (F.softmax(x, dim=-1) + F.softmax(y, dim=-1))
    
    loss = 0.5 * (F.kl_div(F.log_softmax(x, dim=-1), m, reduction="none")
                  + F.kl_div(F.log_softmax(y, dim=-1), m, reduction="none"))
    loss = (loss.sum(-1) * mask).sum(-1) / mask.sum(-1)
    return loss.mean()


class KDLoss:
    def __init__(self, temp_factor=4):
        super().__init__()
        self.temp_factor = temp_factor
    
    def forward(self, x, y):
        x = x / self.temp_factor
        y = y / self.temp_factor
        
        px = F.softmax(x, dim=-1)
        log_px = F.log_softmax(x, dim=-1)
        
        py = F.softmax(y, dim=-1)
        log_py = F.log_softmax(y, dim=-1)
        
        loss = 0.5 * (F.kl_div(log_px, py) + F.kl_div(log_py, px)) * (
                self.temp_factor ** 2)
        return loss


class CLUB(nn.Module):
    """Compute the Contrastive Log-ratio Upper Bound (CLUB) given input pair
        Args:
            hidden_size(int): embedding size
    """
    
    def __init__(self, x_dim=768, y_dim=768, hidden_size=768):
        super().__init__()
        # p_mu outputs mean of q(Y|X)
        # self.p_mu = nn.Sequential(
        #     nn.Linear(x_dim, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, y_dim)
        # )
        # # p_logvar outputs log of variance of q(Y|X)
        # self.p_logvar = nn.Sequential(
        #     nn.Linear(x_dim, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, y_dim),
        #     nn.Tanh()
        # )
    
    def forward(self, x, y):
        """
            CLUB with random shuffle, the Q function in original paper:
                CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]
            Args:
                x (Tensor): x in above equation
                y (Tensor): y in above equation
        """
        # mu, logvar = self.p_mu(x), self.p_logvar(x)  # [bs, dim]
        
        bs = y.size(0)
        random_index = torch.randperm(bs).long()
        
        # # pred v using l
        # pred_tile = mu.unsqueeze(1).repeat(1, bs, 1)  # (bs, bs, emb_size)
        # true_tile = y.unsqueeze(0).repeat(bs, 1, 1)  # (bs, bs, emb_size)
        #
        # positive = - (mu - y) ** 2 / 2. / logvar.exp()
        # # log of conditional probability of negative sample pairs
        # negative = - ((true_tile - pred_tile) ** 2).mean(
        #     dim=1) / 2. / logvar.exp()
        #
        # # lld = torch.mean(torch.sum(positive, -1))
        # upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound  # lld,
        
        # log of conditional probability of positive sample pairs
        # positive = - (mu - y) ** 2 / 2. / logvar.exp()
        
        # log of conditional probability of negative sample pairs
        # negative = - ((mu - y[random_index]) ** 2) / 2. / logvar.exp()
        positive = torch.zeros_like(y)
        negative = - (y - y[random_index]) ** 2 / 2.
        
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class InfoNCE(nn.Module):
    def __init__(self, fx=None, fc=None):
        super().__init__()
        if fx is None:
            fx = Identity()
        
        if fc is None:
            fc = Identity()
        
        self.fx = fx
        self.fc = fc
    
    def forward(self, x, c, mask=None):
        """
        Input:
        :x: BxTxD1 non-contextualized features
        :c: BxTxD2 contextualized features
        :mask: BxT boolean matrix denoting which time stamps are masked
        """
        assert (x.size()[:2] == c.size()[:2]), 'x and c must have same B and T'
        
        B, T, D1 = x.size()
        B, T, D2 = c.size()
        
        if mask is not None:
            err_str = f'size of mask must be {B}x{T} but got {mask.size()}'
            assert (len(mask.size()) == 2), err_str
            
            T_mask = (mask.sum(0) > 0)
            x = x[:, T_mask]
            c = c[:, T_mask]
            T_ = x.size(1)
        else:
            T_ = T
        
        x = self.fx(x.view(-1, D1)).view(B, T_, -1)
        c = self.fc(c.view(-1, D2)).view(B, T_, -1)
        
        x = x.unsqueeze(1)  # Bx1xT_xD
        c = c.unsqueeze(0)  # 1xBxT_xD
        logits = torch.sum(x * c, 3)  # BxBxT
        log_softmax1 = F.log_softmax(logits, 1)  # Select context given feature
        log_softmax2 = F.log_softmax(logits, 0)  # Select feature given context
        avg_log_softmax = 0.5 * (log_softmax1 + 0 * log_softmax2)
        loss = -avg_log_softmax.mean(2).diag().mean()
        
        return loss


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=768):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # shape [sample_size, sample_size, 1]
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.
        
        lower_bound = T0.mean() - (
                T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def batch_off_diagonal(x):
    bs, n, m = x.shape
    assert n == m
    x = x.view(bs, -1)[:, :-1]
    x = x.view(bs, n - 1, n + 1)[:, :, 1:].contiguous()
    return x.view(bs, -1)


class MILoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq_embed, ctx_embed, mask, measure='JSD'):
        seq_lens = torch.clamp(mask.sum(-1), min=1)
        tok_rep = [seq_embed[i][:seq_lens[i]] for i in range(len(seq_lens))]
        tok_rep = torch.cat(tok_rep, dim=0)  # [m, 768]
        
        pos_mask, neg_mask = self.create_masks(seq_lens)

        loss = self.local_global_loss(
            tok_rep, ctx_embed, pos_mask, neg_mask, measure
        )
        return loss
        
    @staticmethod
    def create_masks(lens_a):
        pos_mask = torch.zeros((lens_a.sum(), len(lens_a))).to(lens_a.device)
        neg_mask = torch.ones((lens_a.sum(), len(lens_a))).to(lens_a.device)
        temp = 0
        for idx in range(len(lens_a)):
            for j in range(temp, lens_a[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += lens_a[idx]
    
        return pos_mask, neg_mask
    
    def local_global_loss(self, l_enc, g_enc, pos_mask, neg_mask, measure):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''
        
        res = torch.mm(l_enc, g_enc.t())
        
        # print(l_enc.size(), res.size(), pos_mask.size())
        num_nodes = pos_mask.size(0)
        num_graphs = pos_mask.size(1)
        E_pos = self.get_positive_expectation(
            res * pos_mask, measure, average=False
        ).sum()
        E_pos = E_pos / num_nodes
        E_neg = self.get_negative_expectation(
            res * neg_mask, measure, average=False
        ).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))

        return E_neg - E_pos
    
    def log_sum_exp(self, x, axis=None):
        """Log sum exp function
        Args:
            x: Input.
            axis: Axis over which to perform sum.
        Returns:
            torch.Tensor: log sum exp
        """
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
        return y
    
    def get_positive_expectation(self, p_samples, measure, average=False):
        """Computes the positive part of a divergence / difference.
        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = math.log(2.) - F.softplus(- p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise ValueError

        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, measure, average=False):
        """Computes the negative part of a divergence / difference.
        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - math.log(2.)
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = self.log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            raise ValueError

        if average:
            return Eq.mean()
        else:
            return Eq

