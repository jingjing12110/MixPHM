# @File :r2_loss.py
# @Time :2022/4/28
# @Desc :
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_off_diagonal(x):
    bs, n, m = x.shape
    assert n == m
    x = x.view(bs, -1)[:, :-1]
    x = x.view(bs, n - 1, n + 1)[:, :, 1:].contiguous()
    return x.view(bs, -1)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MILoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq_embed, ctx_embed, mask, measure='JSD'):
        seq_lens = torch.clamp(mask.sum(-1), min=1)
        tok_rep = [seq_embed[i][:seq_lens[i]] for i in range(len(seq_lens))]
        tok_rep = torch.cat(tok_rep, dim=0)  # [m, 768]

        pos_mask, neg_mask = self.create_masks(seq_lens)

        loss = self.compute_mi(
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

    @staticmethod
    def compute_mi(l_enc, g_enc, pos_mask, neg_mask, measure):
        res = torch.mm(l_enc, g_enc.t())

        # print(l_enc.size(), res.size(), pos_mask.size())
        num_nodes = pos_mask.size(0)
        num_graphs = pos_mask.size(1)

        entropy_pos = torch.sum(math.log(2.) - F.softplus(- res * pos_mask)
                                ) / num_nodes

        q_samples = res * neg_mask
        entropy_neg = torch.sum(F.softplus(-q_samples) + q_samples - math.log(2.)
                                ) / (num_nodes * (num_graphs - 1))

        return entropy_neg - entropy_pos


class R2Loss(nn.Module):
    def __init__(self, lam=3.9e-3):
        super().__init__()
        self.lam = lam  # 1/256, 1/384, 1/512, 1/768
        self.mi_loss = MILoss()
    
    def forward(self, up_x, res_x, mask=None):
        scale_factor = 1.0 / up_x.shape[0]
        
        # ****** redundancy ******
        seq_len = torch.clamp(mask.sum(-1), min=1)
        up_x = torch.cat([up_x[i][:seq_len[i]] for i in range(len(seq_len))])
        res_x = torch.cat([res_x[i][:seq_len[i]] for i in range(len(seq_len))])
        
        # ****** self-correlation
        up_x_self = F.normalize(up_x, dim=1)
        res_x_self = F.normalize(res_x, dim=1)
        c = up_x_self.T @ res_x_self
        # c.div_(up_x_self.shape[0])

        # up_c = up_x_self.T @ up_x_self
        # res_c = res_x_self.T @ res_x_self
        #
        # self_off_diag = 0.5 * (off_diagonal(up_c).pow_(2).sum() + off_diagonal(
        #     res_c).pow_(2).sum()) * scale_factor
        # # # self_off_diag = torch.tensor(0., device=up_x.device)

        # # ****** cross-correlation
        # up_x_cross = (up_x - up_x.mean(0)) / (up_x.std(0) + 1e-09)
        # res_x_cross = (res_x - res_x.mean(0)) / (res_x.std(0) + 1e-09)
        #
        # c = up_x_cross.T @ res_x_cross
        # c.div_(up_x_cross.shape[0])

        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(scale_factor)
        # redundancy reduction term
        off_diag = off_diagonal(c).pow_(2).sum().mul(scale_factor)
        # hsic
        # off_diag = off_diagonal(c).add_(1).pow_(2).sum().mul(scale_factor)
        
        # red_loss = on_diag + self.lam * (off_diag + self_off_diag)
        red_loss = self.lam * off_diag

        return red_loss
