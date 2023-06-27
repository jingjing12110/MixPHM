import torch
import torch.nn as nn
import torch.nn.functional as F


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


# TODO added *******************************************************
def loss_bt(z1, z2, mask):
    # to [m, 768]
    seq_lens = torch.clamp(mask.sum(-1), min=1)
    z1 = torch.cat([z1[i][:seq_lens[i]] for i in range(len(seq_lens))], dim=0)
    z2 = torch.cat([z2[i][:seq_lens[i]] for i in range(len(seq_lens))], dim=0)
    
    # Z-score标准化（0-1标准化）
    z1 = (z1 - z1.mean(0, keepdim=True)) / (z1.std(0, keepdim=True) + 1e-09)
    z2 = (z2 - z2.mean(0, keepdim=True)) / (z2.std(0, keepdim=True) + 1e-09)
    
    scale_loss = 1 / z1.shape[0]
    lam = 1 / z1.shape[-1]
    
    c = z1.T @ z2
    c.div_(z1.shape[0])  # [d, d]

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

    # encouraging off_diag to be zero
    off_diag = off_diagonal(c).pow_(2).sum()
    
    return scale_loss * (on_diag + lam * off_diag)


def loss_bt_hsic(z1, z2, mask):
    # to [m, 768]
    seq_lens = torch.clamp(mask.sum(-1), min=1)
    z1 = torch.cat([z1[i][:seq_lens[i]] for i in range(len(seq_lens))], dim=0)
    z2 = torch.cat([z2[i][:seq_lens[i]] for i in range(len(seq_lens))], dim=0)
    
    # Z-score标准化（0-1标准化）
    z1 = (z1 - z1.mean(0, keepdim=True)) / (z1.std(0, keepdim=True) + 1e-09)
    z2 = (z2 - z2.mean(0, keepdim=True)) / (z2.std(0, keepdim=True) + 1e-09)
    
    scale_loss = 1 / z1.shape[0]
    lam = 1 / z1.shape[-1]
    
    c = z1.T @ z2
    c.div_(z1.shape[0])  # [d, d]
    
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    
    # inspired by HSIC encouraging off_diag to be negative ones
    off_diag = off_diagonal(c).add_(1).pow_(2).sum()
    
    return scale_loss * (on_diag + lam * off_diag)


# TODO reserved *******************************************************
class BTLoss(nn.Module):
    def __init__(self, bt_mode='bt', bt_expander=False, projector_hidden=768*4):
        super(BTLoss, self).__init__()
        self.bt_mode = bt_mode
        self.bt_expander = bt_expander
        
        if bt_expander:
            self.projector = nn.Sequential(
                nn.Linear(768, projector_hidden, bias=False),
                nn.BatchNorm1d(projector_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(projector_hidden, projector_hidden, bias=False),
            )
    
    def forward(self, z1, z2, mask):
        seq_lens = torch.clamp(mask.sum(-1), min=1)
        z1 = torch.cat([z1[i][:seq_lens[i]] for i in range(len(seq_lens))], dim=0)
        z2 = torch.cat([z2[i][:seq_lens[i]] for i in range(len(seq_lens))], dim=0)
        
        if self.bt_expander:
            z1, z2 = self.projector(z1), self.projector(z2)

        z1 = (z1 - z1.mean(0, keepdim=True)) / (z1.std(0, keepdim=True) + 1e-09)
        z2 = (z2 - z2.mean(0, keepdim=True)) / (z2.std(0, keepdim=True) + 1e-09)
        
        scale_loss = 1 / z1.shape[0]
        lam = 1 / z1.shape[-1]
        
        c = z1.T @ z2
        c.div_(z1.shape[0])  # [d, d]
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        if self.bt_mode == 'hsic':
            # inspired by HSIC encouraging off_diag to be negative ones
            off_diag = off_diagonal(c).add_(1).pow_(2).sum()
        else:
            # encouraging off_diag to be zero
            off_diag = off_diagonal(c).pow_(2).sum()
        
        return scale_loss * (on_diag + lam * off_diag)

    def t_wise_forward(self, z1, z2, mask):
        if self.bt_expander:
            z1, z2 = self.projector(z1), self.projector(z2)
        
        z1 = z1.transpose(0, 1)  # [seq_len, bs, dim]
        z2 = z2.transpose(0, 1)
        
        z1 = (z1 - z1.mean(1, keepdim=True)) / (z1.std(1, keepdim=True) + 1e-09)
        z2 = (z2 - z2.mean(1, keepdim=True)) / (z2.std(1, keepdim=True) + 1e-09)
        
        scale_loss = 1 / z1.shape[0]
        lam = 1 / z1.shape[-1]
        
        c = (z1.transpose(-2, -1) @ z2).div_(z1.shape[1])
        
        on_diag = torch.diagonal(c, dim1=1, dim2=2).add_(-1).pow_(2).sum(-1)
        if self.bt_mode == 'hsic':
            off_diag = batch_off_diagonal(c).add_(1).pow_(2).sum(-1)
        else:
            off_diag = batch_off_diagonal(c).pow_(2).sum(-1)  #
    
        return torch.mean(scale_loss * (on_diag + lam * off_diag) / mask.sum(0))


def batch_bt_mask_loss(z1, z2, mask):
    z1 = (z1 - z1.mean(1, keepdim=True)) / (z1.std(1, keepdim=True) + 1e-09)
    z2 = (z2 - z2.mean(1, keepdim=True)) / (z2.std(1, keepdim=True) + 1e-09)
    
    z1 = z1 * mask[..., None]
    z2 = z2 * mask[..., None]
    
    scale_loss = 1 / z1.shape[0]
    lam = 1 / z1.shape[-1]
    
    c = z1.transpose(-2, -1) @ z2
    c.div_(z1.shape[1])  # [bs, d, d]
    
    on_diag = torch.diagonal(c, dim1=1, dim2=2).add_(-1).pow_(2).sum(-1)  # [bs]
    
    off_diag = batch_off_diagonal(c).pow_(2).sum(-1)  # bs
    
    return torch.mean(scale_loss * (on_diag + lam * off_diag))

