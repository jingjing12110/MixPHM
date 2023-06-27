import torch
import torch.nn as nn
import torch.nn.functional as F


def pdists(A, squared=False, eps=1e-8):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


def align_loss(x, y, alpha=2):
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    y = y / y.norm(p=2, dim=-1, keepdim=True)
    
    return (x - y).norm(p=2, dim=-1).pow(alpha).mean()


# def uniform_loss(x, t=2):
#     return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
def uniform_loss(x, t=2):
    # x: [bs, n, dim]
    bs = x.shape[0]
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    
    # torch.pdist :
    x = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1, p=2)
    t_mask = torch.ones_like(x).bool().triu(1)
    x = x.view(-1)[t_mask.view(-1)].view(bs, -1)  # [bs, n]
    
    return torch.mean(x.pow(2).mul(-t).exp().mean(-1).log())


def align_mask_loss(x, y, mask, alpha=2):
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    y = y / y.norm(p=2, dim=-1, keepdim=True)
    
    x = (x - y).norm(p=2, dim=-1).pow(alpha)
    return torch.mean(torch.sum(x * mask, dim=-1) / mask.sum(-1))


def uniform_mask_loss(x, mask, t=2):
    # x: [bs, n, dim]
    bs = x.shape[0]
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    
    # torch.pdist :
    x = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1, p=2)
    t_mask = torch.ones_like(x).bool().triu(1)
    x = x.view(-1)[t_mask.view(-1)].view(bs, -1)  # [bs, m]
    
    x = x.pow(2).mul(-t).exp()  # [bs, m]
    
    mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
    mask = mask.view(-1)[t_mask.view(-1)].view(bs, -1)
    
    return torch.mean(torch.log(
        torch.sum(x * mask, dim=-1) / mask.sum(-1)))


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature",
                             torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(
                device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
    
    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        
        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature)  # 2*bs, 2*bs
        
        loss_partial = -torch.log(
            nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def contrastive_loss(emb_i, emb_j, temperature=0.5):
    bs, n = emb_i.shape[0], emb_i.shape[1]
    z_i = F.normalize(emb_i, dim=-1)
    z_j = F.normalize(emb_j, dim=-1)
    
    reps = torch.cat([z_i, z_j], dim=1)
    similarity_matrix = F.cosine_similarity(
        reps.unsqueeze(2), reps.unsqueeze(1), dim=-1)  # simi_mat: (bs, 2*n, 2*n)

    sim_ij = torch.diagonal(similarity_matrix, n, dim1=1, dim2=2)  # bs, n
    sim_ji = torch.diagonal(similarity_matrix, -n, dim1=1, dim2=2)  # bs, n
    positives = torch.cat([sim_ij, sim_ji], dim=1)  # bs, 2*n
    
    nominator = torch.exp(positives / temperature)  # bs, 2*n
    # 主对角线为0，其余位置全为1的mask矩阵
    negatives_mask = torch.eye(n * 2, n * 2).to(emb_i.device).bool()
    negatives_mask = (~negatives_mask).unsqueeze(0).repeat(bs, 1, 1).float()
    # bs, 2*n, 2*n
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)
    # bs, 2*n
    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=-1))
    loss = torch.sum(loss_partial, dim=-1) / (2 * n)
    return loss.mean()

