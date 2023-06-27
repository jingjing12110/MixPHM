import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tools.ops import index_select


class GradientNet(nn.Module):
    def __init__(self):
        super(GradientNet, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
    
    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


def pairwise_distance(points0, points1, normalized=False, clamp=False):
    r"""
    [PyTorch/Numpy] Pairwise distance of two point clouds.

    :param points0: torch.Tensor (d0, ..., dn, num_point0, num_feature)
    :param points1: torch.Tensor (d0, ..., dn, num_point1, num_feature)
    :param normalized: bool (default: False)
        If True, the points are normalized, so a2 and b2 both 1.
        This enables us to use 2 instead of a2 + b2 for simplicity.
    :param clamp: bool (default: False)
        If True, all value will be assured to be non-negative.
    :return: dist: torch.Tensor (d0, ..., dn, num_point0, num_point1)
    """
    ab = torch.matmul(points0, points1.transpose(-1, -2))
    if normalized:
        dist2 = 2 - 2 * ab
    else:
        a2 = torch.sum(points0 ** 2, dim=-1).unsqueeze(-1)
        b2 = torch.sum(points1 ** 2, dim=-1).unsqueeze(-2)
        dist2 = a2 - 2 * ab + b2
    if clamp:
        dist2 = torch.maximum(dist2, torch.zeros_like(dist2))
    
    return dist2


def gaussian_correlation(ref_feats, src_feats, ref_masks, src_masks,
                         normalized=False, dual_normalization=True):
    if not normalized:
        ref_feats = F.normalize(ref_feats, dim=-1)
        src_feats = F.normalize(src_feats, dim=-1)
    
    ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
    src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
    ref_feats = index_select(ref_feats, ref_indices, dim=0)
    src_feats = index_select(src_feats, src_indices, dim=0)
    # correlation matrix
    matching_scores = torch.exp(-pairwise_distance(
        ref_feats, src_feats, normalized=True))
    if dual_normalization:
        ref_matching_scores = matching_scores / matching_scores.sum(
            dim=1, keepdim=True)
        src_matching_scores = matching_scores / matching_scores.sum(
            dim=0, keepdim=True)
        matching_scores = ref_matching_scores * src_matching_scores
    return matching_scores
    # # select top-k proposals
    # corr_scores, corr_indices = matching_scores.view(-1).topk(
    #     k=num_proposal, largest=True)
    #
    # ref_sel_indices = corr_indices // matching_scores.shape[1]
    # src_sel_indices = corr_indices % matching_scores.shape[1]
    # # recover original superpoint indices
    # ref_corr_indices = index_select(ref_indices, ref_sel_indices, dim=0)
    # src_corr_indices = index_select(src_indices, src_sel_indices, dim=0)
    #
    # return ref_corr_indices, src_corr_indices, corr_scores
