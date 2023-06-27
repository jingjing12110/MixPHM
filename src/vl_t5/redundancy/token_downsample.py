# @File :token_downsample.py
# @Time :2022/3/22
# @Desc :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]
    
    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)
    
    dim += value_expand_len
    return values.gather(dim, indices)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)


class AdaptiveTokenSampling(nn.Module):
    """
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/ats_vit.py
    """
    
    def __init__(self, output_num_tokens, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens
    
    def forward(self, attn, value, mask):
        heads, output_num_tokens, eps, device, dtype = \
            attn.shape[1], self.output_num_tokens, \
            self.eps, attn.device, attn.dtype
        
        # first get the attention values for CLS token to all other tokens
        cls_attn = attn[..., 0, 1:]
        
        # calculate the norms of the values, for weighting the scores,
        # as described in the paper
        value_norms = value[..., 1:, :].norm(dim=-1)
        
        # weigh the attention scores by the norm of the values,
        # sum across all heads
        cls_attn = torch.einsum('b h n, b h n -> b n', cls_attn, value_norms)
        
        # normalize to 1
        normed_cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + eps)
        
        # instead of using inverse transform sampling, going to invert the
        # softmax and use gumbel-max sampling instead
        pseudo_logits = torch.log(normed_cls_attn + eps)
        
        # mask out pseudo logits for gumbel-max sampling
        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)
        
        # expand k times, k being the adaptive sampling number
        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k=output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(
            pseudo_logits.shape, device=device, dtype=dtype)
        
        # gumble-max and add one to reserve 0 for padding / mask
        sampled_token_ids = pseudo_logits.argmax(dim=-1) + 1
        
        # calculate unique using torch.unique and
        # then pad the sequence from the right
        unique_sampled_token_ids_list = [
            torch.unique(t, sorted=True) for t in
            torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(
            unique_sampled_token_ids_list, batch_first=True)
        
        # calculate the new mask, based on the padding
        new_mask = unique_sampled_token_ids != 0
        
        # CLS token never gets masked out (gets a value of True)
        new_mask = F.pad(new_mask, (1, 0), value=True)
        
        # prepend a 0 token id to keep the CLS attention scores
        unique_sampled_token_ids = F.pad(
            unique_sampled_token_ids, (1, 0), value=0)
        expanded_unique_sampled_token_ids = repeat(
            unique_sampled_token_ids, 'b n -> b h n', h=heads)
        
        # gather the new attention scores
        new_attn = batched_index_select(
            attn, expanded_unique_sampled_token_ids, dim=2)
        
        # return the sampled attention scores, new mask (denoting padding),
        # as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids

    def get_sampled_token(self, attn, value, mask):
        # first get the attention values for CLS token to all other tokens
        cls_attn = attn[..., 0, 1:]  # [bs, 900]
    
        # calculate the norms of the values, for weighting the scores,
        # as described in the paper
        value_norms = value[..., 1:, :].norm(dim=-1)  # [bs, 900]
    
        # weigh the attention scores by the norm of the values,
        # sum across all heads
        cls_attn = torch.mul(cls_attn, value_norms)  # [bs, 900]
    
        # normalize to 1
        normed_cls_attn = cls_attn / (
                cls_attn.sum(dim=-1, keepdim=True) + self.eps)  # [bs, 900]
    
        # instead of using inverse transform sampling, going to invert the
        # softmax and use gumbel-max sampling instead
        pseudo_logits = torch.log(normed_cls_attn + self.eps)
    
        # mask out pseudo logits for gumbel-max sampling
        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)
    
        # expand k times, k being the adaptive sampling number
        pseudo_logits = repeat(
            pseudo_logits, 'b n -> b k n', k=self.output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(
            pseudo_logits.shape, device=attn.device, dtype=attn.dtype)
    
        # gumble-max and add one to reserve 0 for padding / mask
        sampled_token_ids = pseudo_logits.argmax(dim=-1) + 1
        
        # calculate unique using torch.unique and
        # then pad the sequence from the right
        unique_sampled_token_ids_list = [
            torch.unique(t, sorted=True) for t in
            torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(
            unique_sampled_token_ids_list, batch_first=True)
    
        # calculate the new mask, based on the padding
        new_mask = unique_sampled_token_ids != 0
    
        # CLS token never gets masked out (gets a value of True)
        new_mask = F.pad(new_mask, (1, 0), value=True)
    
        # prepend a 0 token id to keep the CLS attention scores
        unique_sampled_token_ids = F.pad(
            unique_sampled_token_ids, (1, 0), value=0)
    
        # gather the new attention scores
        new_attn = batched_index_select(
            attn, unique_sampled_token_ids, dim=1)
    
        # return the sampled attention scores, new mask (denoting padding),
        # as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids


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
