# @File :sp_moe.py
# @Time :2022/6/27
# @Desc :
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from moe_lib.pw_moe import PWLayer
from moe_lib.gates import *


class SPMoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps=8, d_in=64, dropout=0.2, noise=True):
        super().__init__()
        



class SPMoEAdaptor(nn.Module):
    def __init__(self, b_dim=300, act=None, num_expert=4):
        super().__init__()
        
        self.act = act
        if act is not None:
            self.act = get_activation(act)
        
        self.MoE_A = SPMoEAdaptorLayer(
            n_exps=num_expert,
        )
        
        self.MoE_B = SPMoEAdaptorLayer(
            n_exps=num_expert,
        )
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        residual = x
        
        result = self.MoE_A(x)
        
        if self.act is not None:
            result = self.act(result)
        
        result = self.dropout(result)
        
        result = self.MoE_B(result)
        
        return result + residual

