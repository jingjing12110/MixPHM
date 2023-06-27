import copy

import torch
import torch.nn.functional as F

from torch import nn
from transformers.activations import get_activation


class AdapterExpert(nn.Module):
    def __init__(self, input_size, down_sample, act, dropout=0.5, ):
        super().__init__()
        self.input_size = input_size
        self.down_sample = down_sample
        
        self.act = act
        if act is not None:
            self.act = get_activation(act)
        
        self.dropout = nn.Dropout(dropout)
        
        self.adapter_down = nn.Linear(self.input_size, self.down_sample)
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)
    
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        
        down = self.adapter_down(x)
        
        if self.act is not None:
            down = self.act(down)
        down = self.dropout(down)
        
        up = self.adapter_up(down)
        
        return up + residual


class MultiDownAdapterMoE(nn.Module):
    def __init__(self, input_size, down_sample, num_expert=4, inference_level=0,
                 act=None, dropout=0.5, ):
        super().__init__()
        self.num_expert = num_expert
        self.inference_level = inference_level
        
        if self.num_expert == 2:
            self.down_samples = [64, 128]
        elif self.num_expert == 3:
            self.down_samples = [32, 64, 128]
        elif self.num_expert == 4:
            self.down_samples = [16, 32, 64, 128]
            
        self.experts = nn.ModuleList([
            AdapterExpert(input_size, self.down_samples[i], act, dropout)
            for i in range(self.num_expert)]
        )
        
        self.expert_score_weight = nn.Parameter(
            torch.zeros(self.num_expert),
            requires_grad=False
        )
        self.parameter_dict = None
    
    def forward(self, x):
        # residual = x
        
        expert_output = None
        if self.training:
            # selected expert
            expert_idx = torch.randint(
                low=0, high=self.num_expert, size=(1,)
            ).item()
            
            if self.expert_score_weight.requires_grad:
                self.expert_soup()
                expert_output = self.expert_soup_forward(x)
            else:
                expert_output = self.experts[expert_idx](x)
        else:
            if self.inference_level != 3:
                result = []
                for expert_idx in range(self.num_expert):
                    result.append(self.experts[expert_idx](x))
                result = torch.stack(result, dim=0)  # [num_expert, bs, *, 768]
                
                if self.inference_level == 0:  # token level
                    mask = torch.randint(
                        0, self.num_expert,
                        size=(result.size(1), result.size(2)),
                        device=result.device
                    )
                    for i in range(self.num_expert):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1)
                    expert_output = result.sum(0)
                
                elif self.inference_level == 1:  # sentence level
                    mask = torch.randint(
                        0, self.num_expert,
                        size=(result.size(1),),
                        device=result.device
                    )
                    for i in range(self.num_expert):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                    expert_output = result.sum(0)
                
                elif self.inference_level == 2:
                    # weight = F.softmax(self.expert_score_weight)
                    expert_output = result.sum(0)
                    
            elif self.inference_level == 3:
                self.expert_soup()
                expert_output = self.expert_soup_forward(x)
        
        return expert_output
    
    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight)
        
        self.parameter_dict = {
            "adapter_down.weight": 0, "adapter_down.bias": 0,
            "adapter_up.weight": 0, "adapter_up.bias": 0,
        }
        
        for idx in range(self.num_expert):
            single_expert = self.experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "adapter_down.weight" in s_name:
                    p_name = "adapter_down.weight"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "adapter_down.bias" in s_name:
                    p_name = "adapter_down.bias"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "adapter_up.weight" in s_name:
                    p_name = "adapter_up.weight"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "adapter_up.bias" in s_name:
                    p_name = "adapter_up.bias"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
    
    def expert_soup_forward(self, x):
        adapter_expert = copy.deepcopy(self.experts[0])
        # init one expert with the averaged weights
        for s_name, s_param in adapter_expert.named_parameters():
            s_param.data.copy_(self.parameter_dict[s_name])
        
        return adapter_expert(x)


class AdapterMoE(nn.Module):
    def __init__(self, input_size, down_sample, num_expert=4, inference_level=0,
                 act=None, dropout=0.5, ):
        super().__init__()
        
        self.num_expert = num_expert
        self.inference_level = inference_level
        
        self.experts = nn.ModuleList([
            AdapterExpert(input_size, down_sample, act, dropout)
            for _ in range(self.num_expert)]
        )
        
        self.expert_score_weight = nn.Parameter(
            torch.zeros(self.num_expert),
            requires_grad=False
        )
        self.parameter_dict = None
    
    def forward(self, x):
        residual = x
        
        expert_output = None
        if self.training:
            # selected expert
            expert_idx = torch.randint(
                low=0, high=self.num_expert, size=(1,)
            ).item()
            
            if self.expert_score_weight.requires_grad:
                self.expert_soup()
                expert_output = self.expert_soup_forward(x)
            else:
                expert_output = self.experts[expert_idx](x)
        else:
            if self.inference_level != 3:
                result = []
                for expert_idx in range(self.num_expert):
                    result.append(self.experts[expert_idx](x))
                result = torch.stack(result, dim=0)
                
                if self.inference_level == 0:  # token level
                    mask = torch.randint(
                        0, self.num_expert,
                        size=(result.size(1), result.size(2)),
                        device=result.device
                    )
                    for i in range(self.num_expert):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1)
                    expert_output = result.sum(0)
                
                elif self.inference_level == 1:  # sentence level
                    mask = torch.randint(
                        0, self.num_expert,
                        size=(result.size(1),),
                        device=result.device
                    )
                    for i in range(self.num_expert):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                    expert_output = result.sum(0)
            
            elif self.inference_level == 3:
                self.expert_soup()
                expert_output = self.expert_soup_forward(x)
        
        return expert_output + residual
        
    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight)
        
        self.parameter_dict = {
            "adapter_down.weight": 0, "adapter_down.bias": 0,
            "adapter_up.weight": 0, "adapter_up.bias": 0,
        }
        
        for idx in range(self.num_expert):
            single_expert = self.experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "adapter_down.weight" in s_name:
                    p_name = "adapter_down.weight"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "adapter_down.bias" in s_name:
                    p_name = "adapter_down.bias"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "adapter_up.weight" in s_name:
                    p_name = "adapter_up.weight"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "adapter_up.bias" in s_name:
                    p_name = "adapter_up.bias"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)

    def expert_soup_forward(self, x):
        adapter_expert = copy.deepcopy(self.experts[0])
        # init one expert with the averaged weights
        for s_name, s_param in adapter_expert.named_parameters():
            s_param.data.copy_(self.parameter_dict[s_name])
        
        return adapter_expert(x)
