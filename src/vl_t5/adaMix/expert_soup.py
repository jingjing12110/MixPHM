import copy

import torch
import torch.nn.functional as F

from torch import nn
from transformers.activations import get_activation


class MixtureSoup(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1, inference_level=0):
        super(MixtureSoup, self).__init__()
        
        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(num_local_experts)])
        
        self.num_local_experts = num_local_experts
        self.inference_level = inference_level
        
        self.expert_score_weight = torch.nn.Parameter(
            torch.zeros(self.num_local_experts),
            requires_grad=False
        )
        self.parameter_dict = None

    def get_expert_by_idx(self, idx):
        return self.deepspeed_experts[idx]
    
    def expert_soup_forward(self, hidden_states):
        output = F.linear(
            hidden_states,
            self.parameter_dict["weight"],
            self.parameter_dict["bias"]
        )
        return output
    
    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight)
        self.parameter_dict = {"weight": 0, "bias": 0}
        for idx in range(self.num_local_experts):
            single_expert = self.deepspeed_experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                else:
                    p_name = "bias"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
    
    def forward(self, hidden_states, expert_idx=None):
        expert_output = None
        if self.deepspeed_experts[0].training:
            # selected expert
            if expert_idx is None:
                expert_idx = torch.randint(
                    low=0, high=self.num_local_experts,
                    size=(1,)
                ).item()
            
            if self.expert_score_weight.requires_grad:
                self.expert_soup()
                expert_output = self.expert_soup_forward(hidden_states)
            else:
                expert_output = self.get_expert_by_idx(expert_idx)(hidden_states)
        
        else:
            if self.inference_level != 3:
                result = []
                for expert_idx in range(self.num_local_experts):
                    temp = self.get_expert_by_idx(expert_idx)(hidden_states)
                    result.append(temp)
                result = torch.stack(result, dim=0)
                
                if self.inference_level == 0:  # token level
                    mask = torch.randint(0, self.num_local_experts,
                                         size=(result.size(1), result.size(2)),
                                         device=result.device)
                    for i in range(self.num_local_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1)
                    expert_output = result.sum(0)
                elif self.inference_level == 1:  # sentence level
                    mask = torch.randint(0, self.num_local_experts,
                                         size=(result.size(1),),
                                         device=result.device)
                    for i in range(self.num_local_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                    expert_output = result.sum(0)
            
            elif self.inference_level == 3:
                self.expert_soup()
                expert_output = self.expert_soup_forward(hidden_states)
        
        return expert_output, expert_idx


class ExpertSoup(nn.Module):
    def __init__(self, dim, r, act=None, num_expert=4, inference_level=0,
                 sharing_down=0, sharing_up=0):
        super().__init__()
        
        self.act = act
        if sharing_down == 1:
            self.MoA_A = MixtureSoup(
                nn.Linear(dim, r), 1, inference_level)
        else:
            self.MoA_A = MixtureSoup(
                nn.Linear(dim, r), num_expert,
                inference_level)
        if act is not None:
            self.act = get_activation(act)
        
        if sharing_up == 1:
            self.MoA_B = MixtureSoup(
                nn.Linear(r, dim), 1, inference_level)
        else:
            self.MoA_B = MixtureSoup(
                nn.Linear(r, dim), num_expert, inference_level)
        
        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, is_r2loss=False):
        residual = x
        
        result, _ = self.MoA_A(x)
        
        if self.act is not None:
            result = self.act(result)
        
        # result = self.dropout(result)
        
        # result, _ = self.MoA_B(result, expert_idx)
        result, _ = self.MoA_B(result)
        
        if is_r2loss:
            return result + residual, (result, residual)
        
        return result + residual

