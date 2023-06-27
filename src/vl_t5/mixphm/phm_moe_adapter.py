# @File :phm_moe_adapter.py
# @Time :2022/7/13
# @Desc :
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from vl_t5.compacter.hypercomplex.layers import PHMLinear
from vl_t5.compacter.adapter_utils import Activations

# from moe_lib.gates.noisy_gate import NoisyGate
# from moe_lib.moe import SparseDispatcher


class PHMAdapterExpert(nn.Module):
    """."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        
        self.down_sampler = PHMLinear(
            in_features=self.input_dim,  # 768
            out_features=self.down_sample_size,  # bottleneck
            bias=True,
            c_init=config.phm_c_init,
            phm_dim=config.hypercomplex_division,  # 4
            learn_phm=config.learn_phm,
            w_init=config.hypercomplex_nonlinearity,
            shared_phm_rule=config.shared_phm_rule,
            factorized_phm=config.factorized_phm,
            shared_W_phm=config.shared_W_phm,
            factorized_phm_rule=config.factorized_phm_rule,
            phm_rank=config.phm_rank,
            phm_init_range=config.phm_init_range,
        )
        self.up_sampler = PHMLinear(
            in_features=self.down_sample_size,
            out_features=self.input_dim,
            bias=True,
            c_init=config.phm_c_init,
            phm_dim=config.hypercomplex_division,
            learn_phm=config.learn_phm,
            w_init=config.hypercomplex_nonlinearity,
            shared_phm_rule=config.shared_phm_rule,
            factorized_phm=config.factorized_phm,
            shared_W_phm=config.shared_W_phm,
            factorized_phm_rule=config.factorized_phm_rule,
            phm_rank=config.phm_rank,
            phm_init_range=config.phm_init_range,
        )
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        residual = x
        
        z = self.down_sampler(x)
        
        z = self.activation(z)
        z = self.dropout(z)
        
        z = self.up_sampler(z)
        
        return z + residual


class PHMAdapterMoE(nn.Module):
    def __init__(self, config, num_experts=4, inference_level=0,
                 sharing_down=0, sharing_up=1):
        super(PHMAdapterMoE, self).__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.inference_level = inference_level
        
        self.num_experts = num_experts
        
        # gate mechanism
        self.add_gate = config.add_gate
        if self.add_gate:
            # self.gate = NoisyGate(
            #     self.input_dim, num_experts, world_size=1
            # )
            self.gate = nn.Linear(
                self.input_dim, num_experts, bias=False).float()
        
        # phm_adapter_expert = PHMAdapterExpert(config)
        self.experts = nn.ModuleList([
            PHMAdapterExpert(config)
            for i in range(self.num_experts)
        ])
        
        self.expert_score_weight = nn.Parameter(
            torch.zeros(self.num_experts),
            requires_grad=False
        )
        self.parameter_dict = None
    
    def forward(self, x, is_r2loss=False):
        expert_output = None
        balance_loss = None
        
        if self.add_gate:
            expert_output, _, gate_load = self._forward_gate_token(x)
            
            # bs, seq_len = x.shape[:2]
            # inp = x.view(bs * seq_len, -1)
            #
            # gates, balance_loss = self.gate(inp)
            #
            # dispatcher = SparseDispatcher(self.num_experts, gates)
            # expert_inputs = dispatcher.dispatch(inp)
            # expert_output = [
            #     self.experts[i](expert_inputs[i])
            #     for i in range(self.num_experts)
            # ]
            # expert_output = dispatcher.combine(expert_output)
            # expert_output = expert_output.view(bs, seq_len, -1)
            
            return expert_output, balance_loss
        
        else:
            if self.training:
                # selected expert
                expert_idx = torch.randint(
                    low=0, high=self.num_experts, size=(1,)
                ).item()
                
                if self.expert_score_weight.requires_grad:
                    self.adapter_merging()
                    expert_output = self.expert_forward(x)
                else:
                    expert_output = self.experts[expert_idx](x)
            
            else:
                if self.inference_level != 3:
                    result = []
                    for expert_idx in range(self.num_experts):
                        result.append(self.experts[expert_idx](x))
                    # [num_experts, bs, *, 768]
                    result = torch.stack(result, dim=0)
                    
                    if self.inference_level == 0:  # token level
                        mask = torch.randint(
                            0, self.num_experts,
                            size=(result.size(1), result.size(2)),
                            device=result.device
                        )
                        for i in range(self.num_experts):
                            expert_mask = mask.eq(i)
                            result[i] *= expert_mask.unsqueeze(-1)
                        expert_output = result.sum(0)
                    
                    elif self.inference_level == 1:  # sentence level
                        mask = torch.randint(
                            0, self.num_experts,
                            size=(result.size(1),),
                            device=result.device
                        )
                        for i in range(self.num_experts):
                            expert_mask = mask.eq(i)
                            result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                        expert_output = result.sum(0)

                    elif self.inference_level == 2:
                        # weight = F.softmax(self.expert_score_weight)
                        expert_output = result.mean(0)
                
                elif self.inference_level == 3:
                    self.adapter_merging()
                    expert_output = self.expert_forward(x)
            
            return expert_output, balance_loss

    def expert_forward(self, x):
        adapter_expert = copy.deepcopy(self.experts[0])
        # init one expert with the averaged weights
        for s_name, s_param in adapter_expert.named_parameters():
            # if s_name in self.parameter_dict.keys():
            s_param.data.copy_(self.parameter_dict[s_name])
    
        return adapter_expert(x)

    def adapter_merging(self):
        # weight averaging
        weight = F.softmax(self.expert_score_weight)

        self.parameter_dict = {
            "down_sampler.phm_rule": 0,
            "down_sampler.W_left": 0,
            "down_sampler.W_right": 0,
            "down_sampler.b": 0,
            "up_sampler.phm_rule": 0,
            "up_sampler.W_left": 0,
            "up_sampler.W_right": 0,
            "up_sampler.b": 0,
        }
        for idx in range(self.num_experts):
            single_expert = self.experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "down_sampler.phm_rule" in s_name:
                    p_name = "down_sampler.phm_rule"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "down_sampler.W_left" in s_name:
                    p_name = "down_sampler.W_left"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "down_sampler.W_right" in s_name:
                    p_name = "down_sampler.W_right"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "down_sampler.b" in s_name:
                    p_name = "down_sampler.b"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "up_sampler.phm_rule" in s_name:
                    p_name = "up_sampler.phm_rule"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "up_sampler.W_left" in s_name:
                    p_name = "up_sampler.W_left"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "up_sampler.W_right" in s_name:
                    p_name = "up_sampler.W_right"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "up_sampler.b" in s_name:
                    p_name = "up_sampler.b"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
        
        # self.parameter_dict = {}
        # for idx in range(self.num_experts):
        #     single_expert = self.experts[idx]
        #     for s_name, s_param in single_expert.named_parameters():
        #         if s_name in self.parameter_dict.keys():
        #             self.parameter_dict[s_name] += (weight[idx] * s_param)
        #         else:
        #             self.parameter_dict[s_name] = (weight[idx] * s_param)

    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()
    
        x = x.view(-1, dim)
        logits_gate = self.gate(x)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)  # [bs * seq_len]
        
        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        # a list of length self.num_experts
        x = x.split(num_tokens.tolist(), dim=0)
        
        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)
    
        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)
    
        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx](input_x)
            input_x = input_x * prob_x
            return input_x
    
        x = [forward_expert(x[i], prob_gate[i], i) for i in
             range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, -1)
    
        return x, balance_loss, gate_load
