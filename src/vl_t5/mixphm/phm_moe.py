# @File :phm_moe.py
# @Time :2022/7/11
# @Desc :
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers.activations import get_activation

from vl_t5.compacter.hypercomplex.layers import PHMLinear
from vl_t5.compacter.adapter_utils import Activations
# from redundancy.corr import *
# from losses.r2_loss import off_diagonal, batch_off_diagonal


class PHMExpert(nn.Module):
    def __init__(self, input_dim, output_dim, config,
                 num_experts=4, inference_level=0):
        super(PHMExpert, self).__init__()

        self.num_experts = num_experts
        self.inference_level = inference_level

        # gate mechanism
        self.add_gate = config.add_gate
        if self.add_gate:
            self.gate = nn.Linear(
                input_dim, num_experts, bias=False
            ).float()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        expert = PHMLinear(
            in_features=self.input_dim,
            out_features=self.output_dim,
            bias=True,
            c_init=config.phm_c_init,
            phm_dim=config.hypercomplex_division,  # 4
            learn_phm=config.learn_phm,
            w_init=config.hypercomplex_nonlinearity,
            shared_phm_rule=config.shared_phm_rule,
            factorized_phm=config.factorized_phm,
            shared_W_phm=config.shared_W_phm,  # False
            factorized_phm_rule=config.factorized_phm_rule,
            phm_rank=config.phm_rank,
            phm_init_range=config.phm_init_range,
        )

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(self.num_experts)]
        )

        self.expert_score_weight = nn.Parameter(
            torch.zeros(self.num_experts),
            requires_grad=False
        )
        self.parameter_dict = None

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
            input_x = self.deepspeed_experts[expert_idx](input_x)
            input_x = input_x * prob_x
            return input_x

        x = [forward_expert(x[i], prob_gate[i], i) for i in
             range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, -1)

        return x, balance_loss, gate_load

    def expert_soup_forward(self, hidden_states):
        adapter_expert = copy.deepcopy(self.deepspeed_experts[0])
        # init one expert with the averaged weights
        for s_name, s_param in adapter_expert.named_parameters():
            if s_name in self.parameter_dict.keys():
                s_param.data.copy_(self.parameter_dict[s_name])

        return adapter_expert(hidden_states)

    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight, dim=0)

        if self.config.factorized_phm:
            self.parameter_dict = {
                "phm_rule": 0, "W_left": 0, "W_right": 0, "b": 0}
            if self.config.hypercomplex_nonlinearity == "lora":
                self.parameter_dict["W_lora_fix"] = 0
        else:
            self.parameter_dict = {"phm_rule": 0, "W": 0, "b": 0}

        for idx in range(self.num_experts):
            single_expert = self.deepspeed_experts[idx]

            for s_name, s_param in single_expert.named_parameters():
                if "W_left" in s_name:
                    p_name = "W_left"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "W_right" in s_name:
                    p_name = "W_right"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "W" == s_name:
                    p_name = "W"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "b" in s_name:
                    p_name = "b"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "phm_rule" in s_name:
                    p_name = "phm_rule"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)
                elif "W_lora_fix" in s_name:
                    p_name = "W_lora_fix"
                    self.parameter_dict[p_name] += (weight[idx] * s_param)

    def forward(self, hidden_states, expert_idx=None):
        expert_output = None

        if self.add_gate:
            expert_output, balance_loss, gate_load = self._forward_gate_token(
                hidden_states)

            return expert_output, balance_loss, gate_load

        else:
            if self.training:
                # selected expert
                if expert_idx is None:
                    expert_idx = torch.randint(
                        low=0, high=self.num_experts, size=(1,)
                    ).item()

                if self.expert_score_weight.requires_grad:
                    self.expert_soup()
                    expert_output = self.expert_soup_forward(hidden_states)
                else:
                    expert_output = self.deepspeed_experts[expert_idx](
                        hidden_states)

            else:
                if self.inference_level != 3:
                    result = []
                    for expert_idx in range(self.num_experts):
                        result.append(
                            self.deepspeed_experts[expert_idx](hidden_states)
                        )
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
                        expert_output = result.mean(0)
                        # result_embed = result.permute(1, 2, 3, 0)
                        # result_embed = F.normalize(result_embed, dim=-1)
                        # result_c = result_embed.transpose(-1, -2) @ result_embed
                        #
                        # result_c = (result_c.sum(-1) - torch.diagonal(
                        #     result_c, dim1=2, dim2=3)) / (self.num_experts - 1)
                        #
                        # score = F.softmax(result_c, dim=-1).permute(
                        #     2, 0, 1).unsqueeze(-1)
                        #
                        # expert_output = torch.sum(result * score, dim=0)
                        # expert_idx = score.sum(
                        #     dim=[1, 2]).squeeze().argmax(0).item()
                        # expert_output = result[expert_idx]

                        # # result_embed = result.view(
                        # #     self.num_experts, -1, self.output_dim
                        # # )
                        #
                        # p_corrs = []
                        # for _idx in range(self.num_experts):
                        #     result_idx = F.normalize(result[_idx], dim=-1)
                        #
                        #     c = result_idx.transpose(-1, -2) @ result_idx
                        #     off_diag = batch_off_diagonal(
                        #         c).pow_(2).sum(-1).mean() / result_idx.shape[0]
                        #     p_corrs.append(off_diag)
                        #
                        # # target_embed = hidden_states.mean(1)  # [bs, dim]
                        # # target_embed = F.normalize(target_embed)
                        # #
                        # # target_sim = torch.mm(target_embed, target_embed.t())
                        # # up_index = torch.triu_indices(
                        # #     target_sim.shape[0],
                        # #     target_sim.shape[1]
                        # # )
                        # # target_sim = target_sim[up_index[0], up_index[1]]
                        # #
                        # # p_corrs = []
                        # # for expert_idx in range(self.num_experts):
                        # #     source_embed = result[expert_idx].mean(1)
                        # #     # source_embed = F.normalize(source_embed)
                        # #
                        # #     sim = linear_CKA(target_embed, source_embed)
                        # #
                        # #     # source_sim = torch.mm(
                        # #     #     source_embed, source_embed.t())
                        # #     # up_index = torch.triu_indices(source_sim.shape[0],
                        # #     #                               source_sim.shape[1])
                        # #     # source_sim = source_sim[up_index[0], up_index[1]]
                        # #     #
                        # #     # p_corr = pearson_corr(
                        # #     #     source_sim,
                        # #     #     target_sim
                        # #     # ).item()
                        # #
                        # #     p_corrs.append(1 - sim)
                        #
                        # # expert_idx = p_corrs.index(max(p_corrs))
                        # # expert_output = result[expert_idx]
                        # p_corrs = F.softmax(torch.stack(p_corrs), dim=0)
                        # expert_output = torch.sum(
                        #    result * p_corrs.view(-1, 1, 1, 1), dim=0)

                elif self.inference_level == 3:
                    self.expert_soup()
                    expert_output = self.expert_soup_forward(hidden_states)

            return expert_output, expert_idx


class PHMAdapterMoE(nn.Module):
    def __init__(self, config, num_experts=4, inference_level=0,
                 sharing_down=0, sharing_up=1):
        super(PHMAdapterMoE, self).__init__()
        self.ada_alpha = config.ada_alpha
        self.add_gate = config.add_gate

        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor

        self.activation = Activations(config.non_linearity.lower())

        if sharing_down == 1:
            self.MoA_A = PHMExpert(
                self.input_dim, self.down_sample_size, config,
                1, inference_level
            )
        else:
            self.MoA_A = PHMExpert(
                self.input_dim, self.down_sample_size, config,
                num_experts, inference_level
            )

        if sharing_up == 1:
            self.MoA_B = PHMExpert(
                self.down_sample_size, self.input_dim, config,
                1, inference_level
            )
        else:
            self.MoA_B = PHMExpert(
                self.down_sample_size, self.input_dim, config,
                num_experts, inference_level
            )

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, is_r2loss=False):
        residual = x
        balance_loss = None

        if self.add_gate:
            result, balance_loss_a, gate_load = self.MoA_A(x)

            result = self.activation(result)
            result = self.dropout(result)

            result, balance_loss_b, gate_load = self.MoA_B(result)

            return result + residual, balance_loss_a + balance_loss_b

        else:
            result, expert_idx = self.MoA_A(x)

            result = self.activation(result)
            result = self.dropout(result)

            # result, _ = self.MoA_B(result, expert_idx)
            result, _ = self.MoA_B(result)

            result = self.ada_alpha * result

            if is_r2loss:
                return result + residual, (result, residual)

            return result + residual, balance_loss
