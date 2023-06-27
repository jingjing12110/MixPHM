import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.dropout(x) - self.bias
        return self.lin(x)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    
    def __init__(self, n_exps=8, layers=[768, 300], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        
        self.n_exps = n_exps
        self.noisy_gating = noise
        
        self.experts = nn.ModuleList(
            [PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)]
        )
        
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps),
                                   requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps),
                                    requires_grad=True)
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(
                x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        gates = F.softmax(logits, dim=-1)
        return gates
    
    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_E)
        
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in
                          range(self.n_exps)]  # [(B, 1, D)]
        
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        
        return multiple_outputs.sum(dim=-2)


class MoEAdaptor(nn.Module):
    def __init__(self, b_dim=300, act=None, num_expert=4):
        super().__init__()
        
        self.act = act
        if act is not None:
            self.act = get_activation(act)
            
        self.MoE_A = MoEAdaptorLayer(
            n_exps=num_expert,
            layers=[768, b_dim],
        )
        
        self.MoE_B = MoEAdaptorLayer(
            n_exps=num_expert,
            layers=[b_dim, 768],
        )

        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        residual = x
        
        result = self.MoE_A(x)
        
        if self.act is not None:
            result = self.act(result)

        # result = self.dropout(result)
        
        result = self.MoE_B(result)
       
        return result + residual

