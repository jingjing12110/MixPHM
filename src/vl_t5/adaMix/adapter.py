from torch import nn
from transformers.activations import get_activation


class Adapter(nn.Module):
    def __init__(self, dim, r, act):
        super().__init__()
        self.adapter_A = nn.Linear(dim, r)
        self.act = get_activation(act)
        self.adapter_B = nn.Linear(r, dim)
        
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = x
        
        result = self.adapter_A(x)
        result = self.act(result)

        # result = self.dropout(result)
        
        result = self.adapter_B(result)
        
        return result + residual
