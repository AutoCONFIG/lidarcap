import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba


class MambaTemporal(nn.Module):
    def __init__(
        self,
        n_input=1408,
        n_output=72,
        d_model=1024,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(n_input, d_model)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, n_output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.gelu(self.input_proj(x))
        
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = x + residual
        
        out = self.output_proj(self.dropout(x))
        return out
