import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaTemporal(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, d_state=16, d_conv=4, expand=2):
        """
        Mamba-based temporal fusion module (Bidirectional)
        
        Args:
            n_input: Input feature dimension (e.g., 1024 + 384 = 1408)
            n_output: Output dimension (e.g., 24 * 3 = 72)
            n_hidden: Hidden dimension for the Mamba model
            d_state: SSM state expansion factor
            d_conv: Local convolution width
            expand: Block expansion factor
        """
        super().__init__()
        self.n_hidden = n_hidden
        
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout()
        
        try:
            from mamba_ssm import Mamba
            # 前向Mamba
            self.mamba_fwd = Mamba(
                d_model=n_hidden,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                batch_first=True
            )
            # 后向Mamba
            self.mamba_bwd = Mamba(
                d_model=n_hidden,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                batch_first=True
            )
        except ImportError:
            raise ImportError(
                "mamba_ssm is not installed. Install with: pip install mamba-ssm"
            )
        
        # 双向输出拼接，所以是 n_hidden * 2
        self.linear2 = nn.Linear(n_hidden * 2, n_output)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)
        
        Returns:
            Output tensor of shape (B, T, n_output)
        """
        x = F.relu(self.dropout(self.linear1(x)), inplace=True)
        
        # 前向处理
        x_fwd = self.mamba_fwd(x)
        
        # 后向处理 - 反转时间维度
        x_bwd = self.mamba_bwd(x.flip(1)).flip(1)
        
        # 拼接双向输出
        x = torch.cat([x_fwd, x_bwd], dim=-1)
        
        x = self.linear2(x)
        
        return x
