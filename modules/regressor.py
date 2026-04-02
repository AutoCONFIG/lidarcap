from modules.geometry import rot6d_to_rotmat
from modules.st_gcn import STGCN
from modules.mamba_temporal import MambaTemporal
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from typing import Tuple
from pointnet2_ops import pointnet2_utils
from .Transformer import PCTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_cfg

# 兼容新旧版本的 autocast
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


class DisableAutocast:
    """上下文管理器，用于禁用 autocast"""
    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        torch.set_autocast_enabled(False)
        return self
    def __exit__(self, *args):
        torch.set_autocast_enabled(self.prev)

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

class PoinTr(nn.Module):
    def __init__(self, trans_dim=384, num_query=224, knn_layer=1):
        super().__init__()
        self.trans_dim = trans_dim
        self.knn_layer = knn_layer
        self.num_query = num_query

        self.base_model = PCTransformer(
            in_chans=3,
            embed_dim=self.trans_dim,
            depth=[6, 8],
            drop_rate=0.,
            num_query=self.num_query,
            knn_layer=self.knn_layer
        )

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)

    def forward(self, xyz):
        B, T, N, C = xyz.shape
        xyz = xyz.reshape(B * T, N, C)

        # PoinTr 内部有 CUDA 操作，需要禁用 autocast
        with DisableAutocast():
            q, coarse_point_cloud = self.base_model(xyz)

        B_T, M, C = q.shape

        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1)[0]

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, M, -1),
            q,
            coarse_point_cloud
        ], dim=-1)

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B_T * M, -1))
        rebuild_feature = rebuild_feature.reshape(B_T, M, -1)

        rebuild_feature = rebuild_feature.view(B, T, M, -1)
        coarse_point_cloud = coarse_point_cloud.view(B * T, M, 3)

        with torch.no_grad():
            K_input = 128
            with DisableAutocast():
                sampled_input = fps(xyz, K_input)

        merged_point_cloud = torch.cat([coarse_point_cloud, sampled_input], dim=1)
        merged_point_cloud = merged_point_cloud.view(B, T, -1, 3)

        return rebuild_feature, coarse_point_cloud, merged_point_cloud
        
class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data['human_points']

        B, T, N, _ = x.shape
        x = x.reshape(-1, N, 3)
        xyz, features = self._break_up_pc(x)

        # PointNet2 CUDA 操作不支持半精度，需要禁用 autocast
        with DisableAutocast():
            for module in self.SA_modules:
                xyz, features = module(xyz, features)

        features = features.squeeze(-1).reshape(B, T, -1)
        return features

class Regressor(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        # 从配置文件读取参数
        if cfg is None:
            cfg = get_cfg()

        mamba_cfg = cfg.MAMBA
        pointr_cfg = cfg.PoinTr

        self.encoder = PointNet2Encoder()
        self.pose_s1 = MambaTemporal(
            n_input=1024 + 384,
            n_output=24 * 3,
            d_model=mamba_cfg.d_model,
            d_state=mamba_cfg.d_state,
            d_conv=mamba_cfg.d_conv,
            expand=mamba_cfg.expand,
            n_layers=mamba_cfg.n_layers,
            dropout=mamba_cfg.dropout
        )
        self.pose_s2 = STGCN(3 + 1024 + 384)
        self.pointr = PoinTr(
            trans_dim=pointr_cfg.trans_dim,
            num_query=pointr_cfg.num_query,
            knn_layer=pointr_cfg.knn_layer
        )

    def forward(self, data):
        pred = {}

        # 1. 使用 PoinTr 获取 coarse 点云和其特征
        recon_point_feat, coarse_point_cloud, merged_point_cloud = self.pointr(data['human_points'])  # (B, T, M, C), (B, T, M, 3), (B, T, M+K_input, 3)
        # pred['coarse_points'] = coarse_point_cloud  # 用于点云损失
        pred['gen_points'] = merged_point_cloud  # 加入合并点云，供下游模块使用

        # 2. 编码原始点云，提取原始点云特征
        orig_feat = self.encoder(data)  # (B, T, 1024)
        recon_feat = recon_point_feat.mean(dim=2)  # (B, T, 384)

        # 3. 融合两个点云特征作为 Mamba 输入
        x = torch.cat([orig_feat, recon_feat], dim=-1)  # (B, T, 1024 + 384)

        # 4. 回归关节点
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24 * 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)

        # 5. 姿态估计
        rot6ds = self.pose_s2(torch.cat((
            full_joints.reshape(B, T, 24, 3),
            x.unsqueeze(-2).repeat(1, 1, 24, 1)
        ), dim=-1))  # (B, T, 24, D)

        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))           # (B*T*24, D)
        rotmats = rot6d_to_rotmat(rot6ds)                      # (B*T*24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)

        # 返回最终结果
        pred = {**data, **pred}
        return pred
