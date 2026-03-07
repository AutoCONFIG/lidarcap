from modules.geometry import rot6d_to_rotmat
from modules.st_gcn import STGCN
from modules.mamba import MambaTemporal
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

def mem_check(note=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[GPU] {note:<30s} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

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
        xyz = xyz.reshape(B * T, N, C)  # (B*T, N, 3)
        # mem_check("Input reshaped")

        # Transformer编码器
        q, coarse_point_cloud = self.base_model(xyz)  # B*T, M, C  and B*T, M, 3
        # mem_check("After PCTransformer")

        B_T, M, C = q.shape

        # 提取全局上下文特征
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)  # (B*T, M, 1024)
        # mem_check("After increase_dim")
        global_feature = torch.max(global_feature, dim=1)[0]  # (B*T, 1024)

        # 构建 coarse point 的特征（用于后续下游任务）
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, M, -1),  # (B*T, M, 1024)
            q,                                              # (B*T, M, C)
            coarse_point_cloud                              # (B*T, M, 3)
        ], dim=-1)  # (B*T, M, 1027 + C)
        # mem_check("After feature concat")

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B_T * M, -1))  # (B*T*M, C)
        rebuild_feature = rebuild_feature.reshape(B_T, M, -1)                    # (B*T, M, C)
        # mem_check("After reduce_map")

        # reshape回时间序列结构
        rebuild_feature = rebuild_feature.view(B, T, M, -1)
        coarse_point_cloud = coarse_point_cloud.view(B * T, M, 3)
        # mem_check("Final reshape done")

        # ==== 融合 coarse + 输入点云采样 ====
        with torch.no_grad():
            K_input = 128  # 可调，原始点云中采样点数
            sampled_input = fps(xyz, K_input)  # (B*T, K_input, 3)

        # print("coarse_point_cloud.shape:", coarse_point_cloud.shape)
        # print("sampled_input.shape:", sampled_input.shape)

        # 拼接生成点与原始采样点
        merged_point_cloud = torch.cat([coarse_point_cloud, sampled_input], dim=1)  # (B*T, M+K_input, 3)
        merged_point_cloud = merged_point_cloud.view(B, T, -1, 3)  # (B, T, M+K_input, 3)


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
        # x = data['human_points']  # (B, T, N, 3)
        # 优先使用重建点云（如存在），否则使用原始点云
        # if 'gen_points' in data:
        #     x = data['gen_points']  # (B, T, N, 3)
        # else:
        x = data['human_points']  # (B, T, N, 3)
        
        B, T, N, _ = x.shape
        x = x.reshape(-1, N, 3)  # (B * T, N, 3)
        xyz, features = self._break_up_pc(x)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.squeeze(-1).reshape(B, T, -1)
        return features

# class ModifiedCAA_Module(nn.Module):
#     """Modified Channel-wise Affinity Attention Module (without modifying original features)"""

#     def __init__(self, in_dim, proj_dim=128, gamma=2.0, temperature=0.05):
#         super(ModifiedCAA_Module, self).__init__()

#         # 差异放大系数和 softmax 温度参数作为可学习参数
#         self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
#         self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

#         # 原始点云特征映射
#         self.proj_ori = nn.Sequential(
#             nn.Conv1d(in_dim, proj_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(proj_dim),
#             nn.ReLU()
#         )

#         # 生成点云特征映射
#         self.proj_gen = nn.Sequential(
#             nn.Conv1d(in_dim, proj_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(proj_dim),
#             nn.ReLU()
#         )

#         # 生成点云用于加权的 value 映射
#         self.value_conv = nn.Sequential(
#             nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(in_dim),
#             nn.ReLU()
#         )

#         # （可选）拼接后融合维度 MLP，如果你想输出是 (B, C, N) 而不是 (B, 2C, N)，可用：
#         self.fusion_mlp = nn.Sequential(
#             nn.Conv1d(2 * in_dim, in_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(in_dim),
#             nn.ReLU()
#         )

#     def forward(self, feat_ori, feat_gen):
#         """
#         输入:
#             feat_ori: 原始点云特征 (B, C, N)
#             feat_gen: 生成点云特征 (B, C, N)
#         输出:
#             fused_feat: 融合后的特征 (B, 2C, N)（或者 (B, C, N) 如果启用 fusion_mlp）
#         """
#         B, C, N = feat_ori.size()

#         # 特征投影
#         proj_ori = self.proj_ori(feat_ori)  # (B, C', N)
#         proj_gen = self.proj_gen(feat_gen)  # (B, C', N)

#         # 单位向量归一化
#         proj_ori_norm = F.normalize(proj_ori, p=2, dim=1)
#         proj_gen_norm = F.normalize(proj_gen, p=2, dim=1)

#         # 相似度 → 差异注意力
#         sim = torch.bmm(proj_ori_norm.transpose(1, 2), proj_gen_norm)  # (B, N, N)
#         attn = (1.0 - sim).clamp(min=1e-6)  # 防止为负
#         attn = attn ** self.gamma
#         attn = F.softmax(attn / (self.temperature + 1e-6), dim=-1)

#         # 加权生成特征
#         value = self.value_conv(feat_gen)  # (B, C, N)
#         feat_gen_attn = torch.bmm(value, attn.transpose(1, 2))  # (B, C, N)

#         # 融合：不对 feat_ori 做任何修改
#         fused_feat = torch.cat([feat_ori, feat_gen_attn], dim=1)  # (B, 2C, N)

#         # 如果你启用了 fusion_mlp，并希望输出维度为 C：
#         fused_feat = self.fusion_mlp(fused_feat)  # (B, C, N)

#         return fused_feat


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(n_hidden, n_hidden, n_rnn_layer,
                          batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(n_input, n_hidden)

        self.linear2 = nn.Linear(n_hidden * 2, n_output)

        self.dropout = nn.Dropout()

    def forward(self, x):  # (B, T, D)
        x = self.rnn(F.relu(self.dropout(self.linear1(x)), inplace=True))[0]
        return self.linear2(x)


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = MambaTemporal(1024 + 384, 24 * 3, 1024)  # 使用Mamba替换双向GRU
        self.pose_s2 = STGCN(3 + 1024 + 384)          # 对应拼接后的特征输入
        self.pointr = PoinTr(
            trans_dim=384,
            num_query=224,
            knn_layer=1
        )

    def forward(self, data):
        pred = {}

        # 1. 使用 PoinTr 获取 coarse 点云和其特征
        recon_point_feat, coarse_point_cloud, merged_point_cloud = self.pointr(data['human_points'])  # (B, T, M, C), (B, T, M, 3), (B, T, M+K_input, 3)
        # pred['coarse_points'] = coarse_point_cloud  # 用于点云损失
        pred['gen_points'] = merged_point_cloud  # 加入合并点云，供下游模块使用

        # 2. 编码原始点云，提取原始点云特征
        orig_feat = self.encoder(data)  # (B, T, 1024)
        recon_feat = recon_point_feat.mean(dim=2)  # (B, T, 256)

        # 3. 融合两个点云特征作为 GRU 输入
        x = torch.cat([orig_feat, recon_feat], dim=-1)  # (B, T, 1024 + 256)

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
