import torch
import torch.nn as nn

from modules.smpl import SMPL
from modules.geometry import axis_angle_to_rotation_matrix
from extensions.chamfer_dist import ChamferDistanceL1
from pointnet2_ops import pointnet2_utils


def batch_pc_normalize(pc):
    pc -= pc.mean(1, True)
    return pc / pc.norm(dim=-1, keepdim=True).max(1, True)[0]


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        self.chamfer_loss = ChamferDistanceL1()
        self.smpl = SMPL()

    def forward(self, **kw):
        B, T = kw['human_points'].shape[:2]
        gt_pose = kw['pose']
        gt_rotmats = axis_angle_to_rotation_matrix(
            gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)

        gt_full_joints = kw['full_joints'].reshape(B, T, 24, 3)

        details = {}

        if 'pred_rotmats' in kw:
            # L_{\theta}
            pred_rotmats = kw['pred_rotmats'].reshape(B, T, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            details['loss_param'] = loss_param

            # L_{J_{SMPL}}
            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10)).cuda())
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(B, T, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_full_joints)
            details['loss_smpl_joints'] = loss_smpl_joints

        if 'pred_full_joints' in kw:
            # L_{J}
            pred_full_joints = kw['pred_full_joints']
            loss_full_joints = self.criterion_joints(
                pred_full_joints, gt_full_joints)
            details['loss_full_joints'] = loss_full_joints

        # ==== 点云重建Chamfer损失 ====
        if 'gen_points' in kw:
            pred_points = kw['gen_points'].reshape(B * T, -1, 3)  # (B*T, M, 3)
            M = pred_points.shape[1]
            K = M - 24  # 表面点数
            assert K > 0, "Predicted coarse point count must be > 24 (joint count)"

            smpl_rotmats = gt_rotmats.reshape(B * T, 24, 3, 3)
            smpl_vertices = self.smpl(
                smpl_rotmats,
                torch.zeros((B * T, 10), device=gt_pose.device)
            )  # (B*T, 6890, 3)

            # ==== 下采样 SMPL 顶点点云 ====
            with torch.no_grad():
                smpl_vertices_down = pointnet2_utils.furthest_point_sample(
                    smpl_vertices.contiguous(), K
                )  # (B*T, K)
                smpl_surface = pointnet2_utils.gather_operation(
                    smpl_vertices.transpose(1, 2).contiguous(), smpl_vertices_down
                ).transpose(1, 2).contiguous()  # (B*T, K, 3)

            joint_points = gt_full_joints.reshape(B * T, 24, 3)  # (B*T, 24, 3)

            # 合并监督点云：表面点 + 骨架点
            merged_gt_points = torch.cat([smpl_surface, joint_points], dim=1)  # (B*T, M, 3)

            # 计算 Chamfer 损失
            loss_chamfer = self.chamfer_loss(pred_points, merged_gt_points)
            details['loss_chamfer_smpl'] = loss_chamfer

        # 累加所有损失
        loss = 0
        for _, v in details.items():
            loss += v
        details['loss'] = loss
        return loss, details
