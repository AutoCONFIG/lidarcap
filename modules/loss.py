import torch
import torch.nn as nn

from modules.smpl import SMPL, get_smpl_model
from modules.geometry import axis_angle_to_rotation_matrix
from libs.chamfer_dist import ChamferDistanceL1
from pointnet2_ops import pointnet2_utils
from config import get_cfg


def batch_pc_normalize(pc):
    pc -= pc.mean(1, True)
    return pc / pc.norm(dim=-1, keepdim=True).max(1, True)[0]


class TemporalConsistencyLoss(nn.Module):
    def __init__(
        self,
        velocity_weight=1.0,
        acceleration_weight=0.5,
        bone_length_weight=0.3
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.bone_length_weight = bone_length_weight

        # SMPL骨骼连接对（基于标准SMPL 24关节）
        # 格式: (父关节, 子关节)
        self.skeleton_pairs = [
            # 脊柱
            (0, 1), (0, 2), (0, 3),      # 根节点到髋部
            (1, 4), (2, 5), (3, 6),      # 髋部到大腿
            (4, 7), (5, 8), (6, 9),      # 大腿到小腿
            (7, 10), (8, 11), (9, 12),   # 小腿到脚踝
            # 左臂
            (13, 14), (14, 15),          # 肩->肘->腕
            # 右臂
            (16, 17), (17, 18),          # 肩->肘->腕
            # 手部
            (15, 19), (15, 20),          # 左手
            (18, 21), (18, 22),          # 右手
        ]
    
    def forward(self, joints):
        loss_dict = {}
        
        velocity = joints[:, 1:] - joints[:, :-1]
        
        if velocity.size(1) >= 2:
            velocity_diff = velocity[:, 1:] - velocity[:, :-1]
            loss_dict['loss_velocity'] = torch.mean(velocity_diff ** 2) * self.velocity_weight
        
        if joints.size(1) >= 3:
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            if acceleration.size(1) >= 2:
                accel_diff = acceleration[:, 1:] - acceleration[:, :-1]
                loss_dict['loss_acceleration'] = torch.mean(accel_diff ** 2) * self.acceleration_weight
        
        if joints.size(1) >= 2:
            bone_loss = 0
            for ja, jb in self.skeleton_pairs:
                bone_lengths = torch.norm(
                    joints[:, :, jb] - joints[:, :, ja], dim=-1
                )
                bone_loss += torch.var(bone_lengths, dim=1).mean()
            loss_dict['loss_bone'] = bone_loss * self.bone_length_weight
        
        if len(loss_dict) > 0:
            total = sum(loss_dict.values())
            loss_dict['loss_temporal'] = total
        else:
            loss_dict['loss_temporal'] = torch.tensor(0.0, device=joints.device)
        
        return loss_dict


class Loss(nn.Module):
    _smpl_instance = None  # 类级别的SMPL单例

    def __init__(self, cfg=None):
        super().__init__()

        # 从配置文件读取参数
        if cfg is None:
            cfg = get_cfg()

        temporal_cfg = cfg.TEMPORAL_LOSS

        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        self.chamfer_loss = ChamferDistanceL1()

        # 使用单例SMPL模型
        if Loss._smpl_instance is None:
            Loss._smpl_instance = SMPL()
        self.smpl = Loss._smpl_instance

        # 从配置文件读取时序损失参数
        self.temporal_loss = TemporalConsistencyLoss(
            velocity_weight=temporal_cfg.velocity_weight,
            acceleration_weight=temporal_cfg.acceleration_weight,
            bone_length_weight=temporal_cfg.bone_length_weight
        )
        self.temporal_weight = temporal_cfg.weight

    def forward(self, **kw):
        B, T = kw['human_points'].shape[:2]
        gt_pose = kw['pose']
        gt_rotmats = axis_angle_to_rotation_matrix(
            gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)

        gt_full_joints = kw['full_joints'].reshape(B, T, 24, 3)

        details = {}

        if 'pred_rotmats' in kw:
            pred_rotmats = kw['pred_rotmats'].reshape(B, T, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            details['loss_param'] = loss_param

            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10), device=gt_pose.device))
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(B, T, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_full_joints)
            details['loss_smpl_joints'] = loss_smpl_joints

        if 'pred_full_joints' in kw:
            pred_full_joints = kw['pred_full_joints']
            loss_full_joints = self.criterion_joints(
                pred_full_joints, gt_full_joints)
            details['loss_full_joints'] = loss_full_joints

        if 'gen_points' in kw:
            pred_points = kw['gen_points'].reshape(B * T, -1, 3)
            M = pred_points.shape[1]
            K = M - 24
            assert K > 0, "Predicted coarse point count must be > 24 (joint count)"

            smpl_rotmats = gt_rotmats.reshape(B * T, 24, 3, 3)
            smpl_vertices = self.smpl(
                smpl_rotmats,
                torch.zeros((B * T, 10), device=gt_pose.device)
            )

            with torch.no_grad():
                smpl_vertices_down = pointnet2_utils.furthest_point_sample(
                    smpl_vertices.contiguous(), K
                )
                smpl_surface = pointnet2_utils.gather_operation(
                    smpl_vertices.transpose(1, 2).contiguous(), smpl_vertices_down
                ).transpose(1, 2).contiguous()

            joint_points = gt_full_joints.reshape(B * T, 24, 3)

            merged_gt_points = torch.cat([smpl_surface, joint_points], dim=1)

            loss_chamfer = self.chamfer_loss(pred_points, merged_gt_points)
            details['loss_chamfer_smpl'] = loss_chamfer

        if 'pred_full_joints' in kw:
            pred_full_joints = kw['pred_full_joints']
            temporal_dict = self.temporal_loss(pred_full_joints)
            details.update(temporal_dict)

        loss = 0
        for k, v in details.items():
            if k == 'loss_temporal':
                loss += self.temporal_weight * v
            else:
                loss += v
        
        details['loss'] = loss
        return loss, details
