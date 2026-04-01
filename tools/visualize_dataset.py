"""
数据集可视化工具

功能：
1. 可视化点云序列
2. 可视化关节点骨架
3. 可视化SMPL网格
4. 支持交互式查看和保存图片

运行方式:
    # 使用启动脚本（推荐）
    bash scripts/vis_dataset.sh single     # 可视化单帧
    bash scripts/vis_dataset.sh sequence   # 可视化序列
    bash scripts/vis_dataset.sh animation  # 生成动画
    bash scripts/vis_dataset.sh stats      # 查看统计

    # 或直接运行
    python tools/visualize_dataset.py --mode single --index 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm

from config import get_cfg
from datasets.lidarcap_dataset import TemporalDataset, CachedLidarCapDataset


# SMPL骨骼连接定义 (24关节)
SMPL_SKELETON = [
    # 腿部
    (0, 1), (0, 2), (0, 3),      # 根节点 -> 髋部
    (1, 4), (2, 5), (3, 6),      # 髋部 -> 膝盖
    (4, 7), (5, 8), (6, 9),      # 膝盖 -> 脚踝
    (7, 10), (8, 11), (9, 12),   # 脚踝 -> 脚
    # 躯干
    (12, 15),                     # 颈部 (近似)
    # 左臂
    (13, 14), (14, 15),          # 左锁骨 -> 左肘 -> 左腕
    (15, 19), (15, 20),          # 左手
    # 右臂
    (16, 17), (17, 18),          # 右锁骨 -> 右肘 -> 右腕
    (18, 21), (18, 22),          # 右手
    # 颈部和头部
    (12, 15),                     # 脊柱到颈部
]

# 关节名称 (SMPL 24关节)
JOINT_NAMES = [
    'Root',       # 0
    'LHip',       # 1
    'RHip',       # 2
    'Spine1',     # 3
    'LKnee',      # 4
    'RKnee',      # 5
    'Spine2',     # 6
    'LAnkle',     # 7
    'RAnkle',     # 8
    'Spine3',     # 9
    'LFoot',      # 10
    'RFoot',      # 11
    'Neck',       # 12
    'LCollar',    # 13
    'LShoulder',  # 14
    'LElbow',     # 15
    'RCollar',    # 16
    'RShoulder',  # 17
    'RElbow',     # 18
    'LWrist',     # 19
    'LHand',      # 20
    'RWrist',     # 21
    'RHand',      # 22
    'Head',       # 23
]


class DatasetVisualizer:
    """数据集可视化器"""

    def __init__(self, split='train', preload=False):
        """
        Args:
            split: 'train' 或 'test'
            preload: 是否预加载数据到内存
        """
        self.cfg = get_cfg()
        self.split = split

        print(f"[INFO] 加载{split}数据集...")

        if split == 'train':
            dataset_cfg = self.cfg.TrainDataset
        else:
            dataset_cfg = self.cfg.TestDataset

        if preload:
            self.dataset = CachedLidarCapDataset(
                cfg=dataset_cfg,
                dataset=self.cfg.RUNTIME.dataset,
                train=(split == 'train'),
                preload=True
            )
        else:
            self.dataset = TemporalDataset(
                cfg=dataset_cfg,
                dataset=self.cfg.RUNTIME.dataset,
                train=(split == 'train')
            )

        print(f"[INFO] 数据集大小: {len(self.dataset)} 个序列")
        print(f"[INFO] 序列长度: {self.cfg.TrainDataset.seqlen} 帧")

    def get_sample(self, index):
        """获取单个样本"""
        return self.dataset[index]

    def visualize_single_frame(self, index, frame_idx=0, save_path=None):
        """
        可视化单帧数据

        Args:
            index: 样本索引
            frame_idx: 帧索引 (0-15)
            save_path: 保存路径，None则显示
        """
        sample = self.get_sample(index)

        human_points = sample['human_points'][frame_idx].numpy()  # (N, 3)
        full_joints = sample['full_joints'][frame_idx].numpy()    # (24, 3)
        pose = sample['pose'][frame_idx].numpy()                  # (72,)

        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 3, figure=fig)

        # 1. 原始点云
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.scatter(human_points[:, 0], human_points[:, 1], human_points[:, 2],
                   s=2, c='blue', alpha=0.5)
        ax1.set_title(f'Point Cloud\n({human_points.shape[0]} points)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 2. 关节点
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        self._draw_skeleton(ax2, full_joints)
        ax2.set_title('Joint Skeleton')

        # 3. 点云+关节叠加
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3.scatter(human_points[:, 0], human_points[:, 1], human_points[:, 2],
                   s=1, c='blue', alpha=0.3, label='Point Cloud')
        ax3.scatter(full_joints[:, 0], full_joints[:, 1], full_joints[:, 2],
                   s=50, c='red', marker='o', label='Joints')
        ax3.set_title('Point Cloud + Joints')
        ax3.legend()

        # 4. 关节位置表格
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        table_data = [[JOINT_NAMES[i], f'({full_joints[i, 0]:.3f}, {full_joints[i, 1]:.3f}, {full_joints[i, 2]:.3f})']
                     for i in range(24)]
        table = ax4.table(cellText=table_data, colLabels=['Joint', 'Position (x, y, z)'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax4.set_title('Joint Positions')

        # 5. 点云统计
        ax5 = fig.add_subplot(gs[1, 1])
        coords = ['X', 'Y', 'Z']
        means = [human_points[:, i].mean() for i in range(3)]
        stds = [human_points[:, i].std() for i in range(3)]
        x_pos = np.arange(len(coords))
        ax5.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7)
        ax5.bar(x_pos + 0.2, stds, 0.4, label='Std', alpha=0.7)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(coords)
        ax5.set_ylabel('Value')
        ax5.set_title('Point Cloud Statistics')
        ax5.legend()

        # 6. 姿态参数分布
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(pose, bins=30, alpha=0.7, color='green')
        ax6.set_xlabel('Pose Parameter Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Pose Parameters Distribution')

        plt.suptitle(f'Sample {index}, Frame {frame_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] 保存图片: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_sequence(self, index, save_path=None):
        """
        可视化整个序列 (16帧)

        Args:
            index: 样本索引
            save_path: 保存路径
        """
        sample = self.get_sample(index)

        human_points = sample['human_points'].numpy()  # (T, N, 3)
        full_joints = sample['full_joints'].numpy()    # (T, 24, 3)
        T = human_points.shape[0]

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig)

        # 绘制每帧的骨架
        for t in range(min(T, 12)):
            row = t // 4
            col = t % 4
            ax = fig.add_subplot(gs[row, col], projection='3d')
            self._draw_skeleton(ax, full_joints[t], title=f'Frame {t}')

        # 轨迹图：选定关节随时间的位置变化
        ax_traj = fig.add_subplot(gs[3, 0:2])
        key_joints = [0, 12, 15, 18]  # Root, Neck, LElbow, RElbow
        for j_idx in key_joints:
            ax_traj.plot(range(T), full_joints[:, j_idx, 0], '-',
                        label=f'{JOINT_NAMES[j_idx]} X')
            ax_traj.plot(range(T), full_joints[:, j_idx, 1], '--',
                        label=f'{JOINT_NAMES[j_idx]} Y')
        ax_traj.set_xlabel('Frame')
        ax_traj.set_ylabel('Position')
        ax_traj.set_title('Joint Trajectories')
        ax_traj.legend(loc='upper right', fontsize=8)

        # 运动速度
        velocity = np.linalg.norm(full_joints[1:] - full_joints[:-1], axis=-1).mean(axis=-1)
        ax_vel = fig.add_subplot(gs[3, 2])
        ax_vel.plot(range(1, T), velocity, 'b-', linewidth=2)
        ax_vel.fill_between(range(1, T), 0, velocity, alpha=0.3)
        ax_vel.set_xlabel('Frame')
        ax_vel.set_ylabel('Average Velocity')
        ax_vel.set_title('Motion Speed')

        # 点云数量统计
        ax_pts = fig.add_subplot(gs[3, 3])
        ax_pts.bar(range(T), [human_points[t].shape[0] for t in range(T)])
        ax_pts.set_xlabel('Frame')
        ax_pts.set_ylabel('Point Count')
        ax_pts.set_title('Points per Frame')

        plt.suptitle(f'Sequence {index} (Total {T} frames)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] 保存图片: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_animation(self, index, save_path=None):
        """
        生成骨架动画GIF

        Args:
            index: 样本索引
            save_path: 保存路径
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("[ERROR] 需要安装 Pillow: pip install Pillow")
            return

        sample = self.get_sample(index)
        human_points = sample['human_points'].numpy()
        full_joints = sample['full_joints'].numpy()
        T = human_points.shape[0]

        fig = plt.figure(figsize=(16, 6))

        def update(frame):
            fig.clear()

            # 点云
            ax1 = fig.add_subplot(131, projection='3d')
            pts = human_points[frame]
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c='blue', alpha=0.5)
            ax1.set_title(f'Point Cloud (Frame {frame}/{T-1})')
            self._set_axis_equal(ax1, pts)

            # 骨架
            ax2 = fig.add_subplot(132, projection='3d')
            self._draw_skeleton(ax2, full_joints[frame], title=f'Skeleton (Frame {frame})')

            # 叠加
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='blue', alpha=0.3)
            self._draw_skeleton(ax3, full_joints[frame], title='Overlay')

        anim = FuncAnimation(fig, update, frames=T, interval=100)

        if save_path is None:
            save_path = f'tmp/vis_dataset/animation_sample_{index}.gif'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        print(f"[INFO] 保存动画: {save_path}")
        plt.close()

    def visualize_dataset_statistics(self, num_samples=100, save_path=None):
        """
        可视化数据集统计信息

        Args:
            num_samples: 采样数量
            save_path: 保存路径
        """
        print(f"[INFO] 采样 {num_samples} 个样本统计...")

        joint_positions = []
        point_counts = []
        pose_values = []

        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)

        for idx in tqdm(indices, desc="采样中"):
            sample = self.get_sample(idx)
            joint_positions.append(sample['full_joints'].numpy())
            point_counts.append(sample['human_points'].shape[1])
            pose_values.append(sample['pose'].numpy())

        joint_positions = np.array(joint_positions)  # (N, T, 24, 3)
        pose_values = np.array(pose_values)          # (N, T, 72)

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)

        # 1. 关节位置分布
        ax1 = fig.add_subplot(gs[0, 0])
        for i, coord in enumerate(['X', 'Y', 'Z']):
            ax1.hist(joint_positions[:, :, :, i].flatten(), bins=50, alpha=0.5, label=coord)
        ax1.set_xlabel('Position Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Joint Position Distribution')
        ax1.legend()

        # 2. 姿态参数分布
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(pose_values.flatten(), bins=50, alpha=0.7, color='green')
        ax2.set_xlabel('Pose Parameter Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Pose Parameter Distribution')

        # 3. 点云数量分布
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(point_counts, bins=30, alpha=0.7, color='purple')
        ax3.axvline(np.mean(point_counts), color='red', linestyle='--',
                   label=f'Mean: {np.mean(point_counts):.0f}')
        ax3.set_xlabel('Point Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Point Count per Frame')
        ax3.legend()

        # 4. 各关节位置范围
        ax4 = fig.add_subplot(gs[1, 0])
        joint_ranges = joint_positions.std(axis=(0, 1))  # (24, 3)
        x_pos = np.arange(24)
        width = 0.25
        ax4.bar(x_pos - width, joint_ranges[:, 0], width, label='X', alpha=0.7)
        ax4.bar(x_pos, joint_ranges[:, 1], width, label='Y', alpha=0.7)
        ax4.bar(x_pos + width, joint_ranges[:, 2], width, label='Z', alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([JOINT_NAMES[i][:4] for i in range(24)], rotation=45, fontsize=7)
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Joint Position Variability')
        ax4.legend()

        # 5. 运动幅度统计
        ax5 = fig.add_subplot(gs[1, 1])
        motion = np.linalg.norm(joint_positions[:, 1:] - joint_positions[:, :-1], axis=-1)
        motion_per_joint = motion.mean(axis=(0, 1))  # (24,)
        ax5.bar(range(24), motion_per_joint, alpha=0.7)
        ax5.set_xticks(range(24))
        ax5.set_xticklabels([JOINT_NAMES[i][:4] for i in range(24)], rotation=45, fontsize=7)
        ax5.set_ylabel('Average Motion')
        ax5.set_title('Motion per Joint')

        # 6. 数据摘要
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        summary = f"""
        Dataset Statistics Summary
        =========================
        Total Samples: {len(self.dataset)}
        Sequence Length: {self.cfg.TrainDataset.seqlen}
        Points per Frame: {np.mean(point_counts):.0f} (avg)

        Joint Position Range:
          X: [{joint_positions[:, :, :, 0].min():.3f}, {joint_positions[:, :, :, 0].max():.3f}]
          Y: [{joint_positions[:, :, :, 1].min():.3f}, {joint_positions[:, :, :, 1].max():.3f}]
          Z: [{joint_positions[:, :, :, 2].min():.3f}, {joint_positions[:, :, :, 2].max():.3f}]

        Pose Parameter Range:
          Min: {pose_values.min():.3f}
          Max: {pose_values.max():.3f}
          Mean: {pose_values.mean():.3f}
          Std: {pose_values.std():.3f}
        """
        ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle('Dataset Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] 保存图片: {save_path}")
        else:
            plt.show()

        plt.close()

    def _draw_skeleton(self, ax, joints, title=None):
        """绘制骨架"""
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                  s=30, c='red', marker='o')

        # 绘制骨骼连线
        for i, j in SMPL_SKELETON:
            if i < len(joints) and j < len(joints):
                ax.plot([joints[i, 0], joints[j, 0]],
                       [joints[i, 1], joints[j, 1]],
                       [joints[i, 2], joints[j, 2]], 'r-', alpha=0.5, linewidth=1)

        if title:
            ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        self._set_axis_equal(ax, joints)

    def _set_axis_equal(self, ax, points):
        """设置3D坐标轴等比例"""
        if len(points) > 0:
            max_range = np.array([
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min()
            ]).max() / 2.0

            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)


def main():
    parser = argparse.ArgumentParser(description='数据集可视化工具')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'sequence', 'animation', 'stats'],
                       help='可视化模式: single(单帧), sequence(序列), animation(动画), stats(统计)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='数据集划分')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--frame', type=int, default=0, help='帧索引 (0-15)')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--count', type=int, default=5, help='可视化数量')
    parser.add_argument('--output', type=str, default='tmp/vis_dataset',
                       help='输出目录')
    parser.add_argument('--preload', action='store_true', help='预加载数据到内存')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 创建可视化器
    visualizer = DatasetVisualizer(split=args.split, preload=args.preload)

    if args.mode == 'single':
        # 单帧可视化
        save_path = os.path.join(args.output, f'sample_{args.index}_frame_{args.frame}.png')
        visualizer.visualize_single_frame(args.index, args.frame, save_path)

    elif args.mode == 'sequence':
        # 序列可视化
        for i in range(args.start, min(args.start + args.count, len(visualizer.dataset))):
            save_path = os.path.join(args.output, f'sequence_{i}.png')
            visualizer.visualize_sequence(i, save_path)

    elif args.mode == 'animation':
        # 动画可视化
        save_path = os.path.join(args.output, f'animation_{args.index}.gif')
        visualizer.visualize_animation(args.index, save_path)

    elif args.mode == 'stats':
        # 统计可视化
        save_path = os.path.join(args.output, 'dataset_statistics.png')
        visualizer.visualize_dataset_statistics(save_path=save_path)

    print(f"\n[INFO] 可视化完成! 结果保存在: {args.output}")


if __name__ == '__main__':
    main()
