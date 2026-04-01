"""
可视化Transformer和Mamba的中间结果

功能：
1. Transformer空间特征可视化
2. Mamba时序建模效果可视化
3. 两者协同效果对比

运行方式:
    # 使用启动脚本（推荐）
    bash scripts/vis_intermediate.sh -m output/run_xxx/model/best-valid-loss.pth

    # 或直接运行
    python vis/intermediate.py --model <model_path> --seq 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import argparse
from tqdm import tqdm

from config import get_cfg
from modules.regressor import Regressor
from datasets.lidarcap_dataset import CachedLidarCapDataset, collate
from torch.utils.data import DataLoader


class IntermediateVisualizer:
    def __init__(self, model_path, output_dir='vis_results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] 加载模型: {model_path}")
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        cfg = get_cfg()
        model = Regressor(cfg=cfg)

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        return model
    
    def get_dataloader(self, seq_idx=0):
        cfg = get_cfg()
        dataset = CachedLidarCapDataset(cfg, 'test', seq_idx)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=collate,
            num_workers=0
        )
        return dataloader
    
    @torch.no_grad()
    def extract_intermediate_features(self, data):
        """
        提取中间特征
        
        Returns:
            dict: {
                'pointr_feat': PoinTr提取的空间特征,
                'pointnet_feat': PointNet2特征,
                'fused_feat': 融合后的特征,
                'mamba_output': Mamba输出,
                'pred_joints': 预测关节
            }
        """
        data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in data.items()}
        
        intermediate = {}
        
        # 1. PoinTr空间特征提取
        human_points = data['human_points']
        B, T, N, C = human_points.shape
        
        recon_point_feat, coarse_point_cloud, merged_point_cloud = self.model.pointr(human_points)
        intermediate['pointr_feat'] = recon_point_feat.cpu()  # (B, T, M, 384)
        intermediate['coarse_points'] = coarse_point_cloud.cpu()  # (B*T, M, 3)
        intermediate['merged_points'] = merged_point_cloud.cpu()  # (B, T, M+K, 3)
        
        # 2. PointNet2特征提取
        orig_feat = self.model.encoder(data)
        intermediate['pointnet_feat'] = orig_feat.cpu()  # (B, T, 1024)
        
        # 3. 特征融合
        recon_feat = recon_point_feat.mean(dim=2)
        fused_feat = torch.cat([orig_feat, recon_feat], dim=-1)
        intermediate['fused_feat'] = fused_feat.cpu()  # (B, T, 1408)
        
        # 4. Mamba时序建模
        full_joints = self.model.pose_s1(fused_feat)
        intermediate['mamba_output'] = full_joints.cpu()  # (B, T, 72)
        intermediate['pred_joints'] = full_joints.reshape(B, T, 24, 3).cpu()
        
        # 5. STGCN姿态估计
        rot6ds = self.model.pose_s2(torch.cat((
            full_joints.reshape(B, T, 24, 3),
            fused_feat.unsqueeze(-2).repeat(1, 1, 24, 1)
        ), dim=-1))
        intermediate['rot6ds'] = rot6ds.cpu()
        
        return intermediate
    
    def visualize_spatial_features(self, intermediate, data, frame_idx=0, save_name='spatial_features.png'):
        """
        可视化空间特征提取效果
        
        展示：
        1. 原始点云
        2. PoinTr重建的粗点云
        3. PointNet2特征分布
        """
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig)
        
        # 原始点云
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        points = data['human_points'][0, frame_idx].cpu().numpy()
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)
        ax1.set_title(f'Original Point Cloud\n(Frame {frame_idx})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # PoinTr重建点云
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        merged_points = intermediate['merged_points'][0, frame_idx].numpy()
        ax2.scatter(merged_points[:, 0], merged_points[:, 1], merged_points[:, 2], 
                   s=2, c='red', alpha=0.6)
        ax2.set_title(f'PoinTr Reconstructed\n({merged_points.shape[0]} points)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # GT关节点
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        gt_joints = data['full_joints'][0, frame_idx].cpu().numpy()
        ax3.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2], 
                   s=50, c='blue', marker='o')
        ax3.set_title('GT Joints')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # 预测关节点
        ax4 = fig.add_subplot(gs[0, 3], projection='3d')
        pred_joints = intermediate['pred_joints'][0, frame_idx].numpy()
        ax4.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], 
                   s=50, c='green', marker='o')
        ax4.set_title('Predicted Joints')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        
        # 特征分布 (t-SNE简化版：用前两维)
        ax5 = fig.add_subplot(gs[1, 0])
        pointnet_feat = intermediate['pointnet_feat'][0, frame_idx].numpy()
        ax5.hist(pointnet_feat, bins=50, alpha=0.7, label='PointNet2')
        ax5.set_title('PointNet2 Feature Distribution')
        ax5.set_xlabel('Feature Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # PoinTr特征分布
        ax6 = fig.add_subplot(gs[1, 1])
        pointr_feat = intermediate['pointr_feat'][0, frame_idx].mean(dim=0).numpy()
        ax6.hist(pointr_feat, bins=50, alpha=0.7, color='red', label='PoinTr')
        ax6.set_title('PoinTr Feature Distribution')
        ax6.set_xlabel('Feature Value')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        # 融合特征分布
        ax7 = fig.add_subplot(gs[1, 2])
        fused_feat = intermediate['fused_feat'][0, frame_idx].numpy()
        ax7.hist(fused_feat, bins=50, alpha=0.7, color='purple', label='Fused')
        ax7.set_title('Fused Feature Distribution')
        ax7.set_xlabel('Feature Value')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        
        # 关节误差
        ax8 = fig.add_subplot(gs[1, 3])
        errors = np.linalg.norm(pred_joints - gt_joints, axis=1)
        ax8.bar(range(24), errors)
        ax8.set_title('Per-Joint Error')
        ax8.set_xlabel('Joint Index')
        ax8.set_ylabel('Error (mm)')
        ax8.axhline(y=errors.mean(), color='r', linestyle='--', label=f'Mean: {errors.mean():.2f}mm')
        ax8.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 保存空间特征可视化: {save_path}")
    
    def visualize_temporal_features(self, intermediate, data, save_name='temporal_features.png'):
        """
        可视化时序建模效果
        
        展示：
        1. 关节轨迹随时间变化
        2. 运动平滑度
        3. GT vs Pred对比
        """
        T = data['human_points'].shape[1]
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig)
        
        pred_joints = intermediate['pred_joints'][0].numpy()  # (T, 24, 3)
        gt_joints = data['full_joints'][0].cpu().numpy()  # (T, 24, 3)
        
        # 选择几个关键关节可视化轨迹
        key_joints = [0, 3, 6, 9, 12, 15, 18, 21]  # 头、肩膀、手、髋、膝、脚
        joint_names = ['Root', 'Spine3', 'LShoulder', 'LElbow', 
                       'RShoulder', 'RElbow', 'LHip', 'RKnee']
        
        # 关节轨迹对比 (X, Y, Z分别)
        for idx, (j_idx, j_name) in enumerate(zip(key_joints[:4], joint_names[:4])):
            ax = fig.add_subplot(gs[0, idx])
            ax.plot(range(T), gt_joints[:, j_idx, 0], 'b-', label='GT-X', alpha=0.7)
            ax.plot(range(T), pred_joints[:, j_idx, 0], 'g--', label='Pred-X', alpha=0.7)
            ax.set_title(f'{j_name} - X Coordinate')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Position')
            ax.legend()
        
        # Y坐标轨迹
        for idx, (j_idx, j_name) in enumerate(zip(key_joints[:4], joint_names[:4])):
            ax = fig.add_subplot(gs[1, idx])
            ax.plot(range(T), gt_joints[:, j_idx, 1], 'b-', label='GT-Y', alpha=0.7)
            ax.plot(range(T), pred_joints[:, j_idx, 1], 'g--', label='Pred-Y', alpha=0.7)
            ax.set_title(f'{j_name} - Y Coordinate')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Position')
            ax.legend()
        
        # 运动速度对比
        gt_velocity = np.linalg.norm(gt_joints[1:] - gt_joints[:-1], axis=-1)  # (T-1, 24)
        pred_velocity = np.linalg.norm(pred_joints[1:] - pred_joints[:-1], axis=-1)
        
        ax_vel = fig.add_subplot(gs[2, 0:2])
        for j_idx, j_name in zip(key_joints[:4], joint_names[:4]):
            ax_vel.plot(range(T-1), gt_velocity[:, j_idx], '-', label=f'{j_name} GT', alpha=0.7)
            ax_vel.plot(range(T-1), pred_velocity[:, j_idx], '--', label=f'{j_name} Pred', alpha=0.7)
        ax_vel.set_title('Joint Velocity Over Time')
        ax_vel.set_xlabel('Frame')
        ax_vel.set_ylabel('Velocity')
        ax_vel.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 加速度对比
        gt_accel = np.linalg.norm(gt_velocity[1:] - gt_velocity[:-1], axis=-1)
        pred_accel = np.linalg.norm(pred_velocity[1:] - pred_velocity[:-1], axis=-1)
        
        ax_accel = fig.add_subplot(gs[2, 2])
        ax_accel.plot(range(T-2), gt_accel, 'b-', label='GT', alpha=0.7)
        ax_accel.plot(range(T-2), pred_accel, 'g--', label='Pred', alpha=0.7)
        ax_accel.set_title('Overall Acceleration')
        ax_accel.set_xlabel('Frame')
        ax_accel.set_ylabel('Acceleration')
        ax_accel.legend()
        
        # 时序平滑度指标
        temporal_smoothness = 1 - np.mean(np.abs(pred_velocity - gt_velocity)) / (np.mean(gt_velocity) + 1e-6)
        
        ax_metrics = fig.add_subplot(gs[2, 3])
        metrics = {
            'Temporal Smoothness': temporal_smoothness,
            'Mean Velocity Error': np.mean(np.abs(pred_velocity - gt_velocity)),
            'Mean Position Error': np.mean(np.linalg.norm(pred_joints - gt_joints, axis=-1))
        }
        bars = ax_metrics.bar(metrics.keys(), metrics.values())
        ax_metrics.set_title('Temporal Metrics')
        ax_metrics.set_ylabel('Value')
        for bar, val in zip(bars, metrics.values()):
            ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                          f'{val:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 保存时序特征可视化: {save_path}")
    
    def visualize_comparison(self, intermediate, data, save_name='transformer_vs_mamba.png'):
        """
        对比Transformer和Mamba的贡献
        
        展示：
        1. 空间特征重要性
        2. 时序特征重要性
        3. 融合效果
        """
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, 4, figure=fig)
        
        # 计算不同特征的方差（表示信息量）
        pointnet_var = intermediate['pointnet_feat'].var(dim=-1).mean().item()
        pointr_var = intermediate['pointr_feat'].var(dim=-1).mean().item()
        
        # 特征信息量对比
        ax1 = fig.add_subplot(gs[0, 0])
        features = ['PointNet2', 'PoinTr', 'Fused']
        variances = [pointnet_var, pointr_var, intermediate['fused_feat'].var().item()]
        bars = ax1.bar(features, variances, color=['blue', 'red', 'purple'])
        ax1.set_title('Feature Variance (Information Content)')
        ax1.set_ylabel('Variance')
        for bar, val in zip(bars, variances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 关节误差随时间变化
        pred_joints = intermediate['pred_joints'][0].numpy()
        gt_joints = data['full_joints'][0].cpu().numpy()
        
        errors_per_frame = np.linalg.norm(pred_joints - gt_joints, axis=-1).mean(axis=-1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(range(len(errors_per_frame)), errors_per_frame, 'g-', linewidth=2)
        ax2.axhline(y=errors_per_frame.mean(), color='r', linestyle='--', 
                   label=f'Mean: {errors_per_frame.mean():.2f}mm')
        ax2.fill_between(range(len(errors_per_frame)), 0, errors_per_frame, alpha=0.3)
        ax2.set_title('MPJPE Per Frame')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Error (mm)')
        ax2.legend()
        
        # 空间特征热图 (某一帧)
        ax3 = fig.add_subplot(gs[0, 2])
        feat_frame = intermediate['fused_feat'][0, 0].reshape(44, 32).numpy()
        im = ax3.imshow(feat_frame, cmap='viridis', aspect='auto')
        ax3.set_title('Fused Feature Heatmap (Frame 0)')
        ax3.set_xlabel('Feature Dimension')
        ax3.set_ylabel('Feature Block')
        plt.colorbar(im, ax=ax3)
        
        # 时序特征变化
        ax4 = fig.add_subplot(gs[0, 3])
        temporal_change = torch.norm(
            intermediate['fused_feat'][0, 1:] - intermediate['fused_feat'][0, :-1], 
            dim=-1
        ).numpy()
        ax4.plot(range(len(temporal_change)), temporal_change, 'purple', linewidth=2)
        ax4.set_title('Temporal Feature Change (Motion Saliency)')
        ax4.set_xlabel('Frame Transition')
        ax4.set_ylabel('Feature Distance')
        
        # 3D姿态序列可视化
        ax5 = fig.add_subplot(gs[1, 0:2], projection='3d')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(pred_joints)))
        for t, (pred, color) in enumerate(zip(pred_joints, colors)):
            ax5.scatter(pred[:, 0], pred[:, 1], pred[:, 2], 
                       c=[color], s=20, alpha=0.5, label=f'Frame {t}' if t % 4 == 0 else '')
        ax5.set_title('Predicted Pose Sequence (3D)')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.legend(loc='upper right', fontsize=8)
        
        # GT姿态序列
        ax6 = fig.add_subplot(gs[1, 2:4], projection='3d')
        for t, (gt, color) in enumerate(zip(gt_joints, colors)):
            ax6.scatter(gt[:, 0], gt[:, 1], gt[:, 2], 
                       c=[color], s=20, alpha=0.5, label=f'Frame {t}' if t % 4 == 0 else '')
        ax6.set_title('GT Pose Sequence (3D)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 保存对比可视化: {save_path}")
    
    def visualize_skeleton_sequence(self, intermediate, data, save_name='skeleton_sequence.gif'):
        """
        生成骨架动画，直观展示时序建模效果
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            
            pred_joints = intermediate['pred_joints'][0].numpy()
            gt_joints = data['full_joints'][0].cpu().numpy()
            
            SMPL_SKELETON = [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (2, 5), (3, 6),
                (4, 7), (5, 8), (6, 9),
                (7, 10), (8, 11), (9, 12),
                (0, 13), (0, 16),
                (13, 14), (14, 15),
                (16, 17), (17, 18),
                (15, 19), (15, 20), (18, 21), (18, 22)
            ]
            
            fig = plt.figure(figsize=(16, 8))
            
            def update(frame):
                fig.clear()
                
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                
                pred = pred_joints[frame]
                gt = gt_joints[frame]
                
                ax1.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', s=30)
                for i, j in SMPL_SKELETON:
                    ax1.plot([gt[i, 0], gt[j, 0]], 
                            [gt[i, 1], gt[j, 1]], 
                            [gt[i, 2], gt[j, 2]], 'b-', alpha=0.5)
                ax1.set_title(f'GT Skeleton (Frame {frame})')
                
                ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='green', s=30)
                for i, j in SMPL_SKELETON:
                    ax2.plot([pred[i, 0], pred[j, 0]], 
                            [pred[i, 1], pred[j, 1]], 
                            [pred[i, 2], pred[j, 2]], 'g-', alpha=0.5)
                ax2.set_title(f'Predicted Skeleton (Frame {frame})')
                
                for ax in [ax1, ax2]:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_xlim(-0.8, 0.8)
                    ax.set_ylim(-0.8, 0.8)
                    ax.set_zlim(-0.2, 1.5)
            
            anim = FuncAnimation(fig, update, frames=len(pred_joints), interval=100)
            save_path = os.path.join(self.output_dir, save_name)
            anim.save(save_path, writer=PillowWriter(fps=10))
            plt.close()
            print(f"[INFO] 保存骨架动画: {save_path}")
            
        except ImportError:
            print("[WARNING] 无法生成动画，需要安装 Pillow")
    
    def run_visualization(self, seq_idx=0):
        """运行完整可视化流程"""
        print(f"\n{'='*60}")
        print(f"[INFO] 开始可视化 - 序列 {seq_idx}")
        print(f"{'='*60}\n")
        
        dataloader = self.get_dataloader(seq_idx)
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing")):
            # 提取中间特征
            intermediate = self.extract_intermediate_features(data)
            
            # 生成各类可视化
            self.visualize_spatial_features(
                intermediate, data, 
                save_name=f'seq{seq_idx}_batch{batch_idx}_spatial.png'
            )
            
            self.visualize_temporal_features(
                intermediate, data,
                save_name=f'seq{seq_idx}_batch{batch_idx}_temporal.png'
            )
            
            self.visualize_comparison(
                intermediate, data,
                save_name=f'seq{seq_idx}_batch{batch_idx}_comparison.png'
            )
            
            self.visualize_skeleton_sequence(
                intermediate, data,
                save_name=f'seq{seq_idx}_batch{batch_idx}_skeleton.gif'
            )
            
            # 只处理第一个batch作为示例
            break
        
        print(f"\n{'='*60}")
        print(f"[INFO] 可视化完成！结果保存在: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='可视化Transformer和Mamba中间结果')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--seq', type=int, default=0, help='测试序列索引')
    parser.add_argument('--output', type=str, default='vis_results', help='输出目录')
    
    args = parser.parse_args()
    
    visualizer = IntermediateVisualizer(args.model, args.output)
    visualizer.run_visualization(args.seq)


if __name__ == '__main__':
    main()
