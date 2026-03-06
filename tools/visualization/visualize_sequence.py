#!/usr/bin/env python3
"""
LiDARCap 序列可视化工具
用于可视化连续多帧数据，生成视频或GIF

使用方法:
    python visualize_sequence.py --dataset_path /path/to/data.hdf5 --dataset_id 0001 --start_frame 0 --end_frame 100
"""

import argparse
import h5py
import numpy as np
import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.smpl import SMPL


def create_video_from_frames(frame_dir, output_path, fps=10):
    """将帧合成为视频"""
    try:
        import cv2
        import glob
        
        images = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if not images:
            print("没有找到图像文件")
            return
        
        # 读取第一帧获取尺寸
        frame = cv2.imread(images[0])
        height, width = frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for image_path in tqdm(images, desc="生成视频"):
            frame = cv2.imread(image_path)
            out.write(frame)
        
        out.release()
        print(f"视频已保存到: {output_path}")
        
    except ImportError:
        print("OpenCV未安装，请运行: pip install opencv-python")


def visualize_frame_matplotlib(points, joints, smpl_vertices=None, title="", save_path=None):
    """可视化单帧并保存"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib未安装")
        return
    
    fig = plt.figure(figsize=(18, 6))
    
    # 点云可视化
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 1], cmap='viridis', s=10, alpha=0.6)
    ax1.set_title('LiDAR Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 关节可视化
    ax2 = fig.add_subplot(132, projection='3d')
    if joints is not None:
        ax2.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                   c='red', s=100, marker='^', edgecolors='black')
        # 绘制骨架连接（简单的连接逻辑）
        connections = [
            (0, 1), (1, 4), (4, 7),  # 左腿
            (0, 2), (2, 5), (5, 8),  # 右腿
            (0, 3), (3, 9), (9, 12), (12, 15),  # 脊柱到头部
            (9, 13), (13, 16), (16, 18), (18, 20),  # 左臂
            (9, 14), (14, 17), (17, 19), (19, 21),  # 右臂
        ]
        for start, end in connections:
            if start < len(joints) and end < len(joints):
                ax2.plot([joints[start, 0], joints[end, 0]],
                        [joints[start, 1], joints[end, 1]],
                        [joints[start, 2], joints[end, 2]], 'r-', linewidth=2)
    ax2.set_title('Skeleton Joints')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # SMPL网格可视化
    ax3 = fig.add_subplot(133, projection='3d')
    if smpl_vertices is not None:
        ax3.scatter(smpl_vertices[:, 0], smpl_vertices[:, 1], smpl_vertices[:, 2],
                   c='blue', s=1, alpha=0.5)
        ax3.set_title('SMPL Mesh')
    else:
        ax3.text(0.5, 0.5, 0.5, 'No SMPL Data', ha='center', va='center', fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_sequence(dataset_path, dataset_id, start_frame=0, end_frame=None, 
                       output_dir='visualization', device='cuda'):
    """可视化连续多帧"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开HDF5文件
    h5file = h5py.File(dataset_path, 'r')
    
    # 获取总帧数
    total_frames = len(h5file['pose'])
    print(f"数据集总帧数: {total_frames}")
    
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    
    print(f"可视化帧范围: {start_frame} - {end_frame}")
    
    # 初始化SMPL模型
    smpl = SMPL().to(device)
    
    # 逐帧处理
    for frame_idx in tqdm(range(start_frame, end_frame), desc="生成可视化"):
        # 读取数据
        points = h5file['point_clouds'][frame_idx]
        pose = h5file['pose'][frame_idx]
        beta = h5file['shape'][frame_idx]
        trans = h5file['trans'][frame_idx]
        joints = h5file['full_joints'][frame_idx]
        
        # 生成SMPL顶点
        with torch.no_grad():
            pose_tensor = torch.from_numpy(pose).float().unsqueeze(0).to(device)
            beta_tensor = torch.from_numpy(beta).float().unsqueeze(0).to(device)
            vertices = smpl(pose_tensor, beta_tensor)
            vertices = vertices.cpu().numpy().squeeze()
            if trans is not None:
                vertices += trans
        
        # 可视化并保存
        save_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        visualize_frame_matplotlib(points, joints, vertices, 
                                  f"{dataset_id} - Frame {frame_idx}", save_path)
    
    h5file.close()
    print(f"\n可视化帧已保存到: {output_dir}")
    
    # 询问是否生成视频
    response = input("是否生成视频? (y/n): ")
    if response.lower() == 'y':
        video_path = os.path.join(output_dir, f"{dataset_id}_sequence.mp4")
        create_video_from_frames(output_dir, video_path)


def compare_prediction_gt(pred_rotmats_path, dataset_path, dataset_id, 
                          output_dir='comparison', device='cuda'):
    """对比预测结果和真值"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预测结果
    pred_rotmats = np.load(pred_rotmats_path)
    print(f"预测结果形状: {pred_rotmats.shape}")
    
    # 打开HDF5
    h5file = h5py.File(dataset_path, 'r')
    total_frames = min(len(pred_rotmats), len(h5file['pose']))
    
    smpl = SMPL().to(device)
    
    for frame_idx in tqdm(range(total_frames), desc="生成对比"):
        # 获取GT
        gt_pose = h5file['pose'][frame_idx]
        beta = h5file['shape'][frame_idx]
        trans = h5file['trans'][frame_idx]
        
        # GT顶点
        with torch.no_grad():
            gt_pose_tensor = torch.from_numpy(gt_pose).float().unsqueeze(0).to(device)
            beta_tensor = torch.from_numpy(beta).float().unsqueeze(0).to(device)
            gt_vertices = smpl(gt_pose_tensor, beta_tensor)
            gt_vertices = gt_vertices.cpu().numpy().squeeze()
            
            # 预测顶点
            pred_rotmat = torch.from_numpy(pred_rotmats[frame_idx]).float().unsqueeze(0).to(device)
            pred_vertices = smpl(pred_rotmat, beta_tensor)
            pred_vertices = pred_vertices.cpu().numpy().squeeze()
        
        # 可视化对比
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 5))
            
            # GT
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2],
                       c='blue', s=1, alpha=0.5)
            ax1.set_title('Ground Truth')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 预测
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2],
                       c='red', s=1, alpha=0.5)
            ax2.set_title('Prediction')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            plt.suptitle(f"{dataset_id} - Frame {frame_idx}")
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"compare_{frame_idx:04d}.png")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"可视化失败: {e}")
    
    h5file.close()
    print(f"对比结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='LiDARCap 序列可视化工具')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='HDF5数据集路径')
    parser.add_argument('--dataset_id', type=str, default='sequence',
                       help='数据集ID')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='起始帧')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='结束帧（默认到最后）')
    parser.add_argument('--output_dir', type=str, default='visualization',
                       help='输出目录')
    parser.add_argument('--compare', type=str, default=None,
                       help='预测结果npy文件路径（用于对比模式）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"错误: 文件不存在 {args.dataset_path}")
        return
    
    if args.compare:
        # 对比模式
        compare_prediction_gt(args.compare, args.dataset_path, args.dataset_id, 
                             args.output_dir, args.device)
    else:
        # 普通可视化模式
        visualize_sequence(args.dataset_path, args.dataset_id, 
                          args.start_frame, args.end_frame, args.output_dir, args.device)


if __name__ == '__main__':
    main()
