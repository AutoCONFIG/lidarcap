#!/usr/bin/env python3
"""
LiDARCap 数据集可视化工具
用于可视化 HDF5 数据集中的点云和人体姿态

使用方法:
    python visualize_dataset.py --dataset_path /path/to/data.hdf5 --dataset_id 0001 --frame_idx 0
    
可选参数:
    --dataset_path: HDF5 数据集路径
    --dataset_id: 数据集ID (如 0001)
    --frame_idx: 要可视化的帧索引 (默认 0)
    --mode: 可视化模式 ['pointcloud', 'smpl', 'both'] (默认 'both')
    --save: 保存为PLY文件而不是显示
"""

import argparse
import h5py
import numpy as np
import torch
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.smpl import SMPL


def visualize_with_open3d(points, joints=None, title="Point Cloud"):
    """使用 Open3D 可视化点云和关节"""
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D 未安装，请运行: pip install open3d")
        return
    
    geometries = []
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 根据y坐标着色（高度）
    colors = np.zeros((len(points), 3))
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if y_max > y_min:
        normalized_y = (points[:, 1] - y_min) / (y_max - y_min)
        colors[:, 0] = normalized_y  # 红色通道表示高度
        colors[:, 2] = 1 - normalized_y  # 蓝色通道
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries.append(pcd)
    
    # 添加关节点（如果有）
    if joints is not None:
        joint_pcd = o3d.geometry.PointCloud()
        joint_pcd.points = o3d.utility.Vector3dVector(joints)
        joint_pcd.paint_uniform_color([1, 0, 0])  # 红色关节点
        
        # 增大关节点大小
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(pcd)
        vis.add_geometry(joint_pcd)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 3.0
        
        vis.run()
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries(geometries, window_name=title)


def visualize_with_matplotlib(points, joints=None, title="Point Cloud", save_path=None):
    """使用 Matplotlib 可视化点云（3D散点图）"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib 未安装，请运行: pip install matplotlib")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 1], cmap='viridis', 
                        s=10, alpha=0.6, label='Point Cloud')
    plt.colorbar(scatter, ax=ax, label='Height (Y)')
    
    # 绘制关节点
    if joints is not None:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                  c='red', s=100, marker='^', label='Joints', edgecolors='black')
        
        # 添加关节标签
        for i, joint in enumerate(joints):
            ax.text(joint[0], joint[1], joint[2], f'{i}', fontsize=8, color='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # 设置等比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()


def generate_smpl_mesh(pose, beta, trans, device='cuda'):
    """根据SMPL参数生成人体网格顶点"""
    smpl = SMPL().to(device)
    
    with torch.no_grad():
        pose_tensor = torch.from_numpy(pose).float().unsqueeze(0).to(device)
        beta_tensor = torch.from_numpy(beta).float().unsqueeze(0).to(device)
        
        vertices = smpl(pose_tensor, beta_tensor)
        vertices = vertices.cpu().numpy().squeeze()
        
        # 应用平移
        if trans is not None:
            vertices += trans
    
    return vertices


def save_point_cloud_ply(points, filename):
    """保存点云为PLY格式"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"点云已保存到: {filename}")


def visualize_hdf5_dataset(dataset_path, dataset_id, frame_idx=0, mode='both', save_dir=None):
    """
    可视化HDF5数据集中的数据
    
    Args:
        dataset_path: HDF5文件路径
        dataset_id: 数据集ID
        frame_idx: 帧索引
        mode: 可视化模式 ['pointcloud', 'joints', 'both']
        save_dir: 保存目录（如果为None则直接显示）
    """
    
    # 打开HDF5文件
    h5file = h5py.File(dataset_path, 'r')
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_id}")
    print(f"帧索引: {frame_idx}")
    print(f"{'='*60}\n")
    
    # 读取数据
    point_clouds = h5file['point_clouds'][frame_idx]
    pose = h5file['pose'][frame_idx]
    beta = h5file['shape'][frame_idx]
    trans = h5file['trans'][frame_idx]
    full_joints = h5file['full_joints'][frame_idx]
    
    print(f"点云形状: {point_clouds.shape}")
    print(f"点云范围: X[{point_clouds[:, 0].min():.3f}, {point_clouds[:, 0].max():.3f}], "
          f"Y[{point_clouds[:, 1].min():.3f}, {point_clouds[:, 1].max():.3f}], "
          f"Z[{point_clouds[:, 2].min():.3f}, {point_clouds[:, 2].max():.3f}]")
    print(f"Pose形状: {pose.shape}")
    print(f"Beta形状: {beta.shape}")
    print(f"Trans: {trans}")
    print(f"关节数: {full_joints.shape[0]}")
    
    # 生成SMPL网格（如果需要）
    smpl_vertices = None
    if mode in ['smpl', 'both']:
        print("\n生成SMPL人体网格...")
        try:
            smpl_vertices = generate_smpl_mesh(pose, beta, trans)
            print(f"SMPL顶点数: {smpl_vertices.shape[0]}")
        except Exception as e:
            print(f"生成SMPL网格失败: {e}")
    
    h5file.close()
    
    # 可视化或保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        if mode in ['pointcloud', 'both']:
            save_path = os.path.join(save_dir, f"{dataset_id}_frame{frame_idx}_pointcloud.ply")
            save_point_cloud_ply(point_clouds, save_path)
            
            # 同时保存Matplotlib图像
            img_path = os.path.join(save_dir, f"{dataset_id}_frame{frame_idx}_pointcloud.png")
            visualize_with_matplotlib(point_clouds, full_joints, 
                                     f"{dataset_id} Frame {frame_idx}", img_path)
        
        if smpl_vertices is not None:
            save_path = os.path.join(save_dir, f"{dataset_id}_frame{frame_idx}_smpl.ply")
            save_point_cloud_ply(smpl_vertices, save_path)
    else:
        # 直接显示
        if mode in ['pointcloud', 'both']:
            print("\n显示点云可视化...")
            try:
                visualize_with_open3d(point_clouds, full_joints, 
                                     f"{dataset_id} Frame {frame_idx}")
            except Exception as e:
                print(f"Open3D可视化失败，使用Matplotlib: {e}")
                visualize_with_matplotlib(point_clouds, full_joints, 
                                         f"{dataset_id} Frame {frame_idx}")
        
        if smpl_vertices is not None and mode in ['smpl', 'both']:
            print("\n显示SMPL网格可视化...")
            try:
                visualize_with_open3d(smpl_vertices, None, "SMPL Mesh")
            except Exception as e:
                print(f"Open3D可视化失败，使用Matplotlib: {e}")
                visualize_with_matplotlib(smpl_vertices, None, "SMPL Mesh")


def main():
    parser = argparse.ArgumentParser(description='LiDARCap 数据集可视化工具')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='HDF5数据集路径')
    parser.add_argument('--dataset_id', type=str, default=None,
                       help='数据集ID（如 0001），如果不指定则使用文件中的第一个')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='帧索引（默认0）')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['pointcloud', 'smpl', 'both'],
                       help='可视化模式')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录（如果指定则保存为文件而不是显示）')
    parser.add_argument('--visualizer', type=str, default='open3d',
                       choices=['open3d', 'matplotlib'],
                       help='可视化库选择')
    
    args = parser.parse_args()
    
    # 如果没有指定dataset_id，尝试从文件路径推断
    if args.dataset_id is None:
        import re
        match = re.search(r'(\d{4})', os.path.basename(args.dataset_path))
        if match:
            args.dataset_id = match.group(1)
        else:
            args.dataset_id = 'unknown'
    
    # 检查文件是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 文件不存在 {args.dataset_path}")
        return
    
    # 执行可视化
    visualize_hdf5_dataset(
        args.dataset_path,
        args.dataset_id,
        args.frame_idx,
        args.mode,
        args.save_dir
    )


if __name__ == '__main__':
    main()
