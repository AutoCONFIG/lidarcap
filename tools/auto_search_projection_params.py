#!/usr/bin/env python3
"""
投影参数自动搜索预处理工具 (GPU加速版 v2)

改进点:
- 使用凸包渲染代替膨胀近似，IoU更准确
- 自动估计投影与YOLO的质心偏移，搜索范围自适应
- IoU=0时回退大范围搜索，避免错误参数传播
- GPU真正批量计算参数网格
- 仅读取图像尺寸，不加载完整像素
- 粗到细多轮搜索策略

使用方法:
    bash scripts/run_search_params.sh
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from plyfile import PlyData


# ============== 数据加载 ==============

def load_point_cloud(ply_path):
    ply_data = PlyData.read(ply_path)['vertex'].data
    points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']], axis=1)
    return points.astype(np.float32)


def load_yolo_segment(txt_path, img_width, img_height):
    polygons = []
    if not os.path.exists(txt_path):
        return polygons

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(x) for x in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i + 1] * img_height
            points.append([x, y])
        polygons.append(np.array(points, dtype=np.int32))

    return polygons


def load_pose(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return (
        np.array(data['beta'], dtype=np.float32),
        np.array(data['pose'], dtype=np.float32),
        np.array(data['trans'], dtype=np.float32)
    )


def create_mask_from_polygons(polygons, img_size):
    mask = np.zeros(img_size[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def get_image_size(img_path):
    """仅获取图像尺寸，不加载完整像素数据"""
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            w, h = img.size
        return (h, w, 3)
    return (132, 132, 3)


# ============== GPU投影模块 ==============

class GPUProjector:
    """GPU加速的投影计算器"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 相机参数
        extrinsic = np.array([
            -0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748,
            -0.0052925495236, 0.0017416212982, -0.99998447772, 0.080050847871,
            0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295,
            0, 0, 0, 1
        ]).reshape(4, 4).astype(np.float32)

        intrinsic = np.array([
            9.5632709662202160e+02, 0., 9.6209910493679433e+02,
            0., 9.5687763573729683e+02, 5.9026610775785059e+02,
            0., 0., 1.
        ]).reshape(3, 3).astype(np.float32)

        distortion = np.array([
            -6.1100617222502205e-03, 3.0647823796371821e-02,
            -3.3304524444662654e-04, -4.4038460096976607e-04,
            -2.5974982760794661e-02
        ]).astype(np.float32)

        self.extrinsic = torch.from_numpy(extrinsic).to(self.device)
        self.intrinsic = torch.from_numpy(intrinsic).to(self.device)
        self.distortion = torch.from_numpy(distortion).to(self.device)

        # SMPL模型
        print(f"加载SMPL模型到 {self.device}...")
        from modules.smpl import SMPL
        self.smpl = SMPL().to(self.device)
        self.smpl.eval()
        self.smpl_faces = self.smpl.faces.cpu().numpy()  # (13776, 3) for mesh rendering

        # 缓存基础投影结果
        self._cached_base_proj = None
        self._cached_center = None

    def project_base(self, points: torch.Tensor, top_left: np.ndarray) -> torch.Tensor:
        """
        计算基础投影（不带dx/dy/scale调整）
        points: (N, 3) on GPU
        返回: (N, 2) on GPU
        """
        N = points.shape[0]

        # LiDAR -> Camera
        ones = torch.ones((N, 1), device=self.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=1)  # (N, 4)
        camera_pts = (self.extrinsic @ points_homo.T).T[:, :3]  # (N, 3)

        # Camera -> Pixel (with distortion)
        XX = camera_pts[:, :2] / camera_pts[:, 2:3]  # (N, 2)
        r2 = (XX ** 2).sum(dim=1, keepdim=True)  # (N, 1)

        f = self.intrinsic[[0, 1], [0, 1]]  # (2,)
        c = self.intrinsic[[0, 1], [2, 2]]  # (2,)
        k = self.distortion[[0, 1, 4]]  # (3,)
        p = self.distortion[[2, 3]]  # (2,)

        radial = 1 + (k[0] * r2 + k[1] * r2**2 + k[2] * r2**3)
        tan = 2 * (p[0] * XX[:, [1, 0]] * torch.tensor([1., -1.], device=self.device)).sum(dim=1, keepdim=True)
        XXX = XX * (radial + tan) + r2 * p[[1, 0]]

        image_pts = f * XXX + c  # (N, 2)
        image_pts = image_pts - torch.from_numpy(top_left).float().to(self.device)

        return image_pts

    def cache_projection(self, points: torch.Tensor, top_left: np.ndarray):
        """缓存基础投影结果"""
        self._cached_base_proj = self.project_base(points, top_left)
        self._cached_center = self._cached_base_proj.mean(dim=0).clone()

    def project_with_params(self, dx: float, dy: float, scale: float) -> torch.Tensor:
        """
        应用dx/dy/scale变换（使用缓存的基础投影）
        返回: (N, 2) on GPU，不修改缓存
        """
        if self._cached_base_proj is None:
            raise ValueError("需要先调用 cache_projection")

        # Scale around center, then translate (显式clone避免污染缓存)
        result = self._cached_center + (self._cached_base_proj - self._cached_center) * scale
        result = result.clone()
        result[:, 0] += dx
        result[:, 1] += dy

        return result

    def generate_smpl_mesh(self, beta, pose, trans) -> torch.Tensor:
        """生成SMPL mesh，返回GPU上的tensor"""
        beta_t = torch.from_numpy(np.array(beta, dtype=np.float32)).unsqueeze(0).to(self.device)
        pose_t = torch.from_numpy(np.array(pose, dtype=np.float32)).unsqueeze(0).to(self.device)
        trans_t = torch.from_numpy(np.array(trans, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vertices = self.smpl(pose_t, beta_t)
            vertices = vertices + trans_t.unsqueeze(1)

        return vertices[0]  # (6890, 3)


# ============== Mask生成 ==============

def render_points_mask_cv2(proj_pts: np.ndarray, img_size: Tuple[int, int, int],
                           is_point_cloud: bool = False) -> np.ndarray:
    """
    根据投影点生成mask
    - 点云(~150点): 使用凸包填充
    - SMPL mesh(6890点+faces): 使用三角形面片渲染
    """
    H, W = img_size[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # 过滤有效点
    valid = (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < W) & \
            (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < H)
    valid_pts = proj_pts[valid]

    if len(valid_pts) < 3:
        return mask

    if is_point_cloud:
        # 点云: 使用凸包
        hull = cv2.convexHull(valid_pts.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
    else:
        # SMPL: 使用面片渲染 (通过projector的faces)
        # 这里fallback到凸包，面片渲染在调用处处理
        hull = cv2.convexHull(valid_pts.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)

    return mask


def render_smpl_mesh_mask_cv2(proj_pts: np.ndarray, faces: np.ndarray,
                               img_size: Tuple[int, int, int]) -> np.ndarray:
    """
    SMPL mesh面片渲染：逐三角形填充
    proj_pts: (6890, 2) 投影后的2D坐标
    faces: (13776, 3) 三角面片索引
    """
    H, W = img_size[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    pts_int = proj_pts.astype(np.int32)

    for face in faces:
        tri = pts_int[face]
        # 简单边界检查（允许部分越界的三角形，cv2会裁剪）
        if (tri[:, 0].min() >= -W and tri[:, 0].max() < 2 * W and
            tri[:, 1].min() >= -H and tri[:, 1].max() < 2 * H):
            cv2.fillPoly(mask, [tri], 255)

    return mask


def compute_iou_np(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask的IoU (numpy版)"""
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(intersection / union) if union > 0 else 0.0


def estimate_centroid_offset(proj_pts: np.ndarray, yolo_mask: np.ndarray) -> Tuple[float, float]:
    """
    估计投影点质心与YOLO mask质心的偏移量
    返回: (dx, dy) 从投影质心到YOLO质心的偏移
    """
    # 投影质心
    proj_cx = proj_pts[:, 0].mean()
    proj_cy = proj_pts[:, 1].mean()

    # YOLO mask质心
    yolo_coords = np.argwhere(yolo_mask > 0)
    if len(yolo_coords) == 0:
        return 0.0, 0.0
    yolo_cx = yolo_coords[:, 1].mean()  # column = x
    yolo_cy = yolo_coords[:, 0].mean()  # row = y

    dx = yolo_cx - proj_cx
    dy = yolo_cy - proj_cy

    return float(dx), float(dy)


# ============== 搜索策略 ==============

def search_best_params(
    projector: GPUProjector,
    dx_range: Tuple[float, float],
    dy_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    step: float,
    scale_step: float,
    yolo_mask: np.ndarray,
    img_size: Tuple[int, int, int],
    is_point_cloud: bool = False,
    smpl_faces: Optional[np.ndarray] = None
) -> Tuple[float, float, float, float]:
    """
    粗到细搜索最优投影参数

    策略:
    1. 自动估计质心偏移，将搜索中心对齐到YOLO区域
    2. 粗搜索: 大步长扫描整个范围
    3. 细搜索: 以粗搜索最优结果为中心，小步长精调

    返回: (dx, dy, scale, iou)
    """
    H, W = img_size[:2]

    # 自动估计质心偏移
    base_proj_np = projector._cached_base_proj.cpu().numpy()
    auto_dx, auto_dy = estimate_centroid_offset(base_proj_np, yolo_mask)

    # 将搜索范围平移到对齐YOLO的位置
    dx_range_aligned = (dx_range[0] + auto_dx, dx_range[1] + auto_dy)
    dy_range_aligned = (dy_range[0] + auto_dy, dy_range[1] + auto_dy)

    # ===== 阶段1: 粗搜索 =====
    coarse_step = max(step, 2.0)
    coarse_scale_step = max(scale_step, 0.05)

    coarse_dx, coarse_dy, coarse_scale, coarse_iou = _grid_search(
        projector, dx_range_aligned, dy_range_aligned, scale_range,
        coarse_step, coarse_scale_step, yolo_mask, img_size,
        is_point_cloud, smpl_faces
    )

    # ===== 阶段2: 细搜索 =====
    fine_dx_range = (coarse_dx - coarse_step, coarse_dx + coarse_step)
    fine_dy_range = (coarse_dy - coarse_step, coarse_dy + coarse_step)
    fine_scale_range = (coarse_scale - coarse_scale_step, coarse_scale + coarse_scale_step)

    fine_dx, fine_dy, fine_scale, fine_iou = _grid_search(
        projector, fine_dx_range, fine_dy_range, fine_scale_range,
        step, scale_step, yolo_mask, img_size,
        is_point_cloud, smpl_faces
    )

    # 如果细搜索没找到更好的结果，用粗搜索结果
    if fine_iou < coarse_iou:
        fine_dx, fine_dy, fine_scale, fine_iou = coarse_dx, coarse_dy, coarse_scale, coarse_iou

    return fine_dx, fine_dy, fine_scale, fine_iou


def _grid_search(
    projector: GPUProjector,
    dx_range: Tuple[float, float],
    dy_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    step: float,
    scale_step: float,
    yolo_mask: np.ndarray,
    img_size: Tuple[int, int, int],
    is_point_cloud: bool,
    smpl_faces: Optional[np.ndarray]
) -> Tuple[float, float, float, float]:
    """
    网格搜索最优参数，使用CPU凸包/面片渲染计算IoU
    GPU用于快速计算投影变换
    """
    H, W = img_size[:2]

    # 生成参数网格
    dx_vals = np.arange(dx_range[0], dx_range[1] + step / 2, step)
    dy_vals = np.arange(dy_range[0], dy_range[1] + step / 2, step)
    scale_vals = np.arange(scale_range[0], scale_range[1] + scale_step / 2, scale_step)

    best_iou = 0.0
    best_params = (0.0, 0.0, 1.0)

    base_proj = projector._cached_base_proj  # (N, 2) on GPU
    center = projector._cached_center  # (2,) on GPU

    if base_proj is None:
        return 0.0, 0.0, 1.0, 0.0

    N = base_proj.shape[0]

    # 批量GPU计算所有参数组合的投影结果
    num_dx = len(dx_vals)
    num_dy = len(dy_vals)
    num_scale = len(scale_vals)
    total = num_dx * num_dy * num_scale

    # 分批处理，避免显存爆炸
    # 每批最多处理 batch_limit 个组合
    batch_limit = 2000

    for batch_start in range(0, total, batch_limit):
        batch_end = min(batch_start + batch_limit, total)
        batch_size = batch_end - batch_start

        # 构建当前批次的参数
        indices = np.arange(batch_start, batch_end)
        i_idx = indices // (num_dy * num_scale)
        j_idx = (indices % (num_dy * num_scale)) // num_scale
        k_idx = indices % num_scale

        # 批量计算投影: (batch, N, 2)
        dx_batch = torch.from_numpy(dx_vals[i_idx]).float().to(projector.device)  # (batch,)
        dy_batch = torch.from_numpy(dy_vals[j_idx]).float().to(projector.device)
        scale_batch = torch.from_numpy(scale_vals[k_idx]).float().to(projector.device)

        # base_proj: (N, 2), center: (2,)
        # scale around center: center + (base - center) * scale
        # (batch, N, 2) = (1, N, 2) + (1, N, 2) * (batch, 1, 1)
        shifted = base_proj.unsqueeze(0) - center.unsqueeze(0).unsqueeze(0)  # (1, N, 2)
        scaled = shifted * scale_batch.view(-1, 1, 1)  # (batch, N, 2)
        proj_batch = center.unsqueeze(0).unsqueeze(0) + scaled  # (batch, N, 2)

        # 加上平移
        proj_batch[:, :, 0] += dx_batch.view(-1, 1)
        proj_batch[:, :, 1] += dy_batch.view(-1, 1)

        # 转回CPU计算IoU (凸包/mesh渲染在CPU上更方便)
        proj_batch_np = proj_batch.cpu().numpy()  # (batch, N, 2)

        for b in range(batch_size):
            proj_pts = proj_batch_np[b]  # (N, 2)

            # 生成mask
            if is_point_cloud:
                mask = render_points_mask_cv2(proj_pts, img_size, is_point_cloud=True)
            elif smpl_faces is not None:
                mask = render_smpl_mesh_mask_cv2(proj_pts, smpl_faces, img_size)
            else:
                mask = render_points_mask_cv2(proj_pts, img_size, is_point_cloud=False)

            # 计算IoU
            iou = compute_iou_np(mask, yolo_mask)

            if iou > best_iou:
                best_iou = iou
                best_params = (float(dx_vals[i_idx[b]]),
                              float(dy_vals[j_idx[b]]),
                              float(scale_vals[k_idx[b]]))

    return best_params[0], best_params[1], best_params[2], best_iou


# ============== 主处理器 ==============

class ProjectionParamSearcher:
    """GPU加速的投影参数搜索器"""

    def __init__(self, data_root: str):
        self.data_root = data_root

        # 加载top_left
        top_left_path = os.path.join(data_root, 'lidarhuman26M_top_left.json')
        with open(top_left_path, 'r') as f:
            self.top_left_data = json.load(f)

        # 获取序列
        self.sequences = self._get_available_sequences()

        # GPU投影器
        self.projector = None

        print(f"找到 {len(self.sequences)} 个有YOLO分割的序列")

    def _get_available_sequences(self) -> List[int]:
        yolo_seg_path = os.path.join(self.data_root, 'images_seg')
        if not os.path.exists(yolo_seg_path):
            yolo_seg_path = os.path.join(self.data_root, 'output', 'yolo_seg')

        sequences = []
        if os.path.exists(yolo_seg_path):
            for seq in sorted(os.listdir(yolo_seg_path)):
                seq_path = os.path.join(yolo_seg_path, seq)
                if os.path.isdir(seq_path):
                    txt_files = [f for f in os.listdir(seq_path) if f.endswith('.txt')]
                    if txt_files:
                        sequences.append(int(seq))

        return sorted(sequences)

    def get_frame_list(self, seq: int) -> List[int]:
        yolo_seg_path = os.path.join(self.data_root, 'images_seg')
        if not os.path.exists(yolo_seg_path):
            yolo_seg_path = os.path.join(self.data_root, 'output', 'yolo_seg')

        seq_path = os.path.join(yolo_seg_path, str(seq))
        if os.path.exists(seq_path):
            return sorted([
                int(f.replace('.txt', ''))
                for f in os.listdir(seq_path)
                if f.endswith('.txt')
            ])
        return []

    def save_frame_result(self, result: Dict, output_dir: str):
        seq = result['seq']
        frame = result['frame']
        frame_str = f"{frame:06d}"

        seq_dir = os.path.join(output_dir, str(seq))
        os.makedirs(seq_dir, exist_ok=True)

        output_file = os.path.join(seq_dir, f"{frame_str}.json")

        frame_data = {
            'seq': seq,
            'frame': frame,
            'point_cloud': result.get('point_cloud'),
            'smpl_mesh': result.get('smpl_mesh'),
            'timestamp': datetime.now().isoformat()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)

    def run(
        self,
        sequences: Optional[List[int]] = None,
        first_dx_range: Tuple[float, float] = (-30, 30),
        first_dy_range: Tuple[float, float] = (-30, 30),
        first_scale_range: Tuple[float, float] = (0.85, 1.15),
        refine_dx_range: Tuple[float, float] = (-5, 5),
        refine_dy_range: Tuple[float, float] = (-5, 5),
        refine_scale_range: Tuple[float, float] = (0.98, 1.02),
        step: float = 1.0,
        scale_step: float = 0.01,
        output_dir: Optional[str] = None
    ):
        if sequences is None:
            sequences = self.sequences
        else:
            sequences = [s for s in sequences if s in self.sequences]
            if not sequences:
                print("没有有效的序列可处理")
                return

        if output_dir is None:
            output_dir = os.path.join(self.data_root, 'projection_params')

        os.makedirs(output_dir, exist_ok=True)

        # 初始化GPU投影器
        self.projector = GPUProjector()
        smpl_faces = self.projector.smpl_faces  # (13776, 3)

        total_frames = sum(len(self.get_frame_list(s)) for s in sequences)

        print(f"\n开始处理 {len(sequences)} 个序列，共 {total_frames} 帧")
        print(f"输出目录: {output_dir}")
        print(f"参数: step={step}, scale_step={scale_step}")
        print(f"第一帧范围: dx={first_dx_range}, dy={first_dy_range}, scale={first_scale_range}")
        print(f"微调范围: dx={refine_dx_range}, dy={refine_dy_range}, scale={refine_scale_range}")

        total_start = datetime.now()

        with tqdm(total=total_frames, desc="处理进度") as pbar:
            for seq in sequences:
                frame_list = self.get_frame_list(seq)
                if not frame_list:
                    continue

                prev_pc_params = None
                prev_pc_iou = 0.0
                prev_smpl_params = None
                prev_smpl_iou = 0.0

                for i, frame in enumerate(frame_list):
                    frame_str = f"{frame:06d}"
                    index_key = f"{seq}/{frame_str}"

                    result = {'seq': seq, 'frame': frame}

                    try:
                        # 加载图像尺寸（不加载完整像素）
                        img_path = os.path.join(self.data_root, 'images', str(seq), f"{frame_str}.png")
                        img_size = get_image_size(img_path)

                        # 加载点云
                        pc_path = os.path.join(self.data_root, 'labels', '3d', 'segment', str(seq), f"{frame_str}.ply")
                        point_cloud_gpu = None
                        if os.path.exists(pc_path):
                            pc_np = load_point_cloud(pc_path)
                            point_cloud_gpu = torch.from_numpy(pc_np).float().to(self.projector.device)

                        # 加载SMPL
                        pose_path = os.path.join(self.data_root, 'labels', '3d', 'pose', str(seq), f"{frame_str}.json")
                        smpl_mesh_gpu = None
                        if os.path.exists(pose_path):
                            beta, pose, trans = load_pose(pose_path)
                            smpl_mesh_gpu = self.projector.generate_smpl_mesh(beta, pose, trans)

                        # 加载YOLO
                        yolo_path = os.path.join(self.data_root, 'images_seg', str(seq), f"{frame_str}.txt")
                        if not os.path.exists(yolo_path):
                            yolo_path = os.path.join(self.data_root, 'output', 'yolo_seg', str(seq), f"{frame_str}.txt")

                        yolo_polygons = load_yolo_segment(yolo_path, img_size[1], img_size[0])
                        yolo_mask = create_mask_from_polygons(yolo_polygons, img_size)

                        top_left = np.array(self.top_left_data.get(index_key, [0, 0]))

                        # 优化点云参数
                        if point_cloud_gpu is not None:
                            self.projector.cache_projection(point_cloud_gpu, top_left)

                            # 决定搜索范围: IoU=0或第一帧时用大范围
                            if i == 0 or prev_pc_params is None or prev_pc_iou < 0.01:
                                dx_range = first_dx_range
                                dy_range = first_dy_range
                                scale_range = first_scale_range
                            else:
                                dx_range = (prev_pc_params[0] + refine_dx_range[0],
                                           prev_pc_params[0] + refine_dx_range[1])
                                dy_range = (prev_pc_params[1] + refine_dy_range[0],
                                           prev_pc_params[1] + refine_dy_range[1])
                                scale_range = (prev_pc_params[2] * refine_scale_range[0],
                                              prev_pc_params[2] * refine_scale_range[1])

                            dx, dy, scale, iou = search_best_params(
                                self.projector, dx_range, dy_range, scale_range,
                                step, scale_step, yolo_mask, img_size,
                                is_point_cloud=True, smpl_faces=None
                            )
                            result['point_cloud'] = {'dx': float(dx), 'dy': float(dy),
                                                     'scale': float(scale), 'iou': float(iou)}
                            prev_pc_params = (dx, dy, scale) if iou > 0.01 else None
                            prev_pc_iou = iou

                        # 优化SMPL参数
                        if smpl_mesh_gpu is not None:
                            self.projector.cache_projection(smpl_mesh_gpu, top_left)

                            # 决定搜索范围
                            if i == 0 or prev_smpl_params is None or prev_smpl_iou < 0.01:
                                dx_range = first_dx_range
                                dy_range = first_dy_range
                                scale_range = first_scale_range
                            else:
                                dx_range = (prev_smpl_params[0] + refine_dx_range[0],
                                           prev_smpl_params[0] + refine_dx_range[1])
                                dy_range = (prev_smpl_params[1] + refine_dy_range[0],
                                           prev_smpl_params[1] + refine_dy_range[1])
                                scale_range = (prev_smpl_params[2] * refine_scale_range[0],
                                              prev_smpl_params[2] * refine_scale_range[1])

                            dx, dy, scale, iou = search_best_params(
                                self.projector, dx_range, dy_range, scale_range,
                                step, scale_step, yolo_mask, img_size,
                                is_point_cloud=False, smpl_faces=smpl_faces
                            )
                            result['smpl_mesh'] = {'dx': float(dx), 'dy': float(dy),
                                                   'scale': float(scale), 'iou': float(iou)}
                            prev_smpl_params = (dx, dy, scale) if iou > 0.01 else None
                            prev_smpl_iou = iou

                        # 保存结果
                        self.save_frame_result(result, output_dir)

                    except Exception as e:
                        print(f"\n序列 {seq} 帧 {frame} 处理失败: {e}")
                        import traceback
                        traceback.print_exc()

                    pbar.update(1)

        total_elapsed = (datetime.now() - total_start).total_seconds()

        # 保存汇总
        summary_file = os.path.join(output_dir, '_summary.json')
        summary = {
            'created_at': datetime.now().isoformat(),
            'data_root': self.data_root,
            'sequences': sequences,
            'total_frames': total_frames,
            'total_time_seconds': total_elapsed,
            'params': {
                'first_dx_range': first_dx_range,
                'first_dy_range': first_dy_range,
                'first_scale_range': first_scale_range,
                'refine_dx_range': refine_dx_range,
                'refine_dy_range': refine_dy_range,
                'refine_scale_range': refine_scale_range,
                'step': step,
                'scale_step': scale_step
            }
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n处理完成!")
        print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
        print(f"平均速度: {total_frames/total_elapsed:.1f} 帧/秒")
        print(f"结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='投影参数自动搜索 (GPU加速 v2)')

    parser.add_argument('--data_root', type=str,
                        default='/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seqs', type=str, default=None)

    parser.add_argument('--first_dx', type=float, nargs=2, default=[-30, 30])
    parser.add_argument('--first_dy', type=float, nargs=2, default=[-30, 30])
    parser.add_argument('--first_scale', type=float, nargs=2, default=[0.85, 1.15])

    parser.add_argument('--refine_dx', type=float, nargs=2, default=[-5, 5])
    parser.add_argument('--refine_dy', type=float, nargs=2, default=[-5, 5])
    parser.add_argument('--refine_scale', type=float, nargs=2, default=[0.98, 1.02])

    parser.add_argument('--step', type=float, default=1.0)
    parser.add_argument('--scale_step', type=float, default=0.01)

    args = parser.parse_args()

    sequences = None
    if args.seqs:
        sequences = [int(s.strip()) for s in args.seqs.split(',')]

    searcher = ProjectionParamSearcher(args.data_root)
    searcher.run(
        sequences=sequences,
        first_dx_range=tuple(args.first_dx),
        first_dy_range=tuple(args.first_dy),
        first_scale_range=tuple(args.first_scale),
        refine_dx_range=tuple(args.refine_dx),
        refine_dy_range=tuple(args.refine_dy),
        refine_scale_range=tuple(args.refine_scale),
        step=args.step,
        scale_step=args.scale_step,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
