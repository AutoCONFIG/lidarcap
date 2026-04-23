#!/usr/bin/env python3
"""
交互式投影可视化工具
用于查看点云/SMPL mesh投影与YOLO分割的对比情况

功能:
- 切换序列、帧
- 单窗口分面板显示
- 自动网格搜索优化投影参数
- 点云和SMPL分开调整

使用方法:
    python tools/visualize_projection.py --data_root /path/to/lidarhuman26M
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from plyfile import PlyData

# 中文字体路径
CHINESE_FONT_PATH = '/usr/share/fonts/truetype/noto/NotoSansCJKsc-VF.otf'


def get_chinese_font(size=18):
    """获取中文字体"""
    if os.path.exists(CHINESE_FONT_PATH):
        try:
            return ImageFont.truetype(CHINESE_FONT_PATH, size)
        except:
            pass
    return ImageFont.load_default()


# ============== 投影函数 ==============

def affine(X, matrix):
    """Affine transformation"""
    n = X.shape[0]
    if type(X) == np.ndarray:
        res = np.concatenate((X, np.ones((n, 1))), axis=-1).T
        res = np.dot(matrix, res).T
    else:
        import torch
        res = torch.cat((X, torch.ones((n, 1)).to(X.device)), axis=-1)
        res = matrix.to(X.device).matmul(res.T).T
    return res[..., :-1]


def lidar_to_camera(X, extrinsic_matrix):
    """LiDAR coordinate -> Camera coordinate"""
    return affine(X, extrinsic_matrix)


def camera_to_pixel(X, intrinsic_matrix, distortion_coefficients):
    """Camera coordinate -> Pixel coordinate (with distortion correction)"""
    f = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
    c = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
    k = np.array([distortion_coefficients[0],
                  distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / X[..., 2:]
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                       axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c


def get_camera_params():
    """Get camera parameters (official)"""
    extrinsic_matrix = np.array([
        -0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748,
        -0.0052925495236, 0.0017416212982, -0.99998447772, 0.080050847871,
        0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295,
        0, 0, 0, 1
    ]).reshape(4, 4)

    intrinsic_matrix = np.array([
        9.5632709662202160e+02, 0., 9.6209910493679433e+02,
        0., 9.5687763573729683e+02, 5.9026610775785059e+02,
        0., 0., 1.
    ]).reshape(3, 3)

    distortion_coefficients = np.array([
        -6.1100617222502205e-03, 3.0647823796371821e-02,
        -3.3304524444662654e-04, -4.4038460096976607e-04,
        -2.5974982760794661e-02
    ])

    return extrinsic_matrix, intrinsic_matrix, distortion_coefficients


# ============== Data Loading ==============

def load_point_cloud(ply_path):
    """Load point cloud from PLY file"""
    ply_data = PlyData.read(ply_path)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def load_yolo_segment(txt_path, img_width, img_height):
    """Load YOLO segmentation polygons"""
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
    """Load SMPL pose parameters"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return (
        np.array(data['beta'], dtype=np.float32),
        np.array(data['pose'], dtype=np.float32),
        np.array(data['trans'], dtype=np.float32)
    )


def generate_smpl_mesh(beta, pose, trans, smpl_model):
    """Generate SMPL mesh vertices"""
    import torch
    beta_t = torch.tensor([beta], dtype=torch.float32)
    pose_t = torch.tensor([pose], dtype=torch.float32)
    trans_t = torch.tensor([trans], dtype=torch.float32)

    vertices = smpl_model(pose_t, beta_t)
    vertices = vertices + trans_t.unsqueeze(1)

    return vertices[0].numpy()


# ============== Projection & Mask ==============

def project_points(points, extrinsic, intrinsic, distortion, top_left,
                   dx=0, dy=0, scale=1.0, center=None):
    """Project 3D points to 2D with translation and scale adjustments"""
    if len(points) == 0:
        return np.array([])

    camera_points = lidar_to_camera(points, extrinsic)
    image_points = camera_to_pixel(camera_points, intrinsic, distortion)
    image_points -= top_left

    # Apply scale (around center)
    if scale != 1.0:
        if center is None:
            center = image_points.mean(axis=0)
        image_points = center + (image_points - center) * scale

    # Apply translation
    image_points[:, 0] += dx
    image_points[:, 1] += dy

    return image_points


def create_mask_from_points(points_2d, img_size):
    """Create mask from 2D points using convex hull"""
    mask = np.zeros(img_size[:2], dtype=np.uint8)
    if len(points_2d) < 3:
        return mask

    valid = (points_2d[:, 0] >= -100) & (points_2d[:, 0] < img_size[1] + 100) & \
            (points_2d[:, 1] >= -100) & (points_2d[:, 1] < img_size[0] + 100)
    points_2d = points_2d[valid]

    if len(points_2d) < 3:
        return mask

    hull = cv2.convexHull(points_2d.astype(np.float32).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def create_mask_from_polygons(polygons, img_size):
    """Create mask from polygons"""
    mask = np.zeros(img_size[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def compute_iou(mask1, mask2):
    """Compute IoU between two masks"""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return intersection / union


# ============== Auto Optimization ==============

def optimize_projection(points, yolo_mask, extrinsic, intrinsic, distortion, top_left,
                        dx_range=(-20, 20), dy_range=(-20, 20), scale_range=(0.8, 1.2),
                        step=2, scale_step=0.02):
    """
    Grid search for optimal projection parameters
    Returns: (best_dx, best_dy, best_scale, best_iou)
    """
    img_size = (yolo_mask.shape[0], yolo_mask.shape[1], 3)
    center = None

    best_iou = 0
    best_params = (0, 0, 1.0)

    # Get initial projection for center calculation
    initial_proj = project_points(points, extrinsic, intrinsic, distortion, top_left)
    if len(initial_proj) > 0:
        center = initial_proj.mean(axis=0)

    # Grid search
    dx_values = np.arange(dx_range[0], dx_range[1] + step, step)
    dy_values = np.arange(dy_range[0], dy_range[1] + step, step)
    scale_values = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)

    total = len(dx_values) * len(dy_values) * len(scale_values)
    count = 0

    for dx in dx_values:
        for dy in dy_values:
            for scale in scale_values:
                proj_points = project_points(points, extrinsic, intrinsic, distortion,
                                            top_left, dx, dy, scale, center)
                mask = create_mask_from_points(proj_points, img_size)
                iou = compute_iou(mask, yolo_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_params = (dx, dy, scale)

                count += 1

    return best_params[0], best_params[1], best_params[2], best_iou


# ============== Visualization ==============

def draw_polygon_boundary(img, polygons, color, thickness=2):
    for polygon in polygons:
        cv2.polylines(img, [polygon], True, color, thickness)


def draw_points(img, points_2d, color, radius=2):
    if len(points_2d) == 0:
        return
    h, w = img.shape[:2]
    for pt in points_2d:
        x, y = int(pt[0]), int(pt[1])
        if -100 <= x < w + 100 and -100 <= y < h + 100:
            cv2.circle(img, (x, y), radius, color, -1)


def add_label(img, text, bg_color=(0, 0, 0)):
    """Add label with Chinese support"""
    h, w = img.shape[:2]
    label_h = 35

    new_img = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    new_img[:label_h, :] = bg_color
    new_img[label_h:, :] = img

    pil_img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(18)
    draw.text((8, 7), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class ProjectionVisualizer:
    """Interactive projection visualizer with auto-optimization"""

    def __init__(self, data_root, params_file=None):
        self.data_root = data_root
        self.extrinsic, self.intrinsic, self.distortion = get_camera_params()

        # Load top_left
        top_left_path = os.path.join(data_root, 'lidarhuman26M_top_left.json')
        with open(top_left_path, 'r') as f:
            self.top_left_data = json.load(f)

        # Get sequences
        self.sequences = self._get_available_sequences()
        if not self.sequences:
            raise ValueError("No available sequence data found")

        # State
        self.current_seq_idx = 0
        self.current_frame_idx = 0

        # 点云调整参数 (dx, dy, scale)
        self.pc_dx = 0.0
        self.pc_dy = 0.0
        self.pc_scale = 1.0

        # SMPL调整参数 (dx, dy, scale)
        self.smpl_dx = 0.0
        self.smpl_dy = 0.0
        self.smpl_scale = 1.0

        # 当前调整对象: 0=点云, 1=SMPL
        self.adjust_target = 0
        # 当前调整参数: 0=dx, 1=dy, 2=scale
        self.adjust_param = 0

        # 是否正在优化
        self.optimizing = False

        # 加载预处理参数
        self.params_data = None
        if params_file:
            self.load_params_file(params_file)

        # Load SMPL
        print("加载SMPL模型...")
        from modules.smpl import SMPL
        self.smpl = SMPL()

        self._update_frame_list()

        print(f"找到 {len(self.sequences)} 个序列")
        print(f"当前序列: {self.current_sequence}, 共 {len(self.frame_list)} 帧")

        if self.params_data:
            print(f"已加载预处理参数: {params_file}")

    def _get_available_sequences(self):
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

    def load_params_file(self, params_file):
        """加载预处理参数文件或目录"""
        if os.path.isdir(params_file):
            # 从目录结构加载
            self.params_data = {'results': {}}
            for seq_dir in os.listdir(params_file):
                seq_path = os.path.join(params_file, seq_dir)
                if os.path.isdir(seq_path) and seq_dir.isdigit():
                    self.params_data['results'][seq_dir] = {}
                    for json_file in os.listdir(seq_path):
                        if json_file.endswith('.json'):
                            frame = json_file.replace('.json', '')
                            with open(os.path.join(seq_path, json_file), 'r', encoding='utf-8') as f:
                                self.params_data['results'][seq_dir][frame] = json.load(f)
        elif os.path.exists(params_file):
            # 从单文件加载
            with open(params_file, 'r', encoding='utf-8') as f:
                self.params_data = json.load(f)

    def _apply_loaded_params(self):
        """应用当前帧的预处理参数"""
        if self.params_data is None:
            return

        seq_str = str(self.current_sequence)
        frame_str = str(self.current_frame)

        results = self.params_data.get('results', {})
        seq_data = results.get(seq_str, {})
        frame_data = seq_data.get(frame_str, {})

        # 应用点云参数
        if 'point_cloud' in frame_data:
            pc = frame_data['point_cloud']
            self.pc_dx = pc.get('dx', 0.0)
            self.pc_dy = pc.get('dy', 0.0)
            self.pc_scale = pc.get('scale', 1.0)

        # 应用SMPL参数
        if 'smpl_mesh' in frame_data:
            smpl = frame_data['smpl_mesh']
            self.smpl_dx = smpl.get('dx', 0.0)
            self.smpl_dy = smpl.get('dy', 0.0)
            self.smpl_scale = smpl.get('scale', 1.0)

    def _update_frame_list(self):
        yolo_seg_path = os.path.join(self.data_root, 'images_seg')
        if not os.path.exists(yolo_seg_path):
            yolo_seg_path = os.path.join(self.data_root, 'output', 'yolo_seg')

        seq_path = os.path.join(yolo_seg_path, str(self.current_sequence))
        if os.path.exists(seq_path):
            self.frame_list = sorted([
                int(f.replace('.txt', ''))
                for f in os.listdir(seq_path)
                if f.endswith('.txt')
            ])
        else:
            self.frame_list = []

        self.current_frame_idx = 0

    @property
    def current_sequence(self):
        return self.sequences[self.current_seq_idx]

    @property
    def current_frame(self):
        return self.frame_list[self.current_frame_idx]

    def load_current_data(self):
        seq = self.current_sequence
        frame = self.current_frame
        frame_str = f"{frame:06d}"
        index_key = f"{seq}/{frame_str}"

        data = {}

        # Load RGB
        img_path = os.path.join(self.data_root, 'images', str(seq), f"{frame_str}.png")
        if os.path.exists(img_path):
            data['image'] = np.array(Image.open(img_path).convert('RGB'))
        else:
            data['image'] = None

        # Load point cloud
        pc_path = os.path.join(self.data_root, 'labels', '3d', 'segment', str(seq), f"{frame_str}.ply")
        if os.path.exists(pc_path):
            data['point_cloud'] = load_point_cloud(pc_path)
        else:
            data['point_cloud'] = None

        # Load SMPL
        pose_path = os.path.join(self.data_root, 'labels', '3d', 'pose', str(seq), f"{frame_str}.json")
        if os.path.exists(pose_path):
            beta, pose, trans = load_pose(pose_path)
            data['smpl_mesh'] = generate_smpl_mesh(beta, pose, trans, self.smpl)
        else:
            data['smpl_mesh'] = None

        # Load YOLO
        yolo_path = os.path.join(self.data_root, 'images_seg', str(seq), f"{frame_str}.txt")
        if not os.path.exists(yolo_path):
            yolo_path = os.path.join(self.data_root, 'output', 'yolo_seg', str(seq), f"{frame_str}.txt")

        img_size = data['image'].shape if data['image'] is not None else (132, 132, 3)
        data['yolo_polygons'] = load_yolo_segment(yolo_path, img_size[1], img_size[0])
        data['yolo_mask'] = create_mask_from_polygons(data['yolo_polygons'], img_size)

        # Load top_left
        data['top_left'] = self.top_left_data.get(index_key, [0, 0])

        return data

    def auto_optimize(self, data):
        """Auto optimize both point cloud and SMPL projection"""
        print("\n开始自动优化...")
        self.optimizing = True

        # Optimize point cloud
        if data['point_cloud'] is not None:
            print("优化点云投影参数...")
            dx, dy, scale, iou = optimize_projection(
                data['point_cloud'], data['yolo_mask'],
                self.extrinsic, self.intrinsic, self.distortion, data['top_left']
            )
            self.pc_dx, self.pc_dy, self.pc_scale = dx, dy, scale
            print(f"  点云: dx={dx:.1f}, dy={dy:.1f}, scale={scale:.3f}, IoU={iou:.4f}")

        # Optimize SMPL
        if data['smpl_mesh'] is not None:
            print("优化SMPL投影参数...")
            dx, dy, scale, iou = optimize_projection(
                data['smpl_mesh'], data['yolo_mask'],
                self.extrinsic, self.intrinsic, self.distortion, data['top_left']
            )
            self.smpl_dx, self.smpl_dy, self.smpl_scale = dx, dy, scale
            print(f"  SMPL: dx={dx:.1f}, dy={dy:.1f}, scale={scale:.3f}, IoU={iou:.4f}")

        print("优化完成!\n")
        self.optimizing = False

    def create_visualization(self, data):
        img = data['image']
        if img is None:
            img = np.zeros((132, 132, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        target_size = 400
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Scale YOLO polygons
        scaled_yolo = []
        for polygon in data['yolo_polygons']:
            scaled_yolo.append((polygon * scale).astype(np.int32))

        # ===== Panel 1: RGB + YOLO =====
        panel_rgb = cv2.resize(img.copy(), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if scaled_yolo:
            yolo_mask = create_mask_from_polygons(scaled_yolo, panel_rgb.shape)
            panel_rgb[yolo_mask > 0] = (panel_rgb[yolo_mask > 0] * 0.6 + np.array([0, 100, 0]) * 0.4).astype(np.uint8)
        draw_polygon_boundary(panel_rgb, scaled_yolo, (0, 255, 0), 2)
        panel_rgb = add_label(panel_rgb, "RGB + YOLO分割 (绿色)", (0, 80, 0))

        # ===== Panel 2: Point Cloud =====
        panel_pc = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        panel_pc[:] = 30

        if data['point_cloud'] is not None:
            # Get initial projection for center
            initial_proj = project_points(data['point_cloud'], self.extrinsic,
                                         self.intrinsic, self.distortion, data['top_left'])
            center = initial_proj.mean(axis=0) if len(initial_proj) > 0 else None

            pc_2d = project_points(data['point_cloud'], self.extrinsic, self.intrinsic,
                                   self.distortion, data['top_left'],
                                   self.pc_dx, self.pc_dy, self.pc_scale, center)
            pc_2d_scaled = pc_2d * scale

            draw_points(panel_pc, pc_2d_scaled, (255, 255, 0), 2)
            pc_mask = create_mask_from_points(pc_2d_scaled, panel_pc.shape)
            panel_pc[pc_mask > 0] = (panel_pc[pc_mask > 0] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)

            data['pc_2d'] = pc_2d
            data['pc_2d_scaled'] = pc_2d_scaled
        else:
            data['pc_2d'] = np.array([])
            data['pc_2d_scaled'] = np.array([])

        draw_polygon_boundary(panel_pc, scaled_yolo, (0, 255, 0), 1)
        panel_pc = add_label(panel_pc, "点云投影 (红色)", (0, 0, 120))

        # ===== Panel 3: SMPL =====
        panel_smpl = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        panel_smpl[:] = 30

        if data['smpl_mesh'] is not None:
            initial_proj = project_points(data['smpl_mesh'], self.extrinsic,
                                         self.intrinsic, self.distortion, data['top_left'])
            center = initial_proj.mean(axis=0) if len(initial_proj) > 0 else None

            smpl_2d = project_points(data['smpl_mesh'], self.extrinsic, self.intrinsic,
                                     self.distortion, data['top_left'],
                                     self.smpl_dx, self.smpl_dy, self.smpl_scale, center)
            smpl_2d_scaled = smpl_2d * scale

            smpl_mask = create_mask_from_points(smpl_2d_scaled, panel_smpl.shape)
            panel_smpl[smpl_mask > 0] = (panel_smpl[smpl_mask > 0] * 0.5 + np.array([200, 0, 0]) * 0.5).astype(np.uint8)

            data['smpl_2d'] = smpl_2d
            data['smpl_2d_scaled'] = smpl_2d_scaled
        else:
            data['smpl_2d'] = np.array([])
            data['smpl_2d_scaled'] = np.array([])

        draw_polygon_boundary(panel_smpl, scaled_yolo, (0, 255, 0), 1)
        panel_smpl = add_label(panel_smpl, "SMPL网格投影 (蓝色)", (120, 0, 0))

        # ===== Panel 4: Comparison =====
        panel_compare = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        panel_compare[:] = 30

        if scaled_yolo:
            yolo_mask = create_mask_from_polygons(scaled_yolo, panel_compare.shape)
            panel_compare[yolo_mask > 0] = (panel_compare[yolo_mask > 0] * 0.3 + np.array([0, 100, 0]) * 0.7).astype(np.uint8)

        if len(data.get('pc_2d_scaled', [])) > 0:
            pc_mask = create_mask_from_points(data['pc_2d_scaled'], panel_compare.shape)
            panel_compare[pc_mask > 0] = (panel_compare[pc_mask > 0] * 0.3 + np.array([0, 0, 150]) * 0.7).astype(np.uint8)

        if len(data.get('smpl_2d_scaled', [])) > 0:
            smpl_mask = create_mask_from_points(data['smpl_2d_scaled'], panel_compare.shape)
            panel_compare[smpl_mask > 0] = (panel_compare[smpl_mask > 0] * 0.3 + np.array([150, 0, 0]) * 0.7).astype(np.uint8)

        panel_compare = add_label(panel_compare, "对比叠加", (50, 50, 50))

        # Calculate IoU
        masks = {}
        masks['yolo'] = data['yolo_mask']
        masks['pc'] = create_mask_from_points(data.get('pc_2d', []), img.shape)
        masks['smpl'] = create_mask_from_points(data.get('smpl_2d', []), img.shape)

        iou_pc = compute_iou(masks['pc'], masks['yolo'])
        iou_smpl = compute_iou(masks['smpl'], masks['yolo'])

        return panel_rgb, panel_pc, panel_smpl, panel_compare, iou_pc, iou_smpl

    def create_info_panel(self, iou_pc, iou_smpl, panel_h=280, panel_w=380):
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = 50

        pil_panel = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_panel)
        font = get_chinese_font(16)
        font_small = get_chinese_font(14)

        # Header
        draw.text((15, 10), f"序列: {self.current_sequence}  ({self.current_seq_idx+1}/{len(self.sequences)})", font=font, fill=(255, 255, 255))
        draw.text((15, 32), f"帧: {self.current_frame}  ({self.current_frame_idx+1}/{len(self.frame_list)})", font=font, fill=(255, 255, 255))
        draw.text((15, 54), f"点云-IoU: {iou_pc:.4f}", font=font, fill=(255, 200, 100))
        draw.text((15, 76), f"SMPL-IoU: {iou_smpl:.4f}", font=font, fill=(255, 200, 100))

        # Point cloud params
        target_mark = "►" if self.adjust_target == 0 else " "
        draw.text((15, 105), f"{target_mark} 点云参数:", font=font, fill=(255, 255, 0) if self.adjust_target == 0 else (150, 150, 150))

        params = [('dx', self.pc_dx), ('dy', self.pc_dy), ('scale', self.pc_scale)]
        for i, (name, val) in enumerate(params):
            mark = "►" if self.adjust_target == 0 and self.adjust_param == i else " "
            color = (0, 255, 255) if self.adjust_target == 0 and self.adjust_param == i else (180, 180, 180)
            draw.text((25, 127 + i * 20), f"{mark} {name}: {val:.2f}", font=font_small, fill=color)

        # SMPL params
        target_mark = "►" if self.adjust_target == 1 else " "
        draw.text((15, 195), f"{target_mark} SMPL参数:", font=font, fill=(255, 255, 0) if self.adjust_target == 1 else (150, 150, 150))

        params = [('dx', self.smpl_dx), ('dy', self.smpl_dy), ('scale', self.smpl_scale)]
        for i, (name, val) in enumerate(params):
            mark = "►" if self.adjust_target == 1 and self.adjust_param == i else " "
            color = (0, 255, 255) if self.adjust_target == 1 and self.adjust_param == i else (180, 180, 180)
            draw.text((25, 217 + i * 20), f"{mark} {name}: {val:.2f}", font=font_small, fill=color)

        return cv2.cvtColor(np.array(pil_panel), cv2.COLOR_RGB2BGR)

    def create_help_panel(self, panel_h=280, panel_w=340):
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = 50

        pil_panel = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_panel)
        font = get_chinese_font(14)

        lines = [
            ("═══════ 操作说明 ═══════", (100, 200, 255)),
            ("A / D : 上一帧 / 下一帧", (220, 220, 220)),
            ("W / S : 上一序列 / 下一序列", (220, 220, 220)),
            ("Tab   : 切换点云/SMPL", (220, 220, 220)),
            ("1/2/3 : 切换 dx/dy/scale", (220, 220, 220)),
            ("+ / - : 调整参数值", (220, 220, 220)),
            ("R     : 重置当前参数", (220, 220, 220)),
            ("O     : 自动优化", (100, 255, 100)),
            ("P     : 应用预处理参数", (100, 200, 255)),
            ("Q / ESC : 退出", (220, 220, 220)),
            ("", (220, 220, 220)),
            ("═══════ 图例说明 ═══════", (100, 200, 255)),
            ("绿色 : YOLO 分割", (50, 255, 50)),
            ("红色 : 点云投影", (50, 50, 255)),
            ("蓝色 : SMPL 网格", (255, 50, 50)),
        ]

        y = 10
        for text, color in lines:
            draw.text((15, y), text, font=font, fill=color)
            y += 19

        return cv2.cvtColor(np.array(pil_panel), cv2.COLOR_RGB2BGR)

    def run(self):
        cv2.namedWindow('Projection Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Projection Visualizer', 1400, 900)

        print("\n" + "="*60)
        print("交互式投影可视化工具")
        print("="*60)
        print("A/D: 帧 | W/S: 序列 | Tab: 点云/SMPL | 1/2/3: 参数")
        print("+/-: 调整 | R: 重置 | O: 自动优化 | P: 应用预处理参数 | Q: 退出")
        print("="*60 + "\n")

        # 初始帧应用预处理参数
        self._apply_loaded_params()

        while True:
            data = self.load_current_data()

            panel_rgb, panel_pc, panel_smpl, panel_compare, iou_pc, iou_smpl = \
                self.create_visualization(data)

            panel_info = self.create_info_panel(iou_pc, iou_smpl)
            panel_help = self.create_help_panel()

            max_h = max(panel_rgb.shape[0], panel_pc.shape[0], panel_smpl.shape[0],
                        panel_compare.shape[0], panel_info.shape[0], panel_help.shape[0])

            def pad_height(img, target_h):
                h, w = img.shape[:2]
                if h < target_h:
                    pad = np.zeros((target_h - h, w, 3), dtype=np.uint8)
                    return np.vstack([img, pad])
                return img

            panel_rgb = pad_height(panel_rgb, max_h)
            panel_pc = pad_height(panel_pc, max_h)
            panel_smpl = pad_height(panel_smpl, max_h)
            panel_compare = pad_height(panel_compare, max_h)
            panel_info = pad_height(panel_info, max_h)
            panel_help = pad_height(panel_help, max_h)

            row1 = np.hstack([panel_rgb, panel_pc, panel_smpl])
            row2 = np.hstack([panel_compare, panel_info, panel_help])

            if row2.shape[1] < row1.shape[1]:
                pad = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
                row2 = np.hstack([row2, pad])
            elif row2.shape[1] > row1.shape[1]:
                row2 = row2[:, :row1.shape[1]]

            combined = np.vstack([row1, row2])
            cv2.imshow('Projection Visualizer', combined)

            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('a'):
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                self._apply_loaded_params()
            elif key == ord('d'):
                self.current_frame_idx = min(len(self.frame_list) - 1, self.current_frame_idx + 1)
                self._apply_loaded_params()
            elif key == ord('w'):
                if self.current_seq_idx > 0:
                    self.current_seq_idx -= 1
                    self._update_frame_list()
                    self._apply_loaded_params()
            elif key == ord('s'):
                if self.current_seq_idx < len(self.sequences) - 1:
                    self.current_seq_idx += 1
                    self._update_frame_list()
                    self._apply_loaded_params()
            elif key == 9:  # Tab
                self.adjust_target = 1 - self.adjust_target
            elif key == ord('1'):
                self.adjust_param = 0
            elif key == ord('2'):
                self.adjust_param = 1
            elif key == ord('3'):
                self.adjust_param = 2
            elif key == ord('+') or key == ord('='):
                self._adjust_current_param(1.0 if self.adjust_param < 2 else 0.02)
            elif key == ord('-') or key == ord('_'):
                self._adjust_current_param(-1.0 if self.adjust_param < 2 else -0.02)
            elif key == ord('r'):
                self._reset_current_params()
            elif key == ord('o'):
                self.auto_optimize(data)
            elif key == ord('p'):
                self._apply_loaded_params()
                if self.params_data:
                    print("已应用预处理参数")
                else:
                    print("未加载预处理参数文件")

        cv2.destroyAllWindows()

    def _adjust_current_param(self, delta):
        if self.adjust_target == 0:  # Point cloud
            if self.adjust_param == 0:
                self.pc_dx += delta
            elif self.adjust_param == 1:
                self.pc_dy += delta
            elif self.adjust_param == 2:
                self.pc_scale += delta
                self.pc_scale = max(0.5, min(1.5, self.pc_scale))
        else:  # SMPL
            if self.adjust_param == 0:
                self.smpl_dx += delta
            elif self.adjust_param == 1:
                self.smpl_dy += delta
            elif self.adjust_param == 2:
                self.smpl_scale += delta
                self.smpl_scale = max(0.5, min(1.5, self.smpl_scale))

    def _reset_current_params(self):
        if self.adjust_target == 0:
            self.pc_dx = self.pc_dy = 0.0
            self.pc_scale = 1.0
        else:
            self.smpl_dx = self.smpl_dy = 0.0
            self.smpl_scale = 1.0


def main():
    parser = argparse.ArgumentParser(description='Interactive Projection Visualizer')
    parser.add_argument('--data_root', type=str,
                        default='/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M',
                        help='Dataset root directory')
    parser.add_argument('--params', type=str, default=None,
                        help='预处理参数JSON文件路径')
    args = parser.parse_args()

    visualizer = ProjectionVisualizer(args.data_root, args.params)
    visualizer.run()


if __name__ == '__main__':
    main()
