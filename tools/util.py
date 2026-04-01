import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.geometry import rotation_matrix_to_axis_angle
from modules.smpl import SMPL
from tools import path_util
from tools.multiprocess import multi_func
from tqdm import tqdm
import json
import numpy as np
import pickle
import torch

from config import get_cfg


_smpl_model = None
_face_index = None
_cfg = None


def _get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = get_cfg()
    return _cfg


def _load_smpl_model():
    global _smpl_model, _face_index
    if _smpl_model is None:
        cfg = _get_cfg()
        model_file = cfg.PATHS.SMPL_MODEL
        with open(model_file, 'rb') as f:
            _smpl_model = pickle.load(f, encoding='iso-8859-1')
        _face_index = _smpl_model['f'].astype(np.int64)
    return _smpl_model, _face_index


def get_gt_pose(pose_filename):
    with open(pose_filename) as f:
        content = json.load(f)
        gt_pose = np.array(content['pose'], dtype=np.float32)
        return gt_pose


def get_gt_poses(idx):
    cfg = _get_cfg()
    lidarcap_dataset_folder = cfg.PATHS.DATASET_DIR
    pose_folder = f'{lidarcap_dataset_folder}/labels/3d/pose/{idx}'
    gt_poses = []
    pose_filenames = list(filter(lambda x: x.endswith(
        '.json'), path_util.get_sorted_filenames_by_index(pose_folder)))

    gt_poses = multi_func(get_gt_pose, 32, len(
        pose_filenames), 'get_gt_poses', True, pose_filenames)
    # for pose_filename in tqdm(pose_filenames):
    #     with open(pose_filename) as f:
    #         content = json.load(f)
    #         gt_pose = np.array(content['pose'], dtype=np.float32)
    #         gt_poses.append(gt_pose)
    gt_poses = np.stack(gt_poses)
    return gt_poses




def get_pred_poses(filename, idx=None):
    """
    从文件加载预测的姿态矩阵

    Args:
        filename: 模型文件名或路径
        idx: 数据集索引（可选，用于构建完整路径）

    Returns:
        pred_poses: 预测的姿态数组 (N, 72)
    """
    # 如果 filename 不包含路径分隔符，则构建完整路径
    if idx is not None and '/' not in filename and '\\' not in filename:
        cfg = _get_cfg()
        dataset_dir = cfg.PATHS.DATASET_DIR
        if dataset_dir:
            filename = os.path.join(dataset_dir, 'predictions', str(idx), filename)

    pred_rotmats = np.load(filename).reshape(-1, 24, 3, 3)
    pred_poses = []
    for pred_rotmat in tqdm(pred_rotmats, desc=f'Loading {os.path.basename(filename)}'):
        pred_poses.append(rotation_matrix_to_axis_angle(
            torch.from_numpy(pred_rotmat)).numpy().reshape((72, )))
    pred_poses = np.stack(pred_poses)
    return pred_poses



def poses_to_vertices(poses, trans=None):
    poses = poses.astype(np.float32)
    vertices = []

    n = len(poses)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPL().to(device)
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in tqdm(range(n_batch), desc='poses_to_vertices'):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]).to(device), torch.zeros((cur_n, 10)).to(device))
        vertices.append(cur_vertices.cpu().numpy())

    vertices = np.concatenate(vertices, axis=0)
    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)
    return vertices


def save_smpl_ply(vertices, filename):
    if type(vertices) == torch.Tensor:
        vertices = vertices.squeeze().cpu().detach().numpy()
    if vertices.ndim == 3:
        assert vertices.shape[0] == 1
        vertices = vertices.squeeze(0)
    
    smpl_model, face_index = _load_smpl_model()
    
    face_1 = np.ones((face_index.shape[0], 1))
    face_1 *= 3
    face = np.hstack((face_1, face_index)).astype(int)
    with open(filename, "wb") as zjy_f:
        np.savetxt(zjy_f, vertices, fmt='%f %f %f')
        np.savetxt(zjy_f, face, fmt='%d %d %d %d')
    ply_header = '''ply
format ascii 1.0
element vertex 6890
property float x
property float y
property float z
element face 13776
property list uchar int vertex_indices
end_header
    '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header)
        f.write(old)


def poses_to_joints(poses):
    poses = poses.astype(np.float32)
    joints = []

    n = len(poses)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPL().to(device)
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in tqdm(range(n_batch), desc='poses_to_joints'):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]).to(device), torch.zeros((cur_n, 10)).to(device))
        cur_joints = smpl.get_full_joints(cur_vertices)
        joints.append(cur_joints.cpu().numpy())
    joints = np.concatenate(joints, axis=0)
    return joints
