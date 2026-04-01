from ast import parse
from plyfile import PlyData, PlyElement
from typing import List
import argparse
import numpy as np
import json
import os
import re
import sys
import h5py
import torch
from PIL import Image

# 先添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from config import get_cfg
from tools import multiprocess
from modules.smpl import SMPL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smpl = SMPL().to(device)

MAX_PROCESS_COUNT = 64
ROOT_PATH = None  # 将在main中设置

# img_filenames = []


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def save_ply(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(filename)


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def get_sorted_filenames_by_index(dirname, isabs=True):
    if not os.path.exists(dirname):
        return []
    filenames = os.listdir(dirname)
    filenames = sorted(filenames, key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames


def parse_json(json_filename):
    with open(json_filename) as f:
        content = json.load(f)
        beta = np.array(content['beta'], dtype=np.float32)
        pose = np.array(content['pose'], dtype=np.float32)
        trans = np.array(content['trans'], dtype=np.float32)
    return beta, pose, trans


def fix_points_num(points: np.array, num_points: int):
    points = points[~np.isnan(points).any(axis=-1)]

    origin_num_points = points.shape[0]
    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    if origin_num_points >= num_points:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res


def foo(id, args):
    id = str(id)
    # cur_img_filenames = get_sorted_filenames_by_index(
    #     os.path.join(ROOT_PATH, 'images', id))

    pose_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'pose', id))
    json_filenames = list(filter(lambda x: x.endswith('json'), pose_filenames))
    ply_filenames = list(filter(lambda x: x.endswith('ply'), pose_filenames))

    cur_betas, cur_poses, cur_trans = multiprocess.multi_func(
        parse_json, MAX_PROCESS_COUNT, len(json_filenames), 'Load json files',
        True, json_filenames)
    # cur_vertices = multiprocess.multi_func(
    #     read_ply, MAX_PROCESS_COUNT, len(ply_filenames), 'Load vertices files',
    #     True, ply_filenames)

    depth_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'depth', id))
    cur_depths = depth_filenames

    segment_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'segment', id))
    cur_point_clouds = multiprocess.multi_func(
        read_ply, MAX_PROCESS_COUNT, len(segment_filenames),
        'Load segment files', True, segment_filenames)

    # 加载RGB图像 (统一resize到固定尺寸)
    IMG_SIZE = 128  # 统一图像尺寸
    image_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'images', id))
    cur_images = []
    for img_path in image_filenames:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        cur_images.append(np.array(img))

    # 检查图像数量与标注数量是否匹配
    if len(cur_images) != len(cur_betas):
        print(f"[WARNING] ID {id}: 图像数量({len(cur_images)}) != 标注数量({len(cur_betas)})")
        # 用黑色图像填充缺失的帧
        while len(cur_images) < len(cur_betas):
            cur_images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        # 如果图像多于标注，截断
        cur_images = cur_images[:len(cur_betas)]

    cur_points_nums = [min(args.npoints, points.shape[0])
                       for points in cur_point_clouds]
    cur_point_clouds = [fix_points_num(
        points, args.npoints) for points in cur_point_clouds]

    poses = []
    betas = []
    trans = []
    # vertices = []
    points_nums = []
    point_clouds = []
    depths = []
    full_joints = []
    images = []

    n = len(cur_betas)
    for i in range(n):
        poses.append(cur_poses[i])
        betas.append(cur_betas[i])
        trans.append(cur_trans[i])
        point_clouds.append(cur_point_clouds[i])
        points_nums.append(cur_points_nums[i])
        images.append(cur_images[i])

        np_pose = np.stack([cur_poses[i]])
        full_joints.append(smpl.get_full_joints(smpl(torch.from_numpy(
            np_pose).to(device), torch.zeros((1, 10)).to(device))).cpu().numpy()[0])

    return np.array(poses), np.array(betas), np.array(trans), np.array(point_clouds), np.array(points_nums), cur_depths, np.array(full_joints), np.array(images)


def test(args):
    pass


def get_sorted_ids(s):
    if re.match(r'^([1-9]\d*)-([1-9]\d*)$', s):
        start_index, end_index = s.split('-')
        indexes = list(range(int(start_index), int(end_index) + 1))
    elif re.match(r'^(([0-9]\d*),)*([0-9]\d*)$', s):
        indexes = [int(x) for x in s.split(',')]
    return sorted(indexes)


def dump(args):

    seq_str = '' if args.seqlen == 0 else 'seq{}_'.format(args.seqlen)
    ids = get_sorted_ids(args.ids)

    whole_poses = np.zeros((0, 72))
    whole_betas = np.zeros((0, 10))
    whole_trans = np.zeros((0, 3))
    # whole_vertices = np.zeros((0, 6890, 3))
    whole_point_clouds = np.zeros((0, args.npoints, 3))
    whole_points_nums = np.zeros((0,))
    whole_full_joints = np.zeros((0, 24, 3))
    whole_depths = []
    whole_images = None  # RGB images, 延迟初始化以获取正确尺寸

    for id in ids:
        # poses, betas, trans, vertices, point_clouds, points_nums = foo(
        poses, betas, trans, point_clouds, points_nums, depths, full_joints, images = foo(
            id, args)

        whole_poses = np.concatenate((whole_poses, poses))
        whole_betas = np.concatenate((whole_betas, betas))
        whole_trans = np.concatenate((whole_trans, trans))
        # whole_vertices = np.concatenate(
        #     (whole_vertices, np.stack(vertices)))
        whole_point_clouds = np.concatenate(
            (whole_point_clouds, point_clouds))
        whole_points_nums = np.concatenate(
            (whole_points_nums, points_nums))
        whole_depths += depths
        whole_full_joints = np.concatenate(
            (whole_full_joints, full_joints))

        if whole_images is None:
            whole_images = images
        else:
            whole_images = np.concatenate((whole_images, images))

    whole_filename = args.name + '.hdf5'
    with h5py.File(os.path.join(ROOT_PATH, whole_filename), 'w') as f:
        f.create_dataset('pose', data=whole_poses)
        f.create_dataset('shape', data=whole_betas)
        f.create_dataset('trans', data=whole_trans)
        # f.create_dataset('human_vertex', data=whole_vertices)
        f.create_dataset('point_clouds', data=whole_point_clouds)
        f.create_dataset('points_num', data=whole_points_nums)
        f.create_dataset('depth', data=whole_depths)
        f.create_dataset('full_joints', data=whole_full_joints)
        f.create_dataset('images', data=whole_images)
        print(f"[INFO] 保存RGB图像: shape={whole_images.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                       default='/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M',
                       help='输出目录')
    subparser = parser.add_subparsers()

    parser_dump = subparser.add_parser('dump')
    parser_dump.add_argument('--seqlen', type=int, default=0)
    parser_dump.add_argument('--npoints', type=int, default=512)
    parser_dump.add_argument('--ids', type=str, required=True)
    parser_dump.add_argument('--name', type=str, required=True)
    parser_dump.set_defaults(func=dump)

    parser_test = subparser.add_parser('test')
    parser_test.set_defaults(func=test)

    args = parser.parse_args()

    # 设置ROOT_PATH (使用globals()避免global声明问题)
    globals()['ROOT_PATH'] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    args.func(args)
