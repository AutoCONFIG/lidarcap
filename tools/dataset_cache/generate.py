#!/usr/bin/env python3
"""
LidarCap数据集缓存制作工具

从原始数据直接生成HDF5格式的训练集和测试集缓存文件。

使用方法:
    python tools/create_dataset_cache.py \
        --data-dir /path/to/lidarhuman26M \
        --output-dir ./cache
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def read_ids_from_file(filepath):
    ids = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    return ids


def create_hdf5_from_raw(raw_data_path, output_path, ids, seargs):
    from datasets.preprocess.lidarcap import foo
    import datasets.preprocess.lidarcap as preprocess_module

    raw_data_path = Path(raw_data_path)
    output_path = Path(output_path)

    original_root = preprocess_module.ROOT_PATH
    preprocess_module.ROOT_PATH = str(raw_data_path)

    all_data = {
        'pose': [],
        'shape': [],
        'trans': [],
        'point_clouds': [],
        'points_num': [],
        'full_joints': [],
    }

    dataset_lengths = []

    for id in tqdm(ids, desc="处理数据集"):
        try:
            poses, betas, trans, point_clouds, points_nums, depths, full_joints = foo(id, seargs)

            T = poses.shape[0] * seargs.seqlen

            all_data['pose'].append(poses.reshape(T))
            all_data['shape'].append(betas.reshape(T))
            all_data['trans'].append(trans.reshape(T, 3))
            all_data['point_clouds'].append(point_clouds.reshape(T, seargs.npoints, 3))
            all_data['points_num'].append(points_nums.reshape(T))
            all_data['full_joints'].append(full_joints.reshape(T, 24, 3))

            dataset_lengths.append(T)
        except Exception as e:
            print(f"[错误] 处理数据集 {id} 失败: {e}")
            continue

    preprocess_module.ROOT_PATH = original_root

    if not dataset_lengths:
        print("[错误] 没有成功处理任何数据集!")
        return False

    print(f"[信息] 正在合并数据...")
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)

    total_frames = all_data['pose'].shape[0]
    print(f"[信息] 总帧数: {total_frames}")

    print(f"[信息] 正在创建HDF5文件: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('dataset_ids', data=np.array(ids, dtype=np.int32))
        f.create_dataset('dataset_offsets', data=np.cumsum([0] + dataset_lengths[:-1], dtype=np.int64))
        f.create_dataset('dataset_lengths', data=np.array(dataset_lengths, dtype=np.int64))

        for key, data in all_data.items():
            chunks = (min(seargs.chunk_size, total_frames),) + data.shape[1:]

            if seargs.compress:
                f.create_dataset(
                    key,
                    data=data,
                    chunks=chunks,
                    compression='gzip',
                    compression_opts=4,
                    shuffle=True
                )
            else:
                f.create_dataset(
                    key,
                    data=data,
                    chunks=chunks
                )
            print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")

    print(f"[成功] 创建完成! 输出文件: {output_path}")
    print(f"[信息] 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description='LidarCap数据集缓存制作工具')

    parser.add_argument('--data-dir', dest='data_dir', required=True,
                       help='原始数据目录（包含train.txt和test.txt）')
    parser.add_argument('--output', dest='output_dir', default='./cache',
                       help='输出目录（默认: ./cache）')
    parser.add_argument('--seqlen', dest='seqlen', type=int, default=16,
                       help='序列长度（默认: 16）')
    parser.add_argument('--npoints', dest='npoints', type=int, default=512,
                       help='点云点数（默认: 512）')
    parser.add_argument('--compress', dest='compress', action='store_true', default=True,
                       help='启用gzip压缩（默认启用）')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=1000,
                       help='分块大小（默认: 1000）')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        sys.exit(1)

    train_file = data_dir / 'train.txt'
    test_file = data_dir / 'test.txt'

    if not train_file.exists():
        print(f"[错误] 训练集文件不存在: {train_file}")
        sys.exit(1)

    if not test_file.exists():
        print(f"[错误] 测试集文件不存在: {test_file}")
        sys.exit(1)

    train_ids = read_ids_from_file(train_file)
    test_ids = read_ids_from_file(test_file)

    print("=" * 60)
    print("   LidarCap数据集缓存制作工具")
    print("=" * 60)
    print()
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"序列长度: {args.seqlen}")
    print(f"点云点数: {args.npoints}")
    print(f"压缩: {'启用' if args.compress else '禁用'}")
    print()
    print(f"训练集: {len(train_ids)} 个数据集")
    print(f"  IDs: {', '.join(map(str, train_ids))}")
    print()
    print(f"测试集: {len(test_ids)} 个数据集")
    print(f"  IDs: {', '.join(map(str, test_ids))}")
    print()

    print("[检查] 验证CUDA可用性...")
    try:
        import torch
        assert torch.cuda.is_available(), 'CUDA不可用'
        print("[检查] ✓ CUDA可用")
    except Exception as e:
        print(f"[错误] {e}")
        print("需要CUDA，请确保已正确安装CUDA和PyTorch")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  开始制作训练集缓存...")
    print("=" * 60)

    train_output = output_dir / 'lidarcap_train.hdf5'

    if not create_hdf5_from_raw(data_dir, train_output, train_ids, args):
        print("[错误] 训练集缓存制作失败")
        sys.exit(1)

    print()
    print("[成功] ✓ 训练集缓存制作完成")
    print(f"输出文件: {train_output}")

    print()
    print("=" * 60)
    print("  开始制作测试集缓存...")
    print("=" * 60)

    test_output = output_dir / 'lidarcap_test.hdf5'

    if not create_hdf5_from_raw(data_dir, test_output, test_ids, args):
        print("[错误] 测试集缓存制作失败")
        sys.exit(1)

    print()
    print("[成功] ✓ 测试集缓存制作完成")
    print(f"输出文件: {test_output}")

    print()
    print("=" * 60)
    print("  全部完成!")
    print("=" * 60)
    print()
    print("输出文件:")
    print(f"  训练集: {train_output}")
    print(f"  测试集: {test_output}")
    print()
    print("=" * 60)
    print("  使用方法")
    print("=" * 60)
    print()
    print("1. 更新 base.yaml 配置:")
    print()
    print("   TrainDataset:")
    print(f"     dataset_path: '{train_output}'")
    print()
    print("   TestDataset:")
    print(f"     dataset_path: '{test_output}'")
    print()
    print("2. 或者在代码中使用 CachedLidarCapDataset:")
    print()
    print("   from datasets import CachedLidarCapDataset")
    print()
    print("   train_dataset = CachedLidarCapDataset({")
    print(f"       'dataset_path': '{train_output}',")
    print("       'seqlen': 16")
    print("   })")
    print()
    print("   test_dataset = CachedLidarCapDataset({")
    print(f"       'dataset_path': '{test_output}',")
    print("       'seqlen': 16")
    print("   })")
    print()


if __name__ == '__main__':
    main()
