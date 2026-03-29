"""
配置管理模块
从YAML配置文件加载所有配置，避免硬编码

优先级：
1. 环境变量
2. 配置文件
3. 默认值
"""

import os
import sys
from pathlib import Path
from yacs.config import CfgNode as CN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_default_dataset_dir():
    """获取默认数据集目录"""
    # 优先使用环境变量
    dataset_dir = os.environ.get('LIDARHUMAN_DATASET')
    if dataset_dir and os.path.exists(dataset_dir):
        return dataset_dir
    
    # 尝试Linux常用路径
    linux_path = '/media/yun/41306b47-5fbd-4e11-a4f8-13c59e123adf1/lidarhuman26M'
    if os.path.exists(linux_path):
        return linux_path
    
    # Windows默认路径
    return os.path.join(ROOT_DIR, 'data')


def load_config(config_dir=None, dataset_name=None):
    """
    加载所有配置文件
    
    Args:
        config_dir: 配置文件目录，默认为 config/
        dataset_name: 数据集名称，用于设置数据集路径
    
    Returns:
        cfg: 合并后的配置对象
    """
    if config_dir is None:
        config_dir = os.path.join(ROOT_DIR, 'config')
    
    # 加载基础配置
    cfg = CN()
    
    # 加载各个配置文件
    config_files = ['paths', 'model', 'train', 'dataset', 'runtime']
    
    for config_name in config_files:
        config_path = os.path.join(config_dir, f'{config_name}.yaml')
        if os.path.exists(config_path):
            cfg.merge_from_file(config_path)
        else:
            print(f'[WARNING] 配置文件不存在: {config_path}')
    
    # 设置数据集目录（如果环境变量存在）
    dataset_dir = get_default_dataset_dir()
    if 'PATHS' not in cfg:
        cfg.PATHS = CN()
    if not cfg.PATHS.DATASET_DIR:
        cfg.PATHS.DATASET_DIR = dataset_dir
    
    # 更新数据集路径（使用配置的数据集目录）
    if dataset_name:
        train_dataset = cfg.TrainDataset.get('dataset_path', '')
        test_dataset = cfg.TestDataset.get('dataset_path', '')
        
        if 'lidarcap_train.hdf5' in train_dataset:
            cfg.TrainDataset.dataset_path = os.path.join(dataset_dir, 'lidarcap_train.hdf5')
        
        if 'lidarcap_test.hdf5' in test_dataset:
            cfg.TestDataset.dataset_path = os.path.join(dataset_dir, 'lidarcap_test.hdf5')
    
    # 确保所有路径都是绝对路径
    if 'PATHS' in cfg:
        for key in ['DATA_DIR', 'DATASET_DIR', 'DATASET_CACHE', 'OUTPUT_DIR']:
            if hasattr(cfg.PATHS, key) and getattr(cfg.PATHS, key):
                path = getattr(cfg.PATHS, key)
                if not os.path.isabs(path):
                    setattr(cfg.PATHS, key, os.path.abspath(path))
    
    return cfg


def validate_config(cfg):
    """
    验证配置的完整性
    
    Args:
        cfg: 配置对象
    
    Raises:
        ValueError: 配置无效时
    """
    required_paths = {
        'DATA_DIR': '数据目录',
        'DATASET_DIR': '数据集目录',
        'SMPL_MODEL': 'SMPL模型文件'
    }
    
    if 'PATHS' not in cfg:
        raise ValueError('配置缺少 PATHS 节点')
    
    for path_key, desc in required_paths.items():
        if not hasattr(cfg.PATHS, path_key):
            print(f'[WARNING] 配置缺少路径: {path_key} ({desc})')
        elif not os.path.exists(getattr(cfg.PATHS, path_key)):
            print(f'[WARNING] 路径不存在: {path_key}={getattr(cfg.PATHS, path_key)} ({desc})')


# 全局配置对象（延迟加载）
_cfg = None


def get_cfg():
    """获取全局配置对象"""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
        validate_config(_cfg)
    return _cfg


# 为了向后兼容，保留旧API
DATASET_DIR = get_default_dataset_dir()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SMPL_FILE = os.path.join(DATA_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
SMPL_FACES_FILE = os.path.join(DATA_DIR, 'smpl_faces.npy')
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(DATA_DIR, 'J_regressor_extra.npy')
DATASET_CACHE_DIR = os.path.join(DATA_DIR, 'dataset_cache')
JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18,
               20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]


if __name__ == '__main__':
    cfg = get_cfg()
    print('=== 配置信息 ===')
    if 'PATHS' in cfg:
        print('路径配置:')
        for key, val in cfg.PATHS.items():
            print(f'  {key}: {val}')
    print('\n关节索引:')
    print(f'  JOINTS_IDX: {JOINTS_IDX}')
