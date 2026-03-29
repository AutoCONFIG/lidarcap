"""
配置管理模块
从YAML配置文件加载所有配置
"""

import os
from yacs.config import CfgNode as CN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_dir=None):
    """
    加载所有配置文件
    
    Args:
        config_dir: 配置文件目录，默认为 config/
    
    Returns:
        cfg: 合并后的配置对象
    """
    if config_dir is None:
        config_dir = os.path.join(ROOT_DIR, 'config')
    
    cfg = CN()
    
    config_files = ['train', 'runtime']
    
    for config_name in config_files:
        config_path = os.path.join(config_dir, f'{config_name}.yaml')
        if os.path.exists(config_path):
            cfg.merge_from_file(config_path)
        else:
            raise FileNotFoundError(f'配置文件不存在: {config_path}')
    
    return cfg


def validate_config(cfg):
    """
    验证配置的完整性
    """
    required_sections = ['TRAIN', 'PATHS', 'RUNTIME']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f'配置缺少 {section} 节点')
    
    required_paths = ['DATA_DIR', 'DATASET_DIR', 'SMPL_MODEL']
    for path_key in required_paths:
        if not hasattr(cfg.PATHS, path_key) or not getattr(cfg.PATHS, path_key):
            raise ValueError(f'配置缺少 PATHS.{path_key}')
    
    if 'TrainDataset' not in cfg:
        raise ValueError('配置缺少 TrainDataset 节点')
    if 'TestDataset' not in cfg:
        raise ValueError('配置缺少 TestDataset 节点')


_cfg = None


def get_cfg():
    """获取全局配置对象"""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
        validate_config(_cfg)
    return _cfg


if __name__ == '__main__':
    cfg = get_cfg()
    print('=== 配置信息 ===')
    for section in ['TRAIN', 'PATHS', 'RUNTIME']:
        if section in cfg:
            print(f'{section}:')
            for key, val in cfg[section].items():
                print(f'  {key}: {val}')
