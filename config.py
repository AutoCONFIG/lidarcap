"""
配置管理模块
从YAML配置文件加载所有配置，无默认值
"""

import os
from yacs.config import CfgNode as CN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_dir=None):
    """
    从YAML文件加载配置

    Args:
        config_dir: 配置文件目录，默认为 config/

    Returns:
        cfg: 配置对象
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
    # 必需的配置节
    required_sections = {
        'TRAIN': ['batch_size', 'num_epochs', 'learning_rate'],
        'PATHS': ['SMPL_MODEL'],
        'RUNTIME': ['gpu_id', 'output_dir'],
    }

    for section, keys in required_sections.items():
        if section not in cfg:
            raise ValueError(f'配置缺少必填节: {section}')
        for key in keys:
            if not hasattr(cfg[section], key):
                raise ValueError(f'配置缺少必填项: {section}.{key}')

    # 数据集配置
    if 'TrainDataset' not in cfg:
        raise ValueError('配置缺少必填节: TrainDataset')
    if 'TestDataset' not in cfg:
        raise ValueError('配置缺少必填节: TestDataset')

    # 智能处理 gpu_id：支持 int 或 list/tuple，自动推导
    if hasattr(cfg.RUNTIME, 'gpu_id'):
        gpu_id = cfg.RUNTIME.gpu_id
        if isinstance(gpu_id, (list, tuple)):
            cfg.RUNTIME.gpu_ids = list(gpu_id)
            cfg.RUNTIME.num_gpus = len(gpu_id)
        elif isinstance(gpu_id, int):
            cfg.RUNTIME.gpu_ids = [gpu_id]
            cfg.RUNTIME.num_gpus = 1


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
