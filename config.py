"""
配置管理模块
从YAML配置文件加载所有配置
"""

import os
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    """递归将字典转换为命名空间对象，支持 . 访问"""
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def load_config(config_dir=None):
    """
    从YAML文件加载配置

    Args:
        config_dir: 配置文件目录，默认为 config/

    Returns:
        cfg: 配置对象（支持 . 访问）
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

    cfg_dict = {}

    config_files = ['train', 'runtime']

    for config_name in config_files:
        config_path = os.path.join(config_dir, f'{config_name}.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data:
                    cfg_dict.update(data)
        else:
            raise FileNotFoundError(f'配置文件不存在: {config_path}')

    return dict_to_namespace(cfg_dict)


def validate_config(cfg):
    """
    验证配置的完整性，缺失必填项会报错
    """
    # 必需的配置节和键
    required = {
        'TRAIN': ['batch_size', 'num_epochs', 'learning_rate'],
        'PATHS': ['SMPL_MODEL'],
        'RUNTIME': ['gpu_id', 'output_dir'],
    }

    for section, keys in required.items():
        if not hasattr(cfg, section):
            raise ValueError(f'配置缺少必填节: {section}')
        section_obj = getattr(cfg, section)
        for key in keys:
            if not hasattr(section_obj, key):
                raise ValueError(f'配置缺少必填项: {section}.{key}')

    # 数据集配置
    if not hasattr(cfg, 'TrainDataset'):
        raise ValueError('配置缺少必填节: TrainDataset')
    if not hasattr(cfg, 'TestDataset'):
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
        if hasattr(cfg, section):
            print(f'{section}:')
            section_obj = getattr(cfg, section)
            for key in dir(section_obj):
                if not key.startswith('_'):
                    print(f'  {key}: {getattr(section_obj, key)}')
