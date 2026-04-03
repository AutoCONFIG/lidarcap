"""
配置管理模块
从YAML配置文件加载所有配置
"""

import os
import yaml


class ConfigNode(dict):
    """配置节点，同时支持字典操作和属性访问"""

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigNode):
                self[key] = ConfigNode(value)  # 自动转换为 ConfigNode
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def get(self, key, default=None):
        """支持 dict.get() 方法"""
        try:
            return self[key]
        except KeyError:
            return default


def dict_to_config(d):
    """递归将字典转换为 ConfigNode"""
    if isinstance(d, dict):
        return ConfigNode({k: dict_to_config(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_config(item) for item in d]
    else:
        return d


def load_config(config_dir=None):
    """
    从YAML文件加载配置

    Args:
        config_dir: 配置文件目录，默认为 config/

    Returns:
        cfg: ConfigNode 对象
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
                    # 深度合并
                    def deep_merge(base, update):
                        for k, v in update.items():
                            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                                deep_merge(base[k], v)
                            else:
                                base[k] = v
                    deep_merge(cfg_dict, data)
        else:
            raise FileNotFoundError(f'配置文件不存在: {config_path}')

    return dict_to_config(cfg_dict)


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
        if section not in cfg:
            raise ValueError(f'配置缺少必填节: {section}')
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f'配置缺少必填项: {section}.{key}')

    # 数据集配置
    if 'TrainDataset' not in cfg:
        raise ValueError('配置缺少必填节: TrainDataset')
    if 'TestDataset' not in cfg:
        raise ValueError('配置缺少必填节: TestDataset')

    # 智能处理 gpu_id：支持 int 或 list/tuple，自动推导
    if 'gpu_id' in cfg.RUNTIME:
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
