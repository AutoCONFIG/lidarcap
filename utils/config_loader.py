import yaml
import os
from typing import Dict, Any


def load_config(config_dir: str = 'config') -> Dict[str, Any]:
    """
    加载所有配置文件并合并为一个配置字典
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        合并后的配置字典
    """
    config = {}
    
    config_files = ['train.yaml', 'runtime.yaml']
    
    for filename in config_files:
        filepath = os.path.join(config_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
    
    return config


def merge_args_with_config(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将命令行参数与配置文件合并，命令行参数优先级更高
    
    Args:
        args: 命令行参数对象
        config: 配置字典
        
    Returns:
        合并后的配置字典
    """
    merged = config.copy()
    
    if hasattr(args, 'bs') and args.bs:
        if 'TRAIN' not in merged:
            merged['TRAIN'] = {}
        merged['TRAIN']['batch_size'] = args.bs
    
    if hasattr(args, 'eval_bs') and args.eval_bs:
        if 'TRAIN' not in merged:
            merged['TRAIN'] = {}
        merged['TRAIN']['eval_batch_size'] = args.eval_bs
    
    if hasattr(args, 'epochs') and args.epochs:
        if 'TRAIN' not in merged:
            merged['TRAIN'] = {}
        merged['TRAIN']['num_epochs'] = args.epochs
    
    if hasattr(args, 'threads') and args.threads:
        if 'TRAIN' not in merged:
            merged['TRAIN'] = {}
        merged['TRAIN']['num_workers'] = args.threads
    
    if hasattr(args, 'gpu') and args.gpu is not None:
        if 'RUNTIME' not in merged:
            merged['RUNTIME'] = {}
        merged['RUNTIME']['gpu_id'] = args.gpu
    
    if hasattr(args, 'output_dir') and args.output_dir:
        if 'RUNTIME' not in merged:
            merged['RUNTIME'] = {}
        merged['RUNTIME']['output_dir'] = args.output_dir
    
    if hasattr(args, 'resume') and args.resume:
        if 'RUNTIME' not in merged:
            merged['RUNTIME'] = {}
        merged['RUNTIME']['resume'] = args.resume
    
    if hasattr(args, 'ckpt_path') and args.ckpt_path:
        if 'RUNTIME' not in merged:
            merged['RUNTIME'] = {}
        merged['RUNTIME']['checkpoint_path'] = args.ckpt_path
    
    if hasattr(args, 'debug') and args.debug:
        if 'RUNTIME' not in merged:
            merged['RUNTIME'] = {}
        merged['RUNTIME']['debug'] = True
    
    if hasattr(args, 'preload') and args.preload:
        if 'RUNTIME' not in merged:
            merged['RUNTIME'] = {}
        merged['RUNTIME']['preload'] = True
    
    return merged


def get_config_value(config: Dict[str, Any], keys: str, default=None):
    """
    从嵌套配置字典中获取值
    
    Args:
        config: 配置字典
        keys: 点分隔的键路径，如 'TRAIN.batch_size'
        default: 默认值
        
    Returns:
        配置值
    """
    keys_list = keys.split('.')
    value = config
    
    try:
        for key in keys_list:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
