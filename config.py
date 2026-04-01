"""
配置管理模块
从YAML配置文件加载所有配置
"""

import os
from yacs.config import CfgNode as CN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_cfg_defaults():
    """
    获取默认配置结构
    
    Returns:
        cfg: 默认配置对象
    """
    cfg = CN()
    
    cfg.TRAIN = CN()
    cfg.TRAIN.batch_size = 4
    cfg.TRAIN.eval_batch_size = 4
    cfg.TRAIN.num_epochs = 600
    cfg.TRAIN.num_workers = 6
    cfg.TRAIN.learning_rate = 0.0001
    cfg.TRAIN.weight_decay = 0.0001
    
    cfg.TRAIN.scheduler = CN()
    cfg.TRAIN.scheduler.factor = 0.5
    cfg.TRAIN.scheduler.patience = 5
    cfg.TRAIN.scheduler.threshold = 0.001
    cfg.TRAIN.scheduler.min_lr = 1e-7
    
    cfg.TRAIN.early_stopping = CN()
    cfg.TRAIN.early_stopping.patience = 30
    cfg.TRAIN.early_stopping.min_delta = 0.001
    
    cfg.TRAIN.grad_clip = None
    cfg.TRAIN.use_amp = False
    
    cfg.TRAIN.warmup = CN()
    cfg.TRAIN.warmup.epochs = 0
    cfg.TRAIN.warmup.min_lr = 1e-8
    
    cfg.TRAIN.checkpoint = CN()
    cfg.TRAIN.checkpoint.save_every = 1
    cfg.TRAIN.checkpoint.keep_checkpoints = 5
    
    cfg.TRAIN.log_interval = 100
    cfg.TRAIN.with_body_label = False
    cfg.TRAIN.segment_parallel = False
    cfg.TRAIN.use_drop_first = True
    cfg.TRAIN.use_denoise = False
    cfg.TRAIN.use_replace_noise = False
    cfg.TRAIN.replace_noise_pc_rate = 0.2
    
    cfg.LOSS = CN()
    cfg.LOSS.chamfer_loss = False
    cfg.LOSS.nn_loss = False
    cfg.LOSS.flow_loss = False
    cfg.LOSS.end_loss = False
    cfg.LOSS.pose_loss = True
    cfg.LOSS.zmp_loss = False
    cfg.LOSS.bone_loss = False
    
    cfg.DEVICE = 'cuda'
    
    cfg.MAMBA = CN()
    cfg.MAMBA.d_model = 1024
    cfg.MAMBA.d_state = 16
    cfg.MAMBA.d_conv = 4
    cfg.MAMBA.expand = 2
    cfg.MAMBA.n_layers = 2
    cfg.MAMBA.dropout = 0.1
    
    cfg.TEMPORAL_LOSS = CN()
    cfg.TEMPORAL_LOSS.enabled = True
    cfg.TEMPORAL_LOSS.weight = 0.1
    cfg.TEMPORAL_LOSS.velocity_weight = 1.0
    cfg.TEMPORAL_LOSS.acceleration_weight = 0.5
    cfg.TEMPORAL_LOSS.bone_length_weight = 0.3
    
    cfg.MULTIMODAL = CN()
    cfg.MULTIMODAL.enabled = False
    cfg.MULTIMODAL.rgb_encoder = 'resnet18'
    cfg.MULTIMODAL.fusion_type = 'gcsp'
    
    cfg.MODEL = CN()
    cfg.MODEL.TpointNet2 = False
    cfg.MODEL.Outline = True
    cfg.MODEL.Outline_boundary = False
    cfg.MODEL.add_background = False
    cfg.MODEL.add_twise_noise = False
    cfg.MODEL.use_project_image = False
    cfg.MODEL.use_back_wang = False
    cfg.MODEL.use_transformer = False
    cfg.MODEL.use_attention = False
    cfg.MODEL.use_mdmtransformer = False
    
    cfg.Transformer = CN()
    cfg.Transformer.num_hidden_layers = 3
    cfg.Transformer.hidden_size = 1024
    cfg.Transformer.num_attention_heads = 2
    cfg.Transformer.intermediate_size = 512
    cfg.Transformer.hidden_dropout_prob = 0.1
    
    cfg.MDMTransformer = CN()
    cfg.MDMTransformer.latent_dim = 1536
    cfg.MDMTransformer.dropout = 0.1
    cfg.MDMTransformer.num_heads = 4
    cfg.MDMTransformer.ff_size = 1024
    cfg.MDMTransformer.activation = 'gelu'
    cfg.MDMTransformer.num_layers = 8
    cfg.MDMTransformer.data_rep = 'rot6d'
    cfg.MDMTransformer.njoints = 24
    cfg.MDMTransformer.nfeats = 3
    cfg.MDMTransformer.seqlen = 16
    
    cfg.RUNTIME = CN()
    cfg.RUNTIME.gpu_id = 0
    cfg.RUNTIME.num_gpus = 1
    cfg.RUNTIME.output_dir = './lidarcap_output'
    cfg.RUNTIME.preload = True
    cfg.RUNTIME.debug = False
    cfg.RUNTIME.resume = None
    cfg.RUNTIME.checkpoint_path = None
    cfg.RUNTIME.dataset = 'lidarcap'
    
    cfg.PATHS = CN()
    cfg.PATHS.DATA_DIR = 'data'
    cfg.PATHS.DATASET_DIR = None
    cfg.PATHS.SMPL_MODEL = 'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    cfg.PATHS.SMPL_FACES = 'data/smpl_faces.npy'
    cfg.PATHS.JOINT_REGRESSOR = 'data/J_regressor_extra.npy'
    cfg.PATHS.DATASET_CACHE = 'data/dataset_cache'
    cfg.PATHS.OUTPUT_DIR = 'output'
    
    cfg.JOINTS = CN()
    cfg.JOINTS.INDEX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
    
    cfg.TrainDataset = CN()
    cfg.TrainDataset.dataset_path = ''
    cfg.TrainDataset.use_aug = False
    cfg.TrainDataset.use_rot = False
    cfg.TrainDataset.concat_info = False
    cfg.TrainDataset.use_straight = False
    cfg.TrainDataset.use_pc_w_raw_z = False
    cfg.TrainDataset.ret_raw_pc = True
    cfg.TrainDataset.seqlen = 16
    cfg.TrainDataset.drop_first_n = 0
    cfg.TrainDataset.use_trans_to_normalize = True
    cfg.TrainDataset.replace_noise_pc = False
    cfg.TrainDataset.replace_noise_pc_rate = 0.2
    cfg.TrainDataset.replace_pc_strategy = 'random'
    cfg.TrainDataset.noise_distribution = 'uniform'
    cfg.TrainDataset.add_noise_pc = False
    cfg.TrainDataset.set_body_label_all_one = False
    cfg.TrainDataset.random_permutation = False
    cfg.TrainDataset.use_boundary = False
    cfg.TrainDataset.inside_random = False
    cfg.TrainDataset.use_sample = False
    cfg.TrainDataset.range_image = False
    cfg.TrainDataset.range_image_W = 512
    cfg.TrainDataset.range_image_H = 512
    cfg.TrainDataset.fov_up = 12.5
    cfg.TrainDataset.dataset_ids = ['lidarcap_train']
    
    cfg.TestDataset = CN()
    cfg.TestDataset.dataset_path = ''
    cfg.TestDataset.use_aug = False
    cfg.TestDataset.concat_info = False
    cfg.TestDataset.use_rot = False
    cfg.TestDataset.use_straight = False
    cfg.TestDataset.use_pc_w_raw_z = False
    cfg.TestDataset.ret_raw_pc = True
    cfg.TestDataset.seqlen = 16
    cfg.TestDataset.drop_first_n = 0
    cfg.TestDataset.use_trans_to_normalize = True
    cfg.TestDataset.replace_pc_strategy = 'random'
    cfg.TestDataset.replace_noise_pc = False
    cfg.TestDataset.replace_noise_pc_rate = 0.2
    cfg.TestDataset.noise_distribution = 'uniform'
    cfg.TestDataset.add_noise_pc = False
    cfg.TestDataset.set_body_label_all_one = False
    cfg.TestDataset.random_permutation = False
    cfg.TestDataset.use_boundary = False
    cfg.TestDataset.inside_random = False
    cfg.TestDataset.use_sample = False
    cfg.TestDataset.range_image = False
    cfg.TestDataset.range_image_W = 512
    cfg.TestDataset.range_image_H = 512
    cfg.TestDataset.fov_up = 12.5
    cfg.TestDataset.dataset_ids = ['lidarcap_test']
    
    return cfg


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
    
    cfg = get_cfg_defaults()
    
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
    
    required_paths = ['DATA_DIR', 'SMPL_MODEL']
    for path_key in required_paths:
        if not hasattr(cfg.PATHS, path_key):
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
