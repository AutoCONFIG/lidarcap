import argparse
import h5py
import metric
import numpy as np
import os
import torch
import torch.nn as nn
import logging
import signal
import json
import glob
import datetime
import copy
import time
import threading
import queue
 
from config import get_cfg
from datasets.lidarcap_dataset import collate, TemporalDataset, CachedLidarCapDataset
from modules.geometry import rotation_matrix_to_axis_angle
from modules.regressor import Regressor
from modules.loss import Loss
from tools import common, crafter, multiprocess
from tools.util import save_smpl_ply
from tqdm import tqdm

# 尝试导入混合精度训练支持
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    autocast = None
    GradScaler = None

torch.set_num_threads(1)

def setup_logger(log_dir, debug=False):
    """Setup logger to save logs to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file, mode='a')  # append mode for resume
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Separate file handler for errors only
    error_log_file = os.path.join(log_dir, 'error.log')
    error_handler = logging.FileHandler(error_log_file, mode='a')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

class EarlyStopping:
    """早停机制，当验证损失不再改善时提前终止训练"""
    def __init__(self, patience=10, min_delta=0.001, mode='min', restore_best_weights=True):
        """
        Args:
            patience: 容忍的epoch数，在该epoch数内验证损失没有改善则停止
            min_delta: 认为是改善的最小变化量
            mode: 'min' 或 'max'，监控指标是最小化还是最大化
            restore_best_weights: 停止时是否恢复最佳模型权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, score, model):
        """
        Args:
            score: 当前epoch的监控指标值
            model: 模型对象，用于保存最佳权重
        Returns:
            True: 应该停止训练
            False: 继续训练
        """
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            return False
    
    def reset(self):
        """重置早停计数器"""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
        self.best_weights = None


class MyTrainer(crafter.Trainer):
    def __init__(self, net, loader, loss, optimizer, log_interval,
                 use_amp=False, grad_clip=None):
        """
        Args:
            net: 神经网络模型
            loader: 数据加载器
            loss: 损失函数
            optimizer: 优化器
            log_interval: 日志打印间隔
            use_amp: 是否使用混合精度训练
            grad_clip: 梯度裁剪的最大范数，None表示不裁剪
        """
        super().__init__(net, loader, loss, optimizer, log_interval)
        self.use_amp = use_amp and AMP_AVAILABLE
        self.grad_clip = grad_clip
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            logging.info("混合精度训练已启用")
        if self.grad_clip:
            logging.info(f"梯度裁剪已启用，max_norm={self.grad_clip}")
    
    def forward_backward(self, inputs):
        if self.use_amp:
            with autocast():
                output = self.net(inputs)
                loss, details = self.loss_func(**output)

            # 混合精度反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.net(inputs)
            loss, details = self.loss_func(**output)
            loss.backward()

            # 梯度裁剪
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

            self.optimizer.step()

        # set_to_none=True 可以减少显存访问，略微提升性能
        self.optimizer.zero_grad(set_to_none=True)
        return details

    def forward_val(self, inputs):
        # 验证时也支持混合精度
        if self.use_amp:
            with autocast():
                output = self.net(inputs)
                loss, details = self.loss_func(**output)
        else:
            output = self.net(inputs)
            loss, details = self.loss_func(**output)
        return details

    def forward_net(self, inputs):
        output = self.net(inputs)
        return output

class AsyncCheckpointSaver:
    """异步检查点保存器，使用后台线程保存权重，避免阻塞训练"""
    
    def __init__(self, max_queue_size=5):
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.running = False
        self.saved_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        
    def start(self):
        """启动后台保存线程"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            
    def stop(self, wait=True):
        """停止后台保存线程"""
        self.running = False
        if wait and self.worker_thread is not None:
            # 发送结束信号
            try:
                self.save_queue.put(None, block=False)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=60)
            
    def save_async(self, state_dict, filepath, metadata=None):
        """异步保存检查点
        
        Args:
            state_dict: 要保存的状态字典
            filepath: 保存路径
            metadata: 可选的元数据字典
        """
        if not self.running:
            self.start()
            
        # 将张量移到CPU以避免CUDA内存问题
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu()
            else:
                cpu_state_dict[key] = value
                
        save_info = {
            'state_dict': cpu_state_dict,
            'filepath': filepath,
            'metadata': metadata or {}
        }
        
        try:
            # 如果队列已满，移除最旧的未处理项
            if self.save_queue.full():
                try:
                    old_item = self.save_queue.get_nowait()
                    logging.warning(f"保存队列已满，丢弃旧的保存请求: {old_item['filepath']}")
                except queue.Empty:
                    pass
                    
            self.save_queue.put(save_info, block=True, timeout=1)
        except queue.Full:
            logging.error(f"无法将检查点加入保存队列: {filepath}")
            
    def _worker_loop(self):
        """后台工作线程主循环"""
        while self.running:
            try:
                save_info = self.save_queue.get(block=True, timeout=0.5)
                
                # None 是停止信号
                if save_info is None:
                    break
                    
                self._do_save(save_info)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"保存检查点时发生错误: {e}", exc_info=True)
                with self.lock:
                    self.failed_count += 1
                    
    def _do_save(self, save_info):
        """执行实际的保存操作"""
        state_dict = save_info['state_dict']
        filepath = save_info['filepath']
        metadata = save_info['metadata']
        
        try:
            # 创建保存数据
            save_data = {
                'state_dict': state_dict,
                'metadata': metadata
            }
            
            # 使用临时文件进行原子写入
            temp_path = filepath + '.tmp'
            torch.save(save_data, temp_path)
            
            # 原子重命名
            if os.path.exists(filepath):
                os.replace(temp_path, filepath)
            else:
                os.rename(temp_path, filepath)
                
            logging.info(f"异步保存检查点完成: {filepath}")
            
            with self.lock:
                self.saved_count += 1
                
        except Exception as e:
            logging.error(f"保存检查点失败 {filepath}: {e}", exc_info=True)
            with self.lock:
                self.failed_count += 1
                
    def get_stats(self):
        """获取保存统计信息"""
        with self.lock:
            return {
                'saved_count': self.saved_count,
                'failed_count': self.failed_count,
                'queue_size': self.save_queue.qsize()
            }

class TrainingProgressTracker:
    def __init__(self, model_dir, logger):
        self.model_dir = model_dir
        self.logger = logger
        self.should_stop = False
        self.progress_file = os.path.join(model_dir, 'training_progress.json')
        self.results_file = os.path.join(model_dir, 'epoch_results.json')
        
        # Windows兼容的信号处理
        import platform
        if platform.system() != 'Windows':
            signal.signal(signal.SIGINT, self._signal_handler)
        else:
            # Windows上使用atexit
            import atexit
            atexit.register(self._cleanup)
        
    def _signal_handler(self, signum, frame):
        self.logger.info("\nReceived interrupt signal. Will stop after current epoch...")
        self.should_stop = True
    
    def _cleanup(self):
        """Windows兼容的清理方法"""
        if self.should_stop:
            self.logger.info("Training interrupted. Cleaning up...")
        
    def save_progress(self, epoch, net, optimizer, scheduler, mintloss, minvloss):
        """Save training progress"""
        # 处理 DataParallel 包装的模型
        model_state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()

        progress = {
            'epoch': epoch,
            'mintloss': float(mintloss),
            'minvloss': float(minvloss),
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        
        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(progress, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.model_dir, 'latest_checkpoint.pth')
        torch.save(progress, latest_path)
        
        # Save progress info
        progress_info = {
            'epoch': epoch,
            'mintloss': float(mintloss),
            'minvloss': float(minvloss),
            'checkpoint_path': checkpoint_path
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_info, f, indent=2)
            
        self.logger.info(f"Progress saved at epoch {epoch}")
        
    def load_progress(self, iscuda):
        """Load training progress"""
        if not os.path.exists(self.progress_file):
            return None
            
        with open(self.progress_file, 'r') as f:
            progress_info = json.load(f)
            
        checkpoint_path = progress_info['checkpoint_path']
        if not os.path.exists(checkpoint_path):
            # Try latest checkpoint
            checkpoint_path = os.path.join(self.model_dir, 'latest_checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                self.logger.warning("Checkpoint file not found, starting from scratch")
                return None
                
        checkpoint = torch.load(checkpoint_path, map_location='cpu' if not iscuda else None)
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
        
    def save_epoch_result(self, epoch, train_loss, val_loss, lr=None, train_time=0, val_time=0):
        """Save epoch results with additional metrics"""
        result = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'learning_rate': lr if lr is not None else 0.0,
            'train_time_seconds': train_time,
            'val_time_seconds': val_time,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Load existing results
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)
                
        # Add new result
        results.append(result)
        
        # Save updated results
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    def get_training_history(self):
        """Get training history"""
        if not os.path.exists(self.results_file):
            return []
            
        with open(self.results_file, 'r') as f:
            return json.load(f)

def cleanup_old_checkpoints(model_dir, keep_last=5, logger=None):
    """Clean up old checkpoint files, keeping only the most recent ones and best models"""
    try:
        # Get all checkpoint files
        checkpoint_pattern = os.path.join(model_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= keep_last:
            return
            
        # Sort by epoch number
        def get_epoch_num(filepath):
            filename = os.path.basename(filepath)
            return int(filename.split('_')[-1].split('.')[0])
            
        checkpoint_files.sort(key=get_epoch_num)
        
        # Keep only the last N checkpoints
        files_to_remove = checkpoint_files[:-keep_last]
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                if logger:
                    logger.debug(f"Removed old checkpoint: {os.path.basename(file_path)}")
            except OSError:
                pass
                
    except Exception as e:
        if logger:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

class WarmupScheduler:
    """学习率预热调度器，在前几个epoch线性增加学习率"""
    def __init__(self, optimizer, warmup_epochs, target_lr, min_lr=1e-8):
        """
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: 预热的epoch数
            target_lr: 预热结束后的目标学习率
            min_lr: 初始最小学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 线性增加
            lr = self.min_lr + (self.target_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config-dir', type=str, default='config',
                        help='配置文件目录 (default: config)')
    
    args = parser.parse_args()
    
    cfg = get_cfg()
    
    batch_size = cfg.TRAIN.batch_size
    eval_batch_size = cfg.TRAIN.eval_batch_size
    num_epochs = cfg.TRAIN.num_epochs
    num_workers = cfg.TRAIN.num_workers
    log_interval = cfg.TRAIN.log_interval
    dataset_name = cfg.RUNTIME.dataset
    debug = cfg.RUNTIME.debug
    resume = cfg.RUNTIME.resume
    ckpt_path = cfg.RUNTIME.checkpoint_path
    preload = cfg.RUNTIME.preload
    output_dir = cfg.RUNTIME.output_dir
    gpu_ids = cfg.RUNTIME.gpu_ids  # 自动推导：单卡 [0] 或多卡 [0,1,2,3]

    iscuda = common.torch_set_gpu(gpu_ids)
    common.make_reproducible(iscuda, 0)

    # Handle resume or create new run
    if resume:
        if not os.path.exists(resume):
            raise ValueError(f"Resume path does not exist: {resume}")
        run_dir = resume
        run_id = os.path.basename(run_dir)
        print(f"Resuming training from {run_dir}")
    else:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        run_dir = os.path.join(output_dir, run_id)
        print(f"Starting new training run: {run_id}")
    
    log_dir = os.path.join(run_dir, 'logs')
    logger = setup_logger(log_dir, debug)
    
    if resume:
        logger.info(f"=== RESUMING TRAINING ===")
        logger.info(f"Resume directory: {run_dir}")
    else:
        logger.info(f"=== STARTING NEW TRAINING ===")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {run_dir}")
    
    logger.info(f"Configuration loaded from: {args.config_dir}")
    logger.info(f"Log files saved to: {log_dir}")
    
    model_dir = os.path.join(run_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载配置
    cfg = get_cfg()
    
    lr = cfg.TRAIN.learning_rate
    lr_patience = cfg.TRAIN.scheduler.patience
    lr_factor = cfg.TRAIN.scheduler.factor
    lr_min = float(cfg.TRAIN.scheduler.min_lr)
    lr_threshold = cfg.TRAIN.scheduler.threshold
    
    early_stopping_patience = cfg.TRAIN.early_stopping.patience
    early_stopping_min_delta = cfg.TRAIN.early_stopping.min_delta
    
    grad_clip = cfg.TRAIN.grad_clip
    use_amp = cfg.TRAIN.use_amp
    
    warmup_epochs = cfg.TRAIN.warmup.epochs
    warmup_min_lr = cfg.TRAIN.warmup.min_lr
    
    save_every = cfg.TRAIN.checkpoint.save_every
    keep_checkpoints = cfg.TRAIN.checkpoint.keep_checkpoints
    
    if preload:
        train_dataset = CachedLidarCapDataset(cfg=cfg.TrainDataset, dataset=dataset_name, train=True, preload=True)
    else:
        train_dataset = TemporalDataset(cfg=cfg.TrainDataset, dataset=dataset_name, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    if preload:
        valid_dataset = CachedLidarCapDataset(cfg=cfg.TestDataset, dataset=dataset_name, train=False, preload=True)
    else:
        valid_dataset = TemporalDataset(cfg=cfg.TestDataset, dataset=dataset_name, train=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    loader = {'Train': train_loader, 'Valid': valid_loader}

    # Initialize async checkpoint saver for non-blocking model saves
    logger.info("[INFO] Initializing AsyncCheckpointSaver for non-blocking model saves")
    async_saver = AsyncCheckpointSaver(max_queue_size=10)
    async_saver.start()

    net = Regressor(cfg=cfg)
    loss = Loss(cfg=cfg)

    # 多卡并行：只包装模型，不包装训练器
    # 注意：device_ids 必须是逻辑 GPU 编号 [0, 1, 2, ...]
    # 因为 CUDA_VISIBLE_DEVICES 已经设置了可见 GPU
    if iscuda and len(gpu_ids) > 1:
        net = nn.DataParallel(net, device_ids=list(range(len(gpu_ids))))
        logger.info(f"Using DataParallel on {len(gpu_ids)} GPUs: {gpu_ids} (logical: {list(range(len(gpu_ids)))})")
    if iscuda:
        net = net.cuda()
        loss = loss.cuda()  # Loss 模块也需要移动到 GPU

    # Define optimizer with improved parameters
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=1e-4, eps=1e-8, amsgrad=True)
    
    # 改进的学习率调度器配置
    sc = {
        'factor': lr_factor,
        'patience': lr_patience,
        'threshold': lr_threshold,
        'min_lr': lr_min
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=sc['factor'], patience=sc['patience'],
        threshold_mode='rel', threshold=sc['threshold'], min_lr=sc['min_lr'])

    # 初始化预热调度器
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, lr, warmup_min_lr) if warmup_epochs > 0 else None

    # 初始化早停机制
    early_stopping = None
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='min',
            restore_best_weights=True
        )
        logger.info(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    logger.info("=== 训练配置 ===")
    logger.info(f"学习率: {lr}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"权重衰减: 1e-4")
    logger.info(f"学习率调度器: ReduceLROnPlateau(factor={sc['factor']}, patience={sc['patience']}, min_lr={sc['min_lr']})")
    if warmup_scheduler:
        logger.info(f"预热启用: {warmup_epochs} epochs, 从 {warmup_min_lr} 到 {lr}")
    if use_amp:
        logger.info("混合精度训练: 已启用")
    if grad_clip:
        logger.info(f"梯度裁剪: max_norm={grad_clip}")
    
    # Initialize training manager
    training_manager = TrainingProgressTracker(model_dir, logger)
    
    start_epoch = 1
    mintloss = float('inf')
    minvloss = float('inf')
    
    if resume:
        checkpoint = training_manager.load_progress(iscuda)
        if checkpoint:
            # 加载模型权重时需要处理 DataParallel 的 key 前缀
            state_dict = checkpoint['model_state_dict']
            # 如果保存的是 DataParallel 模型，去除 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            net.load_state_dict(new_state_dict)

            train = MyTrainer(net, loader, loss, optimizer, log_interval,
                             use_amp=use_amp, grad_clip=grad_clip)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            for i in range(len(scheduler.min_lrs)):
                scheduler.min_lrs[i] = float(scheduler.min_lrs[i])

            start_epoch = checkpoint['epoch'] + 1
            mintloss = checkpoint['mintloss']
            minvloss = checkpoint['minvloss']
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")

            history = training_manager.get_training_history()
            if history:
                logger.info("Training history:")
                for result in history[-5:]:
                    logger.info(f"  Epoch {result['epoch']}: Train={result['train_loss']:.6f}, Val={result['val_loss']:.6f}")
        else:
            train = MyTrainer(net, loader, loss, optimizer, log_interval,
                             use_amp=use_amp, grad_clip=grad_clip)

    elif ckpt_path is not None:
        logger.info(f"Loading checkpoint from {ckpt_path}")
        save_model = torch.load(ckpt_path, map_location='cpu' if not iscuda else None)['state_dict']
        model_dict = net.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)

        train = MyTrainer(net, loader, loss, optimizer, log_interval,
                         use_amp=use_amp, grad_clip=grad_clip)
    else:
        train = MyTrainer(net, loader, loss, optimizer, log_interval,
                         use_amp=use_amp, grad_clip=grad_clip)
         
    if ckpt_path is not None:
        logger.info("=== EVALUATION MODE ===")
        logger.info(f"Model loaded from {ckpt_path}")
        logger.info("Use evaluation script instead")
    else:
        logger.info("=== TRAINING MODE ===")
        logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Current best - Train Loss: {mintloss:.6f}, Val Loss: {minvloss:.6f}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Batch size: {batch_size}, Learning rate: {lr}")
        logger.info(f"Using GPU: {gpu_ids if iscuda else 'CPU'}")
        
        epoch = start_epoch
        
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                epoch_start_time = time.time()
                logger.info(f"Starting epoch {epoch}/{num_epochs}")
                logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
                
                # 应用预热学习率
                if warmup_scheduler is not None and epoch <= warmup_scheduler.warmup_epochs:
                    warmup_scheduler.step()
                    logger.info(f"Warmup learning rate: {optimizer.param_groups[0]['lr']:.8f}")
                
                # Training
                train_start_time = time.time()
                train_loss_dict = train(epoch)
                train_time = time.time() - train_start_time
                logger.info(f"Epoch {epoch} - Training completed, loss: {train_loss_dict['loss']:.6f}, time: {train_time:.2f}s")
                
                # Validation
                val_start_time = time.time()
                val_loss_dict = train(epoch, train=False)
                val_time = time.time() - val_start_time
                logger.info(f"Epoch {epoch} - Validation completed, loss: {val_loss_dict['loss']:.6f}, time: {val_time:.2f}s")
                
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(f"Epoch {epoch}/{num_epochs} - "
                           f"Train Loss: {train_loss_dict['loss']:.6f}, "
                           f"Val Loss: {val_loss_dict['loss']:.6f}, "
                           f"LR: {current_lr:.8f}, "
                           f"Epoch Time: {epoch_time:.2f}s")

                # Save epoch results with timing info
                training_manager.save_epoch_result(epoch, train_loss_dict['loss'], val_loss_dict['loss'],
                                                  lr=current_lr, train_time=train_time, val_time=val_time)

                # save model in this epoch
                # if this model is better, then save it as best
                model_state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
                if train_loss_dict['loss'] <= mintloss:
                    mintloss = train_loss_dict['loss']
                    best_save = os.path.join(model_dir, 'best-train-loss.pth')
                    async_saver.save_async(model_state, best_save, metadata={'epoch': epoch, 'loss': mintloss, 'type': 'best_train'})
                    logger.info(f"NEW BEST TRAIN LOSS! Queuing async save at epoch {epoch} (loss: {mintloss:.6f})")

                if val_loss_dict['loss'] <= minvloss:
                    minvloss = val_loss_dict['loss']
                    best_save = os.path.join(model_dir, 'best-valid-loss.pth')
                    async_saver.save_async(model_state, best_save, metadata={'epoch': epoch, 'loss': minvloss, 'type': 'best_valid'})
                    logger.info(f"NEW BEST VALIDATION LOSS! Queuing async save at epoch {epoch} (loss: {minvloss:.6f})")

                # 学习率调度器（不在预热期间使用）
                if warmup_scheduler is None or epoch > warmup_scheduler.warmup_epochs:
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(val_loss_dict['loss'])  # 使用验证损失调整学习率
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    if new_lr != old_lr:
                        logger.info(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")
                
                # 早停检查（使用验证损失）
                if early_stopping is not None:
                    # 获取实际模型用于早停恢复权重
                    actual_model = net.module if isinstance(net, nn.DataParallel) else net
                    if early_stopping(val_loss_dict['loss'], actual_model):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        logger.info(f"No improvement in validation loss for {early_stopping.patience} epochs")
                        logger.info(f"Best validation loss: {early_stopping.best_score:.6f}")
                        break
                
                # Save progress after each epoch (or every N epochs)
                if epoch % save_every == 0:
                    training_manager.save_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)
                
                # 定期清理旧检查点
                if epoch % 10 == 0:
                    cleanup_old_checkpoints(model_dir, keep_last=keep_checkpoints, logger=logger)

                # Check if we should stop (Ctrl+C was pressed)
                if training_manager.should_stop:
                    logger.info(f"Training interrupted at epoch {epoch}. Progress saved.")
                    logger.info(f"To resume training, use: --resume {run_dir}")
                    break
                    
        except KeyboardInterrupt:
            logger.info(f"Training interrupted by user. Progress saved at epoch {epoch}.")
            logger.info(f"To resume training, use: --resume {run_dir}")
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            # Save progress even if training fails
            try:
                training_manager.save_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)
            except Exception as save_error:
                logger.error(f"Failed to save progress during error: {save_error}", exc_info=True)
            raise
        finally:
            # Stop async saver to ensure all saves complete
            logger.info("[INFO] Stopping async checkpoint saver...")
            async_saver.stop(wait=True)
            logger.info("[INFO] Async checkpoint saver stopped.")
             
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(f"Training completed at epoch {epoch}")
        logger.info(f"Best train loss: {mintloss:.6f}")
        logger.info(f"Best valid loss: {minvloss:.6f}")
        logger.info(f"All checkpoints and results saved in: {model_dir}")
        logger.info(f"Log files saved in: {log_dir}")
        
        # Show final training history
        history = training_manager.get_training_history()
        if history and len(history) > 0:
            logger.info("Final training history (last 10 epochs):")
            for result in history[-10:]:
                logger.info(f"  Epoch {result['epoch']}: Train={result['train_loss']:.6f}, Val={result['val_loss']:.6f}, "
                          f"LR={result.get('learning_rate', 0):.8f}")
        
        # Clean up old checkpoints (keep only last N and best models)
        logger.info(f"Cleaning up old checkpoint files, keeping last {keep_checkpoints}...")
        cleanup_old_checkpoints(model_dir, keep_last=keep_checkpoints, logger=logger)
        logger.info("Training pipeline completed successfully!")
