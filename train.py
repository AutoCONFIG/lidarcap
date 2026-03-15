import argparse
import h5py
import metric
import numpy as np
import os
import torch
import logging
import signal
import json
import glob
import datetime
import copy
import time
from collections import deque

from config import DATASET_DIR
from datasets.lidarcap_dataset import collate, TemporalDataset
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
        
        self.optimizer.zero_grad()
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

class TrainingManager:
    def __init__(self, model_dir, logger):
        self.model_dir = model_dir
        self.logger = logger
        self.should_stop = False
        self.progress_file = os.path.join(model_dir, 'training_progress.json')
        self.results_file = os.path.join(model_dir, 'epoch_results.json')
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        self.logger.info("\nReceived interrupt signal. Will stop after current epoch...")
        self.should_stop = True
        
    def save_progress(self, epoch, net, optimizer, scheduler, mintloss, minvloss):
        """Save training progress"""
        progress = {
            'epoch': epoch,
            'mintloss': float(mintloss),
            'minvloss': float(minvloss),
            'model_state_dict': net.state_dict(),
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
        
    def load_progress(self):
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
                
        checkpoint = torch.load(checkpoint_path)
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

    # bs
    parser.add_argument('--bs', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--eval_bs', type=int, default=16,
                        help='input batch size for evaluation')
    # threads
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    # gpu
    parser.add_argument('--gpu', type=int,
                        default=[0], help='-1 for CPU', nargs='+')
    # epochs
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval (default: 100)')
    # dataset
    parser.add_argument("--dataset", type=str, required=True)
    # debug
    parser.add_argument('--debug', action='store_true', help='For debug mode')
    # eval or visual
    parser.add_argument('--eval', default=False, action='store_true',
                        help='evaluation the trained model')

    parser.add_argument('--visual', default=False, action='store_true',
                        help='visualization the result ply')

    # extra things, ignored
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='the saved ckpt needed to be evaluated or visualized')

    # output directory
    parser.add_argument('--output_dir', type=str, default='output',
                        help='output directory for models and results')
    
    # resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='path to the run directory to resume training from (e.g., output/run_1234567890)')
    

    args = parser.parse_args()

    iscuda = common.torch_set_gpu(args.gpu)
    common.make_reproducible(iscuda, 0)

    # Handle resume or create new run
    if args.resume:
        # Resume from existing run
        if not os.path.exists(args.resume):
            raise ValueError(f"Resume path does not exist: {args.resume}")
        run_dir = args.resume
        run_id = os.path.basename(run_dir)
        print(f"Resuming training from {run_dir}")
    else:
        # Create new run
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        run_dir = os.path.join(args.output_dir, run_id)
        print(f"Starting new training run: {run_id}")
    
    # Setup logger with log directory
    log_dir = os.path.join(run_dir, 'logs')
    logger = setup_logger(log_dir, args.debug)
    
    # Log run information
    if args.resume:
        logger.info(f"=== RESUMING TRAINING ===")
        logger.info(f"Resume directory: {run_dir}")
    else:
        logger.info(f"=== STARTING NEW TRAINING ===")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {run_dir}")
    
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Log files saved to: {log_dir}")
    
    # model save models in training
    model_dir = os.path.join(run_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    dataset_name = args.dataset
    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(open('base.yaml'))
    
    # 从配置文件中获取训练策略参数
    lr = cfg.TRAIN.GEN.get('LR', 0.0001)
    lr_patience = cfg.TRAIN.GEN.get('patience', 5)
    lr_factor = cfg.TRAIN.GEN.get('factor', 0.5)
    lr_min = cfg.TRAIN.GEN.get('min_lr', 1e-7)
    lr_threshold = cfg.TRAIN.GEN.get('threshold', 0.001)
    
    early_stopping_patience = cfg.TRAIN.GEN.get('early_stopping', 15)
    early_stopping_min_delta = cfg.TRAIN.GEN.get('early_stopping_min_delta', 0.001)
    
    grad_clip = cfg.TRAIN.GEN.get('grad_clip', None)
    use_amp = cfg.TRAIN.GEN.get('use_amp', False)
    
    warmup_epochs = cfg.TRAIN.GEN.get('warmup_epochs', 0)
    warmup_min_lr = cfg.TRAIN.GEN.get('warmup_min_lr', 1e-8)
    
    save_every = cfg.TRAIN.GEN.get('save_every', 1)
    keep_checkpoints = cfg.TRAIN.GEN.get('keep_checkpoints', 5)
    
    # Load training and validation data
    if args.eval:
        test_dataset = TemporalDataset(cfg.TestDataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.eval_bs, num_workers=args.threads, pin_memory=True, collate_fn=collate)
        loader = {'Test': test_loader}

    else:
        train_dataset = TemporalDataset(cfg=cfg.TrainDataset, dataset=dataset_name, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.threads, pin_memory=True, collate_fn=collate)
        valid_dataset = TemporalDataset(cfg=cfg.TestDataset, dataset=dataset_name, train=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=args.threads, pin_memory=True, collate_fn=collate)
        loader = {'Train': train_loader, 'Valid': valid_loader}

    net = Regressor()
    loss = Loss()

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
    
    # 记录训练配置
    logger.info("=== 训练配置 ===")
    logger.info(f"学习率: {lr}")
    logger.info(f"批次大小: {args.bs}")
    logger.info(f"权重衰减: 1e-4")
    logger.info(f"学习率调度器: ReduceLROnPlateau(factor={sc['factor']}, patience={sc['patience']}, min_lr={sc['min_lr']})")
    if warmup_scheduler:
        logger.info(f"预热启用: {warmup_epochs} epochs, 从 {warmup_min_lr} 到 {lr}")
    if use_amp:
        logger.info("混合精度训练: 已启用")
    if grad_clip:
        logger.info(f"梯度裁剪: max_norm={grad_clip}")
    
    # Initialize training manager
    training_manager = TrainingManager(model_dir, logger)
    
    # Initialize training variables
    start_epoch = 1
    mintloss = float('inf')
    minvloss = float('inf')
    
    # Load checkpoint if resuming or if ckpt_path is provided
    if args.resume:
        checkpoint = training_manager.load_progress()
        if checkpoint:
            # Load model state
            net.load_state_dict(checkpoint['model_state_dict'])
            
            # Move model to GPU first if using CUDA
            train = MyTrainer(net, loader, loss, optimizer, args.log_interval,
                             use_amp=use_amp, grad_clip=grad_clip)
            if iscuda:
                train = train.cuda()
            
            # Load optimizer and scheduler state after moving model to GPU
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            mintloss = checkpoint['mintloss']
            minvloss = checkpoint['minvloss']
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
            
            # Show training history
            history = training_manager.get_training_history()
            if history:
                logger.info("Training history:")
                for result in history[-5:]:  # Show last 5 epochs
                    logger.info(f"  Epoch {result['epoch']}: Train={result['train_loss']:.6f}, Val={result['val_loss']:.6f}")
        else:
            # Instance trainer normally if no checkpoint
            train = MyTrainer(net, loader, loss, optimizer, args.log_interval,
                             use_amp=use_amp, grad_clip=grad_clip)
            if iscuda:
                train = train.cuda()
                
    elif args.ckpt_path is not None:
        logger.info(f"Loading checkpoint from {args.ckpt_path}")
        save_model = torch.load(args.ckpt_path)['state_dict']
        model_dict = net.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
        
        # Instance trainer
        train = MyTrainer(net, loader, loss, optimizer, args.log_interval,
                         use_amp=use_amp, grad_clip=grad_clip)
        if iscuda:
            train = train.cuda()
    else:
        # Instance trainer
        train = MyTrainer(net, loader, loss, optimizer, args.log_interval,
                         use_amp=use_amp, grad_clip=grad_clip)
        if iscuda:
            train = train.cuda()

    if args.eval:
        logger.info("=== EVALUATION MODE ===")
        if args.visual:
            visual_dir = os.path.join('visual', run_id, dataset_name)
            os.makedirs(visual_dir, exist_ok=True)
            logger.info(f"Saving visualizations to {visual_dir}")
            final_loss, pred_rotmats, pred_vertices = train(
                epoch=1, train=False, test=True, visual=True)
            n = len(pred_vertices)
            filenames = [os.path.join(
                visual_dir, '{}.ply'.format(i + 1)) for i in range(n)]
            multiprocess.multi_func(save_smpl_ply, 32, len(
                pred_vertices), 'saving ply', False, pred_vertices, filenames)
            logger.info(f"Visualization files saved to {visual_dir}")

        else:
            final_loss, pred_rotmats = train(
                epoch=1, train=False, visual=False, test=True)
        
        logger.info(f'EVAL LOSS: {final_loss["loss"]}')

        pred_poses = []
        for pred_rotmat in tqdm(pred_rotmats):
            pred_poses.append(rotation_matrix_to_axis_angle(torch.from_numpy(pred_rotmat.reshape(-1, 3, 3))).numpy().reshape((-1, 72)))
        pred_poses = np.stack(pred_poses)

        test_dataset_filename = os.path.join(
            DATASET_DIR, '{}_test.hdf5'.format(dataset_name))
        test_data = h5py.File(test_dataset_filename, 'r')
        gt_poses = test_data['pose'][:]
        
        logger.info("Computing evaluation metrics...")
        metric.output_metric(pred_poses.reshape(-1, 72), gt_poses.reshape(-1, 72))
        logger.info("Evaluation completed")

    else:
        # Training loop
        logger.info("=== TRAINING MODE ===")
        logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Current best - Train Loss: {mintloss:.6f}, Val Loss: {minvloss:.6f}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Batch size: {args.bs}, Learning rate: {lr}")
        logger.info(f"Using GPU: {args.gpu if iscuda else 'CPU'}")
        
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                epoch_start_time = time.time()
                logger.info(f"Starting epoch {epoch}/{args.epochs}")
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
                
                # Log training progress
                logger.info(f"Epoch {epoch}/{args.epochs} - "
                           f"Train Loss: {train_loss_dict['loss']:.6f}, "
                           f"Val Loss: {val_loss_dict['loss']:.6f}, "
                           f"LR: {current_lr:.8f}, "
                           f"Epoch Time: {epoch_time:.2f}s")

                # Save epoch results with timing info
                training_manager.save_epoch_result(epoch, train_loss_dict['loss'], val_loss_dict['loss'],
                                                  lr=current_lr, train_time=train_time, val_time=val_time)

                # save model in this epoch
                # if this model is better, then save it as best
                if train_loss_dict['loss'] <= mintloss:
                    mintloss = train_loss_dict['loss']
                    best_save = os.path.join(model_dir, 'best-train-loss.pth')
                    torch.save({'state_dict': net.state_dict()}, best_save)
                    logger.info(f"NEW BEST TRAIN LOSS! Saving model at epoch {epoch} (loss: {mintloss:.6f})")
                    
                if val_loss_dict['loss'] <= minvloss:
                    minvloss = val_loss_dict['loss']
                    best_save = os.path.join(model_dir, 'best-valid-loss.pth')
                    torch.save({'state_dict': net.state_dict()}, best_save)
                    logger.info(f"NEW BEST VALIDATION LOSS! Saving model at epoch {epoch} (loss: {minvloss:.6f})")

                # 学习率调度器（不在预热期间使用）
                if warmup_scheduler is None or epoch > warmup_scheduler.warmup_epochs:
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(val_loss_dict['loss'])  # 使用验证损失调整学习率
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    if new_lr != old_lr:
                        logger.info(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")
                
                # 早停检查（使用验证损失）
                if early_stopping is not None:
                    if early_stopping(val_loss_dict['loss'], net):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        logger.info(f"No improvement in validation loss for {early_stopping.patience} epochs")
                        logger.info(f"Best validation loss: {early_stopping.best_score:.6f}")
                        break
                
                # Save progress after each epoch (or every N epochs)
                if epoch % save_every == 0:
                    training_manager.save_progress_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)
                
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
            except:
                pass
            raise
            
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
