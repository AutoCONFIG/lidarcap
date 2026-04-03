import argparse
import h5py
import metric
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
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
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    # 兼容旧版本 PyTorch
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False
        autocast = None
        GradScaler = None


class AutoBatchSizeFinder:
    """
    自动寻找最优 batch size 以充分利用 GPU 显存

    通过二分搜索找到最大的可用 batch size，目标是将显存利用率提升到指定比例
    """

    def __init__(self, model, loss_fn, sample_input, optimizer_class,
                 use_amp=False, target_memory_usage=0.90,
                 min_batch_size=1, max_batch_size=64,
                 growth_factor=2.0, warmup_iterations=3, logger=None):
        """
        Args:
            model: 要训练的模型
            loss_fn: 损失函数
            sample_input: 样本输入数据 (单个样本)
            optimizer_class: 优化器工厂函数
            use_amp: 是否使用混合精度
            target_memory_usage: 目标显存利用率 (0.85-0.95)
            min_batch_size: 最小 batch size
            max_batch_size: 最大 batch size
            growth_factor: 增长因子，用于快速探索
            warmup_iterations: 预热迭代次数，用于稳定显存测量
            logger: 日志记录器
        """
        self.model = model
        self.loss_fn = loss_fn
        self.sample_input = sample_input
        self.optimizer_class = optimizer_class
        self.use_amp = use_amp
        self.target_memory_usage = target_memory_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.growth_factor = growth_factor
        self.warmup_iterations = warmup_iterations
        self.logger = logger

        self.best_batch_size = min_batch_size
        self.device = next(model.parameters()).device

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _get_memory_info(self):
        """获取当前 GPU 显存信息，使用 max_memory_allocated 更准确"""
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        # 使用 max_memory_allocated 获取峰值显存，更准确
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        return {
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'free': total - reserved,
            'utilization': max_allocated / total if total > 0 else 0
        }

    def _clear_cache(self):
        """清理 GPU 缓存并重置显存统计"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def _create_batch_input(self, batch_size):
        """创建指定大小的 batch 输入"""
        batch_input = {}
        for key, value in self.sample_input.items():
            if isinstance(value, torch.Tensor):
                # 扩展为 batch，使用 clone() 确保内存独立
                batch_input[key] = value.unsqueeze(0).expand(batch_size, *value.shape).clone()
            elif isinstance(value, (list, tuple)):
                batch_input[key] = [value[0]] * batch_size if len(value) > 0 else []
            else:
                batch_input[key] = value
        return batch_input

    def _try_batch_size(self, batch_size):
        """
        尝试运行指定 batch size 的前向+反向传播

        Returns:
            success: 是否成功
            memory_info: 显存信息 (成功时)
        """
        self._clear_cache()

        try:
            # 创建 batch 输入
            batch_input = self._create_batch_input(batch_size)

            # 创建新的优化器 (避免累积梯度)
            optimizer = self.optimizer_class(self.model.parameters(), lr=1e-4)

            # 预热阶段：运行几次迭代稳定显存
            for _ in range(self.warmup_iterations):
                self.model.train()
                if self.use_amp:
                    with autocast():
                        output = self.model(batch_input)
                        loss, _ = self.loss_fn(**output)
                    scaler = GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(batch_input)
                    loss, _ = self.loss_fn(**output)
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                del output, loss
                torch.cuda.synchronize()

            # 最终测量：再跑一次并测量峰值显存
            torch.cuda.reset_peak_memory_stats()

            self.model.train()
            if self.use_amp:
                with autocast():
                    output = self.model(batch_input)
                    loss, _ = self.loss_fn(**output)
                scaler = GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = self.model(batch_input)
                loss, _ = self.loss_fn(**output)
                loss.backward()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # 同步并获取显存信息
            torch.cuda.synchronize()
            final_memory = self._get_memory_info()

            # 清理
            del batch_input, output, loss, optimizer
            self._clear_cache()

            return True, final_memory

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._clear_cache()
                return False, None
            raise
        except Exception as e:
            self._clear_cache()
            self._log(f"Unexpected error at batch_size={batch_size}: {e}")
            return False, None

    def find_optimal_batch_size(self):
        """
        使用二分搜索找到最优 batch size

        策略:
        1. 快速增长阶段：从 min 开始，按 growth_factor 倍增，直到 OOM
        2. 二分搜索阶段：在可行区间内精确搜索最大可用值
        """
        self._log("=" * 60)
        self._log("Auto Batch Size Finder - 开始自动搜索最优 batch size")
        self._log(f"目标显存利用率: {self.target_memory_usage * 100:.1f}%")
        self._log(f"搜索范围: [{self.min_batch_size}, {self.max_batch_size}]")
        self._log(f"预热迭代次数: {self.warmup_iterations}")
        self._log("=" * 60)

        # 缓存已测试的结果，避免重复测试
        tested_results = {}

        def test_and_cache(bs):
            if bs in tested_results:
                return tested_results[bs]
            result = self._try_batch_size(bs)
            tested_results[bs] = result
            return result

        # 阶段1: 快速增长探索上界
        self._log("\n[阶段1] 快速探索阶段...")
        low = self.min_batch_size
        high = self.max_batch_size

        current = low
        while current <= high:
            success, memory_info = test_and_cache(current)
            if success:
                self.best_batch_size = current
                utilization = memory_info['utilization']
                self._log(f"  batch_size={current}: 成功, 峰值显存={memory_info['max_allocated']/1024**3:.2f}GB, 利用率={utilization*100:.1f}%")

                # 如果已经达到目标利用率，可以提前结束
                if utilization >= self.target_memory_usage:
                    self._log(f"  已达到目标利用率，停止搜索")
                    break

                # 尝试更大的 batch size
                next_size = int(current * self.growth_factor)
                if next_size == current:
                    next_size = current + 1
                current = next_size
            else:
                self._log(f"  batch_size={current}: OOM，设为上界")
                high = current - 1
                break

        if self.best_batch_size == self.min_batch_size:
            success, _ = test_and_cache(self.min_batch_size)
            if not success:
                self._log(f"错误: 最小 batch_size={self.min_batch_size} 都 OOM，请检查模型或降低配置")
                return self.min_batch_size

        # 阶段2: 二分搜索找最大可用值
        self._log(f"\n[阶段2] 二分搜索阶段 (范围: {low}-{high})...")

        while low <= high:
            mid = (low + high) // 2
            success, memory_info = test_and_cache(mid)

            if success:
                utilization = memory_info['utilization']
                self._log(f"  batch_size={mid}: 成功, 峰值显存={memory_info['max_allocated']/1024**3:.2f}GB, 利用率={utilization*100:.1f}%")

                # 尝试更大的值
                low = mid + 1
                self.best_batch_size = mid
            else:
                self._log(f"  batch_size={mid}: OOM")
                high = mid - 1

        # 最终报告
        success, final_memory = test_and_cache(self.best_batch_size)
        self._log("\n" + "=" * 60)
        self._log(f"自动 Batch Size 搜索完成!")
        self._log(f"最优 batch size: {self.best_batch_size}")
        if final_memory:
            self._log(f"峰值显存: {final_memory['max_allocated']/1024**3:.2f} GB / {final_memory['total']/1024**3:.2f} GB")
            self._log(f"显存利用率: {final_memory['utilization']*100:.1f}%")
        self._log("=" * 60)

        return self.best_batch_size


def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    # 初始化分布式进程组
    try:
        # PyTorch 2.3+ 支持 device_id 参数
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f'cuda:{rank}')
        )
    except TypeError:
        # 旧版本 PyTorch 不支持 device_id
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def setup_logger(log_dir, debug=False, rank=0):
    """Setup logger to save logs to file and console"""
    if rank != 0:
        # 非 rank 0 进程不创建 logger
        return None

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
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Error log handler
    error_log_file = os.path.join(log_dir, 'errors.log')
    error_handler = logging.FileHandler(error_log_file, mode='a')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


class EarlyStopping:
    """早停机制，当验证损失连续N个epoch不下降时停止训练"""

    def __init__(self, patience=10, min_delta=0.0, mode='min', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        return False


class WarmupScheduler:
    """学习率预热调度器"""

    def __init__(self, optimizer, warmup_epochs, target_lr, min_lr=1e-8):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 线性插值
            lr = self.min_lr + (self.target_lr - self.min_lr) * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class TrainingProgressTracker:
    """训练进度跟踪器，负责保存和加载训练状态"""

    def __init__(self, model_dir, logger=None):
        self.model_dir = model_dir
        self.logger = logger
        self.progress_file = os.path.join(model_dir, 'progress.json')
        self.history_file = os.path.join(model_dir, 'history.json')

    def save_progress(self, epoch, model, optimizer, scheduler, mintloss, minvloss):
        """保存训练进度"""
        # 处理 DDP 包装的模型
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mintloss': mintloss,
            'minvloss': minvloss
        }

        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # 保存进度信息
        progress = {
            'last_epoch': epoch,
            'mintloss': float(mintloss),
            'minvloss': float(minvloss),
            'last_checkpoint': checkpoint_path
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        if self.logger:
            self.logger.debug(f"Progress saved at epoch {epoch}")

    def load_progress(self, device='cuda'):
        """加载训练进度"""
        if not os.path.exists(self.progress_file):
            return None

        with open(self.progress_file, 'r') as f:
            progress = json.load(f)

        checkpoint_path = progress.get('last_checkpoint')
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            return checkpoint

        return None

    def save_epoch_result(self, epoch, train_loss, val_loss, lr, train_time, val_time):
        """保存每个 epoch 的训练结果"""
        result = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(lr),
            'train_time': float(train_time),
            'val_time': float(val_time),
            'timestamp': datetime.datetime.now().isoformat()
        }

        # 追加到历史文件
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        history.append(result)

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_training_history(self):
        """获取训练历史"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []


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
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _worker(self):
        while self.running or not self.save_queue.empty():
            try:
                item = self.save_queue.get(timeout=1.0)
                filepath = item['filepath']
                state_dict = item['state_dict']
                metadata = item.get('metadata', {})

                try:
                    save_dict = {'state_dict': state_dict, 'metadata': metadata}
                    torch.save(save_dict, filepath)
                    with self.lock:
                        self.saved_count += 1
                    logging.debug(f"异步保存检查点完成: {filepath}")
                except Exception as e:
                    with self.lock:
                        self.failed_count += 1
                    logging.error(f"保存检查点失败 {filepath}: {e}", exc_info=True)

                self.save_queue.task_done()
            except queue.Empty:
                continue

    def save_async(self, state_dict, filepath, metadata=None):
        """异步保存检查点"""
        item = {
            'filepath': filepath,
            'state_dict': state_dict,
            'metadata': metadata or {}
        }
        try:
            # 如果队列满了，移除最旧的项
            if self.save_queue.full():
                try:
                    old_item = self.save_queue.get_nowait()
                    logging.warning(f"保存队列已满，丢弃旧的保存请求: {old_item['filepath']}")
                except queue.Empty:
                    pass

            self.save_queue.put_nowait(item)
        except queue.Full:
            logging.error(f"无法将检查点加入保存队列: {filepath}")

    def get_stats(self):
        with self.lock:
            return {'saved': self.saved_count, 'failed': self.failed_count}


class MyTrainer(crafter.Trainer):
    def __init__(self, net, loader, loss, optimizer, log_interval,
                 use_amp=False, grad_clip=None, rank=0, world_size=1):
        """
        Args:
            net: 神经网络模型
            loader: 数据加载器
            loss: 损失函数
            optimizer: 优化器
            log_interval: 日志打印间隔
            use_amp: 是否使用混合精度训练
            grad_clip: 梯度裁剪的最大范数，None表示不裁剪
            rank: 当前进程的 rank
            world_size: 总进程数
        """
        super().__init__(net, loader, loss, optimizer, log_interval)
        self.use_amp = use_amp and AMP_AVAILABLE
        self.grad_clip = grad_clip
        self.scaler = GradScaler() if self.use_amp else None
        self.rank = rank
        self.world_size = world_size

        if self.use_amp and rank == 0:
            logging.info("混合精度训练已启用")
        if self.grad_clip and rank == 0:
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

    def __call__(self, epoch, train=True, test=False, visual=False):
        """重写 __call__ 方法以支持 DDP 的同步"""
        if train:
            self.net.train()
            key = 'Train'
            # 在训练时设置 sampler 的 epoch，确保每个 epoch 的 shuffle 不同
            if hasattr(self.loader[key].sampler, 'set_epoch'):
                self.loader[key].sampler.set_epoch(epoch)
        elif test:
            self.net.eval()
            key = 'Test'
        else:
            self.net.eval()
            key = 'Valid'

        stats = crafter.defaultdict(list)

        loader = self.loader[key]

        # 只在 rank 0 显示进度条
        if self.rank == 0:
            bar = tqdm(loader, bar_format="{l_bar}{bar:3}{r_bar}", ncols=236, leave=True)
        else:
            bar = loader

        if test:
            rotmats = []
            vertices = []
            from modules.smpl import SMPL, get_smpl_vertices
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            smpl = SMPL().to(device)
            for bi, inputs in enumerate(bar):
                inputs = self.todevice(inputs)
                output = self.forward_net(inputs)
                _, details = self.loss_func(**output)
                pred_rotmats = output['pred_rotmats']
                B, T = pred_rotmats.shape[:2]
                rotmats.append(pred_rotmats.cpu().detach().numpy())

                for k, v in details.items():
                    if type(v) is not dict:
                        if isinstance(v, torch.Tensor):
                            stats[k].append(v.detach().cpu().numpy())
                        else:
                            stats[k].append(v)
                if visual:
                    pred_vertices = get_smpl_vertices(output['trans'].reshape(B * T, 3),
                                                      pred_rotmats.reshape(B * T, 24, 3, 3),
                                                      output['betas'].reshape(B * T, 10), smpl)
                    for index in range(pred_vertices.shape[0]):
                        vertices.append(pred_vertices[index].squeeze().cpu().detach().numpy())

            rotmats = np.concatenate(rotmats, axis=0)
            final_loss = {k: crafter.mean(v) for k, v in stats.items()}

            if visual:
                return final_loss, rotmats, vertices
            return final_loss, rotmats

        if self.rank == 0:
            bar.set_description(f'{key} {epoch:02d}')

        for iter, inputs in enumerate(bar):
            inputs = self.todevice(inputs)

            if train:
                details = self.forward_backward(inputs)
            else:
                with torch.no_grad():
                    details = self.forward_val(inputs)

            for k, v in details.items():
                if type(v) is not dict:
                    if isinstance(v, torch.Tensor):
                        stats[k].append(v.detach().cpu().numpy())
                    else:
                        stats[k].append(v)

            if train and self.rank == 0:
                N = len(stats['loss']) // 10 + 1
                loss = stats['loss']
                bar.set_postfix(loss=f'{crafter.mean(loss[:N]):06.06f} -> '
                                      f'{crafter.mean(loss[-N:]):06.06f} '
                                      f'({crafter.mean(loss):06.06f})')

            if not train and (iter + 1) == len(loader) and self.rank == 0:
                bar.set_postfix(loss=f'{crafter.mean(stats["loss"]):06.06f}')

        final_loss = {k: crafter.mean(v) for k, v in stats.items()}

        # 同步所有进程的损失值
        if self.world_size > 1:
            sync_loss = torch.tensor(final_loss['loss'], device='cuda')
            dist.all_reduce(sync_loss, op=dist.ReduceOp.SUM)
            final_loss['loss'] = (sync_loss / self.world_size).item()

        return final_loss


def train_worker(rank, world_size, cfg, args):
    """DDP 训练工作进程"""
    # 初始化分布式环境
    setup_distributed(rank, world_size)
    torch.cuda.set_device(rank)

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

    # 设置随机种子
    common.make_reproducible(True, 0)

    # Handle resume or create new run
    if resume:
        if not os.path.exists(resume):
            raise ValueError(f"Resume path does not exist: {resume}")
        run_dir = resume
        run_id = os.path.basename(run_dir)
        if rank == 0:
            print(f"Resuming training from {run_dir}")
    else:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        run_dir = os.path.join(output_dir, run_id)
        if rank == 0:
            print(f"Starting new training run: {run_id}")

    log_dir = os.path.join(run_dir, 'logs')
    logger = setup_logger(log_dir, debug, rank)

    if rank == 0:
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
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)

    # 同步确保目录创建完成
    if world_size > 1:
        dist.barrier()

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

    # 加载数据集
    if rank == 0:
        logger.info("加载数据集...")

    if preload:
        train_dataset = CachedLidarCapDataset(cfg=cfg.TrainDataset, dataset=dataset_name, train=True, preload=True)
    else:
        train_dataset = TemporalDataset(cfg=cfg.TrainDataset, dataset=dataset_name, train=True)

    # 自动调整 batch size
    auto_batch_size = cfg.TRAIN.get('auto_batch_size', False)
    if auto_batch_size:
        if rank == 0:
            logger.info("开始自动调整 batch size...")

        # 获取一个样本用于测试
        sample_input = train_dataset[0]
        # 将样本移动到 GPU
        sample_input_gpu = {}
        for key, value in sample_input.items():
            if isinstance(value, torch.Tensor):
                sample_input_gpu[key] = value.cuda()
            else:
                sample_input_gpu[key] = value

        # 创建临时模型用于测试 batch size
        from modules.regressor import Regressor as TempRegressor
        from modules.loss import Loss as TempLoss

        temp_model = TempRegressor(cfg=cfg).cuda()
        temp_loss_fn = TempLoss(cfg=cfg).cuda()

        # 创建优化器工厂函数
        def optimizer_factory(params, **kwargs):
            return torch.optim.Adam(
                params, lr=kwargs.get('lr', cfg.TRAIN.learning_rate),
                weight_decay=cfg.TRAIN.weight_decay
            )

        finder = AutoBatchSizeFinder(
            model=temp_model,
            loss_fn=temp_loss_fn,
            sample_input=sample_input_gpu,
            optimizer_class=optimizer_factory,
            use_amp=cfg.TRAIN.use_amp,
            target_memory_usage=cfg.TRAIN.get('auto_batch_size_target', 0.90),
            min_batch_size=cfg.TRAIN.get('auto_batch_size_min', 1),
            max_batch_size=cfg.TRAIN.get('auto_batch_size_max', 64),
            warmup_iterations=cfg.TRAIN.get('auto_batch_size_warmup', 3),
            logger=logger if rank == 0 else None
        )

        # 在分布式环境中，只在 rank 0 上进行搜索
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            optimal_batch_size = finder.find_optimal_batch_size()
        else:
            optimal_batch_size = None

        # 广播最优 batch size 到所有进程
        if world_size > 1:
            optimal_batch_size_tensor = torch.tensor([optimal_batch_size if rank == 0 else 0], device='cuda')
            dist.broadcast(optimal_batch_size_tensor, src=0)
            optimal_batch_size = int(optimal_batch_size_tensor.item())

        batch_size = int(optimal_batch_size) if optimal_batch_size else cfg.TRAIN.batch_size
        if rank == 0:
            logger.info(f"自动调整后的 batch size: {batch_size}")

        # 同步所有进程
        if world_size > 1:
            dist.barrier()

        # 清理临时模型和样本数据
        del temp_model, temp_loss_fn, sample_input_gpu, finder
        torch.cuda.empty_cache()

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=collate)

    if preload:
        valid_dataset = CachedLidarCapDataset(cfg=cfg.TestDataset, dataset=dataset_name, train=False, preload=True)
    else:
        valid_dataset = TemporalDataset(cfg=cfg.TestDataset, dataset=dataset_name, train=False)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=eval_batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=collate)

    loader = {'Train': train_loader, 'Valid': valid_loader}

    if rank == 0:
        logger.info("[INFO] Initializing AsyncCheckpointSaver for non-blocking model saves")
    async_saver = AsyncCheckpointSaver(max_queue_size=10)
    async_saver.start()

    # 创建模型并移动到当前 GPU
    net = Regressor(cfg=cfg).cuda()
    loss = Loss(cfg=cfg).cuda()

    # 使用 DDP 包装模型
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if rank == 0:
        logger.info(f"Using DistributedDataParallel on {world_size} GPUs")

    # Define optimizer
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=1e-4, eps=1e-8, amsgrad=True)

    # 学习率调度器
    sc = {
        'factor': lr_factor,
        'patience': lr_patience,
        'threshold': lr_threshold,
        'min_lr': lr_min
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=sc['factor'], patience=sc['patience'],
        threshold_mode='rel', threshold=sc['threshold'], min_lr=sc['min_lr'])

    # 预热调度器
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, lr, warmup_min_lr) if warmup_epochs > 0 else None

    # 早停机制
    early_stopping = None
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='min',
            restore_best_weights=True
        )
        if rank == 0:
            logger.info(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

    if rank == 0:
        logger.info("=== 训练配置 ===")
        logger.info(f"学习率: {lr}")
        logger.info(f"批次大小: {batch_size} (每GPU: {batch_size})")
        logger.info(f"有效批次大小: {batch_size * world_size}")
        logger.info(f"权重衰减: 1e-4")
        logger.info(f"学习率调度器: ReduceLROnPlateau(factor={sc['factor']}, patience={sc['patience']}, min_lr={sc['min_lr']})")
        if warmup_scheduler:
            logger.info(f"预热启用: {warmup_epochs} epochs, 从 {warmup_min_lr} 到 {lr}")
        if use_amp:
            logger.info("混合精度训练: 已启用")
        if grad_clip:
            logger.info(f"梯度裁剪: max_norm={grad_clip}")

    training_manager = TrainingProgressTracker(model_dir, logger)

    start_epoch = 1
    mintloss = float('inf')
    minvloss = float('inf')

    train = MyTrainer(net, loader, loss, optimizer, log_interval,
                     use_amp=use_amp, grad_clip=grad_clip, rank=rank, world_size=world_size)

    if resume:
        checkpoint = training_manager.load_progress(f'cuda:{rank}')
        if checkpoint:
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            net.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            for i in range(len(scheduler.min_lrs)):
                scheduler.min_lrs[i] = float(scheduler.min_lrs[i])

            start_epoch = checkpoint['epoch'] + 1
            mintloss = checkpoint['mintloss']
            minvloss = checkpoint['minvloss']

            if rank == 0:
                logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")

                history = training_manager.get_training_history()
                if history:
                    logger.info("Training history:")
                    for result in history[-5:]:
                        logger.info(f"  Epoch {result['epoch']}: Train={result['train_loss']:.6f}, Val={result['val_loss']:.6f}")

    elif ckpt_path is not None:
        if rank == 0:
            logger.info(f"Loading checkpoint from {ckpt_path}")
        save_model = torch.load(ckpt_path, map_location=f'cuda:{rank}')['state_dict']
        model_dict = net.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        net.load_state_dict(model_state)

    if ckpt_path is not None:
        if rank == 0:
            logger.info("=== EVALUATION MODE ===")
            logger.info(f"Model loaded from {ckpt_path}")
            logger.info("Use evaluation script instead")
    else:
        if rank == 0:
            logger.info("=== TRAINING MODE ===")
            logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")
            logger.info(f"Model directory: {model_dir}")
            logger.info(f"Current best - Train Loss: {mintloss:.6f}, Val Loss: {minvloss:.6f}")
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"Batch size: {batch_size * world_size} (per GPU: {batch_size}), Learning rate: {lr}")
            logger.info(f"Using {world_size} GPUs with DDP")

        epoch = start_epoch

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                epoch_start_time = time.time()
                if rank == 0:
                    logger.info(f"Starting epoch {epoch}/{num_epochs}")
                    logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")

                # 应用预热学习率
                if warmup_scheduler is not None and epoch <= warmup_scheduler.warmup_epochs:
                    warmup_scheduler.step()
                    if rank == 0:
                        logger.info(f"Warmup learning rate: {optimizer.param_groups[0]['lr']:.8f}")

                # Training
                train_start_time = time.time()
                train_loss_dict = train(epoch)
                train_time = time.time() - train_start_time

                if rank == 0:
                    logger.info(f"Epoch {epoch} - Training completed, loss: {train_loss_dict['loss']:.6f}, time: {train_time:.2f}s")

                # Validation
                val_start_time = time.time()
                val_loss_dict = train(epoch, train=False)
                val_time = time.time() - val_start_time

                if rank == 0:
                    logger.info(f"Epoch {epoch} - Validation completed, loss: {val_loss_dict['loss']:.6f}, time: {val_time:.2f}s")

                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']

                if rank == 0:
                    logger.info(f"Epoch {epoch}/{num_epochs} - "
                               f"Train Loss: {train_loss_dict['loss']:.6f}, "
                               f"Val Loss: {val_loss_dict['loss']:.6f}, "
                               f"LR: {current_lr:.8f}, "
                               f"Epoch Time: {epoch_time:.2f}s")

                    training_manager.save_epoch_result(epoch, train_loss_dict['loss'], val_loss_dict['loss'],
                                                      lr=current_lr, train_time=train_time, val_time=val_time)

                # Save model
                if rank == 0:
                    model_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
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

                # 学习率调度
                if warmup_scheduler is None or epoch > warmup_scheduler.warmup_epochs:
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(val_loss_dict['loss'])
                    new_lr = optimizer.param_groups[0]['lr']

                    if new_lr != old_lr and rank == 0:
                        logger.info(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

                # 早停检查
                if early_stopping is not None:
                    actual_model = net.module if hasattr(net, 'module') else net
                    if early_stopping(val_loss_dict['loss'], actual_model):
                        if rank == 0:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            logger.info(f"No improvement in validation loss for {early_stopping.patience} epochs")
                            logger.info(f"Best validation loss: {early_stopping.best_score:.6f}")
                        break

                # Save progress
                if epoch % save_every == 0 and rank == 0:
                    training_manager.save_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)

        except KeyboardInterrupt:
            if rank == 0:
                logger.info("Training interrupted by user")
                training_manager.save_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)

        finally:
            async_saver.stop()
            if rank == 0:
                stats = async_saver.get_stats()
                logger.info(f"Checkpoint saver stats: {stats['saved']} saved, {stats['failed']} failed")
                logger.info("Training finished!")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, default='config',
                        help='配置文件目录 (default: config)')
    args = parser.parse_args()

    cfg = get_cfg()
    gpu_ids = cfg.RUNTIME.gpu_ids

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP requires CUDA.")

    world_size = len(gpu_ids)

    # 设置可见 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)

    if world_size > 1:
        # 多 GPU: 使用 DDP
        print(f"Starting DDP training on {world_size} GPUs: {gpu_ids}")
        mp.spawn(train_worker, args=(world_size, cfg, args), nprocs=world_size, join=True)
    else:
        # 单 GPU: 直接运行
        print(f"Starting single GPU training on GPU {gpu_ids[0]}")
        train_worker(0, 1, cfg, args)


if __name__ == '__main__':
    main()
