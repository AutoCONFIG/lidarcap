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

from config import DATASET_DIR
from datasets.lidarcap_dataset import collate, TemporalDataset
from modules.geometry import rotation_matrix_to_axis_angle
from modules.regressor import Regressor
from modules.loss import Loss
from tools import common, crafter, multiprocess
from tools.util import save_smpl_ply
from tqdm import tqdm
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

class MyTrainer(crafter.Trainer):
    def forward_backward(self, inputs):
        output = self.net(inputs)
        loss, details = self.loss_func(**output)
        loss.backward()
        return details

    def forward_val(self, inputs):
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
        
    def save_epoch_result(self, epoch, train_loss, val_loss):
        """Save epoch results"""
        result = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # bs
    parser.add_argument('--bs', type=int, default=16,
                        help='input batch size for training (default: 24)')
    parser.add_argument('--eval_bs', type=int, default=16,
                        help='input batch size for evaluation')
    # threads
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    # gpu
    parser.add_argument('--gpu', type=int,
                        default=[0], help='-1 for CPU', nargs='+')
    # lr
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    # epochs
    parser.add_argument('--epochs', type=int, default=200,
                        help='Traning epochs (default: 200)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Traning epochs (default: 100)')
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
            valid_dataset, batch_size=args.bs, shuffle=False, num_workers=args.threads, pin_memory=True, collate_fn=collate)
        loader = {'Train': train_loader, 'Valid': valid_loader}

    net = Regressor()
    loss = Loss()

    # Define optimizer
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=1e-4)
    sc = {'factor': 0.9, 'patience': 1, 'threshold': 0.01, 'min_lr': 0.00000003}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=sc['factor'], patience=sc['patience'],
        threshold_mode='rel', threshold=sc['threshold'], min_lr=sc['min_lr'])

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
            train = MyTrainer(net, loader, loss, optimizer, args.log_interval)
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
            train = MyTrainer(net, loader, loss, optimizer, args.log_interval)
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
        train = MyTrainer(net, loader, loss, optimizer, args.log_interval)
        if iscuda:
            train = train.cuda()
    else:
        # Instance trainer
        train = MyTrainer(net, loader, loss, optimizer, args.log_interval)
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
        logger.info(f"Batch size: {args.bs}, Learning rate: {args.lr}")
        logger.info(f"Using GPU: {args.gpu if iscuda else 'CPU'}")
        
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                logger.info(f"Starting epoch {epoch}/{args.epochs}")
                
                # Training
                train_loss_dict = train(epoch)
                logger.info(f"Epoch {epoch} - Training completed, loss: {train_loss_dict['loss']:.6f}")
                
                # Validation
                val_loss_dict = train(epoch, train=False)
                logger.info(f"Epoch {epoch} - Validation completed, loss: {val_loss_dict['loss']:.6f}")
                
                # Log training progress
                logger.info(f"Epoch {epoch}/{args.epochs} - "
                           f"Train Loss: {train_loss_dict['loss']:.6f}, "
                           f"Val Loss: {val_loss_dict['loss']:.6f}")

                # Save epoch results
                training_manager.save_epoch_result(epoch, train_loss_dict['loss'], val_loss_dict['loss'])

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

                # scheduler
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(train_loss_dict['loss'])
                new_lr = optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    logger.info(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")
                
                # Save progress after each epoch
                training_manager.save_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)

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
            training_manager.save_progress(epoch, net, optimizer, scheduler, mintloss, minvloss)
            raise
            
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(f"Training completed. Best train loss: {mintloss:.6f}, Best valid loss: {minvloss:.6f}")
        logger.info(f"All checkpoints and results saved in: {model_dir}")
        logger.info(f"Log files saved in: {log_dir}")
        
        # Show final training history
        history = training_manager.get_training_history()
        if history and len(history) > 0:
            logger.info("Final training history (last 10 epochs):")
            for result in history[-10:]:
                logger.info(f"  Epoch {result['epoch']}: Train={result['train_loss']:.6f}, Val={result['val_loss']:.6f}")
        
        # Clean up old checkpoints (keep only last 5 and best models)
        logger.info("Cleaning up old checkpoint files...")
        cleanup_old_checkpoints(model_dir, keep_last=5, logger=logger)
        logger.info("Training pipeline completed successfully!")
