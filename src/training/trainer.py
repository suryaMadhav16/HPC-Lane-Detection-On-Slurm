import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Any, Optional
import yaml
import logging
from tqdm import tqdm
import time

from ..models.lane_detection import LaneDetectionModel
from .losses import CombinedLoss
from .metrics import MetricsTracker, TrainingMetrics

class Trainer:
    """
    Trainer class for Lane Detection model
    """
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(self.config['logging']['log_dir'])
        
        # Create directories
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)

        # Initialize logger
        logging.basicConfig(
            filename=os.path.join(self.config['logging']['log_dir'], 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _setup_device(self) -> torch.device:
        """Setup compute device based on configuration and availability"""
        if self.config['system']['device'] == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            # Set GPU-specific configurations
            torch.backends.cudnn.benchmark = self.config['optimization']['cudnn_benchmark']
            if self.config['optimization'].get('mixed_precision', False):
                torch.backends.cudnn.enabled = True
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(self.config['system'].get('num_threads', 4))
            logging.info(f"Using CPU with {self.config['system'].get('num_threads', 4)} threads")
        
        return device

    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer based on configuration"""    
        if self.config['training']['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        return optimizer

    def train(self, 
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Train the model"""
        model = model.to(self.device)
        criterion = CombinedLoss().to(self.device)
        optimizer = self._setup_optimizer(model)
        
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': []
        }
        
        logging.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_metrics = self._train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch
            )
            
            # Update history
            training_history['train_metrics'].append(train_metrics)
            training_history['train_losses'].append(train_metrics.batch_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            training_history['epochs'].append(epoch + 1)
            
            # Validate if loader provided
            if val_loader is not None:
                val_metrics = self._validate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    epoch=epoch
                )
                training_history['val_metrics'].append(val_metrics)
                training_history['val_losses'].append(val_metrics.batch_loss)
                
                # Save best model
                if val_metrics.batch_loss < best_val_loss:
                    best_val_loss = val_metrics.batch_loss
                    self._save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=True
                    )
            
            # Print epoch summary
            logging.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics.batch_loss:.4f} - "
                f"Time: {train_metrics.time_taken:.2f}s"
            )
            if val_loader is not None:
                logging.info(f"Val Loss: {val_metrics.batch_loss:.4f}")
        
        return training_history
        
    def _train_epoch(self,
                    model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int) -> TrainingMetrics:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        
        system_metrics = self.metrics.get_system_metrics()
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(progress_bar):
            # Get batch data
            images = batch['img'].to(self.device)
            targets = batch['segLabel'].to(self.device)
            
            # Forward pass
            outputs = model(images)
            outputs = F.interpolate(outputs, 
                                 size=targets.shape[1:],
                                 mode='bilinear',
                                 align_corners=False)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        metrics = TrainingMetrics(
            epoch=epoch,
            batch_loss=avg_loss,
            learning_rate=optimizer.param_groups[0]['lr'],
            time_taken=epoch_time,
            memory_used=system_metrics['memory_used'],
            cpu_utilization=system_metrics['cpu_utilization'],
            gpu_utilization=system_metrics.get('gpu_utilization'),
            gpu_memory_used=system_metrics.get('gpu_memory_used')
        )
        
        return metrics

    def _validate(self,
                 model: nn.Module,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 epoch: int) -> TrainingMetrics:
        """Validate the model"""
        model.eval()
        total_loss = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        
        system_metrics = self.metrics.get_system_metrics()
        validation_start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['img'].to(self.device)
                targets = batch['segLabel'].to(self.device)
                
                outputs = model(images)
                outputs = F.interpolate(outputs, 
                                     size=targets.shape[1:],
                                     mode='bilinear',
                                     align_corners=False)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                })

        avg_loss = total_loss / len(val_loader)
        validation_time = time.time() - validation_start_time
        
        metrics = TrainingMetrics(
            epoch=epoch,
            batch_loss=avg_loss,
            learning_rate=0.0,  # Not relevant for validation
            time_taken=validation_time,
            memory_used=system_metrics['memory_used'],
            cpu_utilization=system_metrics['cpu_utilization'],
            gpu_utilization=system_metrics.get('gpu_utilization'),
            gpu_memory_used=system_metrics.get('gpu_memory_used')
        )
        
        return metrics

    def _save_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer,
                        epoch: int,
                        metrics: TrainingMetrics,
                        is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics.__dict__,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.config['logging']['checkpoint_dir'], 'best_model.pth')
        else:
            path = os.path.join(
                self.config['logging']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved: {path}")

    def get_formatted_history(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Format history for visualization"""
        formatted = {}
        
        # Extract metrics
        formatted['train_losses'] = [m.batch_loss for m in history['train_metrics']]
        formatted['val_losses'] = [m.batch_loss for m in history['val_metrics']] if history['val_metrics'] else []
        formatted['learning_rates'] = [m.learning_rate for m in history['train_metrics']]
        formatted['epochs'] = list(range(1, len(history['train_metrics']) + 1))
        
        # System metrics
        formatted['cpu_utilization'] = [m.cpu_utilization for m in history['train_metrics']]
        formatted['memory_used'] = [m.memory_used for m in history['train_metrics']]
        if history['train_metrics'][0].gpu_utilization is not None:
            formatted['gpu_utilization'] = [m.gpu_utilization for m in history['train_metrics']]
        
        # Training time
        formatted['time_per_epoch'] = [m.time_taken for m in history['train_metrics']]
        
        return formatted
