import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import cv2
import torch
import os
from datetime import datetime

class Visualizer:
    """Visualization utility for lane detection results and metrics"""
    
    def __init__(self, save_dir: str):
        """
        Initialize visualizer
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})

    def visualize_prediction(self,
                           image: np.ndarray,
                           prediction: torch.Tensor,
                           target: Optional[torch.Tensor] = None,
                           save_name: Optional[str] = None):
        """Visualize lane detection prediction"""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Prediction
        plt.subplot(132)
        pred_mask = torch.argmax(prediction, dim=0).cpu().numpy()
        plt.imshow(pred_mask, cmap='viridis')
        plt.title('Prediction')
        plt.axis('off')
        
        # Ground truth (if provided)
        if target is not None:
            plt.subplot(133)
            plt.imshow(target.cpu().numpy(), cmap='viridis')
            plt.title('Ground Truth')
            plt.axis('off')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training metrics history"""
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(history['train_accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Learning rate plot
        if 'learning_rate' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['learning_rate'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)

        # IoU plot
        if 'train_iou' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['train_iou'], label='Training IoU')
            if 'val_iou' in history:
                plt.plot(history['val_iou'], label='Validation IoU')
            plt.title('IoU History')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'training_history_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_system_metrics(self, metrics_df: pd.DataFrame):
        """Plot system resource utilization metrics"""
        plt.figure(figsize=(15, 10))
        
        # CPU Usage
        plt.subplot(3, 1, 1)
        plt.plot(metrics_df['timestamp'], metrics_df['cpu_percent'])
        plt.title('CPU Utilization')
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.grid(True)

        # Memory Usage
        plt.subplot(3, 1, 2)
        plt.plot(metrics_df['timestamp'], metrics_df['memory_percent'])
        plt.title('Memory Utilization')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.grid(True)

        # GPU Usage (if available)
        if 'gpu_utilization' in metrics_df.columns:
            plt.subplot(3, 1, 3)
            plt.plot(metrics_df['timestamp'], metrics_df['gpu_utilization'])
            plt.title('GPU Utilization')
            plt.xlabel('Time')
            plt.ylabel('GPU Usage (%)')
            plt.grid(True)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'system_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_comparison_report(self, configurations: List[Dict[str, Any]]):
        """Create comparison report for different training configurations"""
        plt.figure(figsize=(15, 10))
        
        # Training Time Comparison
        plt.subplot(2, 2, 1)
        names = [conf['name'] for conf in configurations]
        times = [conf['total_time'] for conf in configurations]
        plt.bar(names, times)
        plt.title('Training Time Comparison')
        plt.xlabel('Configuration')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)

        # Resource Usage Comparison
        plt.subplot(2, 2, 2)
        cpu_usage = [conf['avg_cpu_usage'] for conf in configurations]
        memory_usage = [conf['avg_memory_usage'] for conf in configurations]
        x = np.arange(len(names))
        width = 0.35
        plt.bar(x - width/2, cpu_usage, width, label='CPU')
        plt.bar(x + width/2, memory_usage, width, label='Memory')
        plt.title('Resource Usage Comparison')
        plt.xlabel('Configuration')
        plt.ylabel('Usage (%)')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        # Final Accuracy Comparison
        plt.subplot(2, 2, 3)
        accuracies = [conf['final_accuracy'] for conf in configurations]
        plt.bar(names, accuracies)
        plt.title('Final Accuracy Comparison')
        plt.xlabel('Configuration')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)

        # Loss Convergence Comparison
        plt.subplot(2, 2, 4)
        for conf in configurations:
            plt.plot(conf['loss_history'], label=conf['name'])
        plt.title('Loss Convergence Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'configuration_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()