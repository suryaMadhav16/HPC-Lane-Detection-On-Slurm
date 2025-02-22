import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from datetime import datetime

from src.data.dataset import LaneDataset
from src.models.lane_detection import LaneDetectionModel
from src.training.trainer import Trainer
from src.utils.profiling import SystemProfiler
from src.utils.visualization import Visualizer

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_system(config: dict) -> torch.device:
    """Setup system based on configuration"""
    if config['system']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = config['optimization']['cudnn_benchmark']
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if 'num_threads' in config['system']:
            torch.set_num_threads(config['system']['num_threads'])
        logging.info(f"Using CPU with {torch.get_num_threads()} threads")
    
    return device

def create_dataloaders(config: dict) -> tuple:
    """Create train and validation dataloaders"""
    train_dataset = LaneDataset(
        dataset_path=config['dataset']['path'],
        train=True,
        size=tuple(config['dataset']['image_size'])
    )    
    
    val_dataset = LaneDataset(
        dataset_path=config['dataset']['path'],
        train=False,
        size=tuple(config['dataset']['image_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['system'].get('pin_memory', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['system'].get('pin_memory', False)
    )
    
    return train_loader, val_loader

def main(config_path: str):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    logging.info(f"Starting training with config: {config_path}")
    
    # Setup system
    device = setup_system(config)
    
    # Initialize profiler
    profiler = SystemProfiler(config['logging']['log_dir'])
    profiler.start_profiling()
    
    # Initialize visualizer
    visualizer = Visualizer(config['logging']['log_dir'])
    
    try:
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(config)
        logging.info("Created data loaders")
        
        # Initialize model
        model = LaneDetectionModel(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone'],
            pretrained=config['model']['pretrained']
        )
        model = model.to(device)        
        logging.info("Model initialized and moved to device")
        
        # Initialize trainer
        trainer = Trainer(config_path)
        
        # Train model
        training_history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )

        formatted_history = trainer.get_formatted_history(training_history)
        
        # Plot training history
        visualizer.plot_training_history(formatted_history)
        
        # Generate and save system metrics report
        system_report = profiler.generate_report()
        logging.info(f"System utilization report: {system_report}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Stop profiling
        profiler.stop_profiling()
        logging.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpu_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)