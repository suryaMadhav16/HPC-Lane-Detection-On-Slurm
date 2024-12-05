# Examples and Configuration Guide for HPC TuSimple

This guide provides detailed examples and configuration templates for different scenarios in the HPC TuSimple project.

## Table of Contents
- [Example Scripts](#example-scripts)
  - [Basic Training](#basic-training)
  - [Distributed Training](#distributed-training)
  - [Model Parallel Training](#model-parallel-training)
  - [Profiling Examples](#profiling-examples)
  - [Visualization Examples](#visualization-examples)
- [Configuration Templates](#configuration-templates)
  - [Hardware Configurations](#hardware-configurations)
  - [Training Configurations](#training-configurations)
  - [Advanced Configurations](#advanced-configurations)

## Example Scripts

### Basic Training

1. **Single GPU Training**
```python
# train_single_gpu.py
from src.models.lane_detection import LaneDetectionModel
from src.training.trainer import Trainer
from src.data.dataset import LaneDataset
from torch.utils.data import DataLoader

def main():
    # Load configuration
    config = {
        'model': {
            'num_classes': 2,
            'backbone': 'resnet50',
            'pretrained': True
        },
        'training': {
            'batch_size': 16,
            'epochs': 10,
            'learning_rate': 0.001
        },
        'system': {
            'device': 'cuda',
            'num_workers': 4
        }
    }
    
    # Initialize model
    model = LaneDetectionModel(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone']
    )
    
    # Create datasets and dataloaders
    train_dataset = LaneDataset(dataset_path='./dataset', train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers']
    )
    
    # Initialize trainer and train
    trainer = Trainer(config)
    trainer.train(model, train_loader)

if __name__ == '__main__':
    main()
```

2. **Mixed Precision Training**
```python
# train_mixed_precision.py
import torch
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, optimizer, criterion):
    scaler = GradScaler()
    model.train()
    
    for batch in train_loader:
        images, targets = batch['img'], batch['segLabel']
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Scaled backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
```

### Distributed Training

1. **DDP Training Script**
```python
# train_ddp.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def train_ddp(rank, world_size, config):
    setup_ddp(rank, world_size)
    
    # Create model and move to GPU
    model = LaneDetectionModel(config['model'])
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create data loader with DistributedSampler
    train_dataset = LaneDataset(config['dataset']['path'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler
    )
    
    trainer = Trainer(config)
    trainer.train(model, train_loader)
    
    dist.destroy_process_group()
```

### Model Parallel Training

```python
# train_model_parallel.py
class ModelParallelLaneDetection(LaneDetectionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Split model across GPUs
        self.encoder = self.encoder.to('cuda:0')
        self.coord_att = self.coord_att.to('cuda:1')
        self.decoder = self.decoder.to('cuda:2')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.encoder(x)
        
        x = x.to('cuda:1')
        x = self.coord_att(x)
        
        x = x.to('cuda:2')
        x = self.decoder(x)
        return x

def train_model_parallel():
    model = ModelParallelLaneDetection(num_classes=2)
    # Training loop implementation
```

### Profiling Examples

1. **System Profiling**
```python
# profile_training.py
from src.utils.profiling import SystemProfiler

def profile_training_run():
    profiler = SystemProfiler(log_dir='./logs')
    profiler.start_profiling()
    
    # Training code here
    
    profiler.stop_profiling()
    report = profiler.generate_report()
    print(f"Training Profile:\n{report}")

@profiler.profile_operation("forward_pass")
def model_forward_pass(model, input_data):
    return model(input_data)
```

2. **Memory Profiling**
```python
# profile_memory.py
import torch
from src.utils.profiling import SystemProfiler

def profile_memory_usage():
    profiler = SystemProfiler(log_dir='./logs')
    
    # Track initial memory
    initial_memory = profiler.get_system_profile()
    
    # Create and train model
    model = LaneDetectionModel()
    
    # Track peak memory
    peak_memory = profiler.get_system_profile()
    
    return {
        'initial_memory': initial_memory,
        'peak_memory': peak_memory
    }
```

### Visualization Examples

1. **Training Metrics Visualization**
```python
# visualize_training.py
from src.utils.visualization import Visualizer

def visualize_training_results(history):
    visualizer = Visualizer(save_dir='./visualizations')
    
    # Plot training history
    visualizer.plot_training_history(history)
    
    # Plot system metrics
    visualizer.plot_system_metrics(history['system_metrics'])
```

2. **Prediction Visualization**
```python
# visualize_predictions.py
def visualize_model_predictions(model, dataloader, visualizer):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['img']
            targets = batch['segLabel']
            predictions = model(images)
            
            visualizer.visualize_prediction(
                images[0],
                predictions[0],
                targets[0],
                save_name='prediction_example'
            )
            break
```

## Configuration Templates

### Hardware Configurations

1. **High-End Multi-GPU Setup**
```yaml
# configs/high_end_gpu.yaml
system:
  device: 'cuda'
  cuda_devices: [0, 1, 2, 3]
  num_workers: 8
  pin_memory: true
  
optimization:
  cudnn_benchmark: true
  mixed_precision: true
  memory_efficient: false
  
training:
  batch_size: 64
  gradient_accumulation: 1
```

2. **Limited Resource Setup**
```yaml
# configs/limited_resource.yaml
system:
  device: 'cuda'
  cuda_devices: [0]
  num_workers: 2
  pin_memory: false
  
optimization:
  cudnn_benchmark: false
  mixed_precision: true
  memory_efficient: true
  
training:
  batch_size: 8
  gradient_accumulation: 4
```

3. **CPU-Only Setup**
```yaml
# configs/cpu_only.yaml
system:
  device: 'cpu'
  num_threads: 4
  pin_memory: false
  
optimization:
  memory_efficient: true
  
training:
  batch_size: 4
  gradient_accumulation: 2
```

### Training Configurations

1. **Fast Training Configuration**
```yaml
# configs/fast_training.yaml
training:
  epochs: 10
  learning_rate: 0.001
  optimizer: 'adamw'
  scheduler:
    type: 'onecycle'
    max_lr: 0.01
    pct_start: 0.3

model:
  backbone: 'resnet18'
  pretrained: true
  num_classes: 2
```

2. **High Accuracy Configuration**
```yaml
# configs/high_accuracy.yaml
training:
  epochs: 50
  learning_rate: 0.0001
  optimizer: 'adamw'
  scheduler:
    type: 'cosine'
    T_max: 50
    eta_min: 0.00001

model:
  backbone: 'resnet50'
  pretrained: true
  num_classes: 2
  dropout: 0.2
```

### Advanced Configurations

1. **DDP Training Configuration**
```yaml
# configs/ddp_training.yaml
distributed:
  backend: 'nccl'
  world_size: 4
  init_method: 'env://'
  sync_bn: true
  
optimization:
  gradient_clip: 1.0
  accumulation_steps: 1
  mixed_precision: true
  
training:
  batch_size_per_gpu: 16
  total_batch_size: 64
```

2. **Model Parallel Configuration**
```yaml
# configs/model_parallel.yaml
model_parallel:
  enabled: true
  num_gpus: 3
  partition_size: [0.4, 0.2, 0.4]  # Distribution of model across GPUs
  pipeline_chunks: 4
  
optimization:
  checkpoint_activation: true
  memory_efficient: true
  
training:
  micro_batch_size: 4
  global_batch_size: 32
```

3. **Hybrid Parallelism Configuration**
```yaml
# configs/hybrid_parallel.yaml
hybrid_parallel:
  enabled: true
  data_parallel_size: 2
  model_parallel_size: 2
  pipeline_parallel_size: 2
  
optimization:
  mixed_precision: true
  gradient_clip: 1.0
  checkpoint_activation: true
  
training:
  micro_batch_size: 4
  global_batch_size: 32
  gradient_accumulation: 4
```

Each of these configurations and examples can be customized based on your specific needs and hardware setup. Remember to adjust parameters like batch sizes, number of workers, and optimization settings based on your available resources and requirements.