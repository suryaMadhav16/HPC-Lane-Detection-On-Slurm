# Lane Detection and Prediction Through Parallel Computing

## Project Overview

This project implements an optimized lane detection system for autonomous driving applications using the TuSimple dataset. The implementation focuses on performance optimization through various parallelization techniques and provides comprehensive benchmarking tools.

The goal is to optimize lane detection using parallel computing techniques, experiment with different data loading strategies, and implement multi-CPU and multi-GPU training to maximize computational efficiency.

## Dataset: TuSimple Lane Detection

- **6,408 highway images** at **1280×720 resolution**
- **Sequential video frames** where only the last frame per clip is labeled
- **23GB total dataset size**
- Label format in JSON with lane coordinates and vertical positions
- Key challenges include weather variability, occlusions, traffic variations, and complex road structures

## Model Architecture

### Overview
- **ResNet-18 & ResNet-50** backbones for feature extraction
- Feature extraction refined using **Coordinate Attention Mechanism**
- **U-Net-like upsampling network** for **segmentation mask prediction**
- **Loss functions**: Dice Loss & IoU Loss

### Coordinate Attention Mechanism
The model implements a spatial attention mechanism called Coordinate Attention that enhances the network's ability to focus on relevant spatial features:

```python
class CoordAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 32):
        super(CoordAttention, self).__init__()
        
        # Pooling layers
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Channel reduction
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
```

### Lane Detection Model Architecture
The main model uses a ResNet backbone (either ResNet-18 or ResNet-50) with a custom decoder for lane segmentation:

1. **Encoder**: ResNet backbone for feature extraction
2. **Attention**: Coordinate Attention to refine features
3. **Decoder**: Series of upsampling blocks to generate segmentation masks
4. **Output**: Final convolutional layer for class prediction

The model is flexible and can work with different ResNet variants:

```python
if backbone == 'resnet50':
    resnet = models.resnet50(pretrained=pretrained)
    self.encoder = nn.Sequential(*list(resnet.children())[:-2])
    encoder_channels = 2048
elif backbone == 'resnet18':
    resnet = models.resnet18(pretrained=pretrained)
    self.encoder = nn.Sequential(*list(resnet.children())[:-2])
    encoder_channels = 512
```

The decoder architecture adapts based on the chosen backbone, with ResNet-50 having a deeper decoder structure compared to ResNet-18.

## Data Preprocessing

The project implements a custom dataset class for the TuSimple lane detection dataset with the following preprocessing steps:

### 1. Image Preprocessing
- **Resizing**: Images are resized from 1280×720 to 800×360 pixels
- **Color Conversion**: BGR to RGB conversion for PyTorch compatibility
- **Normalization**: Pixel values converted to floating-point format

### 2. Label Preprocessing
- **Binary Segmentation Masks**: Convert lane markings to binary masks
- **Tensor Conversion**: Conversion to torch tensors for model input
- **Size Matching**: Labels resized to match the input image dimensions

```python
# Load and process image
image = cv2.imread(img_path)        
raw_image = image.copy()
image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_LINEAR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create binary segmentation
binary_seg = seg_image.copy()
binary_seg[binary_seg > 0] = 1

# Convert to tensors
image_tensor = torch.from_numpy(image).float().permute((2, 0, 1))
seg_tensor = torch.from_numpy(binary_seg).to(torch.int64)
```

## Loss Functions

The project implements multiple loss functions specifically designed for lane segmentation:

### 1. Dice Loss
The Dice Loss measures the overlap between predicted and ground truth segmentations:

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert target to one-hot encoding if needed
        if len(target.shape) == 3:
            target = F.one_hot(target, num_classes=pred.size(1))  # [B, H, W, C]
            target = target.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Calculate Dice score
        pred = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
        target = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]
        
        intersection = (pred * target).sum(dim=2)  # [B, C]
        union = pred.sum(dim=2) + target.sum(dim=2)  # [B, C]
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        return 1 - dice.mean()
```

### 2. IoU Loss
The IoU (Intersection over Union) Loss measures the overlap ratio between predictions and ground truth:

```python
class IoULoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate IoU for each class
        intersection = (pred * target).sum(dim=2)
        union = (pred + target - pred * target).sum(dim=2)
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou.mean()
```

### 3. Combined Loss
The Combined Loss uses a weighted combination of Cross Entropy Loss and Dice Loss for better segmentation results:

```python
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
```

## Project Structure

```
hpc-tusimple/
├── configs/                 # Configuration files
│   ├── base_config.yaml    # Base configuration template
│   ├── cpu_config.yaml     # CPU-specific settings
│   └── gpu_config.yaml     # GPU-specific settings
├── src/
│   ├── data/               # Data handling
│   │   ├── dataset.py      # TuSimple dataset implementation
│   │   └── transforms.py   # Data transformations
│   ├── models/             # Model architectures
│   │   ├── attention.py    # Coordinate Attention mechanism
│   │   └── lane_detection.py # Main model implementation
│   ├── training/           # Training components
│   │   ├── trainer.py      # Training loop implementation
│   │   ├── losses.py       # Loss functions
│   │   ├── metrics.py      # Performance metrics
│   │   └── distributed.py  # Distributed training utilities
│   └── utils/              # Utility functions
│       ├── profiling.py    # System profiling tools
│       └── visualization.py # Visualization utilities
├── benchmark.py            # Benchmarking script
├── main.py                 # Main training script
├── setup.py                # Package setup
└── README.md               # Documentation
```

## Data Loading Strategies

This project implements and compares multiple data loading strategies for the TuSimple dataset:

### 1. Baseline PyTorch DataLoader
- Standard implementation using PyTorch's DataLoader
- Single-process loading with minimal optimization

### 2. Optimized DataLoader
- Uses PyTorch's DataLoader with optimized parameters
- Multi-worker processing with `num_workers` set to CPU count
- Pinned memory for faster CPU-to-GPU transfers
- Prefetching to overlap data loading and model computation

### 3. Dask-based Parallel Processing
- Uses Dask for distributed computing
- Implements custom DaskLaneDataset with parallel processing
- Works with a local Dask cluster for distributed operations

### 4. Memory-Mapped (Memmap) Loading
- Implements memory-mapped file access for faster data loading
- Reduces disk I/O overhead by mapping files directly to memory

### Performance Comparison

| Method | Loading Time | Memory Usage | CPU Usage |
|--------|-------------|-------------|-----------|
| **Baseline DataLoader** | **1.10s** | **1.02GB** | **15.9%** |
| **Optimized DataLoader** | **1.11s** | **18GB** | **17.49%** |
| **Dask Parallel** | **1.65s** | **18GB** | **14.76%** |
| **Memmap Loader** | **0.67s** | **18GB** | **15.53%** |

The Memmap loader provides the fastest loading times but requires significantly more memory, while the Dask implementation introduces overhead that makes it less efficient for this specific workload.

## Parallelization Techniques

The project implements multiple parallelization strategies to optimize training performance:

### 1. Data Parallelism (DDP)
Implements PyTorch's DistributedDataParallel for training across multiple GPUs:

- Initializes process groups for communication
- Creates distributed samplers for dataset partitioning
- Replicates model across GPUs and synchronizes gradients
- Scales batch size according to the number of GPUs

```python
def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
```

### 2. Model Parallelism
Implements model partitioning across multiple GPUs:

- Divides model layers across multiple GPUs
- Manages cross-device data transfer during forward passes
- Custom implementation of model partitioning logic

```python
def partition_model(self, model: torch.nn.Module) -> torch.nn.ModuleList:
    # Example partitioning strategy
    layers = []
    if hasattr(model, 'encoder'):
        layers.append(model.encoder)
    if hasattr(model, 'coord_att'):
        layers.append(model.coord_att)
    if hasattr(model, 'up1'):
        layers.extend([model.up1, model.up2, model.up3])
    if hasattr(model, 'final_conv'):
        layers.append(model.final_conv)
    
    # Distribute layers across GPUs
    partitions = torch.nn.ModuleList()
    layers_per_partition = max(1, len(layers) // self.num_gpus)
    
    for i in range(0, len(layers), layers_per_partition):
        partition = torch.nn.Sequential(*layers[i:i + layers_per_partition])
        partition.to(f'cuda:{i // layers_per_partition}')
        partitions.append(partition)
    
    return partitions
```

### 3. Multi-CPU Training
Optimizes CPU utilization during training:

- Thread management for CPU-based operations
- Pinned memory configuration for faster data transfers
- Process-based parallelism for data loading

### 4. Mixed Precision Training
Implements mixed precision training to reduce memory usage and increase speed:

- Uses FP16 (half precision) for most operations
- Maintains FP32 (full precision) for critical operations
- Automatically handles scaling to prevent underflow

## Benchmarking and Performance Analysis

The project includes a comprehensive benchmarking system that analyzes training performance across different configurations:

### Multi-CPU Performance

| CPUs | Elapsed Time (s) | Speedup |
|------|----------------|---------|
| **2** | **1.2s** | **2.5x** |
| **4** | **0.4s** | **3.5x** |
| **8** | **0.5s** | **2.8x** |
| **16** | **0.7s** | **2.0x** |

Optimal performance is achieved with 4 CPUs, with diminishing returns beyond that point.

### Multi-GPU Performance

| GPUs | Training Time (s) | Speedup | Efficiency (%) |
|------|----------------|---------|--------------|
| **1** | **403.41s** | **1.0x** | **100%** |
| **2** | **253.62s** | **1.59x** | **79.4%** |
| **3** | **174.74s** | **2.31x** | **77.1%** |
| **4** | **168.99s** | **2.39x** | **59.7%** |

3 GPUs provide the best balance between speedup and efficiency, with 4 GPUs showing significantly diminished returns.

### Mixed Precision Training

| Mode | Training Time (s) | Speedup (%) |
|------|----------------|-------------|
| **32-bit** | **96.63s** | **Baseline** |
| **Mixed Precision (16-bit)** | **78.61s** | **18.7% faster** |

Mixed precision training provides a significant speedup without affecting model accuracy.

## System Profiling

The project includes a system profiling module that monitors resource utilization during training:

- Tracks CPU utilization, memory usage, and execution times
- Monitors GPU memory and utilization when available
- Records detailed timing information for operations
- Generates comprehensive performance reports

This information is used to analyze the efficiency of different parallelization strategies and identify bottlenecks in the training pipeline.

## Conclusion and Recommendations

### Data Loading Optimization
- **Memmap**: Provides the fastest loading times (0.67s) but has high memory usage (18GB)
- **Dask**: Shows inefficiency due to overhead for this specific workload
- **Optimized DataLoader**: Offers the best balance between speed and memory usage

### CPU and GPU Utilization
- **CPU Optimization**: 4 CPUs provide optimal parallelization, with diminishing returns beyond this point
- **GPU Scaling**: 3 GPUs offer the best balance between speedup (2.31x) and efficiency (77.1%)
- **Mixed Precision**: Provides significant speedup (18.7%) with no sacrifice in accuracy

### Implementation Complexity vs. Benefits

| Method | Speedup | Memory Usage | Implementation Complexity |
|--------|--------|-------------|---------------------------|
| **Baseline** | **1.0x** | **1GB** | **Simple** |
| **Optimized Loader** | **1.1x** | **18GB** | **Moderate** |
| **Dask** | **0.9x** | **18GB** | **Complex** |
| **Memmap** | **1.65x** | **18GB** | **High Complexity** |

### Best Approach
Based on the comprehensive benchmarking, the recommended approach for optimal performance is:

1. **Data Loading**: Use the Optimized DataLoader with pinned memory and multiple workers
2. **Model Architecture**: Use ResNet-50 with Coordinate Attention for the best accuracy-performance balance
3. **Parallelization**: Implement Distributed Data Parallel (DDP) with 3 GPUs
4. **Optimization**: Enable mixed precision training for additional speedup
5. **CPU Utilization**: Configure with 4 CPU threads for data loading

This hybrid approach offers the best balance between performance, memory usage, and implementation complexity, achieving near-linear speedup while maintaining model accuracy.
