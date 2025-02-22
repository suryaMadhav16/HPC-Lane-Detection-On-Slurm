"""
Dataset Loading Performance Analysis

This module implements and evaluates different parallelization strategies for the TuSimple dataset loading pipeline.
It includes comprehensive metrics tracking and visualization capabilities.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import cv2
import logging
import dask.array as da
from dask.distributed import Client, LocalCluster
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import contextlib

from src.data.dataset import LaneDataset
from src.utils.profiling import SystemProfiler
from src.utils.dask_utils import DaskProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_loading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def dask_cluster():
    """Context manager for Dask cluster setup and cleanup."""
    try:
        # Start local cluster with diagnostics
        cluster = LocalCluster(
            n_workers=mp.cpu_count(),
            threads_per_worker=1,
            memory_limit='2GB',
            dashboard_address=':8787'  # Remove if not needed
        )
        client = Client(cluster)
        logger.info(f"Dask cluster started with {mp.cpu_count()} workers")
        yield client
    except Exception as e:
        logger.error(f"Error setting up Dask cluster: {e}")
        raise
    finally:
        try:
            client.close()
            cluster.close()
            logger.info("Dask cluster shutdown completed")
        except Exception as e:
            logger.warning(f"Error during Dask cluster cleanup: {e}")

class PerformanceTracker:
    """Tracks and stores performance metrics during data loading."""
    
    def __init__(self):
        self.batch_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_usage = []
        self.start_time = None
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Initialize GPU monitoring if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_gpu = True
            logger.info("GPU monitoring initialized")
        except ImportError:
            logger.warning("pynvml not installed - GPU monitoring disabled")
            self.has_gpu = False
        except Exception as e:
            logger.warning(f"GPU monitoring not available: {e}")
            self.has_gpu = False
    
    def start_batch(self):
        """Start timing for current batch."""
        self.start_time = time.time()
    
    def end_batch(self):
        """Record metrics for the current batch."""
        if self.start_time is None:
            raise ValueError("start_batch must be called first")
        
        duration = time.time() - self.start_time
        self.batch_times.append(duration)
        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
        self.cpu_usage.append(psutil.cpu_percent())
        
        if self.has_gpu:
            try:
                import pynvml
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.gpu_usage.append(info.used / info.total * 100)
            except Exception as e:
                logger.error(f"Error recording GPU metrics: {e}")
                self.gpu_usage.append(0)
        else:
            self.gpu_usage.append(0)
        
        self.start_time = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return comprehensive metrics summary."""
        return {
            'batch_times': self.batch_times,
            'avg_batch_time': np.mean(self.batch_times),
            'std_batch_time': np.std(self.batch_times),
            'memory_usage': self.memory_usage,
            'avg_memory': np.mean(self.memory_usage),
            'peak_memory': max(self.memory_usage),
            'cpu_usage': self.cpu_usage,
            'avg_cpu': np.mean(self.cpu_usage),
            'gpu_usage': self.gpu_usage,
            'avg_gpu': np.mean(self.gpu_usage) if self.gpu_usage else 0
        }

class DaskLaneDataset(LaneDataset):
    """Dataset implementation using Dask for parallel processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client: Optional[Client] = None
        self._setup_dask()
    
    def _setup_dask(self):
        """Set up Dask client for parallel processing."""
        try:
            # Configure Dask cluster settings
            cluster = LocalCluster(
                n_workers=mp.cpu_count(),
                threads_per_worker=1,
                memory_limit='2GB'
            )
            self.client = Client(cluster)
            logger.info(f"Dask cluster initialized with {mp.cpu_count()} workers")
        except Exception as e:
            logger.error(f"Failed to initialize Dask cluster: {e}")
            raise
    
    def __getitem__(self, idx):
        try:
            data = super().__getitem__(idx)
            image = da.from_array(data['img'].numpy(), chunks='auto')
            
            # Process image using Dask
            resized = da.map_overlap(
                cv2.resize,
                image,
                chunks=image.chunks,
                dtype=np.uint8,
                dsize=self._image_size
            )
            
            result = resized.compute()
            data['img'] = torch.from_numpy(result)
            return data
            
        except Exception as e:
            logger.error(f"Error in DaskLaneDataset.__getitem__: {e}")
            raise
    
    def __del__(self):
        """Cleanup Dask resources."""
        if self.client is not None:
            try:
                self.client.close()
                logger.info("Dask client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Dask client: {e}")

class MPLaneDataset(LaneDataset):
    """Dataset implementation using ProcessPoolExecutor."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            image = cv2.resize(image, self._image_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            img_path = self._data[idx][0]
            future = self.pool.submit(self.preprocess_image, img_path)
            image = future.result(timeout=10)  # Add timeout to prevent hanging
            return {'img': torch.from_numpy(image)}
            
        except Exception as e:
            logger.error(f"Error in MPLaneDataset.__getitem__: {e}")
            raise
    
    def __del__(self):
        """Cleanup ProcessPoolExecutor."""
        try:
            self.pool.shutdown(wait=True)
            logger.info("ProcessPoolExecutor shutdown completed")
        except Exception as e:
            logger.warning(f"Error during ProcessPoolExecutor shutdown: {e}")

class ParallelDatasetBenchmark:
    """Benchmarks different dataset loading strategies."""
    
    def __init__(self, dataset_path: str, output_dir: str = './benchmark_results'):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = SystemProfiler(str(self.output_dir))
    
    def run_benchmark(self, 
                     batch_size: int = 32, 
                     num_workers: int = None,
                     num_batches: int = 100) -> Dict[str, Dict]:
        """Run benchmark for all dataset loading strategies."""
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        logger.info(f"Starting benchmark with {num_workers} workers")
        results = {}
        
        strategies = [
            ('baseline', LaneDataset, {'num_workers': 0, 'pin_memory': False}),
            ('optimized', LaneDataset, {'num_workers': num_workers, 'pin_memory': True, 'prefetch_factor': 2}),
            ('multiprocessing', MPLaneDataset, {'num_workers': num_workers}),
            ('dask', DaskLaneDataset, {'num_workers': num_workers})
        ]
        
        for name, dataset_class, kwargs in strategies:
            try:
                results[name] = self._test_strategy(
                    dataset_class,
                    batch_size=batch_size,
                    num_batches=num_batches,
                    **kwargs
                )
                logger.info(f"Successfully tested {name} strategy")
            except Exception as e:
                logger.error(f"Error testing {name} strategy: {e}")
                results[name] = {'error': str(e)}
        
        self._save_results(results)
        return results
    
    def _test_strategy(self, 
                      dataset_class, 
                      batch_size: int,
                      num_workers: int,
                      num_batches: int,
                      **kwargs) -> Dict[str, Any]:
        """Test a specific dataset loading strategy."""
        logger.info(f"Testing {dataset_class.__name__}")
        tracker = PerformanceTracker()
        
        try:
            dataset = dataset_class(
                dataset_path=self.dataset_path,
                train=True,
                size=(800, 360)
            )
            
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                **kwargs
            )
            
            for i, batch in enumerate(tqdm(loader)):
                if i >= num_batches:
                    break
                
                tracker.start_batch()
                if isinstance(batch, dict) and 'img' in batch:
                    _ = batch['img'].float()
                tracker.end_batch()
            
            return tracker.get_metrics()
            
        except Exception as e:
            logger.error(f"Error testing {dataset_class.__name__}: {e}")
            raise

    def _save_results(self, results: Dict[str, Dict]):
        """Save benchmark results."""
        try:
            # Save summary metrics
            summary = pd.DataFrame({
                strategy: {
                    k: v for k, v in metrics.items() 
                    if not isinstance(v, list) and k != 'error'
                }
                for strategy, metrics in results.items()
                if 'error' not in metrics
            })
            summary.to_csv(self.output_dir / 'benchmark_summary.csv')
            
            # Save detailed metrics
            for strategy, metrics in results.items():
                if 'error' not in metrics:
                    detailed = pd.DataFrame({
                        'batch_times': metrics['batch_times'],
                        'memory_usage': metrics['memory_usage'],
                        'cpu_usage': metrics['cpu_usage'],
                        'gpu_usage': metrics['gpu_usage']
                    })
                    detailed.to_csv(self.output_dir / f'{strategy}_detailed.csv')
            
            logger.info("Results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def main():
    """Main execution function."""
    dataset_path = '/path/to/tusimple'  # Update with actual path
    output_dir = './benchmark_results'
    
    benchmark = ParallelDatasetBenchmark(dataset_path, output_dir)
    
    try:
        logger.info("Starting benchmark")
        results = benchmark.run_benchmark(
            batch_size=32,
            num_workers=4,
            num_batches=100
        )
        
        print("\nBenchmark Results:")
        for strategy, metrics in results.items():
            print(f"\n{strategy.upper()}:")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
            else:
                for metric, value in metrics.items():
                    if not isinstance(value, list):
                        print(f"{metric}: {value:.4f}")
        
        logger.info("Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
