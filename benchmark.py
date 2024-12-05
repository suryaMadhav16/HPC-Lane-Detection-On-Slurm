import os
import yaml
import torch
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any
import logging

from src.training.trainer import Trainer
from src.utils.profiling import SystemProfiler
from src.utils.visualization import Visualizer
from src.models.lane_detection import LaneDetectionModel
from src.data.dataset import LaneDataset
from src.training.distributed import DDPWrapper, ModelParallelWrapper

class BenchmarkRunner:
    """Runner for benchmarking different training configurations"""
    
    def __init__(self, base_config_path: str, output_dir: str):
        """
        Initialize benchmark runner
        
        Args:
            base_config_path (str): Path to base configuration file
            output_dir (str): Directory for saving benchmark results
        """
        self.base_config = self._load_config(base_config_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(output_dir, 'benchmark.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.profiler = SystemProfiler(output_dir)
        self.visualizer = Visualizer(output_dir)
        self.results = []

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_config(self, config: dict, name: str):
        """Save configuration to file"""
        path = os.path.join(self.output_dir, f'config_{name}.yaml')
        with open(path, 'w') as f:
            yaml.dump(config, f)

    def run_single_gpu(self) -> Dict[str, Any]:
        """Benchmark single GPU training"""
        config = self.base_config.copy()
        config['system']['device'] = 'cuda'
        config['system']['cuda_devices'] = [0]
        
        logging.info("Starting single GPU benchmark")
        self._save_config(config, 'single_gpu')
        
        trainer = Trainer(config)
        model = LaneDetectionModel(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone']
        ).to('cuda')
        
        self.profiler.start_profiling()
        
        try:
            results = trainer.train(
                model=model,
                train_loader=self._create_dataloader(config),
                val_loader=self._create_dataloader(config, train=False)
            )
            
            system_metrics = self.profiler.generate_report()
            
            return {
                'name': 'single_gpu',
                'config': config,
                'results': results,
                'system_metrics': system_metrics
            }
            
        except Exception as e:
            logging.error(f"Error in single GPU benchmark: {str(e)}")
            raise
        finally:
            self.profiler.stop_profiling()

    def run_multi_gpu_ddp(self, num_gpus: int) -> Dict[str, Any]:
        """Benchmark distributed data parallel training"""
        if num_gpus > torch.cuda.device_count():
            logging.warning(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
            return None
            
        logging.info(f"Starting DDP benchmark with {num_gpus} GPUs")
        
        config = self.base_config.copy()
        config['system']['device'] = 'cuda'
        config['system']['cuda_devices'] = list(range(num_gpus))
        
        self._save_config(config, f'ddp_{num_gpus}gpu')
        
        def create_model():
            return LaneDetectionModel(
                num_classes=config['model']['num_classes'],
                backbone=config['model']['backbone']
            )
        
        ddp_wrapper = DDPWrapper(
            model_fn=create_model,
            world_size=num_gpus,
            config=config
        )
        
        self.profiler.start_profiling()
        
        try:
            results = []
            def train_fn(rank, world_size):
                trainer = Trainer(config)
                model = ddp_wrapper.setup_model(rank)
                
                train_loader = ddp_wrapper.prepare_dataloader(
                    self._create_dataset(config),
                    config['dataset']['batch_size'] // world_size,
                    rank
                )
                
                val_loader = ddp_wrapper.prepare_dataloader(
                    self._create_dataset(config, train=False),
                    config['dataset']['batch_size'] // world_size,
                    rank
                )
                
                result = trainer.train(model, train_loader, val_loader)
                results.append(result)
            
            DDPWrapper.run_distributed(num_gpus, train_fn)
            
            system_metrics = self.profiler.generate_report()
            
            return {
                'name': f'ddp_{num_gpus}gpu',
                'config': config,
                'results': results,
                'system_metrics': system_metrics
            }
            
        except Exception as e:
            logging.error(f"Error in DDP benchmark: {str(e)}")
            raise
        finally:
            self.profiler.stop_profiling()

    def run_model_parallel(self, num_gpus: int) -> Dict[str, Any]:
        """Benchmark model parallel training"""
        if num_gpus > torch.cuda.device_count():
            logging.warning(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
            return None
            
        logging.info(f"Starting model parallel benchmark with {num_gpus} GPUs")
        
        config = self.base_config.copy()
        config['system']['device'] = 'cuda'
        config['system']['cuda_devices'] = list(range(num_gpus))
        
        self._save_config(config, f'mp_{num_gpus}gpu')
        
        mp_wrapper = ModelParallelWrapper(num_gpus)
        model = LaneDetectionModel(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone']
        )
        
        partitions = mp_wrapper.partition_model(model)
        
        self.profiler.start_profiling()
        
        try:
            trainer = Trainer(config)
            results = trainer.train(
                model=partitions,
                train_loader=self._create_dataloader(config),
                val_loader=self._create_dataloader(config, train=False)
            )
            
            system_metrics = self.profiler.generate_report()
            
            return {
                'name': f'mp_{num_gpus}gpu',
                'config': config,
                'results': results,
                'system_metrics': system_metrics
            }
            
        except Exception as e:
            logging.error(f"Error in model parallel benchmark: {str(e)}")
            raise
        finally:
            self.profiler.stop_profiling()

    def run_cpu_benchmark(self) -> Dict[str, Any]:
        """Run CPU-only benchmark"""
        logging.info("Starting CPU benchmark")
        
        config = self.base_config.copy()
        config['system']['device'] = 'cpu'
        self._save_config(config, 'cpu')
        
        trainer = Trainer(config)
        model = LaneDetectionModel(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone']
        )
        
        self.profiler.start_profiling()
        
        try:
            results = trainer.train(
                model=model,
                train_loader=self._create_dataloader(config),
                val_loader=self._create_dataloader(config, train=False)
            )
            
            system_metrics = self.profiler.generate_report()
            
            return {
                'name': 'cpu',
                'config': config,
                'results': results,
                'system_metrics': system_metrics
            }
            
        except Exception as e:
            logging.error(f"Error in CPU benchmark: {str(e)}")
            raise
        finally:
            self.profiler.stop_profiling()

    def _create_dataset(self, config: dict, train: bool = True) -> LaneDataset:
        """Create dataset instance"""
        return LaneDataset(
            dataset_path=config['dataset']['path'],
            train=train,
            size=tuple(config['dataset']['image_size'])
        )

    def _create_dataloader(self, config: dict, train: bool = True):
        """Create data loader"""
        dataset = self._create_dataset(config, train)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=train,
            num_workers=config['dataset']['num_workers'],
            pin_memory=config['system'].get('pin_memory', False)
        )

    def run_all_benchmarks(self, gpu_configs: List[int]):
        """Run all benchmark configurations"""
        try:
            # CPU benchmark
            cpu_results = self.run_cpu_benchmark()
            if cpu_results:
                self.results.append(cpu_results)
            
            # GPU benchmarks if available
            if torch.cuda.is_available():
                # Single GPU
                single_gpu_results = self.run_single_gpu()
                if single_gpu_results:
                    self.results.append(single_gpu_results)
                
                # Multi-GPU configurations
                for num_gpus in gpu_configs:
                    ddp_results = self.run_multi_gpu_ddp(num_gpus)
                    if ddp_results:
                        self.results.append(ddp_results)
                    
                    mp_results = self.run_model_parallel(num_gpus)
                    if mp_results:
                        self.results.append(mp_results)
            
            # Generate comparison visualizations
            self.visualizer.create_comparison_report(self.results)
            
            # Save results
            self._save_results()
            
        except Exception as e:
            logging.error(f"Error in benchmarking: {str(e)}")
            raise

    def _save_results(self):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'benchmark_results_{timestamp}.json')
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logging.info(f"Saved benchmark results to {results_file}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Run training benchmarks')
    parser.add_argument('--config', type=str, required=True, help='Path to base configuration file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for saving results')
    parser.add_argument('--gpu-configs', type=int, nargs='+', default=[2, 4, 8],
                      help='List of GPU configurations to test (e.g., 2 4 8)')
    
    args = parser.parse_args()
    
    try:
        runner = BenchmarkRunner(args.config, args.output_dir)
        runner.run_all_benchmarks(args.gpu_configs)
    except Exception as e:
        logging.error(f"Benchmark failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()