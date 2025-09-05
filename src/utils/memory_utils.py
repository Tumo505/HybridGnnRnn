"""
Memory Utilities for M1 MacBook Pro Optimization

This module provides utilities for monitoring and optimizing memory usage
during training and inference on memory-constrained devices.
"""

import gc
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from functools import wraps
import time
import warnings

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitor system and GPU memory usage during training.
    """
    
    def __init__(
        self,
        max_memory_gb: float = 16.0,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        log_interval: int = 100
    ):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_gb (float): Maximum available memory in GB
            warning_threshold (float): Threshold to issue warnings (fraction of max)
            critical_threshold (float): Threshold for critical memory usage
            log_interval (int): Interval for logging memory usage
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.log_interval = log_interval
        
        self.memory_history = []
        self.step_count = 0
        
        # Detect device type
        self.device_type = self._detect_device_type()
        
        logger.info(f"MemoryMonitor initialized for {self.device_type} with {max_memory_gb}GB limit")
    
    def _detect_device_type(self) -> str:
        """Detect the type of device being used."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        # System memory
        system_memory = psutil.virtual_memory()
        
        stats = {
            'system_total_gb': system_memory.total / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_percent': system_memory.percent,
            'process_memory_gb': psutil.Process().memory_info().rss / (1024**3)
        }
        
        # Device-specific memory
        if self.device_type == "cuda":
            if torch.cuda.is_available():
                stats.update({
                    'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'gpu_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
                })
        elif self.device_type == "mps":
            # MPS doesn't have direct memory query APIs
            # We'll estimate based on allocated tensors
            stats.update({
                'mps_estimated_gb': self._estimate_mps_memory() / (1024**3)
            })
        
        return stats
    
    def _estimate_mps_memory(self) -> float:
        """Estimate MPS memory usage by tracking tensor allocations."""
        total_memory = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.device.type == 'mps':
                total_memory += obj.numel() * obj.element_size()
        return total_memory
    
    def check_memory(self, context: str = ""):
        """
        Check current memory usage and issue warnings if necessary.
        
        Args:
            context (str): Context string for logging
        """
        self.step_count += 1
        stats = self.get_memory_usage()
        
        # Calculate memory pressure
        memory_pressure = stats['system_used_gb'] / self.max_memory_gb
        
        # Store in history
        self.memory_history.append({
            'step': self.step_count,
            'context': context,
            'stats': stats,
            'pressure': memory_pressure
        })
        
        # Check thresholds
        if memory_pressure > self.critical_threshold:
            logger.critical(f"CRITICAL MEMORY USAGE: {memory_pressure:.1%} "
                          f"({stats['system_used_gb']:.1f}GB/{self.max_memory_gb}GB) - {context}")
            self.emergency_cleanup()
        elif memory_pressure > self.warning_threshold:
            logger.warning(f"High memory usage: {memory_pressure:.1%} "
                         f"({stats['system_used_gb']:.1f}GB/{self.max_memory_gb}GB) - {context}")
        
        # Log periodically
        if self.step_count % self.log_interval == 0:
            logger.info(f"Memory usage: {memory_pressure:.1%} "
                       f"({stats['system_used_gb']:.1f}GB/{self.max_memory_gb}GB) - {context}")
    
    def emergency_cleanup(self):
        """Emergency memory cleanup procedures."""
        logger.info("Performing emergency memory cleanup...")
        
        # Force garbage collection
        gc.collect()
        
        # Clear device caches
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()
        
        # Additional cleanup
        self._clear_tensor_cache()
        
        # Re-check memory
        stats = self.get_memory_usage()
        new_pressure = stats['system_used_gb'] / self.max_memory_gb
        logger.info(f"After cleanup: {new_pressure:.1%} "
                   f"({stats['system_used_gb']:.1f}GB/{self.max_memory_gb}GB)")
    
    def _clear_tensor_cache(self):
        """Clear any cached tensors."""
        # This is a placeholder for more sophisticated tensor cache clearing
        # In practice, you might want to clear specific model caches
        pass
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive memory usage report.
        
        Returns:
            Dict[str, Any]: Memory usage report
        """
        if not self.memory_history:
            return {"error": "No memory history available"}
        
        pressures = [entry['pressure'] for entry in self.memory_history]
        
        report = {
            'max_memory_gb': self.max_memory_gb,
            'device_type': self.device_type,
            'total_steps': len(self.memory_history),
            'peak_pressure': max(pressures),
            'avg_pressure': np.mean(pressures),
            'min_pressure': min(pressures),
            'warning_violations': sum(1 for p in pressures if p > self.warning_threshold),
            'critical_violations': sum(1 for p in pressures if p > self.critical_threshold),
            'current_stats': self.get_memory_usage()
        }
        
        return report


def memory_efficient_decorator(max_memory_gb: float = 16.0):
    """
    Decorator for memory-efficient function execution.
    
    Args:
        max_memory_gb (float): Maximum memory limit in GB
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor(max_memory_gb=max_memory_gb)
            
            # Pre-execution cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            monitor.check_memory(f"Before {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                monitor.check_memory(f"After {func.__name__}")
                return result
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"Out of memory in {func.__name__}: {e}")
                    monitor.emergency_cleanup()
                    raise MemoryError(f"Out of memory in {func.__name__}: {e}")
                else:
                    raise
            finally:
                # Post-execution cleanup
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return wrapper
    return decorator


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader wrapper.
    """
    
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        memory_monitor: MemoryMonitor,
        cleanup_interval: int = 10
    ):
        """
        Initialize memory-efficient data loader.
        
        Args:
            dataloader: Original data loader
            memory_monitor: Memory monitor instance
            cleanup_interval: Interval for memory cleanup
        """
        self.dataloader = dataloader
        self.memory_monitor = memory_monitor
        self.cleanup_interval = cleanup_interval
        self.batch_count = 0
    
    def __iter__(self):
        for batch in self.dataloader:
            self.batch_count += 1
            
            # Periodic memory check and cleanup
            if self.batch_count % self.cleanup_interval == 0:
                self.memory_monitor.check_memory(f"Batch {self.batch_count}")
                
                # Cleanup if memory pressure is high
                stats = self.memory_monitor.get_memory_usage()
                pressure = stats['system_used_gb'] / self.memory_monitor.max_memory_gb
                
                if pressure > self.memory_monitor.warning_threshold:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)


def optimize_model_for_memory(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply memory optimizations to a model.
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        torch.nn.Module: Optimized model
    """
    # Enable gradient checkpointing for supported models
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Set model to use memory efficient attention if available
    for module in model.modules():
        if hasattr(module, 'memory_efficient'):
            module.memory_efficient = True
    
    # Optimize for inference if not training
    if not model.training:
        model = torch.jit.script(model)
        logger.info("Applied TorchScript optimization")
    
    return model


def chunk_tensor_processing(
    tensor: torch.Tensor,
    processing_fn: callable,
    chunk_size: int = 1000,
    dim: int = 0
) -> torch.Tensor:
    """
    Process large tensors in chunks to reduce memory usage.
    
    Args:
        tensor: Input tensor to process
        processing_fn: Function to apply to each chunk
        chunk_size: Size of each chunk
        dim: Dimension along which to chunk
        
    Returns:
        torch.Tensor: Processed tensor
    """
    chunks = torch.split(tensor, chunk_size, dim=dim)
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            processed_chunk = processing_fn(chunk)
            processed_chunks.append(processed_chunk)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM in chunk {i}, reducing chunk size")
                # Recursively process with smaller chunks
                smaller_chunks = torch.split(chunk, chunk_size // 2, dim=dim)
                for small_chunk in smaller_chunks:
                    processed_small = processing_fn(small_chunk)
                    processed_chunks.append(processed_small)
            else:
                raise
    
    return torch.cat(processed_chunks, dim=dim)


def get_memory_efficient_settings(available_memory_gb: float) -> Dict[str, Any]:
    """
    Get recommended memory-efficient settings based on available memory.
    
    Args:
        available_memory_gb: Available memory in GB
        
    Returns:
        Dict[str, Any]: Recommended settings
    """
    if available_memory_gb < 8:
        return {
            'batch_size': 2,
            'max_nodes_per_graph': 1000,
            'gradient_accumulation_steps': 8,
            'use_checkpoint': True,
            'precision': '16-mixed',
            'subsample_rate': 0.05
        }
    elif available_memory_gb < 16:
        return {
            'batch_size': 4,
            'max_nodes_per_graph': 2000,
            'gradient_accumulation_steps': 4,
            'use_checkpoint': True,
            'precision': '16-mixed',
            'subsample_rate': 0.1
        }
    else:
        return {
            'batch_size': 8,
            'max_nodes_per_graph': 5000,
            'gradient_accumulation_steps': 2,
            'use_checkpoint': False,
            'precision': '32-true',
            'subsample_rate': 0.2
        }


class MemoryTracker:
    """
    Track memory usage of specific operations or model components.
    """
    
    def __init__(self, name: str = "MemoryTracker"):
        self.name = name
        self.measurements = []
        self.start_memory = None
    
    def start(self):
        """Start tracking memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.start_memory = torch.cuda.memory_allocated()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            self.start_memory = psutil.Process().memory_info().rss
        else:
            self.start_memory = psutil.Process().memory_info().rss
    
    def stop(self, operation_name: str = "operation"):
        """Stop tracking and record memory usage."""
        if self.start_memory is None:
            logger.warning("Memory tracking not started")
            return
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory) / (1024**2)  # MB
        else:
            end_memory = psutil.Process().memory_info().rss
            memory_used = (end_memory - self.start_memory) / (1024**2)  # MB
        
        self.measurements.append({
            'operation': operation_name,
            'memory_mb': memory_used,
            'timestamp': time.time()
        })
        
        logger.info(f"{self.name} - {operation_name}: {memory_used:.1f} MB")
        self.start_memory = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of memory measurements."""
        if not self.measurements:
            return {"error": "No measurements recorded"}
        
        memory_values = [m['memory_mb'] for m in self.measurements]
        
        return {
            'total_operations': len(self.measurements),
            'total_memory_mb': sum(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'min_memory_mb': min(memory_values),
            'measurements': self.measurements
        }


if __name__ == "__main__":
    # Example usage
    print("Memory utilities module loaded successfully!")
    
    # Test memory monitor
    monitor = MemoryMonitor(max_memory_gb=16.0)
    monitor.check_memory("Test")
    
    # Get memory report
    report = monitor.get_memory_report()
    print(f"Current memory usage: {report['current_stats']['system_used_gb']:.1f}GB")
    
    # Test memory tracker
    tracker = MemoryTracker("TestTracker")
    tracker.start()
    
    # Simulate some work
    dummy_tensor = torch.randn(1000, 1000)
    result = dummy_tensor @ dummy_tensor.T
    
    tracker.stop("Matrix multiplication")
    summary = tracker.get_summary()
    print(f"Memory used: {summary['total_memory_mb']:.1f}MB")
