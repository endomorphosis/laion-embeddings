import os
import sys
import json
import random
import datasets
import asyncio
import subprocess
import aiohttp
import requests
import torch
import faiss
import math
import timeit
import time
import gc
import numpy as np
import psutil
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from aiohttp import ClientSession, ClientTimeout
import multiprocessing
from multiprocessing import Pool
import transformers
from transformers import AutoTokenizer, AutoModel
import datasets
import ipfs_accelerate_py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datasets import Dataset, concatenate_datasets, load_dataset
try:
    from .ipfs_multiformats import ipfs_multiformats_py
    from .ipfs_multiformats import *
except Exception as e:
    try:
        from ipfs_multiformats import ipfs_multiformats_py
        from ipfs_multiformats import *
    except Exception as e:
        try:
            import ipfs_multiformats
            ipfs_multiformats_py = getattr(ipfs_multiformats, 'ipfs_multiformats_py', None)
        except Exception as e:
            ipfs_multiformats_py = None
    pass

try:
    from .chunker import chunker
    from .chunker import *
except Exception as e:
    try:
        from chunker import chunker
        from chunker import *
    except Exception as e:
        try:
            import chunker
        except Exception as e:
            chunker = None
    pass
try:
    from .elasticsearch_kit import elasticsearch_kit
    from .elasticsearch_kit import *
except Exception as e:
    try:
        from elasticsearch_kit import elasticsearch_kit
        from elasticsearch_kit import *
    except Exception as e:
        elasticsearch_kit = None
    pass

try:
    from .qdrant_kit import qdrant_kit_py
    from .qdrant_kit import *
except Exception as e:
    try:
        from qdrant_kit import qdrant_kit_py
        from qdrant_kit import *
    except Exception as e:
        qdrant_kit_py = None
    pass

try:
    from .faiss_kit import faiss_kit_py
    from .faiss_kit import *
except Exception as e:
    try:
        from faiss_kit import faiss_kit_py
        from faiss_kit import *
    except Exception as e:
        faiss_kit_py = None
    pass


from multiprocessing import Process
from ipfs_kit_py.ipfs_kit import ipfs_kit
from ipfs_embeddings_py.ipfs_datasets import ipfs_datasets_py
from .qdrant_kit import qdrant_kit_py
from .faiss_kit import faiss_kit_py

# ==============================================================================
# ADAPTIVE BATCH PROCESSING OPTIMIZATION
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for batch processing optimization"""
    batch_size: int
    processing_time: float
    memory_usage_mb: float
    throughput: float  # items per second
    success_rate: float
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MemoryMonitor:
    """Monitor system memory usage for adaptive batch sizing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".MemoryMonitor")
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        try:
            return psutil.virtual_memory().available / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Failed to get available memory: {e}")
            return 1024.0  # Default fallback
    
    def get_memory_percent(self) -> float:
        """Get memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except Exception as e:
            self.logger.warning(f"Failed to get memory percentage: {e}")
            return 0.0

class AdaptiveBatchProcessor:
    """Intelligent batch size optimization based on performance metrics and memory usage"""
    
    def __init__(self, max_memory_percent: float = 80.0, min_batch_size: int = 1, max_batch_size: int = 512):
        self.max_memory_percent = max_memory_percent
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.optimal_batch_sizes: Dict[str, int] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.memory_monitor = MemoryMonitor()
        self.logger = logging.getLogger(__name__ + ".AdaptiveBatchProcessor")
    
    async def find_optimal_batch_size(self, operation_key: str, test_function, initial_batch_size: int = 32) -> int:
        """Find optimal batch size for a given operation using binary search with performance monitoring"""
        start_time = time.time()
        
        try:
            # Start with conservative batch size based on memory
            current_batch_size = min(initial_batch_size, self.get_memory_aware_batch_size())
            
            # Track performance for this operation
            if operation_key not in self.performance_history:
                self.performance_history[operation_key] = []
            
            best_batch_size = current_batch_size
            best_throughput = 0.0
            
            # Binary search for optimal batch size
            min_size = self.min_batch_size
            max_size = min(self.max_batch_size, self.get_memory_aware_batch_size())
            
            while min_size <= max_size:
                test_batch_size = (min_size + max_size) // 2
                
                # Test performance at this batch size
                metrics = await self._test_batch_performance(test_function, test_batch_size, operation_key)
                
                if metrics.success_rate > 0.8 and metrics.throughput > best_throughput:
                    best_batch_size = test_batch_size
                    best_throughput = metrics.throughput
                    min_size = test_batch_size + 1  # Try larger batch
                else:
                    max_size = test_batch_size - 1  # Try smaller batch
                
                # Memory safety check
                if self.memory_monitor.get_memory_percent() > self.max_memory_percent:
                    self.logger.warning(f"Memory usage too high ({self.memory_monitor.get_memory_percent():.1f}%), reducing batch size")
                    max_size = test_batch_size - 1
            
            self.optimal_batch_sizes[operation_key] = best_batch_size
            self.logger.info(f"Optimal batch size for {operation_key}: {best_batch_size} (throughput: {best_throughput:.2f} items/sec)")
            
            return best_batch_size
            
        except Exception as e:
            self.logger.error(f"Error finding optimal batch size for {operation_key}: {e}")
            return self.min_batch_size
    
    def get_adaptive_batch_size(self, operation_key: str, default_size: int = 32) -> int:
        """Get adaptive batch size based on current memory and historical performance"""
        try:
            # Check if we have an optimal size for this operation
            if operation_key in self.optimal_batch_sizes:
                optimal_size = self.optimal_batch_sizes[operation_key]
            else:
                optimal_size = default_size
            
            # Adjust based on current memory usage
            memory_adjusted_size = self.get_memory_aware_batch_size()
            
            # Use the smaller of optimal and memory-constrained size
            adaptive_size = min(optimal_size, memory_adjusted_size)
            
            # Ensure within bounds
            adaptive_size = max(self.min_batch_size, min(adaptive_size, self.max_batch_size))
            
            return adaptive_size
            
        except Exception as e:
            self.logger.warning(f"Error calculating adaptive batch size: {e}")
            return default_size
    
    def get_memory_aware_batch_size(self) -> int:
        """Calculate batch size based on available memory"""
        try:
            memory_percent = self.memory_monitor.get_memory_percent()
            available_mb = self.memory_monitor.get_available_memory_mb()
            
            # Conservative batch sizing based on memory pressure
            if memory_percent > 90:
                return self.min_batch_size
            elif memory_percent > 80:
                return max(self.min_batch_size, self.max_batch_size // 4)
            elif memory_percent > 70:
                return max(self.min_batch_size, self.max_batch_size // 2)
            elif available_mb < 500:  # Less than 500MB available
                return max(self.min_batch_size, self.max_batch_size // 4)
            else:
                return self.max_batch_size
                
        except Exception as e:
            self.logger.warning(f"Error calculating memory-aware batch size: {e}")
            return self.min_batch_size
    
    async def _test_batch_performance(self, test_function, batch_size: int, operation_key: str) -> PerformanceMetrics:
        """Test performance of a batch operation"""
        start_time = time.time()
        start_memory = self.memory_monitor.get_memory_usage_mb()
        
        try:
            # Create test data
            test_data = [f"test_item_{i}" for i in range(batch_size)]
            
            # Run test function
            success = await test_function(test_data)
            
            end_time = time.time()
            end_memory = self.memory_monitor.get_memory_usage_mb()
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = batch_size / processing_time if processing_time > 0 else 0
            success_rate = 1.0 if success else 0.0
            
            metrics = PerformanceMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                throughput=throughput,
                success_rate=success_rate
            )
            
            # Store metrics
            self.performance_history[operation_key].append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Batch performance test failed for size {batch_size}: {e}")
            return PerformanceMetrics(
                batch_size=batch_size,
                processing_time=0.0,
                memory_usage_mb=0.0,
                throughput=0.0,
                success_rate=0.0
            )
    
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

# ==============================================================================
# ADAPTIVE QUEUE MANAGEMENT SYSTEM (P1 PRIORITY)
# ==============================================================================

@dataclass
class QueueMetrics:
    """Metrics for queue performance monitoring"""
    queue_size: int
    max_size: int
    throughput: float
    wait_time: float
    memory_usage_mb: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)

class AdaptiveQueueManager:
    """
    Intelligent queue management with adaptive sizing based on performance metrics,
    memory usage, and system load. Part of P1 performance optimization priorities.
    """
    
    def __init__(self, 
                 max_memory_percent: float = 75.0,
                 min_queue_size: int = 10,
                 max_queue_size: int = 1000,
                 target_throughput: float = 10.0):
        self.max_memory_percent = max_memory_percent
        self.min_queue_size = min_queue_size
        self.max_queue_size = max_queue_size
        self.target_throughput = target_throughput
        
        # Queue configurations by operation type
        self.queue_configs: Dict[str, Dict[str, Any]] = {}
        self.queue_metrics_history: Dict[str, List[QueueMetrics]] = {}
        self.optimal_queue_sizes: Dict[str, int] = {}
        
        self.memory_monitor = MemoryMonitor()
        self.logger = logging.getLogger(__name__ + ".AdaptiveQueueManager")
    
    def get_adaptive_queue_size(self, operation_key: str, current_load: int = 0) -> int:
        """
        Calculate optimal queue size based on current conditions and performance history.
        
        Args:
            operation_key: Unique identifier for the operation type
            current_load: Current number of items being processed
            
        Returns:
            Optimal queue size for current conditions
        """
        try:
            # Start with cached optimal size if available
            base_size = self.optimal_queue_sizes.get(operation_key, self.min_queue_size * 4)
            
            # Adjust based on memory pressure
            memory_percent = self.memory_monitor.get_memory_percent()
            memory_factor = self._calculate_memory_factor(memory_percent)
            
            # Adjust based on historical performance
            performance_factor = self._calculate_performance_factor(operation_key)
            
            # Adjust based on current system load
            load_factor = self._calculate_load_factor(current_load)
            
            # Calculate adaptive size
            adaptive_size = int(base_size * memory_factor * performance_factor * load_factor)
            
            # Ensure within bounds
            adaptive_size = max(self.min_queue_size, min(adaptive_size, self.max_queue_size))
            
            self.logger.debug(f"Adaptive queue size for {operation_key}: {adaptive_size} "
                            f"(base: {base_size}, memory: {memory_factor:.2f}, "
                            f"performance: {performance_factor:.2f}, load: {load_factor:.2f})")
            
            return adaptive_size
            
        except Exception as e:
            self.logger.warning(f"Error calculating adaptive queue size: {e}")
            return self.min_queue_size * 4
    
    def _calculate_memory_factor(self, memory_percent: float) -> float:
        """Calculate memory pressure factor for queue sizing"""
        if memory_percent > 90:
            return 0.3  # Severely reduce queue size
        elif memory_percent > 80:
            return 0.5  # Moderately reduce queue size
        elif memory_percent > 70:
            return 0.7  # Slightly reduce queue size
        elif memory_percent < 50:
            return 1.2  # Can increase queue size
        else:
            return 1.0  # Normal operation
    
    def _calculate_performance_factor(self, operation_key: str) -> float:
        """Calculate performance factor based on historical metrics"""
        if operation_key not in self.queue_metrics_history:
            return 1.0
        
        recent_metrics = self.queue_metrics_history[operation_key][-10:]  # Last 10 measurements
        if not recent_metrics:
            return 1.0
        
        # Calculate average throughput and error rate
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # High throughput = can handle larger queues
        throughput_factor = min(2.0, avg_throughput / self.target_throughput)
        
        # High error rate = should reduce queue size
        error_factor = max(0.5, 1.0 - avg_error_rate)
        
        return throughput_factor * error_factor
    
    def _calculate_load_factor(self, current_load: int) -> float:
        """Calculate load factor based on current system utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # High CPU usage = reduce queue size
            if cpu_percent > 90:
                return 0.6
            elif cpu_percent > 80:
                return 0.8
            elif cpu_percent < 30:
                return 1.2
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def record_queue_metrics(self, operation_key: str, metrics: QueueMetrics):
        """Record queue performance metrics for future optimization"""
        if operation_key not in self.queue_metrics_history:
            self.queue_metrics_history[operation_key] = []
        
        self.queue_metrics_history[operation_key].append(metrics)
        
        # Keep only recent metrics (last 100 measurements)
        if len(self.queue_metrics_history[operation_key]) > 100:
            self.queue_metrics_history[operation_key] = self.queue_metrics_history[operation_key][-100:]
        
        # Update optimal queue size based on performance
        self._update_optimal_queue_size(operation_key)
    
    def _update_optimal_queue_size(self, operation_key: str):
        """Update optimal queue size based on performance metrics"""
        if operation_key not in self.queue_metrics_history:
            return
        
        recent_metrics = self.queue_metrics_history[operation_key][-20:]  # Last 20 measurements
        if len(recent_metrics) < 5:
            return  # Need more data
        
        # Find queue size with best performance (highest throughput, lowest error rate)
        best_score = 0
        best_size = self.min_queue_size * 4
        
        for metrics in recent_metrics:
            # Score based on throughput and inverse error rate
            score = metrics.throughput * (1.0 - metrics.error_rate)
            if score > best_score:
                best_score = score
                best_size = metrics.queue_size
        
        self.optimal_queue_sizes[operation_key] = best_size
        self.logger.debug(f"Updated optimal queue size for {operation_key}: {best_size}")
    
    def get_queue_recommendations(self, operation_key: str) -> Dict[str, Any]:
        """Get queue configuration recommendations for an operation"""
        current_size = self.get_adaptive_queue_size(operation_key)
        
        recommendations = {
            'recommended_size': current_size,
            'memory_pressure': self.memory_monitor.get_memory_percent(),
            'historical_performance': self._get_performance_summary(operation_key),
            'optimization_suggestions': []
        }
        
        # Add specific suggestions
        if self.memory_monitor.get_memory_percent() > 80:
            recommendations['optimization_suggestions'].append(
                "High memory usage detected - consider reducing queue size or implementing memory cleanup"
            )
        
        if operation_key in self.queue_metrics_history:
            recent_metrics = self.queue_metrics_history[operation_key][-5:]
            if recent_metrics:
                avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
                if avg_error_rate > 0.1:
                    recommendations['optimization_suggestions'].append(
                        f"High error rate ({avg_error_rate:.1%}) - investigate error causes or reduce processing load"
                    )
        
        return recommendations
    
    def _get_performance_summary(self, operation_key: str) -> Dict[str, float]:
        """Get performance summary for an operation"""
        if operation_key not in self.queue_metrics_history or not self.queue_metrics_history[operation_key]:
            return {}
        
        recent_metrics = self.queue_metrics_history[operation_key][-20:]
        
        return {
            'avg_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'avg_wait_time': sum(m.wait_time for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        }
    
    def cleanup_old_metrics(self, max_age_hours: int = 24):
        """Clean up old metrics to prevent memory bloat"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for operation_key in list(self.queue_metrics_history.keys()):
            self.queue_metrics_history[operation_key] = [
                m for m in self.queue_metrics_history[operation_key] 
                if m.timestamp > cutoff_time
            ]
            
            # Remove empty histories
            if not self.queue_metrics_history[operation_key]:
                del self.queue_metrics_history[operation_key]
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary for monitoring"""
        memory_percent = self.memory_monitor.get_memory_percent()
        
        health_status = "healthy"
        if memory_percent > 90:
            health_status = "critical"
        elif memory_percent > 80:
            health_status = "warning"
        
        active_operations = len(self.queue_metrics_history)
        total_metrics = sum(len(metrics) for metrics in self.queue_metrics_history.values())
        
        return {
            'health_status': health_status,
            'memory_percent': memory_percent,
            'active_operations': active_operations,
            'total_metrics_recorded': total_metrics,
            'optimal_queue_sizes': dict(self.optimal_queue_sizes),
            'recommendations': self._get_system_recommendations()
        }
    
    def _get_system_recommendations(self) -> List[str]:
        """Get system-level optimization recommendations"""
        recommendations = []
        memory_percent = self.memory_monitor.get_memory_percent()
        
        if memory_percent > 85:
            recommendations.append("Critical: Memory usage is very high - consider reducing queue sizes globally")
        elif memory_percent > 75:
            recommendations.append("Warning: Memory usage is elevated - monitor queue performance closely")
        
        if len(self.queue_metrics_history) > 50:
            recommendations.append("Info: Many active operations detected - consider queue consolidation")
        
        # Check for operations with consistently poor performance
        for operation_key, metrics in self.queue_metrics_history.items():
            if len(metrics) > 5:
                recent_metrics = metrics[-10:]
                avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
                if avg_error_rate > 0.15:
                    recommendations.append(f"Operation '{operation_key}' has high error rate ({avg_error_rate:.1%})")
        
        return recommendations

# ==============================================================================
# ENHANCED ERROR HANDLING FRAMEWORK
# ==============================================================================

class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

class ProcessingError(Exception):
    """Raised when processing operations fail"""
    pass

class MemoryError(Exception):
    """Raised when memory constraints are exceeded"""
    pass

def validate_batch_input(batch: Any, max_size: int = 1000) -> List[str]:
    """Validate and sanitize batch input"""
    if batch is None:
        raise ValidationError("Batch cannot be None")
    
    if isinstance(batch, str):
        batch = [batch]
    
    if not isinstance(batch, (list, tuple)):
        raise ValidationError(f"Batch must be a list or tuple, got {type(batch)}")
    
    if len(batch) == 0:
        raise ValidationError("Batch cannot be empty")
    
    if len(batch) > max_size:
        raise ValidationError(f"Batch size {len(batch)} exceeds maximum {max_size}")
    
    # Convert all items to strings and validate
    validated_batch = []
    for i, item in enumerate(batch):
        if item is None:
            logger.warning(f"Skipping None item at index {i}")
            continue
        
        try:
            str_item = str(item).strip()
            if str_item:
                validated_batch.append(str_item)
        except Exception as e:
            logger.warning(f"Failed to convert item at index {i} to string: {e}")
    
    if not validated_batch:
        raise ValidationError("No valid items in batch after validation")
    
    return validated_batch

def safe_execute_with_retry(func, *args, max_retries: int = 3, delay: float = 1.0, **kwargs):
    """Execute function with exponential backoff retry logic"""
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
    
    # This should never happen due to the loop logic, but type safety
    if last_exception is None:
        last_exception = RuntimeError("Unexpected error: no exception recorded")
    raise last_exception

# Timeout constants for batch size optimization and network operations
BATCH_SIZE_OPTIMIZATION_TIMEOUT = 180  # 3 minutes for batch size optimization
NETWORK_REQUEST_TIMEOUT = 60  # 1 minute for individual network requests
ADAPTIVE_BATCH_TIMEOUT = 300  # 5 minutes for adaptive batch processing
TEST_BATCH_TIMEOUT = 45  # 45 seconds for individual batch tests

class BatchSizeTimeoutError(Exception):
    """Custom exception for batch size optimization timeouts"""
    pass

async def safe_async_execute_with_timeout(coroutine, timeout: float, operation_name: str = "operation"):
    """Execute async operation with timeout protection"""
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        error_msg = f"Timeout ({timeout}s) exceeded for {operation_name}"
        logger.error(error_msg)
        raise BatchSizeTimeoutError(error_msg)
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}")
        raise

async def safe_async_execute_with_retry(func, *args, max_retries: int = 3, delay: float = 1.0, timeout: Optional[float] = None, **kwargs):
    """Execute async function with exponential backoff retry logic and optional timeout"""
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries + 1):
        try:
            if timeout:
                # Apply timeout to each individual attempt
                return await safe_async_execute_with_timeout(
                    func(*args, **kwargs), 
                    timeout=timeout, 
                    operation_name=f"{func.__name__}_attempt_{attempt + 1}"
                )
            else:
                return await func(*args, **kwargs)
        except (Exception, BatchSizeTimeoutError) as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
    
    # This should never happen due to the loop logic, but type safety
    if last_exception is None:
        last_exception = RuntimeError("Unexpected error: no exception recorded")
    raise last_exception

class ipfs_embeddings_py:
    def __init__(self, resources, metadata):
        # Initialize with error handling for optional components
        try:
            if ipfs_multiformats_py is not None:
                self.multiformats = ipfs_multiformats_py(resources, metadata)
            else:
                self.multiformats = None
        except (NameError, AttributeError, TypeError):
            self.multiformats = None
            
        try:
            if ipfs_multiformats_py is not None:
                self.multiformats_py = ipfs_multiformats_py(resources, metadata)
            else:
                self.multiformats_py = None
        except (NameError, AttributeError, TypeError):
            self.multiformats_py = None
            
        self.datasets = Dataset
        self.ipfs_datasets = ipfs_datasets_py(resources, metadata)
        
        try:
            if chunker is not None and callable(chunker):
                self.chunker = chunker(resources, metadata)
            else:
                self.chunker = None
        except (NameError, AttributeError, TypeError):
            self.chunker = None
            
        self.qdrant_kit_py = qdrant_kit_py(resources, metadata)
        self.elasticsearch_kit = elasticsearch_kit(resources, metadata)
        self.faiss_kit = faiss_kit_py(resources, metadata)
        self.ipfs_accelerate_py = ipfs_accelerate_py.ipfs_accelerate_py(resources=resources, metadata=metadata)
        # Create wrapper methods to handle signature differences
        # self.process_new_dataset_shard = self.ipfs_datasets.process_new_dataset_shard
        self.process_index_shard = self.ipfs_datasets.process_index_shard
        self.ipfs_parquet_to_car = self.ipfs_datasets.ipfs_parquet_to_car_py
        self.ipfs_parquet_to_car_test = self.ipfs_datasets.ipfs_parquet_to_car_py.test
        self.ipfs_parquet_to_car_install = self.ipfs_datasets.ipfs_parquet_to_car_py.install
        if "ipfs_embeddings" not in dir(self) and "ipfs_embeddings" in self.__dict__.keys():
            self.ipfs_embeddings = self
        self.parquet_to_car = self.ipfs_parquet_to_car
        # self.elasticsearch = elasticsearch_kit(resources, metadata)
        self.consumer_task_done = {}
        self.producer_task_done = False
        self.save_to_disk_task_done = False
        self.tei_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.local_endpoints = {}
        self.index =  {}
        self.schemas = {}
        self.queues = {}
        self.caches = {}
        self.chunk_cache = {}
        self.chunk_embeddings = {}
        self.cid_chunk_list = []
        self.cid_chunk_set = set()
        self.batch_sizes = {}
        self.cid_list = set()
        self.cid_set = set()
        self.new_dataset = None
        self.all_cid_list = {}
        self.all_cid_set = {}
        self.cid_chunk_queue = None
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.endpoint_handler = {}
        self.new_dataset = {}
        self.new_dataset_children = {}
        self.saved = False
        self.resources = resources
        self.metadata = metadata
        
        # Initialize adaptive batch processing system
        self.adaptive_batch_processor = AdaptiveBatchProcessor(
            max_memory_percent=80.0,
            min_batch_size=1,
            max_batch_size=512
        )
        self.memory_monitor = self.adaptive_batch_processor.memory_monitor
        
        # Initialize adaptive queue management system
        self.adaptive_queue_manager = AdaptiveQueueManager(
            min_queue_size=10,
            max_queue_size=1000
        )
        
        # Enhanced error handling tracking
        self.processing_errors = {}
        self.validation_errors = {}
        
        # Initialize endpoints
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.init_endpoints = self.init_endpoints       
        return None
    
    async def process_new_dataset_shard(self, dataset, split=None):
        """Wrapper method for process_new_dataset_shard with correct signature"""
        try:
            # Call the underlying method with proper parameter mapping
            # The original method expects (shard, datatype, split) but we provide (dataset, split)
            # We'll map dataset -> shard and use None for datatype
            result = self.ipfs_datasets.process_new_dataset_shard(dataset, datatype=None, split=split)
            return result
        except Exception as e:
            logger.error(f"Error processing new dataset shard: {e}")
            return None
    
    async def init_endpoints(self, models, endpoint_list=None):
        try:
            # Since ipfs_kit.init_endpoints is not async, call it directly
            results = self.ipfs_kit.init_endpoints(models, endpoint_list)
            return results
        except Exception as e:
            logger.error(f"Error initializing endpoints: {e}")
            return None

    def load_index(self, index):
        self.index = index
        return None 
    
    async def load_dataset(self, dataset, split=None):
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.safe_dataset_column_names(self.dataset)
        if isinstance(columns, list):
            columns.append("cid")
        else:
            columns = list(columns) + ["cid"] if columns else ["cid"]
        return None

    def index_cid(self, samples):
        results = []
        if samples is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
        if isinstance(samples, list):
            for this_sample in samples:
                if self.multiformats and hasattr(self.multiformats, 'get_cid'):
                    this_sample_cid = self.multiformats.get_cid(this_sample)
                else:
                    # Fallback CID generation if multiformats is not available
                    import hashlib
                    sample_str = str(this_sample) if not isinstance(this_sample, str) else this_sample
                    this_sample_cid = f"bafkrei{hashlib.sha256(sample_str.encode()).hexdigest()[:50]}"
                self.cid_index[this_sample_cid] = this_sample
                results.append(this_sample_cid)
        else:
            raise ValueError("samples must be a list or string")
        return results
    
    async def parse_knn_errors(self, request, model, endpoint, endpoint_type=None):
        fatal = False
        return fatal
    
    
    async def parse_knn(self, request, model, endpoint, endpoint_type=None):
        token_length_size = 0
        incoming_batch_size = len(request)
        endpoint_batch_size = self.batch_sizes[model]
        embeddings = request
        embeddings_request = embeddings
        endpoint_context_size = 0
        response = None  # Initialize response variable
        
        if endpoint_type is None:
            raise ValueError("Endpoint type must be defined")
        
        if endpoint_type == "local_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                if "cuda" in endpoint or "cpu" in endpoint:
                    response = self.request_local_endpoint(model, endpoint, endpoint_type)
                elif "openvino:" in endpoint:
                    response = self.request_openvino_endpoint(model, endpoint, endpoint_type)
                elif "llama_cpp" in endpoint:
                    response = self.request_llama_cpp_endpoint(model, endpoint, endpoint_type)
                else:
                    response = ValueError("Endpoint not found")
        elif endpoint_type == "tei_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                response = self.request_tei_endpoint(model, endpoint, endpoint_type)
        elif endpoint_type == "openvino_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                response = self.request_openvino_endpoint(model, endpoint, endpoint_type)
        elif endpoint_type == "libp2p_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                response = self.request_libp2p_endpoint(model, endpoint, endpoint_type)
        
        # Ensure response is not None before proceeding
        if response is None:
            response = ValueError(f"No valid endpoint response for {endpoint_type}")
        
        errors = await self.parse_knn_errors(response, model, endpoint, endpoint_type)
        
        return not errors
    
    async def request_knn(self, request_batch, model, endpoint, endpoint_type):
        request = None
        if endpoint_type is None:
            request = None
            pass
        elif endpoint_type == "tei_endpoints":
            request = await self.request_tei_endpoint(model, len(request_batch))
            pass
        elif endpoint_type == "openvino_endpoints":
            request = await self.request_openvino_endpoint(model, len(request_batch))
            pass
        elif endpoint_type == "libp2p_endpoints":
            request = await self.request_libp2p_endpoint(model, len(request_batch))
            pass
        elif endpoint_type == "local_endpoints":
            request = await self.request_local_endpoint(model, len(request_batch))
            pass
        else:
            request = None
            pass
        if request is not None:
            return request
        else:   
            return None

    async def max_batch_size(self, model, endpoint=None, endpoint_type=None):
        """
        Determine optimal batch size for a model and endpoint using adaptive processing.
        
        This method replaces the legacy exponential batch size testing with intelligent
        adaptive batch processing that considers memory usage, performance metrics, and
        hardware capabilities.
        
        Args:
            model: The model name/identifier
            endpoint: The endpoint URL or identifier (optional)
            endpoint_type: Type of endpoint (optional, will be auto-detected)
            
        Returns:
            Optimal batch size for the given configuration
        """
        try:
            # Wrap the entire batch size optimization in a timeout
            return await safe_async_execute_with_timeout(
                self._max_batch_size_implementation(model, endpoint, endpoint_type),
                timeout=BATCH_SIZE_OPTIMIZATION_TIMEOUT,
                operation_name=f"batch_size_optimization_for_{model}"
            )
            
        except BatchSizeTimeoutError as e:
            logger.error(f"Batch size optimization timed out for model {model}: {e}")
            # Fall back to safe default batch size
            fallback_size = 8  # Conservative fallback
            if endpoint and hasattr(self, 'endpoint_status'):
                self.endpoint_status[endpoint] = fallback_size
            return fallback_size
            
        except Exception as e:
            logger.error(f"Error determining max batch size for model {model}: {e}")
            # Fall back to memory-aware batch size if available
            try:
                if hasattr(self, 'adaptive_batch_processor'):
                    fallback_size = self.adaptive_batch_processor.get_memory_aware_batch_size()
                else:
                    fallback_size = 16  # Reasonable default
            except Exception:
                fallback_size = 8  # Conservative fallback
                
            if endpoint and hasattr(self, 'endpoint_status'):
                self.endpoint_status[endpoint] = fallback_size
                
            return fallback_size

    async def _max_batch_size_implementation(self, model, endpoint=None, endpoint_type=None):
        """
        Internal implementation of batch size optimization with timeout protection.
        
        This method contains the core logic wrapped by timeout protection in max_batch_size.
        """
        # Input validation
        if not model:
            logger.error("Model parameter is required")
            if hasattr(self, 'adaptive_batch_processor'):
                return self.adaptive_batch_processor.min_batch_size
            return 8  # Safe default
        
        # Auto-detect endpoint type if not provided
        if endpoint and endpoint_type is None:
            if "/embed" in str(endpoint):
                endpoint_type = "tei_endpoints"
            elif "/infer" in str(endpoint):
                endpoint_type = "openvino_endpoints"
            elif "http" in str(endpoint):
                endpoint_type = "tei_endpoints"
            elif any(x in str(endpoint) for x in ["cuda", "cpu", "local"]):
                endpoint_type = "local_endpoints"
            elif "libp2p" in str(endpoint):
                endpoint_type = "libp2p_endpoints"
            else:
                logger.warning(f"Could not determine endpoint type for {endpoint}")
                endpoint_type = "tei_endpoints"  # Default fallback
        
        # Choose endpoint if not provided
        if endpoint is None:
            endpoint = self.choose_endpoint(model)
            
        if not endpoint:
            logger.warning(f"No endpoint available for model {model}")
            if hasattr(self, 'adaptive_batch_processor'):
                return self.adaptive_batch_processor.min_batch_size
            return 8  # Safe default
        
        # Create operation key for caching optimal batch sizes
        operation_key = f"{model}_{endpoint_type}_{endpoint}"
        
        # Check if we have a cached optimal batch size
        if hasattr(self, 'adaptive_batch_processor') and operation_key in self.adaptive_batch_processor.optimal_batch_sizes:
            cached_size = self.adaptive_batch_processor.optimal_batch_sizes[operation_key]
            logger.info(f"Using cached optimal batch size for {operation_key}: {cached_size}")
            
            # Update endpoint status for backwards compatibility
            if hasattr(self, 'endpoint_status'):
                self.endpoint_status[endpoint] = cached_size
                
            return cached_size
        
        # Prepare test data for batch size optimization with timeout protection
        token_length_size = await safe_async_execute_with_timeout(
            self._get_token_length_size_async(model, endpoint, endpoint_type),
            timeout=30,  # 30 seconds for token length determination
            operation_name=f"token_length_determination_for_{model}"
        )
        
        test_text = await safe_async_execute_with_timeout(
            self._generate_test_text_async(model, token_length_size),
            timeout=30,  # 30 seconds for test text generation
            operation_name=f"test_text_generation_for_{model}"
        )
        
        # Create test function for adaptive batch processor with timeout protection
        async def test_batch_function(test_batch):
            """Test function for the adaptive batch processor with timeout protection"""
            try:
                # Wrap batch testing in timeout
                result = await safe_async_execute_with_timeout(
                    self._execute_batch_test(test_batch, model, endpoint, endpoint_type),
                    timeout=TEST_BATCH_TIMEOUT,
                    operation_name=f"batch_test_size_{len(test_batch)}"
                )
                return result
                
            except (BatchSizeTimeoutError, Exception) as e:
                logger.warning(f"Batch test failed for size {len(test_batch)}: {e}")
                return False
        
        # Use adaptive batch processor to find optimal size with timeout protection
        logger.info(f"Finding optimal batch size for {operation_key}...")
        
        # Create test data (use shorter test text for memory efficiency)
        test_data = [test_text[:min(512, len(test_text))] for _ in range(50)]  # Sample size for testing
        
        # Get initial batch size safely
        if hasattr(self, 'adaptive_batch_processor'):
            initial_batch_size = min(32, self.adaptive_batch_processor.get_memory_aware_batch_size())
        else:
            initial_batch_size = 16  # Conservative default
        
        # Execute adaptive batch processing with timeout protection
        optimal_batch_size = await safe_async_execute_with_timeout(
            self._find_optimal_batch_size_with_protection(
                operation_key, test_batch_function, initial_batch_size
            ),
            timeout=ADAPTIVE_BATCH_TIMEOUT,
            operation_name=f"adaptive_batch_processing_for_{operation_key}"
        )
        
        # Update endpoint status for backwards compatibility
        if hasattr(self, 'endpoint_status'):
            self.endpoint_status[endpoint] = optimal_batch_size
        
        logger.info(f"Optimal batch size determined for {operation_key}: {optimal_batch_size}")
        return optimal_batch_size

    async def _execute_batch_test(self, test_batch, model, endpoint, endpoint_type):
        """Execute a single batch test with proper validation and timeout protection"""
        # Validate batch input
        validated_batch = validate_batch_input(test_batch)
        
        # Test the actual processing pipeline with timeout protection for network calls
        if endpoint_type == "local_endpoints" and any(x in str(endpoint) for x in ["cuda", "cpu", "local"]):
            # Test local processing with timeout
            results = await safe_async_execute_with_retry(
                self.index_knn, 
                validated_batch, 
                model, 
                endpoint,
                timeout=NETWORK_REQUEST_TIMEOUT,
                max_retries=2
            )
        else:
            # Test remote endpoint processing with timeout
            results = await safe_async_execute_with_retry(
                self.request_knn, 
                validated_batch, 
                model, 
                endpoint, 
                endpoint_type,
                timeout=NETWORK_REQUEST_TIMEOUT,
                max_retries=2
            )
        
        # Return success status
        return results is not None and len(results) > 0

    async def _find_optimal_batch_size_with_protection(self, operation_key, test_batch_function, initial_batch_size):
        """Find optimal batch size with timeout and error protection"""
        if hasattr(self, 'adaptive_batch_processor'):
            return await self.adaptive_batch_processor.find_optimal_batch_size(
                operation_key=operation_key,
                test_function=test_batch_function,
                initial_batch_size=initial_batch_size
            )
        else:
            # Fallback implementation if adaptive_batch_processor is not available
            logger.warning(f"Adaptive batch processor not available, using fallback for {operation_key}")
            return initial_batch_size

    async def _get_token_length_size_async(self, model: str, endpoint: str, endpoint_type: Optional[str]) -> int:
        """Async wrapper for token length size determination with timeout protection"""
        return self._get_token_length_size(model, endpoint, endpoint_type)

    async def _generate_test_text_async(self, model: str, token_length_size: int) -> str:
        """Async wrapper for test text generation with timeout protection"""
        return self._generate_test_text(model, token_length_size)
    
    def _get_token_length_size(self, model: str, endpoint: str, endpoint_type: Optional[str]) -> int:
        """Get token length size for model/endpoint configuration"""
        try:
            # Check specific endpoint types for token length configuration
            if endpoint_type == "tei_endpoints" and hasattr(self, 'tei_endpoints'):
                for endpoint_info in getattr(self, 'tei_endpoints', []):
                    if isinstance(endpoint_info, (list, tuple)) and len(endpoint_info) >= 3:
                        if endpoint_info[0] == model and endpoint_info[1] == endpoint:
                            return round(endpoint_info[2] * 0.99)
                            
            elif endpoint_type == "openvino_endpoints" and hasattr(self, 'openvino_endpoints'):
                for endpoint_info in getattr(self, 'openvino_endpoints', []):
                    if isinstance(endpoint_info, (list, tuple)) and len(endpoint_info) >= 3:
                        if endpoint_info[0] == model and endpoint_info[1] == endpoint:
                            return round(endpoint_info[2] * 0.99)
                            
            elif endpoint_type == "local_endpoints" and hasattr(self, 'local_endpoints'):
                for endpoint_info in getattr(self, 'local_endpoints', []):
                    if isinstance(endpoint_info, (list, tuple)) and len(endpoint_info) >= 3:
                        if endpoint_info[0] == model and endpoint_info[1] == endpoint:
                            return round(endpoint_info[2] * 0.99)
                            
            elif endpoint_type == "libp2p_endpoints" and hasattr(self, 'libp2p_endpoints'):
                for endpoint_info in getattr(self, 'libp2p_endpoints', []):
                    if isinstance(endpoint_info, (list, tuple)) and len(endpoint_info) >= 3:
                        if endpoint_info[0] == model and endpoint_info[1] == endpoint:
                            return round(endpoint_info[2] * 0.99)
            
            # Check if we have endpoint status information
            if hasattr(self, 'endpoint_status') and endpoint in self.endpoint_status:
                status_value = self.endpoint_status[endpoint]
                if isinstance(status_value, (int, float)) and status_value > 0:
                    return round(status_value * 0.99)
            
            # Model-based fallbacks for common models
            if "gte-small" in model:
                return 512
            elif "gte-large" in model:
                return 8192
            elif "gte-Qwen2" in model or "Qwen2" in model:
                return 32768
            elif "bge-m3" in model:
                return 4096
            
            # Default fallback
            return 512
            
        except Exception as e:
            logger.warning(f"Could not determine token length size for {model}@{endpoint}: {e}")
            return 512
    
    def _generate_test_text(self, model: str, token_length_size: int) -> str:
        """Generate test text for batch size optimization"""
        try:
            # Initialize tokenizer if needed
            if model not in self.tokenizer:
                self.tokenizer[model] = {}
            if "cpu" not in self.tokenizer[model]:
                self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu')
            
            # Generate test tokens
            find_token_str = "z"  # Simple test character
            find_token_int = self.tokenizer[model]["cpu"].encode(find_token_str)
            
            # Extract token ID (handle different tokenizer outputs)
            if len(find_token_int) >= 2:
                token_id = find_token_int[1]
            else:
                token_id = find_token_int[0]
            
            # Create test tokens (use smaller size for memory efficiency)
            test_token_count = min(token_length_size, 512)  # Limit for memory efficiency
            test_tokens = [token_id] * test_token_count
            
            # Decode to text
            test_text = self.tokenizer[model]["cpu"].decode(test_tokens)
            return test_text
            
        except Exception as e:
            logger.warning(f"Could not generate test text for model {model}: {e}")
            # Return simple fallback text
            return "z" * 100
    
    async def save_chunks_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(60)
            if self.saved == False:
                if len(self.chunk_cache) > 0: 
                    for this_cid in list(self.chunk_cache.keys()):
                        this_chunk = self.chunk_cache[this_cid]
                        this_cid_dataset = datasets.Dataset.from_dict({"items":this_chunk["items"]})
                        this_cid_dataset.to_parquet(os.path.join(dst_path, "checkpoints", "sparse_chunks", this_cid + ".parquet"))
                        print("Saved " + str(len(this_cid_dataset)) + " chunks to disk for CID " + this_cid + " at " + dst_path)
                        self.cid_chunk_set.add(this_cid)
                        self.cid_chunk_list.append(this_cid)
                        del self.chunk_cache[this_cid]
                        del this_cid_dataset
                    self.saved = True
        return None

    async def index_knn(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        query_response = None  # Initialize query_response variable
        
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is list or type(samples) is iter:
            this_query = {"inputs": samples}
            all_endpoints = { "tei_endpoints":  [ x[2] for x in self.tei_endpoints ], "openvino_endpoints":  [ x[2] for x in self.openvino_endpoints ], "libp2p_endpoints":  [ x[2] for x in self.libp2p_endpoints ], "local_endpoints": [ x[2] for x in self.local_endpoints ] }
            if len(all_endpoints["local_endpoints"]) > 0 and "cuda" in chosen_endpoint or "cpu" in chosen_endpoint:
                if chosen_endpoint is None:
                    if "chosen_endpoint" not in list(dir(self)) or self.chosen_local_endpoint is None or self.chosen_local_endpoint_model != model:
                        self.chosen_local_endpoint_model = model
                        self.chosen_local_endpoint = AutoModel.from_pretrained(model)
                        if model not in self.tokenizer.keys():
                            self.tokenizer[model] = {}
                        if "cpu" not in self.tokenizer[model].keys():
                            self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True)
                    chosen_endpoint = self.chosen_local_endpoint
                    chosen_endpoint.eval()
                    inputs = self.tokenizer[model]["cpu"](samples, return_tensors="pt")
                    with torch.no_grad():
                        output = chosen_endpoint(**inputs).last_hidden_state.mean(dim=1).tolist()
                        query_response = output[0]
            if len(all_endpoints["tei_endpoints"]) > 0 and "/embed" in chosen_endpoint and "cuda" not in chosen_endpoint and "cpu" not in chosen_endpoint:
                try:
                    query_response = await self.make_post_request(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            if len(all_endpoints["openvino_endpoints"]) > 0 and "/infer" in chosen_endpoint and "cuda" not in chosen_endpoint and "cpu" not in chosen_endpoint:
                try:
                    query_response = await self.make_post_request_openvino(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            if len(all_endpoints["libp2p_endpoints"]) > 0 and "/infer" not in chosen_endpoint and "/embed" not in chosen_endpoint and "cuda" not in chosen_endpoint and "cpu" not in chosen_endpoint:
                try:
                    query_response = await self.make_post_request_libp2p(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        return knn_stack
    
    async def index_knn_openvino(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        query_response = None  # Initialize query_response variable
        
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is list or type(samples) is iter:
            this_query = {"inputs": samples}
            if "cuda" in chosen_endpoint or "cpu" in chosen_endpoint:
                if model not in self.local_endpoints.keys():
                    self.local_endpoints[model] = {}
                if model not in self.tokenizer.keys():
                    self.tokenizer[model] = {}
                if chosen_endpoint not in self.local_endpoints[model].keys():
                    self.local_endpoints[model][chosen_endpoint] = AutoModel.from_pretrained(model, device=chosen_endpoint)
                    self.tokenizer[model][chosen_endpoint] = AutoTokenizer.from_pretrained(model, device=chosen_endpoint, use_fast=True)
                query_response = await self.make_local_request(model, chosen_endpoint, samples)
                knn_stack = query_response
                return knn_stack
            else:
                try:
                    if model not in self.tokenizer.keys():
                        self.tokenizer[model] = {}
                    if "cpu" not in self.tokenizer[model].keys():
                        self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True)
                        pass
                    if len(samples) > 1:
                        raise ValueError("samples must be a list of one item")
                    inputs = []
                    for sample in samples:
                        max_length = 0
                        for resource in self.resources["tei_endpoints"]:
                            if model in resource and chosen_endpoint in resource:                            
                                max_length = resource[2]
                        input = self.tokenizer[model]["cpu"](sample, max_length=max_length, truncation=True, return_tensors='pt')
                        for item in list(input.keys()):
                            data = input[item].tolist()
                            data_len = len(data[0])
                            this_input = {
                                "name": item,
                                "shape": [1, data_len],
                                "datatype": "INT64",
                                "data": data
                            }
                            inputs.append(this_input)
                    data = {"inputs": inputs}
                    query_response = await self.make_post_request_openvino(chosen_endpoint, data)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            
            if isinstance(query_response, ValueError):
                # If query_response is a ValueError, return it directly
                return query_response
            elif isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            elif isinstance(query_response, dict) and "outputs" in query_response:
                query_response_outputs = query_response["outputs"]
                data = query_response_outputs[0]
                vectors = data["data"]
                knn_stack = [vectors]
            else:
                # Handle unexpected response format
                logger.warning(f"Unexpected query_response format: {type(query_response)}")
                knn_stack = []
            pass
        return knn_stack
    
    async def make_post_request(self, endpoint, data):
        """Make HTTP POST request with enhanced timeout protection"""
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=NETWORK_REQUEST_TIMEOUT)  # Use our defined timeout constant
        
        try:
            return await safe_async_execute_with_timeout(
                self._execute_post_request(endpoint, data, headers, timeout),
                timeout=NETWORK_REQUEST_TIMEOUT + 10,  # Add buffer for overall operation
                operation_name=f"post_request_to_{endpoint}"
            )
        except BatchSizeTimeoutError as e:
            logger.error(f"POST request to {endpoint} timed out: {e}")
            return ValueError(f"Request timeout: {e}")
        except Exception as e:
            logger.error(f"POST request to {endpoint} failed: {e}")
            return ValueError(f"Request failed: {e}")

    async def _execute_post_request(self, endpoint, data, headers, timeout):
        """Execute the actual POST request with proper error handling"""
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status} from {endpoint}"
                        logger.error(error_msg)
                        return ValueError(error_msg)
                    return await response.json()
            except aiohttp.ClientPayloadError as e:
                error_msg = f"ClientPayloadError: {str(e)}"
                logger.error(f"Payload error for {endpoint}: {error_msg}")
                return ValueError(error_msg)
            except asyncio.TimeoutError as e:
                error_msg = f"Timeout error: {str(e)}"
                logger.error(f"Timeout for {endpoint}: {error_msg}")
                return ValueError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error for {endpoint}: {error_msg}")
                if "Can not write request body" in str(e):
                    logger.warning(f"Endpoint {endpoint} is not accepting requests")
                return ValueError(error_msg)

    async def make_post_request_openvino(self, endpoint, data):
        """Make HTTP POST request to OpenVINO endpoint with enhanced timeout protection"""
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=NETWORK_REQUEST_TIMEOUT)  # Use our defined timeout constant
        
        try:
            return await safe_async_execute_with_timeout(
                self._execute_openvino_request(endpoint, data, headers, timeout),
                timeout=NETWORK_REQUEST_TIMEOUT + 10,  # Add buffer for overall operation
                operation_name=f"openvino_request_to_{endpoint}"
            )
        except BatchSizeTimeoutError as e:
            logger.error(f"OpenVINO request to {endpoint} timed out: {e}")
            return ValueError(f"OpenVINO request timeout: {e}")
        except Exception as e:
            logger.error(f"OpenVINO request to {endpoint} failed: {e}")
            return ValueError(f"OpenVINO request failed: {e}")

    async def _execute_openvino_request(self, endpoint, data, headers, timeout):
        """Execute the actual OpenVINO POST request with proper error handling"""
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status} from OpenVINO endpoint {endpoint}"
                        logger.error(error_msg)
                        return ValueError(error_msg)
                    return await response.json()
            except aiohttp.ClientPayloadError as e:
                error_msg = f"ClientPayloadError: {str(e)}"
                logger.error(f"Payload error for OpenVINO {endpoint}: {error_msg}")
                return ValueError(error_msg)
            except asyncio.TimeoutError as e:
                error_msg = f"Timeout error: {str(e)}"
                logger.error(f"Timeout for OpenVINO {endpoint}: {error_msg}")
                return ValueError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error for OpenVINO {endpoint}: {error_msg}")
                if "Can not write request body" in str(e):
                    logger.warning(f"OpenVINO endpoint {endpoint} is not accepting requests")
                return ValueError(error_msg)

    async def request_llama_cpp_endpoint(self, model, endpoint, endpoint_type):
        """Request llama.cpp endpoint with timeout protection"""
        try:
            # Prepare request data
            request_data = {
                "model": model,
                "endpoint_type": endpoint_type,
                "timestamp": time.time()
            }
            
            # Use timeout protection for the request
            result = await safe_async_execute_with_timeout(
                self._execute_llama_cpp_request(endpoint, request_data),
                timeout=NETWORK_REQUEST_TIMEOUT,
                operation_name=f"llama_cpp_request_to_{endpoint}"
            )
            
            return result
            
        except BatchSizeTimeoutError as e:
            logger.error(f"Llama.cpp endpoint request timed out: {e}")
            return {"error": f"Request timeout: {e}"}
        except Exception as e:
            logger.error(f"Llama.cpp endpoint request failed: {e}")
            return {"error": f"Request failed: {e}"}

    async def _execute_llama_cpp_request(self, endpoint, data):
        """Execute llama.cpp request with proper error handling"""
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=NETWORK_REQUEST_TIMEOUT)
        
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status} from {endpoint}"
                        logger.error(error_msg)
                        return {"error": error_msg}
                    return await response.json()
            except Exception as e:
                error_msg = f"Llama.cpp request error: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

    async def make_post_request_libp2p(self, endpoint, data):
        """Make HTTP POST request to libp2p endpoint with timeout protection"""
        try:
            # Use timeout protection for libp2p requests
            result = await safe_async_execute_with_timeout(
                self._execute_libp2p_request(endpoint, data),
                timeout=NETWORK_REQUEST_TIMEOUT,
                operation_name=f"libp2p_request_to_{endpoint}"
            )
            
            return result
            
        except BatchSizeTimeoutError as e:
            logger.error(f"Libp2p request timed out: {e}")
            return {"error": f"Libp2p request timeout: {e}"}
        except Exception as e:
            logger.error(f"Libp2p request failed: {e}")
            return {"error": f"Libp2p request failed: {e}"}

    async def _execute_libp2p_request(self, endpoint, data):
        """Execute libp2p request with proper error handling"""
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=NETWORK_REQUEST_TIMEOUT)
        
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status} from {endpoint}"
                        logger.error(error_msg)
                        return {"error": error_msg}
                    return await response.json()
            except Exception as e:
                error_msg = f"Libp2p request error: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

    async def make_local_request(self, model, endpoint, samples):
        """Make local request with timeout protection"""
        try:
            # Prepare local request data
            request_data = {
                "model": model,
                "samples": samples,
                "timestamp": time.time()
            }
            
            # Use timeout protection for local requests
            result = await safe_async_execute_with_timeout(
                self._execute_local_request(endpoint, request_data),
                timeout=NETWORK_REQUEST_TIMEOUT,
                operation_name=f"local_request_to_{endpoint}"
            )
            
            return result
            
        except BatchSizeTimeoutError as e:
            logger.error(f"Local request timed out: {e}")
            return {"error": f"Local request timeout: {e}", "outputs": []}
        except Exception as e:
            logger.error(f"Local request failed: {e}")
            return {"error": f"Local request failed: {e}", "outputs": []}

    async def _execute_local_request(self, endpoint, data):
        """Execute local request with proper error handling"""
        try:
            # For local requests, we can simulate processing or call local services
            # This is a placeholder implementation that should be customized based on actual local service
            if "local" in endpoint.lower():
                # Simulate local processing
                await asyncio.sleep(0.1)  # Simulate processing time
                return {
                    "status": "success",
                    "outputs": data.get("samples", []),
                    "processed_at": time.time()
                }
            else:
                # For HTTP local endpoints
                headers = {'Content-Type': 'application/json'}
                timeout = ClientTimeout(total=NETWORK_REQUEST_TIMEOUT)
                
                async with ClientSession(timeout=timeout) as session:
                    async with session.post(endpoint, headers=headers, json=data) as response:
                        if response.status != 200:
                            error_msg = f"HTTP {response.status} from {endpoint}"
                            logger.error(error_msg)
                            return {"error": error_msg, "outputs": []}
                        return await response.json()
                        
        except Exception as e:
            error_msg = f"Local request processing error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "outputs": []}
    
    def safe_dataset_column_names(self, dataset):
        """Safely get column names from dataset"""
        if dataset is None:
            return []
        
        if hasattr(dataset, 'column_names'):
            column_names = dataset.column_names
            if isinstance(column_names, dict):
                # For DatasetDict, return combined column names
                all_columns = set()
                for split_columns in column_names.values():
                    if isinstance(split_columns, list):
                        all_columns.update(split_columns)
                return list(all_columns)
            elif isinstance(column_names, list):
                return column_names
        
        # Fallback: try to get columns from first item
        try:
            first_item = next(iter(dataset))
            if isinstance(first_item, dict):
                return list(first_item.keys())
        except Exception:
            pass
        
        return []
    
    def add_https_endpoint(self, model, endpoint, ctx_length):
        """Add HTTPS endpoint (typically TEI endpoint) to the system"""
        if model not in self.tei_endpoints:
            self.tei_endpoints[model] = {}
        self.tei_endpoints[model][endpoint] = ctx_length
        # Initialize endpoint status with ctx_length as max batch size
        self.endpoint_status[endpoint] = ctx_length
        return None

    def add_tei_endpoint(self, model, endpoint, ctx_length):
        """Add TEI endpoint to the system"""
        if model not in self.tei_endpoints:
            self.tei_endpoints[model] = {}
        self.tei_endpoints[model][endpoint] = ctx_length
        # Initialize endpoint status with ctx_length as max batch size
        self.endpoint_status[endpoint] = ctx_length
        return None

    def add_endpoint(self, model, endpoint, context_length, endpoint_type):
        """Generic method to add endpoint of any type"""
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if endpoint_type not in list(dir(self)):
                    self.__dict__[endpoint_type] = {}
                if model not in list(self.__dict__[endpoint_type].keys()):
                    self.__dict__[endpoint_type][model] = {}
                if endpoint not in list(self.__dict__[endpoint_type][model].keys()):
                    self.__dict__[endpoint_type][model][endpoint] = context_length
                self.endpoint_status[endpoint] = context_length
                success = True
            except Exception as e:
                print(e)
                pass
            return success        
        return None

    def rm_endpoint(self, model, endpoint, endpoint_type):
        """Generic method to remove endpoint of any type"""
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if model in self.__dict__[endpoint_type] and endpoint in self.__dict__[endpoint_type][model]:
                    del self.__dict__[endpoint_type][model][endpoint]
                if endpoint in self.endpoint_status:
                    del self.endpoint_status[endpoint]
                success = True
            except Exception as e:
                print(e)
                pass
            return success
        return None
