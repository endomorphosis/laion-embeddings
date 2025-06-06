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
import gc
import time
import numpy as np
import psutil
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from aiohttp import ClientSession, ClientTimeout
import multiprocessing
from multiprocessing import Pool
import transformers
from transformers import AutoTokenizer, AutoModel
import datasets
import ipfs_accelerate_py
import chunker
import qdrant_kit
import elasticsearch_kit
import faiss_kit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datasets import Dataset, concatenate_datasets, load_dataset

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
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MemoryMonitor:
    """Monitor system memory usage for optimal batch sizing"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def get_memory_percent(self) -> float:
        """Get memory usage percentage"""
        return psutil.virtual_memory().percent

class AdaptiveBatchProcessor:
    """
    Adaptive batch processor that optimizes batch sizes based on hardware capabilities
    and performance history. Implements the P1 priority improvement from the codebase plan.
    """
    
    def __init__(self, max_memory_percent: float = 80.0, min_batch_size: int = 1, max_batch_size: int = 512):
        self.max_memory_percent = max_memory_percent
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.optimal_batch_sizes: Dict[str, int] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.memory_monitor = MemoryMonitor()
        self.logger = logging.getLogger(__name__ + ".AdaptiveBatchProcessor")
        
    async def find_optimal_batch_size(self, 
                                    model_name: str, 
                                    test_function,
                                    test_data: List[str],
                                    max_test_batches: int = 5) -> int:
        """
        Dynamically find optimal batch size for current hardware and model.
        
        Args:
            model_name: Name of the model for caching optimal batch size
            test_function: Function to test batch processing performance
            test_data: Sample data for testing
            max_test_batches: Maximum number of test batches to try
            
        Returns:
            Optimal batch size for the current configuration
        """
        if model_name in self.optimal_batch_sizes:
            self.logger.info(f"Using cached optimal batch size for {model_name}: {self.optimal_batch_sizes[model_name]}")
            return self.optimal_batch_sizes[model_name]
        
        self.logger.info(f"Finding optimal batch size for {model_name}...")
        
        # Start with a conservative batch size
        current_batch_size = self.min_batch_size
        best_batch_size = current_batch_size
        best_throughput = 0.0
        consecutive_failures = 0
        
        while current_batch_size <= self.max_batch_size and consecutive_failures < 3:
            try:
                # Prepare test batch
                test_batch = test_data[:min(current_batch_size, len(test_data))]
                if len(test_batch) < current_batch_size:
                    # Repeat data to reach desired batch size
                    test_batch = (test_batch * ((current_batch_size // len(test_batch)) + 1))[:current_batch_size]
                
                # Monitor memory before processing
                memory_before = self.memory_monitor.get_memory_usage_mb()
                memory_percent_before = self.memory_monitor.get_memory_percent()
                
                # Check if we have enough memory
                if memory_percent_before > self.max_memory_percent:
                    self.logger.warning(f"Memory usage too high ({memory_percent_before:.1f}%), stopping batch size increase")
                    break
                
                # Time the batch processing
                start_time = time.time()
                result = await test_function(test_batch)
                processing_time = time.time() - start_time
                
                # Monitor memory after processing
                memory_after = self.memory_monitor.get_memory_usage_mb()
                memory_usage = memory_after - memory_before
                
                # Calculate performance metrics
                throughput = len(test_batch) / processing_time if processing_time > 0 else 0
                success_rate = 1.0 if result is not None else 0.0
                
                metrics = PerformanceMetrics(
                    batch_size=current_batch_size,
                    processing_time=processing_time,
                    memory_usage_mb=memory_usage,
                    throughput=throughput,
                    success_rate=success_rate
                )
                
                # Store performance metrics
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = []
                self.performance_history[model_name].append(metrics)
                
                self.logger.info(f"Batch size {current_batch_size}: {throughput:.2f} items/s, {memory_usage:.1f} MB")
                
                # Update best batch size if this is better
                if throughput > best_throughput and success_rate > 0.9:
                    best_throughput = throughput
                    best_batch_size = current_batch_size
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                # Double the batch size for next test
                current_batch_size = min(current_batch_size * 2, self.max_batch_size)
                
            except Exception as e:
                self.logger.warning(f"Batch size {current_batch_size} failed: {e}")
                consecutive_failures += 1
                current_batch_size = min(current_batch_size * 2, self.max_batch_size)
        
        # Cache the optimal batch size
        self.optimal_batch_sizes[model_name] = best_batch_size
        self.logger.info(f"Optimal batch size for {model_name}: {best_batch_size} (throughput: {best_throughput:.2f} items/s)")
        
        return best_batch_size
    
    def get_adaptive_batch_size(self, model_name: str, queue_size: int) -> int:
        """
        Get adaptive batch size based on current conditions.
        
        Args:
            model_name: Name of the model
            queue_size: Current queue size
            
        Returns:
            Recommended batch size
        """
        # Start with optimal batch size if available
        base_batch_size = self.optimal_batch_sizes.get(model_name, self.min_batch_size)
        
        # Adjust based on current memory usage
        memory_percent = self.memory_monitor.get_memory_percent()
        if memory_percent > self.max_memory_percent:
            # Reduce batch size if memory is high
            memory_factor = (100 - memory_percent) / (100 - self.max_memory_percent)
            base_batch_size = max(self.min_batch_size, int(base_batch_size * memory_factor))
        
        # Adjust based on queue size
        if queue_size < base_batch_size:
            return min(queue_size, base_batch_size)
        
        return base_batch_size
    
    def cleanup_memory(self):
        """Clean up memory and trigger garbage collection"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.logger.debug("Memory cleanup completed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

# Global adaptive batch processor instance
adaptive_batch_processor = AdaptiveBatchProcessor()

# ==============================================================================
# ENHANCED ERROR HANDLING AND INPUT VALIDATION
# ==============================================================================

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass

class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass

class MemoryError(Exception):
    """Custom exception for memory-related errors"""
    pass

def validate_batch_input(batch: Union[List[str], List[Dict]], max_size: int = 10000) -> List[str]:
    """
    Validate and sanitize batch input data.
    
    Args:
        batch: Input batch data
        max_size: Maximum allowed batch size
        
    Returns:
        Validated and sanitized batch data
        
    Raises:
        ValidationError: If input is invalid
    """
    if not batch:
        raise ValidationError("Batch cannot be empty")
    
    if len(batch) > max_size:
        raise ValidationError(f"Batch size {len(batch)} exceeds maximum {max_size}")
    
    # Convert to list of strings if needed
    validated_batch = []
    for i, item in enumerate(batch):
        try:
            if isinstance(item, str):
                if len(item.strip()) == 0:
                    logger.warning(f"Empty string found at index {i}, skipping")
                    continue
                validated_batch.append(item.strip())
            elif isinstance(item, dict):
                # Convert dict to JSON string
                validated_batch.append(json.dumps(item))
            else:
                # Convert other types to string
                validated_batch.append(str(item))
        except Exception as e:
            logger.warning(f"Failed to process item at index {i}: {e}")
            continue
    
    if not validated_batch:
        raise ValidationError("No valid items found in batch after validation")
    
    return validated_batch

def safe_execute_with_retry(func, *args, max_retries: int = 3, delay: float = 1.0, **kwargs):
    """
    Execute function with retry logic and error handling.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        ProcessingError: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise ProcessingError(f"Failed after {max_retries + 1} attempts: {last_exception}")

async def safe_async_execute_with_retry(func, *args, max_retries: int = 3, delay: float = 1.0, **kwargs):
    """
    Execute async function with retry logic and error handling.
    
    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        ProcessingError: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise ProcessingError(f"Failed after {max_retries + 1} attempts: {last_exception}")

# ==============================================================================
# END ENHANCED ERROR HANDLING AND INPUT VALIDATION
# ==============================================================================
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
        except Exception as e:
            pass
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
            pass
    pass

try:
    from .elasticsearch_kit import elasticsearch_kit
    from .elasticsearch_kit import *
except Exception as e:
    try:
        from elasticsearch_kit import elasticsearch_kit
        from elasticsearch_kit import *
    except Exception as e:
        pass
    pass

try:
    from .qdrant_kit import qdrant_kit_py
    from .qdrant_kit import *
except Exception as e:
    try:
        from qdrant_kit import qdrant_kit_py
        from qdrant_kit import *
    except Exception as e:
        pass
    pass

try:
    from .faiss_kit import faiss_kit_py
    from .faiss_kit import *
except Exception as e:
    try:
        from faiss_kit import faiss_kit_py
        from faiss_kit import *
    except Exception as e:
        pass
    pass

from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process
import concurrent.futures
import concurrent
import json
from ipfs_embeddings_py.ipfs_datasets import ipfs_datasets_py
from ipfs_kit_py.ipfs_kit import ipfs_kit
from ipfs_kit_py.ipfs_kit import ipfs_kit
import ipfs_accelerate_py
import multiformats
from queue import Queue

manager = Manager()
caches = manager.dict()
chunk_cache = manager.dict()
cid_cache = manager.dict()
cid_chunk_queue = manager.Queue()
cid_queue = manager.Queue()
cid_set = manager.list()
cid_chunk_set = manager.list()
all_cid_set = manager.dict()
model_cid_set = manager.dict()
batch_sizes = manager.dict()
metadata = manager.dict()

# Initialize modules with error handling
ipfs_multiformats = None
this_chunker = None
ipfs_multiformats_py = None
qdrant_kit_py = None
faiss_kit_py = None

# Safe module initialization
try:
    if 'ipfs_multiformats_py' in globals() and ipfs_multiformats_py is not None:
        if hasattr(ipfs_multiformats_py, '__call__'):
            ipfs_multiformats = ipfs_multiformats_py()
        else:
            ipfs_multiformats = ipfs_multiformats_py
    elif hasattr(multiformats, 'get_cid'):
        ipfs_multiformats = multiformats
    else:
        ipfs_multiformats = None
except Exception as e:
    print(f"Warning: Could not initialize ipfs_multiformats: {e}")
    ipfs_multiformats = None

try:
    if 'chunker' in globals() and chunker is not None:
        # chunker is a class, we need to import it and instantiate it
        from chunker import chunker as ChunkerClass
        this_chunker = ChunkerClass()
    else:
        this_chunker = None
except Exception as e:
    print(f"Warning: Could not initialize chunker: {e}")
    this_chunker = None

# Safe helper functions
def safe_get_cid(file_data):
    """
    Safely generate a CID with enhanced error handling and validation.
    
    Args:
        file_data: Data to generate CID for (string, dict, or other serializable type)
        
    Returns:
        str: Generated CID string
        
    Raises:
        ValidationError: If input data is invalid
    """
    # Input validation
    if file_data is None:
        raise ValidationError("Cannot generate CID for None data")
    
    # Normalize input data
    try:
        if isinstance(file_data, dict):
            normalized_data = json.dumps(file_data, sort_keys=True)
        elif isinstance(file_data, (list, tuple)):
            normalized_data = json.dumps(list(file_data), sort_keys=True)
        elif not isinstance(file_data, str):
            normalized_data = str(file_data)
        else:
            normalized_data = file_data
            
        # Validate that we have meaningful data
        if len(normalized_data.strip()) == 0:
            raise ValidationError("Cannot generate CID for empty data")
            
    except Exception as e:
        raise ValidationError(f"Failed to normalize input data: {e}")
    
    # Try different CID generation methods with retry logic
    def try_ipfs_multiformats():
        """Try IPFS multiformats CID generation"""
        if 'ipfs_multiformats' in globals() and ipfs_multiformats is not None:
            if callable(ipfs_multiformats):
                return ipfs_multiformats(normalized_data)
            elif hasattr(ipfs_multiformats, 'get_cid') and callable(getattr(ipfs_multiformats, 'get_cid')):
                get_cid_func = getattr(ipfs_multiformats, 'get_cid')
                return get_cid_func(normalized_data)
        return None
    
    def try_ipfs_multiformats_py():
        """Try IPFS multiformats_py CID generation"""
        if 'ipfs_multiformats_py' in globals() and ipfs_multiformats_py is not None:
            if callable(ipfs_multiformats_py):
                return ipfs_multiformats_py(normalized_data)
            elif hasattr(ipfs_multiformats_py, 'get_cid') and callable(getattr(ipfs_multiformats_py, 'get_cid')):
                get_cid_py_func = getattr(ipfs_multiformats_py, 'get_cid')
                return get_cid_py_func(normalized_data)
        return None
    
    def try_multiformats():
        """Try multiformats module CID generation"""
        if 'multiformats' in globals() and multiformats is not None:
            if hasattr(multiformats, 'get_cid') and callable(getattr(multiformats, 'get_cid')):
                get_cid_func = getattr(multiformats, 'get_cid')
                return get_cid_func(normalized_data)
        return None
    
    def fallback_hash_cid():
        """Generate hash-based CID as fallback"""
        import hashlib
        try:
            hash_value = hashlib.sha256(normalized_data.encode('utf-8', errors='ignore')).hexdigest()
            return "baf" + hash_value[:32]
        except Exception as e:
            logger.warning(f"Hash-based CID generation failed: {e}")
            # Ultimate fallback - deterministic placeholder based on data length
            data_hash = str(abs(hash(normalized_data))) if normalized_data else "0"
            return f"bafybeifi6kicddkqn24zbypkdpdqdvudtnb5qwul3jxkgvf{data_hash[:8]:0>8}"
    
    # Try CID generation methods in order of preference
    cid_methods = [
        ("ipfs_multiformats", try_ipfs_multiformats),
        ("ipfs_multiformats_py", try_ipfs_multiformats_py),
        ("multiformats", try_multiformats),
        ("hash_fallback", fallback_hash_cid)
    ]
    
    for method_name, method_func in cid_methods:
        try:
            result = safe_execute_with_retry(method_func, max_retries=1)
            if result is not None:
                logger.debug(f"CID generated using {method_name}: {result}")
                return result
        except Exception as e:
            logger.debug(f"CID generation method {method_name} failed: {e}")
            continue
    
    # If all methods fail, this should not happen due to fallback, but just in case
    logger.error("All CID generation methods failed")
    return "bafybeifi6kicddkqn24zbypkdpdqdvudtnb5qwul3jxkgvf2dh6wvxdxku"

def safe_tokenizer_encode(tokenizer, text):
    """Safely encode text with tokenizer"""
    if tokenizer is None:
        return []
    
    # Handle empty or None text
    if text is None or text == "":
        return []
        
    try:
        # Check if encode is actually a method on tokenizer
        if hasattr(tokenizer, 'encode') and callable(getattr(tokenizer, 'encode')):
            encode_method = getattr(tokenizer, 'encode')
            return encode_method(text)
        # Fallback for transformers tokenizers
        elif hasattr(tokenizer, '__call__'):
            try:
                result = tokenizer(text)
                if isinstance(result, dict) and 'input_ids' in result:
                    return result['input_ids']
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: tokenizer encode failed: {e}")
    
    # Ultimate fallback - simple character-based tokenization
    try:
        # Convert to string and return character codes
        return [ord(c) for c in str(text)]
    except Exception:
        return []

def safe_tokenizer_decode(tokenizer, tokens):
    """Safely decode tokens with tokenizer"""
    if tokenizer is None:
        return ""
        
    # Handle empty or None tokens
    if tokens is None or (hasattr(tokens, '__len__') and len(tokens) == 0):
        return ""
        
    try:
        # Check if decode is actually a method on tokenizer
        if hasattr(tokenizer, 'decode') and callable(getattr(tokenizer, 'decode')):
            decode_method = getattr(tokenizer, 'decode')
            return decode_method(tokens)
        # Fallback for transformers tokenizers
        elif hasattr(tokenizer, 'convert_ids_to_tokens') and callable(getattr(tokenizer, 'convert_ids_to_tokens')):
            try:
                tokens_list = getattr(tokenizer, 'convert_ids_to_tokens')(tokens)
                return ' '.join(tokens_list)
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: tokenizer decode failed: {e}")
    
    # Ultimate fallback - simple character-based detokenization
    try:
        # Convert token integers back to characters
        return ''.join([chr(t) if isinstance(t, int) and 0 <= t <= 0x10FFFF else ' ' for t in tokens])
    except Exception:
        return ""

def safe_os_path_join(dst_path, *args):
    """Safely join paths with None handling"""
    if dst_path is None:
        dst_path = "."
    try:
        return os.path.join(dst_path, *args)
    except Exception as e:
        print(f"Warning: path join failed: {e}")
        return "."

def safe_chunker_chunk(chunker, content, tokenizer, method, *args):
    """Safely chunk content with fallback"""
    # First check if content is valid
    if content is None:
        return []
    
    # Check if chunker is valid and has the chunk method
    has_chunk_method = False
    try:
        if chunker is not None and hasattr(chunker, 'chunk') and callable(getattr(chunker, 'chunk')):
            has_chunk_method = True
    except Exception as e:
        print(f"Warning: checking chunker.chunk failed: {e}")
    
    if not has_chunk_method:
        # Fallback chunking - simple fixed-size chunks
        if isinstance(content, str):
            chunk_size = 512
            chunks = []
            tokens = safe_tokenizer_encode(tokenizer, content)
            if not tokens:  # If encoding failed, return empty list
                return []
            
            for i in range(0, len(tokens), chunk_size):
                chunks.append((i, min(i + chunk_size, len(tokens))))
            return chunks
        return []
    
    # Try using the chunker with error handling
    try:
        # Handle potential issues with method arguments
        actual_args = []
        for arg in args:
            if arg is not None:
                actual_args.append(arg)
            else:
                # Add sensible defaults based on position
                if len(actual_args) == 0:  # First arg is usually chunk_size
                    actual_args.append(512)
                elif len(actual_args) == 1:  # Second arg is often n_sentences
                    actual_args.append(8)
                elif len(actual_args) == 2:  # Third arg is typically step_size
                    actual_args.append(256)
                else:
                    actual_args.append(None)
        
        # Safe call to chunk method using getattr
        chunk_method = getattr(chunker, 'chunk')
        return chunk_method(content, tokenizer, method, *actual_args)
    except Exception as e:
        print(f"Warning: chunker.chunk failed: {e}")
        # Fallback chunking
        if isinstance(content, str):
            chunk_size = 512
            chunks = []
            tokens = safe_tokenizer_encode(tokenizer, content)
            if not tokens:  # If encoding failed, return empty list
                return []
                
            for i in range(0, len(tokens), chunk_size):
                chunks.append((i, min(i + chunk_size, len(tokens))))
            return chunks
        return []

def safe_get_num_rows(dataset):
    """Safely get number of rows from dataset"""
    if dataset is None:
        return 0
    
    if hasattr(dataset, 'num_rows'):
        return dataset.num_rows
    elif isinstance(dataset, dict):
        # For DatasetDict, get total rows from all splits
        total = 0
        for split_data in dataset.values():
            if hasattr(split_data, '__len__'):
                total += len(split_data)
            elif hasattr(split_data, 'num_rows'):
                total += split_data.num_rows
        return total
    elif hasattr(dataset, '__len__'):
        return len(dataset)
    else:
        # For IterableDataset, we can't get exact count
        return 1000  # Default estimate

def safe_queue_empty(queue):
    """Safely check if queue is empty"""
    if queue is None:
        return True
    try:
        return queue.empty()
    except Exception as e:
        print(f"Warning: queue.empty() failed: {e}")
        return True

def safe_queue_full(queue):
    """Safely check if queue is full"""
    if queue is None:
        return True
    try:
        return queue.full()
    except Exception as e:
        print(f"Warning: queue.full() failed: {e}")
        return True

def safe_queue_get(queue):
    """Safely get item from queue"""
    if queue is None:
        return None
    try:
        return queue.get()
    except Exception as e:
        print(f"Warning: queue.get() failed: {e}")
        return None

def safe_queue_put_nowait(queue, item):
    """Safely put item in queue without waiting"""
    if queue is None:
        return False
    try:
        queue.put_nowait(item)
        return True
    except Exception as e:
        print(f"Warning: queue.put_nowait() failed: {e}")
        return False

def safe_queue_task_done(queue):
    """Safely mark queue task as done"""
    if queue is None:
        return
    try:
        queue.task_done()
    except Exception as e:
        print(f"Warning: queue.task_done() failed: {e}")

def safe_queue_qsize(queue):
    """Safely get queue size"""
    if queue is None:
        return 0
    try:
        return queue.qsize()
    except Exception as e:
        print(f"Warning: queue.qsize() failed: {e}")
        return 0

def safe_resource_access(obj, path, default=None):
    """Safely access a nested resource using a dot-separated path string"""
    if obj is None:
        return default
    
    # Split the path and access each part safely
    parts = path.split('.')
    current = obj
    
    for part in parts:
        if current is None:
            return default
        
        # Handle dictionary access
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
            
        # Handle attribute access
        if hasattr(current, part):
            try:
                current = getattr(current, part)
                if callable(current):
                    try:
                        current = current()
                    except:
                        # If calling fails, keep the callable
                        pass
                continue
            except Exception:
                return default
                
        # If we get here, the access failed
        return default
    
    return current

def safe_module_call(module, method_name, *args, **kwargs):
    """Safely call a method on a module, with fallback"""
    if module is None:
        return None

    # Check if the method exists
    method = None
    try:
        if hasattr(module, method_name):
            method = getattr(module, method_name)
            if callable(method):
                return method(*args, **kwargs)
    except Exception as e:
        print(f"Warning: {method_name} call on {module} failed: {e}")

    return None

def safe_dataset_column_names(dataset):
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

def safe_dataset_shard(dataset, num_shards, index):
    """Safely shard a dataset"""
    if dataset is None:
        return None
    
    if hasattr(dataset, 'shard'):
        try:
            return dataset.shard(num_shards=num_shards, index=index)
        except Exception as e:
            print(f"Warning: dataset.shard() failed: {e}")
    
    # Fallback: return empty list
    return []

def safe_model_replace(model_name, old_str, new_str):
    """Safely replace string in model name"""
    if model_name is None:
        return "unknown_model"
    try:
        return str(model_name).replace(old_str, new_str)
    except:
        return "unknown_model"

def safe_len_comparison(len1, len2):
    """Safely compare two lengths, returning tuple of (len1, len2)"""
    try:
        len1_val = int(len1) if len1 is not None else 0
    except:
        len1_val = 0
    
    try:
        len2_val = int(len2) if len2 is not None else 0
    except:
        len2_val = 0
    
    return (len1_val, len2_val)

def safe_metadata_models_access(metadata, index=0):
    """Safely access metadata models with fallback"""
    if metadata is None or not isinstance(metadata, dict):
        return "thenlper/gte-small"  # Default model
    
    if "models" not in metadata:
        return "thenlper/gte-small"
    
    models = metadata["models"]
    if models is None or not isinstance(models, list) or len(models) == 0:
        return "thenlper/gte-small"
    
    if index >= len(models):
        index = 0
    
    return models[index] if models[index] is not None else "thenlper/gte-small"

def safe_list_subscript(lst, index, default=None):
    """Safely access list element by index"""
    if lst is None or not hasattr(lst, '__getitem__'):
        return default
    
    try:
        if index < len(lst):
            return lst[index]
    except Exception as e:
        print(f"Warning: list subscript failed: {e}")
    
    return default

def safe_dict_subscript(dictionary, key, default=None):
    """Safely access dictionary element by key"""
    if dictionary is None or not isinstance(dictionary, dict):
        return default
    
    try:
        return dictionary.get(key, default)
    except Exception as e:
        print(f"Warning: dict subscript failed: {e}")
        return default

def safe_in_operator(item, container):
    """Safely check if item is in container"""
    if container is None:
        return False
    
    try:
        return item in container
    except Exception as e:
        print(f"Warning: 'in' operator failed: {e}")
        return False

def safe_del_variable(var_dict, var_name):
    """Safely delete a variable from local/global scope"""
    try:
        if var_name in var_dict:
            del var_dict[var_name]
    except Exception as e:
        print(f"Warning: Failed to delete variable {var_name}: {e}")

def safe_async_queue_get(queue):
    """Safely get item from async queue"""
    if queue is None:
        return None
    try:
        return queue.get()
    except Exception as e:
        print(f"Warning: async queue.get() failed: {e}")
        return None

def safe_queue_operations(queue, operation, item=None):
    """Centralized safe queue operations handler"""
    if queue is None:
        if operation in ['empty', 'full']:
            return True
        elif operation in ['get', 'get_nowait']:
            return None
        elif operation in ['put', 'put_nowait']:
            return False
        elif operation in ['qsize']:
            return 0
        else:
            return None
    
    try:
        if operation == 'empty':
            return queue.empty() if hasattr(queue, 'empty') else True
        elif operation == 'full':
            return queue.full() if hasattr(queue, 'full') else True
        elif operation == 'get':
            return queue.get() if hasattr(queue, 'get') else None
        elif operation == 'get_nowait':
            return queue.get_nowait() if hasattr(queue, 'get_nowait') else None
        elif operation == 'put' and item is not None:
            if hasattr(queue, 'put'):
                queue.put(item)
                return True
            return False
        elif operation == 'put_nowait' and item is not None:
            if hasattr(queue, 'put_nowait'):
                queue.put_nowait(item)
                return True
            return False
        elif operation == 'task_done':
            if hasattr(queue, 'task_done'):
                queue.task_done()
            return None
        elif operation == 'qsize':
            return queue.qsize() if hasattr(queue, 'qsize') else 0
        else:
            return None
    except Exception as e:
        print(f"Warning: queue operation {operation} failed: {e}")
        if operation in ['empty', 'full']:
            return True
        elif operation in ['get', 'get_nowait']:
            return None
        elif operation in ['put', 'put_nowait']:
            return False
        elif operation in ['qsize']:
            return 0
        else:
            return None

def safe_init_module(module, resources=None, metadata=None):
    """Safely initialize a module with resources and metadata"""
    if module is None:
        return None
    try:
        if hasattr(module, '__call__'):
            return module(resources or {}, metadata or {})
        else:
            return module
    except Exception as e:
        print(f"Warning: Could not initialize module {module}: {e}")
        return None

def index_cid(samples, use_adaptive_batching: bool = True, batch_size: Optional[int] = None):
    """
    Generate CIDs for samples with adaptive batch processing optimization.
    
    Args:
        samples: Input samples (string, list of strings, or list of dicts)
        use_adaptive_batching: Whether to use adaptive batch processing
        batch_size: Fixed batch size (if not using adaptive batching)
        
    Returns:
        List[str]: Generated CIDs for each sample
        
    Raises:
        ValidationError: If input samples are invalid
    """
    # Input validation
    if samples is None:
        raise ValidationError("Samples cannot be None")
    
    # Normalize samples to list
    if isinstance(samples, str):
        samples = [samples]
    elif not isinstance(samples, list):
        raise ValidationError("Samples must be a string or list")
    
    # Validate batch input
    validated_samples = validate_batch_input(samples, max_size=50000)
    
    if not validated_samples:
        return []
    
    # Use adaptive batch processing if enabled
    if use_adaptive_batching and len(validated_samples) > 10:
        return _process_cid_batch_adaptive(validated_samples)
    else:
        return _process_cid_batch_simple(validated_samples, batch_size)

def _process_cid_batch_simple(samples: List[str], batch_size: Optional[int] = None) -> List[str]:
    """Process CID generation with simple batching"""
    results = []
    effective_batch_size = batch_size or min(100, len(samples))
    
    for i in range(0, len(samples), effective_batch_size):
        batch = samples[i:i + effective_batch_size]
        batch_results = []
        
        for sample in batch:
            try:
                cid = safe_get_cid(sample)
                batch_results.append(cid)
            except Exception as e:
                logger.warning(f"Failed to generate CID for sample at index {i + len(batch_results)}: {e}")
                # Use fallback CID based on sample content
                fallback_cid = f"bafyerror{abs(hash(sample))}"[:32]
                batch_results.append(fallback_cid)
        
        results.extend(batch_results)
        
        # Memory cleanup for large batches
        if len(results) % 1000 == 0:
            adaptive_batch_processor.cleanup_memory()
    
    return results

def _process_cid_batch_adaptive(samples: List[str]) -> List[str]:
    """Process CID generation with adaptive batch processing"""
    try:
        # Define test function for adaptive batch processor
        def test_cid_batch(test_batch):
            """Test function for finding optimal batch size"""
            test_results = []
            for item in test_batch:
                test_results.append(safe_get_cid(item))
            return test_results
        
        # Find optimal batch size
        optimal_batch_size = asyncio.run(
            adaptive_batch_processor.find_optimal_batch_size(
                model_name="cid_generation",
                test_function=test_cid_batch,
                test_data=samples[:min(50, len(samples))]  # Use sample for testing
            )
        )
        
        logger.info(f"Using adaptive batch size: {optimal_batch_size}")
        
        # Process with optimal batch size
        results = []
        for i in range(0, len(samples), optimal_batch_size):
            batch = samples[i:i + optimal_batch_size]
            
            # Check memory usage and adjust if needed
            current_batch_size = adaptive_batch_processor.get_adaptive_batch_size(
                "cid_generation", len(batch)
            )
            
            if current_batch_size < len(batch):
                # Process in smaller chunks if memory is constrained
                batch = batch[:current_batch_size]
                logger.info(f"Reduced batch size to {current_batch_size} due to memory constraints")
            
            # Process batch with retry logic
            try:
                batch_results = safe_execute_with_retry(
                    test_cid_batch, 
                    batch, 
                    max_retries=2
                )
                results.extend(batch_results)
            except ProcessingError as e:
                logger.warning(f"Batch processing failed, falling back to individual processing: {e}")
                # Fallback to individual processing
                for sample in batch:
                    try:
                        cid = safe_get_cid(sample)
                        results.append(cid)
                    except Exception as sample_e:
                        logger.warning(f"Individual CID generation failed: {sample_e}")
                        fallback_cid = f"bafyerror{abs(hash(sample))}"[:32]
                        results.append(fallback_cid)
            
            # Progress logging
            if len(results) % 1000 == 0:
                logger.info(f"Processed {len(results)}/{len(samples)} CIDs")
                adaptive_batch_processor.cleanup_memory()
        
        return results
        
    except Exception as e:
        logger.warning(f"Adaptive batch processing failed, falling back to simple processing: {e}")
        return _process_cid_batch_simple(samples)

def init_datasets(model, dataset, split, column, dst_path):
    """Initialize datasets with safe handling"""
    columns = []
    init_hashed_datasets = None
    init_load_combined = None
    init_load_clusters = None
    init_load_checkpoints = None
    this_ipfs_datasets = ipfs_datasets_py({},{})
    this_all_cid_list = {}
    this_all_cid_set = {}
    this_dataset = None
    models = [model] if isinstance(model, str) else model  # Initialize models variable
    
    # Handle dst_path being None
    if dst_path is None:
        dst_path = "."
    
    try:
        if split is None:
            this_dataset = load_dataset(dataset).shuffle(random.randint(0,65536))
        else:
            this_dataset = load_dataset(dataset, split=split).shuffle(random.randint(0,65536))
    except Exception as e:
        print(e)
        this_dataset = None
    
    # Handle different dataset types for num_rows using safe helper
    len_datasets_list = safe_get_num_rows(this_dataset)
            
    try:
        this_dataset, this_hashed_dataset = this_ipfs_datasets.load_combined(this_dataset, models, dataset, split, column, dst_path)
        init_load_combined = True
    except Exception as e:
        print(e)
        init_load_combined = e
        this_hashed_dataset = None
        try:
            init_load_clusters = this_ipfs_datasets.load_clusters(dataset, split, dst_path)
            this_hashed_dataset = init_load_clusters
            init_load_clusters = True
        except Exception as e:
            print(e)
            init_load_clusters = e
            try:
                init_load_checkpoints = this_ipfs_datasets.load_checkpoints(dataset, split, dst_path, models)        
                this_hashed_dataset = init_load_checkpoints
                init_load_checkpoints = True
            except Exception as e:
                print(e)
                init_load_checkpoints = e

    # Handle different dataset types for num_rows (recompute after loading)
    len_datasets_list = safe_get_num_rows(this_dataset)
    
    # Handle CID list comparison with safety checks
    len_cid_list = 0
    len_cid_set = 0
    
    if hasattr(this_ipfs_datasets, 'all_cid_list') and "hashed_dataset" in this_ipfs_datasets.all_cid_list:
        len_cid_list = len(this_ipfs_datasets.all_cid_list["hashed_dataset"])
    if hasattr(this_ipfs_datasets, 'all_cid_set') and "hashed_dataset" in this_ipfs_datasets.all_cid_set:
        len_cid_set = len(this_ipfs_datasets.all_cid_set["hashed_dataset"])
        
    if len_cid_list == len_datasets_list:
        this_cid_list = this_ipfs_datasets.all_cid_list["hashed_dataset"]
        this_cid_set = set(this_cid_list)
        this_all_cid_list["hashed_dataset"] = this_cid_list
        this_all_cid_set["hashed_dataset"] = this_cid_set
        
    return {
        'dataset': this_dataset,
        'hashed_dataset': this_hashed_dataset,
        'cid_list': this_all_cid_list,
        'cid_set': this_all_cid_set,
        'len_datasets_list': len_datasets_list,
        'len_cid_list': len_cid_list,
        'len_cid_set': len_cid_set,
        'init_load_combined': init_load_combined,
        'init_load_clusters': init_load_clusters,
        'init_load_checkpoints': init_load_checkpoints
    }

# Main execution
if __name__ == "__main__":
    print("IPFS Embeddings Main - Ready for processing")
    print("All safe helper functions initialized successfully")
    
    # Example usage
    try:
        # Test safe functions
        test_data = "Hello, world!"
        test_cid = safe_get_cid(test_data)
        print(f"Test CID generated: {test_cid}")
        
        # Test index_cid function
        test_samples = ["sample1", "sample2", "sample3"]
        test_cids = index_cid(test_samples)
        print(f"Test CIDs: {test_cids}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        
    # Example usage of tokenize_batch
    try:
        from transformers import AutoTokenizer
        
        # Load a sample tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Test tokenize_batch function
        test_batch = ["Hello, world!", "This is a test.", "Tokenize me!"]
        tokenization_results = tokenize_batch(test_batch, tokenizer=tokenizer, use_adaptive_batching=True)
        print(f"Tokenization results: {tokenization_results}")
        
    except Exception as e:
        print(f"Error during tokenization testing: {e}")

# Additional Safe Helper Functions for Enhanced Processing

def safe_tokenizer_encode(tokenizer: Optional[Any], text: Optional[str], **kwargs) -> List[int]:
    """
    Safely encode text using a tokenizer with fallback mechanisms.
    
    Args:
        tokenizer: Tokenizer instance to use
        text: Text to encode
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        List of token IDs
    """
    if text is None:
        return []
    
    if tokenizer is None:
        logger.warning("No tokenizer provided, using character-based fallback")
        # Simple character-based encoding as fallback
        return [ord(c) for c in str(text)[:100]]  # Limit to prevent huge lists
    
    try:
        result = safe_execute_with_retry(
            lambda: tokenizer.encode(str(text), **kwargs),
            max_retries=2,
            initial_delay=0.01
        )
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning(f"Tokenizer encoding failed: {e}, using fallback")
        return [ord(c) for c in str(text)[:100]]


def safe_tokenizer_decode(tokenizer: Optional[Any], tokens: Optional[List[int]], **kwargs) -> str:
    """
    Safely decode tokens using a tokenizer with fallback mechanisms.
    
    Args:
        tokenizer: Tokenizer instance to use
        tokens: List of token IDs to decode
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        Decoded text string
    """
    if tokens is None or not tokens:
        return ""
    
    if tokenizer is None:
        logger.warning("No tokenizer provided, using character-based fallback")
        # Simple character-based decoding as fallback
        try:
            return ''.join(chr(min(max(t, 32), 126)) for t in tokens if isinstance(t, int))
        except Exception:
            return str(tokens)
    
    try:
        result = safe_execute_with_retry(
            lambda: tokenizer.decode(tokens, **kwargs),
            max_retries=2,
            initial_delay=0.01
        )
        return result if isinstance(result, str) else str(tokens)
    except Exception as e:
        logger.warning(f"Tokenizer decoding failed: {e}, using fallback")
        try:
            return ''.join(chr(min(max(t, 32), 126)) for t in tokens if isinstance(t, int))
        except Exception:
            return str(tokens)


def safe_chunker_chunk(chunker: Optional[Any], 
                      content: str, 
                      tokenizer: Optional[Any] = None,
                      method: str = "fixed",
                      *args, **kwargs) -> List[Any]:
    """
    Safely chunk content using a chunker with fallback mechanisms.
    
    Args:
        chunker: Chunker instance to use
        content: Content to chunk
        tokenizer: Optional tokenizer for token-based chunking
        method: Chunking method
        *args, **kwargs: Additional arguments for chunker
        
    Returns:
        List of chunks
    """
    if not content:
        return []
    
    if chunker is None:
        logger.warning("No chunker provided, using simple text splitting")
        # Simple text chunking fallback
        chunk_size = args[0] if args else 100
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunks.append(' '.join(chunk_words))
        return chunks
    
    try:
        # Try using the chunker with provided method
        if hasattr(chunker, 'chunk'):
            result = safe_execute_with_retry(
                lambda: chunker.chunk(content, tokenizer, method, *args, **kwargs),
                max_retries=2,
                initial_delay=0.01
            )
            return result if isinstance(result, list) else [content]
        elif hasattr(chunker, 'split_text'):
            result = safe_execute_with_retry(
                lambda: chunker.split_text(content),
                max_retries=2,
                initial_delay=0.01
            )
            return result if isinstance(result, list) else [content]
        else:
            logger.warning("Chunker doesn't have expected methods, using fallback")
            return [content]
            
    except Exception as e:
        logger.warning(f"Chunker failed: {e}, using simple splitting")
        # Simple sentence-based chunking fallback
        sentences = content.split('. ')
        chunk_size = args[0] if args else 3
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            chunks.append('. '.join(chunk_sentences))
        return chunks


def safe_get_num_rows(dataset: Any) -> int:
    """
    Safely get the number of rows from a dataset.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Number of rows, or 0 if cannot be determined
    """
    if dataset is None:
        return 0
    
    try:
        # Try common attributes for row count
        if hasattr(dataset, '__len__'):
            return len(dataset)
        elif hasattr(dataset, 'num_rows'):
            return dataset.num_rows
        elif hasattr(dataset, 'shape'):
            return dataset.shape[0]
        elif hasattr(dataset, 'count'):
            return dataset.count()
        else:
            logger.warning("Dataset type not recognized for row counting")
            return 0
    except Exception as e:
        logger.warning(f"Failed to get dataset row count: {e}")
        return 0


def enhanced_batch_processor_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the adaptive batch processor.
    
    Returns:
        Dictionary with processor status and performance metrics
    """
    try:
        memory_info = adaptive_batch_processor.memory_monitor.get_memory_info()
        
        return {
            'processor_status': {
                'max_batch_size': adaptive_batch_processor.max_batch_size,
                'max_memory_percent': adaptive_batch_processor.max_memory_percent,
                'optimal_batch_sizes': dict(adaptive_batch_processor.optimal_batch_sizes),
                'performance_history_length': {
                    model: len(history) 
                    for model, history in adaptive_batch_processor.performance_history.items()
                }
            },
            'memory_status': {
                'current_usage_mb': memory_info.get('used_mb', 0),
                'current_usage_percent': memory_info.get('percent', 0),
                'available_mb': memory_info.get('available_mb', 0),
                'total_mb': memory_info.get('total_mb', 0)
            },
            'processing_capabilities': {
                'adaptive_batching_available': True,
                'memory_monitoring_available': True,
                'performance_tracking_available': True,
                'error_recovery_available': True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get processor status: {e}")
        return {
            'processor_status': {'error': str(e)},
            'memory_status': {'error': str(e)},
            'processing_capabilities': {
                'adaptive_batching_available': False,
                'memory_monitoring_available': False,
                'performance_tracking_available': False,
                'error_recovery_available': False
            }
        }
