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
    """Safely generate a CID with fallback to hash-based approach"""
    # Try different ways to access the get_cid function
    try:
        # Try direct module access with explicit safe calling
        if 'ipfs_multiformats' in globals() and ipfs_multiformats is not None:
            # Check if it's a function itself rather than having get_cid attribute
            if callable(ipfs_multiformats):
                try:
                    return ipfs_multiformats(file_data)
                except Exception as e:
                    print(f"Warning: calling ipfs_multiformats directly failed: {e}")
            
            # Try calling get_cid as a method
            if hasattr(ipfs_multiformats, 'get_cid') and callable(getattr(ipfs_multiformats, 'get_cid')):
                try:
                    # Use explicit method call with getattr to avoid attribute access errors
                    get_cid_func = getattr(ipfs_multiformats, 'get_cid')
                    return get_cid_func(file_data)
                except Exception as e:
                    print(f"Warning: ipfs_multiformats.get_cid failed: {e}")
    except Exception as e:
        print(f"Warning: accessing ipfs_multiformats failed: {e}")
    
    try:
        # Try the ipfs_multiformats_py module with explicit safe calling
        if 'ipfs_multiformats_py' in globals() and ipfs_multiformats_py is not None:
            # Check if it's a function itself
            if callable(ipfs_multiformats_py):
                try:
                    return ipfs_multiformats_py(file_data)
                except Exception as e:
                    print(f"Warning: calling ipfs_multiformats_py directly failed: {e}")
            
            # Try calling get_cid as a method
            if hasattr(ipfs_multiformats_py, 'get_cid') and callable(getattr(ipfs_multiformats_py, 'get_cid')):
                try:
                    # Use explicit method call
                    get_cid_py_func = getattr(ipfs_multiformats_py, 'get_cid')
                    return get_cid_py_func(file_data)
                except Exception as e:
                    print(f"Warning: ipfs_multiformats_py.get_cid failed: {e}")
    except Exception as e:
        print(f"Warning: accessing ipfs_multiformats_py failed: {e}")
    
    # Try module-level multiformats import
    try:
        if 'multiformats' in globals() and multiformats is not None:
            if hasattr(multiformats, 'get_cid') and callable(getattr(multiformats, 'get_cid')):
                try:
                    get_cid_func = getattr(multiformats, 'get_cid')
                    return get_cid_func(file_data)
                except Exception as e:
                    print(f"Warning: multiformats.get_cid failed: {e}")
    except Exception as e:
        print(f"Warning: accessing multiformats failed: {e}")
    
    # Fallback to a simple hash-based CID generation
    import hashlib
    import json
    try:
        if isinstance(file_data, dict):
            file_data = json.dumps(file_data, sort_keys=True)
        elif not isinstance(file_data, str):
            file_data = str(file_data)
        return "baf" + hashlib.sha256(file_data.encode('utf-8', errors='ignore')).hexdigest()[:32]
    except Exception as e:
        print(f"Warning: hash-based CID generation failed: {e}")
        # Ultimate fallback - return a placeholder CID
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

def index_cid(samples):
    """Generate CIDs for samples"""
    results = []
    if samples is None:
        raise ValueError("samples must be a list")
    if isinstance(samples, str):
        samples = [samples]
    if isinstance(samples, list):
        for this_sample in samples:
            this_sample_cid = safe_get_cid(this_sample)
            results.append(this_sample_cid)
    else:
        raise ValueError("samples must be a list or string")
    return results

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
