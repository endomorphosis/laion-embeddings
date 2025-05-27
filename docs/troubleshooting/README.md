# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the LAION Embeddings project.

## Quick Diagnostics

### Health Check Script

```python
#!/usr/bin/env python3
"""
Quick diagnostic script for LAION Embeddings.
"""

import requests
import subprocess
import sys
import os
import psutil
import GPUtil

def check_service_health():
    """Check if the main service is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ Service is healthy")
            print(f"  Status: {data.get('status', 'unknown')}")
            if 'endpoints' in data:
                for endpoint, status in data['endpoints'].items():
                    print(f"  {endpoint}: {status}")
            return True
        else:
            print(f"✗ Service returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Service connection failed: {e}")
        return False

def check_system_resources():
    """Check system resources."""
    print("\n=== System Resources ===")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.percent}% used ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
    
    # GPU
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            print("GPUs:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                print(f"    Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
                print(f"    Load: {gpu.load*100:.1f}%")
        else:
            print("No GPUs detected")
    except:
        print("GPU information unavailable")

def check_ports():
    """Check if required ports are in use."""
    import socket
    
    print("\n=== Port Status ===")
    ports_to_check = [8000, 8001, 8002, 5001]  # Main service + common endpoints + IPFS
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        if result == 0:
            print(f"✓ Port {port} is open")
        else:
            print(f"✗ Port {port} is closed")
        sock.close()

def check_dependencies():
    """Check Python dependencies."""
    print("\n=== Dependencies ===")
    
    required_packages = [
        'fastapi', 'uvicorn', 'requests', 'numpy', 'pandas',
        'torch', 'transformers', 'sentence-transformers',
        'ipfshttpclient', 'datasets'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")

def main():
    print("=== LAION Embeddings Diagnostic ===")
    
    check_service_health()
    check_system_resources()
    check_ports()
    check_dependencies()
    
    print("\n=== End Diagnostic ===")

if __name__ == "__main__":
    main()
```

## Common Issues

### 1. Service Won't Start

**Symptoms:**
- Service fails to start
- Port already in use errors
- Import errors

**Solutions:**

```bash
# Check if port is in use
lsof -i :8000

# Kill process using port
kill -9 <PID>

# Check Python environment
python --version
pip list | grep -E "(fastapi|torch|transformers)"

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Environment Issues:**
```bash
# Check virtual environment
which python
echo $VIRTUAL_ENV

# Activate correct environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 2. Model Loading Failures

**Symptoms:**
- Models fail to load
- CUDA out of memory
- Slow model initialization

**Solutions:**

**Memory Issues:**
```python
# Check available GPU memory
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        memory_free = torch.cuda.get_device_properties(i).total_memory
        memory_used = torch.cuda.memory_allocated(i)
        print(f"GPU {i}: {memory_used / 1e9:.1f}GB / {memory_free / 1e9:.1f}GB")
```

**Model Configuration:**
```yaml
# config.yaml - Reduce model memory usage
models:
  gte-small:
    device: "cuda:0"
    max_length: 512
    batch_size: 16  # Reduce if OOM
    
  gte-large-en-v1.5:
    device: "cpu"  # Move to CPU if GPU memory insufficient
    batch_size: 8
```

**Force CPU Mode:**
```bash
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### 3. Embedding Creation Errors

**Symptoms:**
- API returns 500 errors
- Timeout errors
- Inconsistent results

**Debugging:**

```python
# Test individual components
import requests

def debug_embedding_creation():
    # Test basic endpoint
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.status_code}")
    
    # Test simple embedding
    payload = {
        "texts": ["test text"],
        "model": "gte-small",
        "normalize": True
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/create_embeddings/",
            json=payload,
            timeout=30
        )
        print(f"Embedding creation: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Embedding shape: {len(result['embeddings'][0])}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

debug_embedding_creation()
```

**Common Fixes:**

1. **Timeout Issues:**
```yaml
# Increase timeouts
server:
  timeout: 300
  keep_alive: 60
```

2. **Batch Size Too Large:**
```python
# Reduce batch size
payload = {
    "texts": texts[:10],  # Process smaller batches
    "model": "gte-small"
}
```

3. **Text Length Issues:**
```python
# Truncate long texts
max_length = 512
texts = [text[:max_length] for text in original_texts]
```

### 4. IPFS Integration Issues

**Symptoms:**
- IPFS connection errors
- CID generation failures
- Upload/download timeouts

**Solutions:**

```bash
# Check IPFS daemon
ipfs daemon --enable-gc &

# Test IPFS connection
curl http://localhost:5001/api/v0/version

# Check IPFS peers
ipfs swarm peers | wc -l
```

**Python IPFS Debugging:**
```python
import ipfshttpclient

def debug_ipfs():
    try:
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        
        # Test basic connection
        version = client.version()
        print(f"IPFS version: {version}")
        
        # Test add/get
        test_data = b"test content"
        result = client.add_bytes(test_data)
        print(f"Added test data: {result}")
        
        retrieved = client.cat(result)
        print(f"Retrieved: {retrieved == test_data}")
        
    except Exception as e:
        print(f"IPFS error: {e}")

debug_ipfs()
```

### 5. Performance Issues

**Symptoms:**
- Slow response times
- High memory usage
- CPU/GPU bottlenecks

**Performance Profiling:**

```python
import time
import psutil
import functools

def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        
        print(f"{func.__name__}:")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB ({mem_after - mem_before:+.1f}MB)")
        
        return result
    return wrapper

# Use decorator on functions to profile
@profile_function
def create_embeddings(texts):
    # Your embedding code
    pass
```

**Optimization Strategies:**

1. **Batch Size Optimization:**
```python
# Find optimal batch size
def find_optimal_batch_size():
    batch_sizes = [1, 4, 8, 16, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            start = time.time()
            # Test with this batch size
            process_batch(texts[:batch_size])
            duration = time.time() - start
            throughput = batch_size / duration
            results[batch_size] = throughput
        except Exception as e:
            results[batch_size] = 0
    
    optimal = max(results, key=results.get)
    print(f"Optimal batch size: {optimal}")
    return optimal
```

2. **Memory Management:**
```python
import gc
import torch

def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

### 6. Docker Issues

**Symptoms:**
- Container won't start
- Port mapping issues
- Volume mount problems

**Debugging:**

```bash
# Check container status
docker ps -a

# View container logs
docker logs <container_name>

# Debug inside container
docker exec -it <container_name> /bin/bash

# Check resource usage
docker stats <container_name>
```

**Common Docker Fixes:**

1. **Port Conflicts:**
```bash
# Use different port
docker run -p 8001:8000 laion-embeddings
```

2. **Memory Limits:**
```bash
# Increase memory limit
docker run --memory=8g laion-embeddings
```

3. **GPU Access:**
```bash
# Enable GPU access
docker run --gpus all laion-embeddings
```

### 7. Configuration Issues

**Symptoms:**
- Settings not applied
- Endpoint configuration errors
- Model loading problems

**Configuration Debugging:**

```python
import yaml
import os

def debug_config():
    # Check environment variables
    print("Environment variables:")
    for key in os.environ:
        if 'EMBEDDINGS' in key or 'MODEL' in key:
            print(f"  {key}: {os.environ[key]}")
    
    # Check config file
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("Config file loaded successfully")
        print(f"Models configured: {list(config.get('models', {}).keys())}")
    except Exception as e:
        print(f"Config file error: {e}")

debug_config()
```

**Config Validation:**
```python
def validate_config(config):
    """Validate configuration structure."""
    required_sections = ['models', 'server', 'storage']
    missing = [section for section in required_sections if section not in config]
    
    if missing:
        print(f"Missing config sections: {missing}")
        return False
    
    # Validate models
    for model_name, model_config in config['models'].items():
        required_model_keys = ['endpoint_type', 'device']
        missing_keys = [key for key in required_model_keys if key not in model_config]
        if missing_keys:
            print(f"Model {model_name} missing keys: {missing_keys}")
            return False
    
    print("Configuration is valid")
    return True
```

## Error Codes Reference

### HTTP Status Codes

| Code | Meaning | Common Cause | Solution |
|------|---------|--------------|----------|
| 400 | Bad Request | Invalid input parameters | Check request format |
| 422 | Validation Error | Missing/invalid fields | Validate request data |
| 500 | Internal Server Error | Model/processing failure | Check logs and resources |
| 503 | Service Unavailable | Model not loaded | Wait for model loading |
| 504 | Gateway Timeout | Request timeout | Reduce batch size |

### Model Loading Errors

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Insufficient GPU memory | Reduce batch size or use CPU |
| Model not found | Missing model files | Download model or check path |
| Device unavailable | CUDA/device not available | Use CPU or install CUDA |
| Import error | Missing dependencies | Install required packages |

## Log Analysis

### Enable Debug Logging

```python
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# For specific modules
logging.getLogger('transformers').setLevel(logging.DEBUG)
logging.getLogger('torch').setLevel(logging.DEBUG)
```

### Log File Analysis

```bash
# Watch logs in real-time
tail -f embeddings.log

# Search for errors
grep -i error embeddings.log

# Filter by timestamp
grep "2024-01-15" embeddings.log
```

## Performance Monitoring

### System Monitoring Script

```python
import psutil
import time
import threading

class SystemMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        
    def start_monitoring(self):
        self.running = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.start()
        
    def stop_monitoring(self):
        self.running = False
        
    def _monitor_loop(self):
        while self.running:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            print(f"CPU: {cpu}% | Memory: {memory.percent}%")
            
            time.sleep(self.interval)

# Usage
monitor = SystemMonitor()
monitor.start_monitoring()
# ... run your processing ...
monitor.stop_monitoring()
```

## Getting Help

### Community Resources

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check component-specific docs
3. **Discussions**: Community Q&A

### Creating Bug Reports

Include:
1. System information (OS, Python version, GPU)
2. Complete error messages and stack traces
3. Configuration files (sanitized)
4. Steps to reproduce
5. Expected vs actual behavior

### Sample Bug Report Template

```markdown
## Bug Report

**Environment:**
- OS: Ubuntu 20.04
- Python: 3.9.7
- CUDA: 11.8
- GPU: RTX 3080 (10GB)

**Configuration:**
```yaml
# Your config.yaml here
```

**Error:**
```
Complete error message and stack trace
```

**Steps to Reproduce:**
1. Start service with config.yaml
2. Send request with payload X
3. Error occurs

**Expected:** Service should return embeddings
**Actual:** 500 error with CUDA OOM
```

This troubleshooting guide covers the most common issues. For specific problems not covered here, check the component-specific documentation or create an issue with detailed information.
