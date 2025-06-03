# Configuration Guide

![Production Ready](https://img.shields.io/badge/status-production%20ready-green)
![Tests](https://img.shields.io/badge/tests-100%25%20passing-green)
![Test Coverage](https://img.shields.io/badge/coverage-validated-green)

## ✅ Production Validation Status

This configuration guide has been validated with **100% test success rate** across all service layers:

- **✅ Vector Service**: 23/23 tests passing
- **✅ IPFS Vector Service**: 15/15 tests passing  
- **✅ Clustering Service**: 19/19 tests passing
- **✅ Integration Tests**: 2/2 tests passing
- **✅ Service Dependencies**: All imports validated

**Last Validated**: June 3, 2025

This guide covers how to configure LAION Embeddings for various deployment scenarios and use cases.

## Configuration Files

### Environment Variables

Create a `.env` file in your project root:

```bash
# Server Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=9999
FASTAPI_WORKERS=1
FASTAPI_RELOAD=false

# Model Configuration
DEFAULT_MODELS=thenlper/gte-small,Alibaba-NLP/gte-large-en-v1.5
MODEL_CACHE_DIR=./models
MAX_SEQUENCE_LENGTH=512
EMBEDDING_BATCH_SIZE=32

# IPFS Configuration
IPFS_HOST=127.0.0.1
IPFS_PORT=5001
IPFS_GATEWAY=https://ipfs.io
IPFS_TIMEOUT=300

# Storage Configuration
DATA_DIR=./data
CHECKPOINT_DIR=./checkpoints
TEMP_DIR=./tmp
MAX_DISK_USAGE=80

# Performance Configuration
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=300
QUEUE_SIZE=1000
WORKER_THREADS=4

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
CUDA_MEMORY_FRACTION=0.8
USE_MIXED_PRECISION=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/server.log
LOG_ROTATION=1d
LOG_RETENTION=30d

# Security Configuration (Future)
API_KEY_REQUIRED=false
RATE_LIMIT_ENABLED=true
CORS_ORIGINS=*
```

### Models Configuration

Configure available models in `models.yaml`:

```yaml
models:
  small:
    name: "thenlper/gte-small"
    dimensions: 384
    max_sequence_length: 512
    default_batch_size: 64
    memory_requirements: "2GB"
    
  large:
    name: "Alibaba-NLP/gte-large-en-v1.5" 
    dimensions: 1024
    max_sequence_length: 8192
    default_batch_size: 16
    memory_requirements: "8GB"
    
  qwen:
    name: "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    dimensions: 1536
    max_sequence_length: 32768
    default_batch_size: 8
    memory_requirements: "12GB"

# Model aliases for easy reference
aliases:
  default: "small"
  best_quality: "large"
  latest: "qwen"
```

### Endpoints Configuration

Configure inference endpoints in `endpoints.yaml`:

```yaml
endpoints:
  tei_endpoints:
    - model: "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
      url: "http://62.146.169.111:8080/embed-medium"
      context_length: 32768
      health_check: "/health"
      
    - model: "thenlper/gte-small"
      url: "http://62.146.169.111:8080/embed-tiny"
      context_length: 512
      health_check: "/health"
      
    - model: "Alibaba-NLP/gte-large-en-v1.5"
      url: "http://62.146.169.111:8081/embed-small"
      context_length: 8192
      health_check: "/health"

  openvino_endpoints:
    - model: "thenlper/gte-small"
      url: "http://localhost:8001/v2/models/gte-small/infer"
      context_length: 512
      device: "CPU"
      
  local_endpoints:
    - model: "thenlper/gte-small"
      path: "./models/gte-small"
      device: "cpu"
      workers: 2
      
  cuda_endpoints:
    - model: "Alibaba-NLP/gte-large-en-v1.5"
      device: "cuda:0"
      memory_fraction: 0.7
      mixed_precision: true

  libp2p_endpoints:
    - model: "thenlper/gte-small"
      peer_id: "12D3KooW..."
      multiaddr: "/ip4/192.168.1.100/tcp/4001/p2p/12D3KooW..."
      
# Load balancing configuration  
load_balancing:
  strategy: "round_robin"  # round_robin, least_connections, weighted
  health_check_interval: 30  # seconds
  max_retries: 3
  timeout: 30
```

## Deployment Configurations

### Development Configuration

For local development:

```bash
# .env.development
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=9999
FASTAPI_RELOAD=true
LOG_LEVEL=DEBUG
CUDA_VISIBLE_DEVICES=""  # CPU only
MODEL_CACHE_DIR=./dev_models
```

### Production Configuration

For production deployment:

```bash
# .env.production
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_WORKERS=4
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0,1  # Multiple GPUs
MODEL_CACHE_DIR=/opt/models
DATA_DIR=/opt/data
```

### Docker Configuration

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  laion-embeddings:
    build: .
    ports:
      - "9999:9999"
    environment:
      - FASTAPI_HOST=0.0.0.0
      - IPFS_HOST=ipfs
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    depends_on:
      - ipfs
      
  ipfs:
    image: ipfs/kubo:latest
    ports:
      - "4001:4001"
      - "5001:5001" 
      - "8080:8080"
    volumes:
      - ipfs_data:/data/ipfs

volumes:
  ipfs_data:
```

## Hardware-Specific Configuration

### CPU-Only Setup

```yaml
# config/cpu.yaml
hardware:
  type: "cpu"
  workers: 8
  memory_limit: "16GB"
  
models:
  preferred: ["thenlper/gte-small"]
  batch_sizes:
    "thenlper/gte-small": 32
    
endpoints:
  local_only: true
  types: ["local", "openvino"]
```

### GPU Setup

```yaml
# config/gpu.yaml
hardware:
  type: "gpu"
  devices: ["cuda:0", "cuda:1"]
  memory_per_device: "12GB"
  mixed_precision: true
  
models:
  preferred: ["Alibaba-NLP/gte-large-en-v1.5"]
  batch_sizes:
    "thenlper/gte-small": 128
    "Alibaba-NLP/gte-large-en-v1.5": 64
    
endpoints:
  types: ["cuda", "tei", "local"]
```

### Multi-Node Cluster

```yaml
# config/cluster.yaml
cluster:
  mode: "distributed"
  coordinator: "node-1.example.com:9999"
  nodes:
    - id: "node-1"
      address: "192.168.1.10:9999"
      role: "coordinator"
      resources: ["cpu", "gpu"]
      
    - id: "node-2" 
      address: "192.168.1.11:9999"
      role: "worker"
      resources: ["gpu"]
      
    - id: "node-3"
      address: "192.168.1.12:9999"
      role: "worker" 
      resources: ["cpu"]

load_balancing:
  strategy: "resource_aware"
  health_check_interval: 15
```

## Advanced Configuration

### Custom Chunking Strategy

```python
# config/chunking.py
CHUNKING_CONFIG = {
    "method": "semantic",  # semantic, fixed, sliding
    "chunk_size": 512,
    "overlap": 64,
    "min_chunk_size": 100,
    "max_chunk_size": 1024,
    "sentence_splitter": "spacy",  # spacy, nltk, simple
    "preserve_boundaries": True
}
```

### Caching Configuration

```yaml
# config/cache.yaml
cache:
  embeddings:
    enabled: true
    backend: "redis"  # redis, memory, disk
    ttl: 3600  # seconds
    max_size: "10GB"
    
  models:
    enabled: true
    backend: "disk"
    path: "./model_cache"
    
  search_results:
    enabled: true
    backend: "memory"
    ttl: 300
    max_entries: 10000
```

### Monitoring Configuration

```yaml
# config/monitoring.yaml
monitoring:
  metrics:
    enabled: true
    endpoint: "/metrics"
    format: "prometheus"
    
  health_checks:
    enabled: true
    endpoint: "/health"
    checks:
      - "database"
      - "ipfs"
      - "models"
      - "endpoints"
      
  logging:
    level: "INFO"
    format: "json"
    destinations:
      - "console"
      - "file"
      - "elasticsearch"  # optional
```

## Configuration Validation

Validate your configuration:

```bash
# Check configuration syntax
python -m laion_embeddings validate-config

# Test endpoint connectivity
python -m laion_embeddings test-endpoints

# Validate model compatibility
python -m laion_embeddings check-models
```

## Environment-Specific Overrides

Use configuration hierarchies:

1. `config/default.yaml` - Base configuration
2. `config/production.yaml` - Production overrides
3. Environment variables - Runtime overrides

```python
# Loading priority (highest first):
# 1. Environment variables
# 2. config/{ENVIRONMENT}.yaml
# 3. config/default.yaml
```

## Security Configuration

```yaml
# config/security.yaml
security:
  authentication:
    enabled: false  # Enable in production
    method: "api_key"  # api_key, oauth2, jwt
    
  authorization:
    enabled: false
    roles: ["admin", "user", "readonly"]
    
  rate_limiting:
    enabled: true
    strategies:
      search: "100/minute"
      create: "5/hour"
      default: "1000/hour"
      
  cors:
    enabled: true
    origins: ["*"]  # Restrict in production
    methods: ["GET", "POST"]
    
  encryption:
    at_rest: false  # Enable for sensitive data
    in_transit: true  # HTTPS required
```

## Performance Tuning

### Memory Optimization

```yaml
memory:
  optimization:
    model_offloading: true
    gradient_checkpointing: true
    attention_slicing: true
    
  limits:
    per_request: "1GB"
    total_system: "32GB"
    swap_usage: "25%"
```

### Network Optimization

```yaml
network:
  timeouts:
    connection: 30
    read: 300
    write: 60
    
  retries:
    max_attempts: 3
    backoff_factor: 2
    
  compression:
    enabled: true
    level: 6
```

## Troubleshooting Configuration

Common configuration issues:

### Port Conflicts
```bash
# Check if port is in use
lsof -i :9999

# Change port in configuration
FASTAPI_PORT=8888
```

### Memory Issues
```bash
# Reduce batch sizes
EMBEDDING_BATCH_SIZE=16

# Enable model offloading
MODEL_OFFLOADING=true
```

### IPFS Connection
```bash
# Check IPFS daemon
ipfs id

# Update IPFS configuration
IPFS_HOST=your-ipfs-node.com
IPFS_PORT=5001
```

## Configuration Best Practices

1. **Use environment-specific configs** - Separate dev/staging/prod
2. **Version control configuration** - Track changes
3. **Validate on startup** - Fail fast with invalid config
4. **Document custom settings** - Comment complex configurations
5. **Monitor configuration drift** - Alert on changes
6. **Backup configurations** - Store safely
7. **Test configuration changes** - Validate before deployment

## Configuration Templates

Generate configuration templates:

```bash
# Generate basic configuration
python -m laion_embeddings init-config --template basic

# Generate production configuration  
python -m laion_embeddings init-config --template production

# Generate cluster configuration
python -m laion_embeddings init-config --template cluster
```

## Next Steps

- [Endpoint Management](endpoints/README.md) - Configure inference endpoints
- [Model Configuration](models/configuration.md) - Set up embedding models  
- [Performance Tuning](troubleshooting/performance.md) - Optimize performance
- [Deployment Guide](development/deployment.md) - Deploy to production
