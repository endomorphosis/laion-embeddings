# Endpoint Management

LAION Embeddings supports multiple endpoint types for flexible deployment and scalability. The system automatically detects available hardware and configures appropriate endpoints.

## Overview

The endpoint management system supports:

- **TEI (Text Embeddings Inference)** - Docker-based GPU endpoints
- **OpenVINO** - Intel-optimized inference endpoints  
- **Local CUDA** - Direct GPU compute endpoints
- **Local CPU** - CPU-only endpoints
- **LibP2P** - Decentralized peer-to-peer endpoints
- **Intel IPEX** - Intel Extension for PyTorch endpoints
- **Llama.cpp** - Quantized model endpoints

## Endpoint Types

### TEI Endpoints

Text Embeddings Inference provides high-performance GPU-accelerated embedding generation using Docker containers.

#### Configuration

```python
resources = {
    "tei_endpoints": [
        ["thenlper/gte-small", "http://localhost:8080/embed", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "http://localhost:8081/embed", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://localhost:8082/embed", 32768],
    ]
}
```

#### Launch Script

Use the provided `launch_tei.sh` script to start TEI containers:

```bash
#!/bin/bash
# Assumes 2 GPU setup
hf_token=YOUR_HF_TOKEN
volume=/storage/hf_models

# Launch gte-large-en-v1.5 on both GPUs
model=Alibaba-NLP/gte-large-en-v1.5
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 8080:80 \
  -v $volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id $model --max-batch-tokens 8192 --payload-limit 32000000 &

docker run --gpus all -e CUDA_VISIBLE_DEVICES=1 -p 8081:80 \
  -v $volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id $model --max-batch-tokens 8192 --payload-limit 32000000 &

# Launch gte-small on both GPUs  
model=thenlper/gte-small
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 8084:80 \
  -v $volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id $model --max-batch-tokens 512 --payload-limit 32000000 &

docker run --gpus all -e CUDA_VISIBLE_DEVICES=1 -p 8085:80 \
  -v $volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id $model --max-batch-tokens 512 --payload-limit 32000000 &
```

#### Request Format

```python
# TEI endpoint request
data = {
    "inputs": ["Sample text to embed", "Another text sample"],
    "normalize": True,
    "truncate": True
}

response = await make_post_request("http://localhost:8080/embed", data)
embeddings = response["embeddings"]
```

### OpenVINO Endpoints

Intel OpenVINO provides optimized inference for Intel hardware including CPUs, GPUs, and VPUs.

#### Configuration

```python
resources = {
    "openvino_endpoints": [
        ["thenlper/gte-small", "https://gte-small-ov.example.com/v2/models/gte-small/infer", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "https://gte-large-ov.example.com/v2/models/gte-large/infer", 8192],
    ]
}
```

#### Request Format

OpenVINO endpoints use the Triton Inference Server protocol:

```python
data = {
    "inputs": [
        {
            "name": "input_ids",
            "shape": [1, 128],
            "datatype": "INT64", 
            "data": input_ids_list
        },
        {
            "name": "attention_mask",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": attention_mask_list
        }
    ]
}

response = await make_post_request_openvino(endpoint, data)
embeddings = response["outputs"][0]["data"]
```

### Local CUDA Endpoints

Direct GPU compute using PyTorch CUDA for maximum performance and control.

#### Configuration

```python
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "cuda:0", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cuda:0", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:0", 32768],
        ["thenlper/gte-small", "cuda:1", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cuda:1", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:1", 32768],
    ]
}
```

#### Implementation

Local CUDA endpoints automatically:
- Load models to specified GPU devices
- Manage tokenizers per device
- Handle batch processing with memory optimization
- Implement proper cleanup and memory management

```python
async def make_local_request(self, model, endpoint, data):
    device = torch.device(endpoint)
    inputs = self.tokenizer[model][endpoint](
        data, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    self.local_endpoints[model][endpoint].to(device).eval()
    with torch.no_grad():
        outputs = self.local_endpoints[model][endpoint](**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        
        # Memory cleanup
        del inputs, outputs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return embeddings
```

### Local CPU Endpoints

CPU-only inference for environments without GPU acceleration.

#### Configuration

```python
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "cpu", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cpu", 32768],
    ]
}
```

#### Features

- Automatic fallback when GPU unavailable
- Optimized batch sizes for CPU processing
- Multi-threading support
- Memory-efficient processing

### Intel IPEX Endpoints

Intel Extension for PyTorch provides optimizations for Intel hardware.

#### Configuration

```python
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "ipex", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "ipex", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "ipex", 32768],
    ]
}
```

#### Requirements

- Intel Extension for PyTorch installed
- Compatible Intel hardware (CPUs/GPUs)
- Proper environment configuration

### Llama.cpp Endpoints

Quantized model inference using the llama.cpp backend.

#### Configuration

```python
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "llama_cpp", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "llama_cpp", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "llama_cpp", 32768],
    ]
}
```

#### Features

- Reduced memory usage through quantization
- CPU-optimized inference
- Support for various quantization formats
- Fast inference on consumer hardware

### LibP2P Endpoints

Decentralized peer-to-peer endpoints for distributed inference.

#### Configuration

```python
resources = {
    "libp2p_endpoints": [
        ["thenlper/gte-small", "/ip4/127.0.0.1/tcp/4001/p2p/QmHash", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "/ip4/127.0.0.1/tcp/4002/p2p/QmHash", 8192],
    ]
}
```

#### Features

- Decentralized compute network
- IPFS integration
- Fault-tolerant distributed processing
- Automatic peer discovery

## Hardware Detection

The system automatically detects available hardware and configures appropriate endpoints:

```python
async def test_hardware(self):
    """Test available hardware and software capabilities"""
    results = {
        "cuda": await self.test_cuda(),
        "openvino": await self.test_local_openvino(), 
        "llama_cpp": await self.test_llama_cpp(),
        "ipex": await self.test_ipex(),
    }
    return results

async def test_cuda(self):
    """Test CUDA availability and GPU count"""
    try:
        gpus = torch.cuda.device_count()
        return gpus > 0
    except Exception as e:
        return False

async def test_local_openvino(self):
    """Test OpenVINO installation"""
    try:
        import openvino
        return True
    except ImportError:
        return False

async def test_ipex(self):
    """Test Intel Extension for PyTorch"""
    try:
        import intel_extension_for_pytorch as ipex
        return True
    except ImportError:
        return False
```

## Endpoint Selection

The system intelligently selects endpoints based on:

1. **Hardware availability** - Detected GPU/CPU capabilities
2. **Model compatibility** - Supported models per endpoint type
3. **Load balancing** - Distribute requests across available endpoints
4. **Endpoint health** - Monitor endpoint status and responsiveness

```python
def choose_endpoint(self, model, endpoint_type=None):
    """Select best available endpoint for model"""
    if endpoint_type is None:
        # Try in order of preference: TEI -> Local CUDA -> OpenVINO -> CPU
        for ep_type in ["tei_endpoints", "local_endpoints", "openvino_endpoints"]:
            filtered_endpoints = self.get_available_endpoints(model, ep_type)
            if filtered_endpoints:
                return random.choice(filtered_endpoints)
    else:
        return self.get_available_endpoints(model, endpoint_type)
    
    return None

def get_available_endpoints(self, model, endpoint_type):
    """Get healthy endpoints for model and type"""
    endpoints = self.endpoints.get(endpoint_type, [])
    model_endpoints = [ep for ep in endpoints if ep[0] == model]
    healthy_endpoints = [ep for ep in model_endpoints 
                        if self.endpoint_status.get(ep[1], 0) >= 1]
    return healthy_endpoints
```

## Batch Size Optimization

Each endpoint type has optimized batch sizes based on hardware capabilities:

```python
async def max_batch_size(self, model, endpoint, endpoint_type=None):
    """Determine maximum batch size for endpoint"""
    # Start with small batch and increase until memory limit
    exponent = 0
    batch_size = 2**exponent
    
    # Get context length for model
    if "cuda" in endpoint or "cpu" in endpoint:
        token_length = self.local_endpoints[model][endpoint].config.max_position_embeddings
    else:
        token_length = self.get_endpoint_context_length(endpoint)
    
    # Test increasing batch sizes
    while batch_size <= 1024:  # Maximum reasonable batch size
        try:
            test_batch = ["test text"] * batch_size
            await self.test_endpoint_batch(model, endpoint, test_batch)
            exponent += 1
            batch_size = 2**exponent
        except Exception as e:
            # Hit memory limit, use previous batch size
            return max(1, 2**(exponent-1))
    
    return batch_size
```

## Error Handling

Robust error handling ensures system reliability:

```python
async def make_request_with_retry(self, model, endpoint, data, max_retries=3):
    """Make request with automatic retry and fallback"""
    for attempt in range(max_retries):
        try:
            if "cuda" in endpoint or "cpu" in endpoint:
                return await self.make_local_request(model, endpoint, data)
            elif "/embed" in endpoint:
                return await self.make_post_request(endpoint, {"inputs": data})
            elif "/infer" in endpoint:
                return await self.make_post_request_openvino(endpoint, data)
        except Exception as e:
            if attempt == max_retries - 1:
                # Try fallback endpoint
                fallback = self.choose_endpoint(model, "local_endpoints")
                if fallback and fallback != endpoint:
                    return await self.make_local_request(model, fallback, data)
                raise e
            
            # Wait before retry
            await asyncio.sleep(2 ** attempt)
```

## Monitoring and Health Checks

Continuous monitoring ensures endpoint health:

```python
async def monitor_endpoints(self):
    """Monitor endpoint health and update status"""
    for endpoint_type, endpoints in self.endpoints.items():
        for model, endpoint, context_length in endpoints:
            try:
                # Test endpoint with small sample
                test_data = ["health check"]
                await asyncio.wait_for(
                    self.make_request(model, endpoint, test_data),
                    timeout=30.0
                )
                self.endpoint_status[endpoint] = context_length
            except Exception as e:
                # Mark endpoint as unhealthy
                self.endpoint_status[endpoint] = 0
                print(f"Endpoint {endpoint} failed health check: {e}")

# Run monitoring in background
asyncio.create_task(self.monitor_endpoints())
```

## Best Practices

### Performance Optimization

1. **Use TEI endpoints** for production GPU deployments
2. **Distribute models** across multiple GPUs when available  
3. **Monitor batch sizes** to maximize throughput
4. **Enable mixed precision** for supported models
5. **Use appropriate context lengths** for your use case

### Resource Management

1. **Set memory limits** to prevent OOM errors
2. **Implement proper cleanup** for GPU memory
3. **Use async processing** for concurrent requests
4. **Monitor endpoint health** continuously
5. **Implement graceful fallbacks** for failed endpoints

### Security

1. **Use authentication** for remote endpoints
2. **Validate input data** before processing
3. **Implement rate limiting** for public endpoints
4. **Monitor resource usage** to prevent abuse
5. **Use HTTPS** for remote connections

## Troubleshooting

### Common Issues

#### GPU Out of Memory
```python
# Reduce batch size
self.batch_sizes[model][endpoint] = max(1, self.batch_sizes[model][endpoint] // 2)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
with torch.cuda.amp.autocast():
    outputs = model(**inputs)
```

#### Endpoint Timeouts
```python
# Increase timeout values
timeout = ClientTimeout(total=300)

# Implement retry logic
for attempt in range(3):
    try:
        response = await session.post(url, json=data, timeout=timeout)
        break
    except asyncio.TimeoutError:
        if attempt == 2:
            raise
        await asyncio.sleep(2 ** attempt)
```

#### Model Loading Failures
```python
# Check available memory
if torch.cuda.is_available():
    memory_free = torch.cuda.get_device_properties(0).total_memory
    memory_used = torch.cuda.memory_allocated(0)
    memory_available = memory_free - memory_used
    
    if memory_available < model_size_estimate:
        # Use CPU or smaller model
        device = "cpu"
```

## Configuration Examples

### Development Setup
```python
# Single GPU development
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "cuda:0", 512],
        ["thenlper/gte-small", "cpu", 512],  # Fallback
    ],
    "tei_endpoints": [],
    "openvino_endpoints": [],
    "libp2p_endpoints": []
}
```

### Production Setup
```python
# Multi-GPU production with TEI
resources = {
    "tei_endpoints": [
        ["Alibaba-NLP/gte-large-en-v1.5", "http://gpu1:8080/embed", 8192],
        ["Alibaba-NLP/gte-large-en-v1.5", "http://gpu2:8081/embed", 8192],
        ["thenlper/gte-small", "http://gpu3:8082/embed", 512],
        ["thenlper/gte-small", "http://gpu4:8083/embed", 512],
    ],
    "local_endpoints": [
        ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],  # Fallback
    ],
    "openvino_endpoints": [
        ["thenlper/gte-small", "http://intel-server:8000/infer", 512],
    ]
}
```

### Edge Deployment
```python
# Optimized for resource-constrained environments
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "cpu", 512],
    ],
    "llama_cpp_endpoints": [
        ["thenlper/gte-small", "llama_cpp", 512],  # Quantized model
    ]
}
```
