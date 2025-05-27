# Model Configuration and Custom Models

This guide covers advanced model configuration, adding custom models, and optimizing model performance.

## Supported Model Types

### 1. Sentence Transformers Models

**Compatible Models:**
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `thenlper/gte-small`
- `thenlper/gte-large-en-v1.5`
- `thenlper/gte-Qwen2-1.5B-instruct`

**Configuration Example:**
```yaml
models:
  custom-sentence-transformer:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    endpoint_type: "sentence_transformers"
    device: "cuda:0"
    max_length: 512
    batch_size: 32
    normalize_embeddings: true
    model_kwargs:
      trust_remote_code: false
      use_auth_token: false
```

### 2. Hugging Face Transformers Models

**Configuration Example:**
```yaml
models:
  custom-transformer:
    model_name: "microsoft/DialoGPT-medium"
    endpoint_type: "transformers"
    device: "cuda:0"
    tokenizer_name: "microsoft/DialoGPT-medium"  # Optional, defaults to model_name
    max_length: 1024
    batch_size: 16
    model_kwargs:
      torch_dtype: "float16"
      low_cpu_mem_usage: true
      trust_remote_code: false
```

### 3. OpenAI Models (via API)

**Configuration Example:**
```yaml
models:
  openai-ada:
    endpoint_type: "openai"
    model_name: "text-embedding-ada-002"
    api_key: "${OPENAI_API_KEY}"  # Use environment variable
    batch_size: 100  # OpenAI allows larger batches
    rate_limit: 60   # Requests per minute
    timeout: 30
```

### 4. Text Embeddings Inference (TEI)

**Configuration Example:**
```yaml
models:
  tei-model:
    endpoint_type: "tei"
    model_name: "thenlper/gte-small"
    host: "localhost"
    port: 8080
    max_length: 512
    batch_size: 32
    docker_config:
      image: "ghcr.io/huggingface/text-embeddings-inference:latest"
      volumes:
        - "${HOME}/.cache/huggingface:/data"
      environment:
        - "MODEL_ID=thenlper/gte-small"
        - "MAX_BATCH_TOKENS=16384"
```

## Adding Custom Models

### Step 1: Prepare Model Files

**Option A: From Hugging Face Hub**
```python
from transformers import AutoModel, AutoTokenizer

model_name = "your-org/your-model"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/your-model")
tokenizer.save_pretrained("./models/your-model")
```

**Option B: Local Model Files**
```
models/
└── your-model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

### Step 2: Create Model Configuration

```yaml
models:
  your-custom-model:
    model_name: "./models/your-model"  # Local path
    endpoint_type: "sentence_transformers"
    device: "cuda:0"
    max_length: 512
    batch_size: 16
    
    # Custom pooling strategy
    pooling_mode: "mean"  # Options: mean, max, cls
    
    # Model-specific parameters
    model_kwargs:
      torch_dtype: "float16"
      attn_implementation: "flash_attention_2"  # If supported
      
    # Tokenizer options
    tokenizer_kwargs:
      padding: true
      truncation: true
      return_tensors: "pt"
```

### Step 3: Implement Custom Model Class

For advanced customization, create a custom model class:

```python
# custom_models/my_model.py

import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

class CustomEmbeddingModel:
    def __init__(self, model_path, device="cpu", **kwargs):
        self.device = device
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = kwargs.get('max_length', 512)
        
    def encode(self, texts, batch_size=32, normalize=True):
        """Encode texts to embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Custom pooling strategy
                embeddings = self._pool_embeddings(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )
                
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _pool_embeddings(self, hidden_states, attention_mask):
        """Custom pooling strategy."""
        # Mean pooling with attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# Register custom model
def create_custom_model(config):
    return CustomEmbeddingModel(
        model_path=config['model_name'],
        device=config['device'],
        **config.get('model_kwargs', {})
    )
```

### Step 4: Register Custom Model

```python
# In main.py or model registry

from custom_models.my_model import create_custom_model

# Register custom model factory
MODEL_FACTORIES = {
    'sentence_transformers': create_sentence_transformer_model,
    'transformers': create_transformer_model,
    'custom': create_custom_model,  # Add your custom factory
    'tei': create_tei_model,
    'openai': create_openai_model,
}

def load_model(model_config):
    endpoint_type = model_config.get('endpoint_type', 'sentence_transformers')
    factory = MODEL_FACTORIES.get(endpoint_type)
    
    if not factory:
        raise ValueError(f"Unknown endpoint type: {endpoint_type}")
    
    return factory(model_config)
```

## Model Optimization

### 1. Quantization

**INT8 Quantization:**
```yaml
models:
  quantized-model:
    model_name: "thenlper/gte-small"
    endpoint_type: "sentence_transformers"
    device: "cuda:0"
    model_kwargs:
      torch_dtype: "int8"
      load_in_8bit: true
```

**FP16 Precision:**
```yaml
models:
  fp16-model:
    model_name: "thenlper/gte-large-en-v1.5"
    endpoint_type: "sentence_transformers"
    device: "cuda:0"
    model_kwargs:
      torch_dtype: "float16"
      autocast: true
```

### 2. ONNX Optimization

**Convert to ONNX:**
```python
import torch
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Convert model to ONNX
model_name = "thenlper/gte-small"
onnx_model = ORTModelForFeatureExtraction.from_pretrained(
    model_name,
    export=True,
    provider="CUDAExecutionProvider"  # or "CPUExecutionProvider"
)

# Save ONNX model
onnx_model.save_pretrained("./models/gte-small-onnx")
```

**ONNX Configuration:**
```yaml
models:
  onnx-model:
    model_name: "./models/gte-small-onnx"
    endpoint_type: "onnx"
    device: "cuda:0"
    batch_size: 64  # ONNX can handle larger batches
    providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

### 3. TensorRT Optimization

**TensorRT Configuration:**
```yaml
models:
  tensorrt-model:
    model_name: "thenlper/gte-small"
    endpoint_type: "tensorrt"
    device: "cuda:0"
    tensorrt_config:
      precision: "fp16"
      max_batch_size: 128
      max_sequence_length: 512
      optimization_level: 5
```

### 4. Intel OpenVINO

**OpenVINO Configuration:**
```yaml
models:
  openvino-model:
    model_name: "thenlper/gte-small"
    endpoint_type: "openvino"
    device: "CPU"  # or "GPU" for Intel GPUs
    openvino_config:
      precision: "FP16"
      num_threads: 8
      enable_mmap: true
```

## Multi-Model Configuration

### Load Balancing

```yaml
models:
  # Primary model
  gte-small-gpu:
    model_name: "thenlper/gte-small"
    endpoint_type: "sentence_transformers"
    device: "cuda:0"
    batch_size: 32
    priority: 1
    
  # Fallback model
  gte-small-cpu:
    model_name: "thenlper/gte-small"
    endpoint_type: "sentence_transformers"
    device: "cpu"
    batch_size: 16
    priority: 2
    
  # High-capacity model
  gte-large:
    model_name: "thenlper/gte-large-en-v1.5"
    endpoint_type: "sentence_transformers"
    device: "cuda:1"
    batch_size: 16
    priority: 1
    max_requests_per_minute: 100

# Load balancing strategy
load_balancing:
  strategy: "round_robin"  # Options: round_robin, least_loaded, priority
  health_check_interval: 30
  fallback_enabled: true
```

### Model Routing

```python
class ModelRouter:
    def __init__(self, models_config):
        self.models = {}
        self.load_models(models_config)
    
    def route_request(self, request):
        """Route request to appropriate model based on criteria."""
        
        # Route based on text length
        text_length = len(request.get('texts', [''])[0])
        
        if text_length > 1000:
            return self.models['gte-large']
        elif text_length > 500:
            return self.models['gte-small-gpu']
        else:
            return self.models['gte-small-cpu']
    
    def route_by_language(self, request):
        """Route based on detected language."""
        texts = request.get('texts', [])
        
        # Simple language detection (you'd use a proper detector)
        if any('中文' in text for text in texts):
            return self.models['multilingual-model']
        else:
            return self.models['english-model']
```

## Performance Benchmarking

### Benchmark Script

```python
import time
import statistics
from datetime import datetime

class ModelBenchmark:
    def __init__(self, model, test_texts):
        self.model = model
        self.test_texts = test_texts
        self.results = {}
    
    def run_benchmark(self, batch_sizes=[1, 8, 16, 32, 64]):
        """Run comprehensive benchmark."""
        
        print(f"Benchmarking model: {self.model.model_name}")
        print(f"Test texts: {len(self.test_texts)}")
        print(f"Device: {self.model.device}")
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Warm up
            self._warmup(batch_size)
            
            # Benchmark
            times = []
            memory_usage = []
            
            for i in range(5):  # 5 runs
                batch_texts = self.test_texts[:batch_size]
                
                start_time = time.time()
                embeddings = self.model.encode(batch_texts)
                end_time = time.time()
                
                batch_time = end_time - start_time
                times.append(batch_time)
                
                # Calculate throughput
                throughput = len(batch_texts) / batch_time
                
                print(f"  Run {i+1}: {batch_time:.3f}s, {throughput:.1f} texts/sec")
            
            # Statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            avg_throughput = batch_size / avg_time
            
            self.results[batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'avg_throughput': avg_throughput,
                'times': times
            }
            
            print(f"  Average: {avg_time:.3f}±{std_time:.3f}s, {avg_throughput:.1f} texts/sec")
    
    def _warmup(self, batch_size, warmup_runs=2):
        """Warm up model."""
        for _ in range(warmup_runs):
            batch_texts = self.test_texts[:batch_size]
            self.model.encode(batch_texts)
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        print(f"{'Batch Size':<12} {'Avg Time':<12} {'Throughput':<15} {'Memory':<10}")
        print("-" * 50)
        
        for batch_size, results in self.results.items():
            print(f"{batch_size:<12} {results['avg_time']:<12.3f} {results['avg_throughput']:<15.1f}")
        
        # Find optimal batch size
        optimal_batch = max(self.results.keys(), 
                          key=lambda x: self.results[x]['avg_throughput'])
        print(f"\nOptimal batch size: {optimal_batch}")
        print(f"Peak throughput: {self.results[optimal_batch]['avg_throughput']:.1f} texts/sec")

# Usage
def benchmark_model():
    # Load model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('thenlper/gte-small')
    
    # Create test texts
    test_texts = [f"This is test sentence number {i}" for i in range(100)]
    
    # Run benchmark
    benchmark = ModelBenchmark(model, test_texts)
    benchmark.run_benchmark()
    benchmark.print_summary()

if __name__ == "__main__":
    benchmark_model()
```

## Model Monitoring

### Health Checks

```python
class ModelHealthMonitor:
    def __init__(self, model):
        self.model = model
        self.test_text = "Health check test"
        
    def health_check(self):
        """Comprehensive health check."""
        checks = {
            'model_loaded': self._check_model_loaded(),
            'inference_working': self._check_inference(),
            'memory_usage': self._check_memory(),
            'response_time': self._check_response_time()
        }
        
        all_healthy = all(check['status'] for check in checks.values())
        
        return {
            'healthy': all_healthy,
            'checks': checks,
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_model_loaded(self):
        try:
            # Check if model is accessible
            hasattr(self.model, 'encode')
            return {'status': True, 'message': 'Model loaded successfully'}
        except Exception as e:
            return {'status': False, 'message': f'Model load error: {e}'}
    
    def _check_inference(self):
        try:
            embeddings = self.model.encode([self.test_text])
            if len(embeddings) > 0 and len(embeddings[0]) > 0:
                return {'status': True, 'message': 'Inference working'}
            else:
                return {'status': False, 'message': 'Empty embeddings returned'}
        except Exception as e:
            return {'status': False, 'message': f'Inference error: {e}'}
    
    def _check_response_time(self):
        try:
            start = time.time()
            self.model.encode([self.test_text])
            response_time = time.time() - start
            
            if response_time < 5.0:  # 5 second threshold
                return {'status': True, 'message': f'Response time: {response_time:.2f}s'}
            else:
                return {'status': False, 'message': f'Slow response: {response_time:.2f}s'}
        except Exception as e:
            return {'status': False, 'message': f'Response time check error: {e}'}
    
    def _check_memory(self):
        try:
            import torch
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                usage_percent = (memory_used / memory_total) * 100
                
                if usage_percent < 90:
                    return {'status': True, 'message': f'GPU memory: {usage_percent:.1f}%'}
                else:
                    return {'status': False, 'message': f'High GPU memory: {usage_percent:.1f}%'}
            else:
                return {'status': True, 'message': 'CPU mode, memory check skipped'}
        except Exception as e:
            return {'status': False, 'message': f'Memory check error: {e}'}
```

## Configuration Validation

```python
import jsonschema

# Model configuration schema
MODEL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "minLength": 1},
        "endpoint_type": {
            "type": "string",
            "enum": ["sentence_transformers", "transformers", "tei", "openai", "custom"]
        },
        "device": {"type": "string"},
        "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
        "max_length": {"type": "integer", "minimum": 1, "maximum": 8192},
        "normalize_embeddings": {"type": "boolean"},
        "model_kwargs": {"type": "object"},
        "tokenizer_kwargs": {"type": "object"}
    },
    "required": ["model_name", "endpoint_type"],
    "additionalProperties": True
}

def validate_model_config(config):
    """Validate model configuration."""
    try:
        jsonschema.validate(config, MODEL_CONFIG_SCHEMA)
        return True, "Configuration is valid"
    except jsonschema.ValidationError as e:
        return False, f"Configuration error: {e.message}"

# Usage
config = {
    "model_name": "thenlper/gte-small",
    "endpoint_type": "sentence_transformers",
    "device": "cuda:0",
    "batch_size": 32
}

is_valid, message = validate_model_config(config)
print(message)
```

This comprehensive guide covers all aspects of model configuration and customization. For additional help with specific models or advanced configurations, refer to the model-specific documentation or community resources.
