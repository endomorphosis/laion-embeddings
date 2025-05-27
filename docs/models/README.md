# Supported Models

LAION Embeddings supports multiple state-of-the-art embedding models for generating high-quality vector representations from text data.

## Overview

The system currently supports three main embedding models, each optimized for different use cases:

| Model | Dimensions | Context Length | Use Case | Memory |
|-------|------------|----------------|----------|---------|
| gte-small | 384 | 512 | Fast inference, low memory | 2GB |
| gte-large-en-v1.5 | 1024 | 8192 | High quality, English | 8GB |
| gte-Qwen2-1.5B-instruct | 1536 | 32768 | Latest, multilingual | 12GB |

## Model Details

### thenlper/gte-small

**General Text Embeddings (Small)**

- **Dimensions**: 384
- **Max Sequence Length**: 512 tokens
- **Model Size**: ~134M parameters
- **Memory Requirements**: ~2GB
- **Languages**: Primarily English
- **Performance**: Fast inference, good for real-time applications

**Best For:**
- Real-time applications
- Resource-constrained environments
- Quick prototyping
- Large-scale batch processing

**Performance Benchmarks:**
- Inference speed: ~1000 sequences/second (CPU)
- Memory usage: 2GB RAM
- Accuracy: Good for general text similarity

### Alibaba-NLP/gte-large-en-v1.5

**General Text Embeddings (Large, English v1.5)**

- **Dimensions**: 1024
- **Max Sequence Length**: 8192 tokens
- **Model Size**: ~434M parameters
- **Memory Requirements**: ~8GB
- **Languages**: English (optimized)
- **Performance**: High quality embeddings

**Best For:**
- High-accuracy applications
- Long document processing
- English text analysis
- Research and development

**Performance Benchmarks:**
- Inference speed: ~200 sequences/second (CPU)
- Memory usage: 8GB RAM
- Accuracy: Excellent for English text

### Alibaba-NLP/gte-Qwen2-1.5B-instruct

**Qwen2 General Text Embeddings (1.5B, Instruct)**

- **Dimensions**: 1536
- **Max Sequence Length**: 32768 tokens
- **Model Size**: ~1.5B parameters
- **Memory Requirements**: ~12GB
- **Languages**: Multilingual
- **Performance**: State-of-the-art quality

**Best For:**
- Cutting-edge applications
- Very long documents
- Multilingual content
- Instruction-following tasks

**Performance Benchmarks:**
- Inference speed: ~50 sequences/second (CPU)
- Memory usage: 12GB RAM
- Accuracy: State-of-the-art across languages

## Model Selection Guide

### By Use Case

#### Real-time Applications
**Recommended**: `thenlper/gte-small`
- Fast inference
- Low latency
- Minimal memory footprint

#### High Accuracy Requirements
**Recommended**: `Alibaba-NLP/gte-large-en-v1.5`
- Superior embedding quality
- Good balance of speed and accuracy
- Proven performance

#### Multilingual Content
**Recommended**: `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
- Best multilingual support
- Latest model architecture
- Highest quality embeddings

#### Long Documents
**Recommended**: `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
- 32K context length
- Better understanding of long texts
- Maintains quality across entire document

### By Resource Constraints

#### Limited Memory (< 4GB)
**Use**: `thenlper/gte-small`
```yaml
model_config:
  name: "thenlper/gte-small"
  batch_size: 64
  max_workers: 2
```

#### Moderate Memory (4-10GB)
**Use**: `Alibaba-NLP/gte-large-en-v1.5`
```yaml
model_config:
  name: "Alibaba-NLP/gte-large-en-v1.5"
  batch_size: 32
  max_workers: 1
```

#### High Memory (> 10GB)
**Use**: `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
```yaml
model_config:
  name: "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
  batch_size: 16
  max_workers: 1
```

## Model Configuration

### Default Configuration

```python
DEFAULT_MODELS = {
    "thenlper/gte-small": {
        "dimensions": 384,
        "max_length": 512,
        "batch_size": 64,
        "device": "cpu"
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "dimensions": 1024,
        "max_length": 8192,
        "batch_size": 32,
        "device": "cpu"
    },
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "dimensions": 1536,
        "max_length": 32768,
        "batch_size": 16,
        "device": "cpu"
    }
}
```

### GPU Configuration

For GPU acceleration:

```python
GPU_MODELS = {
    "thenlper/gte-small": {
        "device": "cuda:0",
        "batch_size": 128,
        "mixed_precision": True
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "device": "cuda:0", 
        "batch_size": 64,
        "mixed_precision": True
    },
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "device": "cuda:0",
        "batch_size": 32,
        "mixed_precision": True
    }
}
```

## Model Loading

### Automatic Download

Models are automatically downloaded on first use:

```python
from transformers import AutoModel, AutoTokenizer

# This will download the model if not already cached
model = AutoModel.from_pretrained("thenlper/gte-small")
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
```

### Manual Pre-download

Pre-download models for offline use:

```bash
# Download all supported models
python -c "
from transformers import AutoModel, AutoTokenizer
models = [
    'thenlper/gte-small',
    'Alibaba-NLP/gte-large-en-v1.5', 
    'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
]
for model_name in models:
    print(f'Downloading {model_name}...')
    AutoModel.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)
    print(f'Downloaded {model_name}')
"
```

### Custom Cache Directory

```bash
export TRANSFORMERS_CACHE=/path/to/model/cache
export HF_HOME=/path/to/hf/cache
```

## Performance Comparison

### Inference Speed (sequences/second)

| Model | CPU (8 cores) | GPU (RTX 3080) | Batch Size |
|-------|---------------|----------------|------------|
| gte-small | 1000 | 5000 | 64 |
| gte-large-en-v1.5 | 200 | 1000 | 32 |
| gte-Qwen2-1.5B-instruct | 50 | 250 | 16 |

### Memory Usage

| Model | CPU Memory | GPU Memory | Model Size |
|-------|------------|------------|------------|
| gte-small | 2GB | 1GB | 500MB |
| gte-large-en-v1.5 | 8GB | 4GB | 1.7GB |
| gte-Qwen2-1.5B-instruct | 12GB | 8GB | 6GB |

### Quality Metrics

| Model | MTEB Score | STS Benchmark | Retrieval Accuracy |
|-------|------------|---------------|-------------------|
| gte-small | 61.3 | 82.1 | Good |
| gte-large-en-v1.5 | 65.7 | 85.4 | Very Good |
| gte-Qwen2-1.5B-instruct | 68.2 | 87.9 | Excellent |

## Model-Specific Features

### gte-small Features
- Fast inference
- Lightweight deployment
- Good general-purpose embeddings
- Optimized for speed

### gte-large-en-v1.5 Features
- High-quality English embeddings
- Better semantic understanding
- Improved similarity detection
- Good for retrieval tasks

### gte-Qwen2-1.5B-instruct Features
- Instruction-following capabilities
- Multilingual support
- Very long context support
- State-of-the-art performance
- Better handling of complex queries

## Integration Examples

### Single Model Usage

```python
# Using gte-small for fast processing
resources = {
    "models": ["thenlper/gte-small"],
    "batch_size": 64
}

embeddings = create_embeddings.create_embeddings(resources, metadata)
await embeddings.process_dataset(dataset="my_dataset", column="text")
```

### Multi-Model Processing

```python
# Using multiple models for comparison
resources = {
    "models": [
        "thenlper/gte-small",
        "Alibaba-NLP/gte-large-en-v1.5"
    ]
}

# This will create embeddings with both models
embeddings = create_embeddings.create_embeddings(resources, metadata)
await embeddings.process_dataset(
    dataset="my_dataset",
    column="text",
    dst_path="./multi_model_output"
)
```

### Dynamic Model Selection

```python
def select_model(text_length, quality_requirement):
    if text_length < 100 and quality_requirement == "fast":
        return "thenlper/gte-small"
    elif text_length < 1000 and quality_requirement == "balanced":
        return "Alibaba-NLP/gte-large-en-v1.5"
    else:
        return "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# Dynamic selection based on content
model = select_model(len(text), "high_quality")
```

## Model Updates and Versioning

### Version Management

Models are versioned through HuggingFace model hub:
- `thenlper/gte-small` (latest)
- `Alibaba-NLP/gte-large-en-v1.5` (version 1.5)
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (Qwen2 series)

### Update Strategy

```python
# Check for model updates
def check_model_updates():
    from transformers import AutoModel
    
    models = ["thenlper/gte-small"]
    for model_name in models:
        try:
            # This will check for updates
            AutoModel.from_pretrained(
                model_name,
                force_download=False,
                resume_download=True
            )
        except Exception as e:
            print(f"Update check failed for {model_name}: {e}")
```

## Custom Model Integration

### Adding New Models

To add support for new models:

1. **Update model configuration**:
```python
CUSTOM_MODELS = {
    "new_model/name": {
        "dimensions": 768,
        "max_length": 1024,
        "batch_size": 32
    }
}
```

2. **Add model validation**:
```python
def validate_custom_model(model_name):
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
```

3. **Update endpoint configuration**:
```yaml
endpoints:
  local_endpoints:
    - model: "new_model/name"
      path: "./models/new_model"
      device: "cpu"
```

## Troubleshooting

### Common Issues

#### Model Download Failures
```bash
# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf ~/.cache/huggingface/
python -c "from transformers import AutoModel; AutoModel.from_pretrained('thenlper/gte-small')"
```

#### Memory Issues
```python
# Reduce batch size
model_config = {
    "batch_size": 8,  # Reduced from default
    "max_workers": 1
}
```

#### GPU Out of Memory
```python
# Enable mixed precision
model_config = {
    "mixed_precision": True,
    "gradient_checkpointing": True
}
```

### Performance Issues

#### Slow Inference
- Use smaller model (gte-small)
- Increase batch size
- Use GPU acceleration
- Optimize sequence length

#### High Memory Usage
- Use model quantization
- Enable gradient checkpointing
- Reduce batch size
- Use CPU offloading

## Best Practices

### Model Selection
1. **Start with gte-small** for prototyping
2. **Upgrade to gte-large** for production quality
3. **Use gte-Qwen2** for cutting-edge applications
4. **Consider resource constraints** in selection

### Performance Optimization
1. **Use appropriate batch sizes** for your hardware
2. **Enable GPU acceleration** when available
3. **Cache model instances** to avoid reloading
4. **Monitor memory usage** and adjust accordingly

### Quality Optimization
1. **Choose model based on content type**
2. **Use appropriate sequence lengths**
3. **Consider ensemble methods** for critical applications
4. **Validate results** with human evaluation

## Related Documentation

- [Model Configuration](configuration.md) - Detailed configuration options
- [Endpoint Management](../endpoints/README.md) - Managing model endpoints
- [Performance Tuning](../troubleshooting/performance.md) - Optimization guides
- [Custom Models](custom-models.md) - Adding custom models
