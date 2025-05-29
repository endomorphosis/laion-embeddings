# Development Guide

This guide covers development workflows, testing procedures, and deployment strategies for the LAION Embeddings project.

## Recent Updates (May 28, 2025)

- **Tokenization Workflow Validation**: Complete end-to-end validation of the token processing pipeline
- **Enhanced Testing Infrastructure**: Comprehensive test suite for tokenization, chunking, and CID generation
- **Production-Ready Error Handling**: Robust error handling with fallback mechanisms across all safe_* functions
- **Performance Optimizations**: Improved batch processing with validated token workflows

## Getting Started

### Development Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/laion-embeddings.git
   cd laion-embeddings
   ```

2. **Create Development Environment**
   ```bash
   # Using conda
   conda create -n laion-embeddings python=3.9
   conda activate laion-embeddings
   
   # Using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .  # Install in editable mode
   ```

4. **Install Additional Development Tools**
   ```bash
   # Code formatting and linting
   pip install black isort flake8 mypy
   
   # Testing frameworks
   pip install pytest pytest-asyncio pytest-cov
   
   # Documentation
   pip install sphinx sphinx-rtd-theme
   ```

### Project Structure

```
laion-embeddings/
├── docs/                           # Documentation
├── ipfs_embeddings_py/            # Core package
│   ├── __init__.py
│   ├── ipfs_embeddings.py         # Main embeddings class
│   ├── ipfs_datasets.py           # Dataset handling
│   ├── ipfs_multiformats.py       # IPFS content addressing
│   ├── ipfs_parquet_to_car.py     # Data format conversion
│   ├── qdrant_kit.py              # Qdrant integration
│   ├── elasticsearch_kit.py       # Elasticsearch integration
│   └── faiss_kit.py               # FAISS integration
├── create_embeddings/             # Embedding creation component
├── search_embeddings/             # Search functionality
├── sparse_embeddings/             # Sparse embedding processing
├── shard_embeddings/              # Dataset sharding
├── ipfs_cluster_index/            # IPFS cluster management
├── storacha_clusters/             # Storacha integration
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
├── main.py                        # FastAPI application
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
└── README.md                      # Main documentation
```

## Development Workflow

### Git Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-embedding-model
   ```

2. **Make Changes and Commit**
   ```bash
   git add .
   git commit -m "Add support for new embedding model"
   ```

3. **Push and Create Pull Request**
   ```bash
   git push origin feature/new-embedding-model
   # Create PR through GitHub interface
   ```

### Code Style and Standards

#### Python Code Formatting

```bash
# Format code with black
black ipfs_embeddings_py/ tests/

# Sort imports with isort
isort ipfs_embeddings_py/ tests/

# Check linting with flake8
flake8 ipfs_embeddings_py/ tests/

# Type checking with mypy
mypy ipfs_embeddings_py/
```

#### Pre-commit Hooks

Set up pre-commit hooks to automatically format code:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### Adding New Features

#### 1. Adding a New Embedding Model

```python
# Example: Adding support for a new model
class NewEmbeddingModel:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        # Implementation here
        pass
    
    async def embed_batch(self, texts, batch_size=32):
        """Generate embeddings for a batch of texts"""
        # Implementation here
        pass
    
    def get_context_length(self):
        """Return maximum context length"""
        return self.config.get("context_length", 512)

# Register the model in the main embeddings class
class ipfs_embeddings_py:
    def __init__(self, resources, metadata):
        # ... existing code ...
        self.supported_models["new-model"] = NewEmbeddingModel
```

#### 2. Adding a New Storage Backend

```python
# Example: Adding a new storage backend
class NewStorageBackend:
    def __init__(self, config):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize storage client"""
        # Implementation here
        pass
    
    async def store(self, key, data):
        """Store data with given key"""
        # Implementation here
        pass
    
    async def retrieve(self, key):
        """Retrieve data by key"""
        # Implementation here
        pass
    
    async def list_keys(self, prefix=None):
        """List stored keys"""
        # Implementation here
        pass

# Register the backend
storage_backends = {
    "faiss": faiss_kit_py,
    "qdrant": qdrant_kit_py,
    "elasticsearch": elasticsearch_kit,
    "new_backend": NewStorageBackend
}
```

## Testing

### Test Structure

```
tests/
├── unit/                          # Unit tests
│   ├── test_embeddings.py
│   ├── test_datasets.py
│   ├── test_multiformats.py
│   ├── test_tokenization.py       # NEW: Tokenization workflow tests
│   └── test_storage_backends.py
├── integration/                   # Integration tests
│   ├── test_end_to_end.py
│   ├── test_api_endpoints.py
│   ├── test_tokenization_workflow.py  # NEW: Full workflow validation
│   └── test_storage_integration.py
├── performance/                   # Performance tests
│   ├── test_embedding_speed.py
│   ├── test_tokenization_performance.py  # NEW: Token processing performance
│   └── test_search_performance.py
├── validation/                    # NEW: Workflow validation tests
│   ├── basic_validation.py       # Basic tokenization validation
│   ├── comprehensive_test_suite.py  # Complete workflow testing
│   └── file_based_test.py        # File-based validation tests
├── fixtures/                      # Test data and fixtures
│   ├── sample_datasets.py
│   └── mock_embeddings.py
└── conftest.py                   # Pytest configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_embeddings.py

# Run tokenization workflow validation
pytest tests/validation/ -v

# Run basic tokenization validation
python test/basic_validation.py

# Run comprehensive workflow tests
python test/comprehensive_test_suite.py

# Run with coverage
pytest --cov=ipfs_embeddings_py --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
pytest -m "tokenization"  # Run only tokenization tests
```

### Test Configuration

`conftest.py`:
```python
import pytest
import asyncio
from ipfs_embeddings_py import ipfs_embeddings_py

@pytest.fixture
def resources():
    return {
        "local_endpoints": [
            ["thenlper/gte-small", "cpu", 512]
        ],
        "tei_endpoints": [],
        "storage_path": "/tmp/test_storage"
    }

@pytest.fixture
def metadata():
    return {
        "dataset": "test_dataset",
        "models": ["thenlper/gte-small"],
        "chunk_settings": {
            "chunk_size": 256,
            "method": "fixed"
        }
    }

@pytest.fixture
async def embeddings_client(resources, metadata):
    client = ipfs_embeddings_py(resources, metadata)
    await client.init_endpoints()
    yield client
    # Cleanup
    await client.cleanup()

@pytest.fixture
def sample_texts():
    return [
        "This is a sample text for testing.",
        "Another example sentence for embeddings.",
        "Testing the embedding generation process."
    ]
```

### Writing Tests

#### Unit Test Example

```python
import pytest
import numpy as np
from ipfs_embeddings_py.ipfs_multiformats import ipfs_multiformats_py

class TestMultiformats:
    def test_cid_generation(self):
        """Test CID generation for text content"""
        multiformats = ipfs_multiformats_py()
        
        text = "Sample text content"
        cid = multiformats.get_cid(text)
        
        # CID should be a string
        assert isinstance(cid, str)
        
        # CID should be deterministic
        cid2 = multiformats.get_cid(text)
        assert cid == cid2
    
    def test_file_hash(self, tmp_path):
        """Test file hashing functionality"""
        multiformats = ipfs_multiformats_py()
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Get hash
        file_hash = multiformats.get_file_sha256(str(test_file))
        
        # Hash should be bytes
        assert isinstance(file_hash, bytes)
        assert len(file_hash) == 32  # SHA-256 is 32 bytes
```

#### Integration Test Example

```python
import pytest
from ipfs_embeddings_py import ipfs_embeddings_py

@pytest.mark.asyncio
class TestEmbeddingIntegration:
    async def test_end_to_end_embedding(self, embeddings_client, sample_texts):
        """Test complete embedding generation pipeline"""
        
        # Generate embeddings
        results = await embeddings_client.embed_texts(
            texts=sample_texts,
            model="thenlper/gte-small"
        )
        
        # Verify results
        assert len(results) == len(sample_texts)
        
        for result in results:
            assert "embedding" in result
            assert "cid" in result
            assert isinstance(result["embedding"], list)
            assert len(result["embedding"]) == 384  # gte-small dimension
    
    async def test_storage_retrieval(self, embeddings_client, sample_texts):
        """Test storing and retrieving embeddings"""
        
        # Store embeddings
        storage_keys = await embeddings_client.store_embeddings(
            texts=sample_texts,
            model="thenlper/gte-small"
        )
        
        # Retrieve embeddings
        for key in storage_keys:
            retrieved = await embeddings_client.retrieve_embedding(key)
            assert retrieved is not None
            assert "embedding" in retrieved
```

### Performance Testing

```python
import pytest
import time
import numpy as np
from ipfs_embeddings_py import ipfs_embeddings_py

@pytest.mark.performance
class TestPerformance:
    @pytest.mark.asyncio
    async def test_embedding_throughput(self, embeddings_client):
        """Test embedding generation throughput"""
        
        # Generate test texts
        texts = [f"Sample text {i}" for i in range(1000)]
        
        # Measure time
        start_time = time.time()
        results = await embeddings_client.embed_texts(
            texts=texts,
            model="thenlper/gte-small",
            batch_size=32
        )
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        throughput = len(texts) / total_time
        
        print(f"Processed {len(texts)} texts in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} texts/second")
        
        # Performance assertions
        assert throughput > 10  # At least 10 texts per second
        assert len(results) == len(texts)
    
    @pytest.mark.asyncio
    async def test_search_performance(self, embeddings_client):
        """Test search performance with large index"""
        
        # Create large embedding index
        texts = [f"Document {i} with content" for i in range(10000)]
        await embeddings_client.index_texts(texts)
        
        # Test search performance
        query = "sample search query"
        
        start_time = time.time()
        results = await embeddings_client.search(query, top_k=10)
        end_time = time.time()
        
        search_time = end_time - start_time
        
        print(f"Search completed in {search_time:.3f}s")
        
        # Performance assertions
        assert search_time < 1.0  # Search should complete in under 1 second
        assert len(results) == 10
```

### Tokenization Workflow Testing

The project includes comprehensive testing for the tokenization workflow, validating the complete sequence: 
Text → Tokenization → Chunking → CID → Batch → Embeddings.

#### Basic Tokenization Validation

```python
import pytest
from ipfs_embeddings_py.chunker import chunker_py
from ipfs_embeddings_py.ipfs_multiformats import ipfs_multiformats_py

@pytest.mark.tokenization
class TestTokenizationWorkflow:
    def test_safe_tokenizer_encode(self):
        """Test safe tokenization encoding with error handling"""
        chunker = chunker_py()
        
        # Test normal case
        text = "This is a test sentence."
        result = chunker.safe_tokenizer_encode(text)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'tokens' in result
        assert 'success' in result
        assert result['success'] is True
    
    def test_safe_tokenizer_decode(self):
        """Test safe tokenization decoding with validation"""
        chunker = chunker_py()
        
        # First encode
        text = "This is a test sentence."
        encoded = chunker.safe_tokenizer_encode(text)
        
        # Then decode
        decoded = chunker.safe_tokenizer_decode(encoded['tokens'])
        
        assert decoded is not None
        assert decoded['success'] is True
        assert decoded['text'].strip() == text.strip()
    
    def test_safe_chunker_chunk(self):
        """Test safe chunking with error handling"""
        chunker = chunker_py()
        
        text = "This is a longer text that should be chunked into smaller pieces for processing."
        chunks = chunker.safe_chunker_chunk(text, chunk_size=256)
        
        assert chunks is not None
        assert isinstance(chunks, dict)
        assert 'chunks' in chunks
        assert 'success' in chunks
        assert chunks['success'] is True
        assert len(chunks['chunks']) > 0
    
    def test_safe_get_cid(self):
        """Test safe CID generation with validation"""
        multiformats = ipfs_multiformats_py()
        
        text = "Test content for CID generation"
        cid_result = multiformats.safe_get_cid(text)
        
        assert cid_result is not None
        assert isinstance(cid_result, dict)
        assert 'cid' in cid_result
        assert 'success' in cid_result
        assert cid_result['success'] is True
        assert isinstance(cid_result['cid'], str)
    
    def test_complete_workflow_sequence(self):
        """Test the complete tokenization workflow sequence"""
        chunker = chunker_py()
        multiformats = ipfs_multiformats_py()
        
        # Input text
        text = "Complete workflow test with multiple processing steps."
        
        # Step 1: Tokenization
        encoded = chunker.safe_tokenizer_encode(text)
        assert encoded['success'] is True
        
        # Step 2: Chunking
        chunks = chunker.safe_chunker_chunk(text, chunk_size=256)
        assert chunks['success'] is True
        
        # Step 3: CID generation for each chunk
        for chunk in chunks['chunks']:
            cid_result = multiformats.safe_get_cid(chunk)
            assert cid_result['success'] is True
            assert len(cid_result['cid']) > 0
        
        # Step 4: Decoding validation
        decoded = chunker.safe_tokenizer_decode(encoded['tokens'])
        assert decoded['success'] is True
```

#### Running Workflow Validation Tests

```bash
# Run basic validation
python test/basic_validation.py

# Run comprehensive test suite
python test/comprehensive_test_suite.py

# Run specific tokenization tests
pytest -m tokenization -v

# Run validation tests with coverage
pytest tests/validation/ --cov=ipfs_embeddings_py --cov-report=html
```

### Test Markers

`pytest.ini`:
```ini
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    tokenization: Tokenization workflow tests
    slow: Tests that take a long time to run
    requires_gpu: Tests that require GPU
    requires_network: Tests that require network access
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

### Documentation Standards

1. **Docstring Format**: Use Google-style docstrings

```python
def embed_text(self, text: str, model: str = "gte-small") -> Dict[str, Any]:
    """Generate embedding for a single text.
    
    Args:
        text: Input text to embed
        model: Name of the embedding model to use
        
    Returns:
        Dictionary containing embedding vector and metadata
        
    Raises:
        ValueError: If model is not supported
        RuntimeError: If embedding generation fails
        
    Example:
        >>> embedder = EmbeddingGenerator()
        >>> result = embedder.embed_text("Hello world")
        >>> print(result["embedding"][:5])
        [0.1, 0.2, 0.3, 0.4, 0.5]
    """
```

2. **Type Hints**: Use comprehensive type hints

```python
from typing import List, Dict, Any, Optional, Union
import numpy as np

async def process_batch(
    self,
    texts: List[str],
    model: str,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process a batch of texts."""
```

## Deployment

### Docker Deployment

#### Production Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for CAR conversion
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  laion-embeddings:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    volumes:
      - ./storage:/app/storage
      - ./config:/app/config
    depends_on:
      - qdrant
      - elasticsearch
    restart: unless-stopped
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped
  
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    restart: unless-stopped

volumes:
  qdrant_data:
  elasticsearch_data:
```

### Kubernetes Deployment

#### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: laion-embeddings
  labels:
    app: laion-embeddings
spec:
  replicas: 3
  selector:
    matchLabels:
      app: laion-embeddings
  template:
    metadata:
      labels:
        app: laion-embeddings
    spec:
      containers:
      - name: laion-embeddings
        image: laion-embeddings:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: laion-embeddings-service
spec:
  selector:
    app: laion-embeddings
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Environment Configuration

#### Production Environment Variables

```bash
# Application Settings
export ENVIRONMENT=production
export LOG_LEVEL=info
export DEBUG=false

# Storage Settings
export STORAGE_PATH=/app/storage
export CACHE_SIZE=10000
export BATCH_SIZE=32

# Model Settings
export DEFAULT_MODEL=thenlper/gte-small
export MODEL_CACHE_PATH=/app/models

# Performance Settings
export MAX_WORKERS=8
export QUEUE_SIZE=1000
export TIMEOUT=300

# Security Settings
export API_KEY_REQUIRED=true
export CORS_ORIGINS="https://yourdomain.com"
```

### Monitoring and Logging

#### Logging Configuration

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Configure logging for production"""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
```

#### Metrics Collection

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
embedding_requests = Counter(
    'embedding_requests_total',
    'Total embedding requests',
    ['model', 'status']
)

embedding_duration = Histogram(
    'embedding_duration_seconds',
    'Time spent generating embeddings',
    ['model']
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Collect metrics for each request"""
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Record metrics
    if "/embed" in str(request.url):
        model = request.query_params.get("model", "unknown")
        status = "success" if response.status_code == 200 else "error"
        
        embedding_requests.labels(model=model, status=status).inc()
        embedding_duration.labels(model=model).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")
```

## Contributing Guidelines

### Pull Request Process

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/your-feature`
3. **Make Changes**: Follow coding standards and add tests
4. **Run Tests**: Ensure all tests pass
5. **Update Documentation**: Update relevant documentation
6. **Submit PR**: Create pull request with clear description

### Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

### Issue Reporting

When reporting issues, include:

1. **Environment Details**: OS, Python version, package versions
2. **Reproduction Steps**: Minimal code to reproduce the issue
3. **Expected vs Actual Behavior**: Clear description of the problem
4. **Error Messages**: Full error traces and logs
5. **Additional Context**: Any relevant configuration or data

## Release Process

### Version Management

1. **Update Version**: Update version in `__init__.py`
2. **Update Changelog**: Document changes in `CHANGELOG.md`
3. **Create Release**: Tag release in Git
4. **Build Packages**: Build wheel and source distributions
5. **Deploy**: Upload to PyPI and update Docker images

### Automated Release Pipeline

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install build twine
      
      - name: Run tests
        run: pytest
      
      - name: Build package
        run: python -m build
      
      - name: Upload to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```
