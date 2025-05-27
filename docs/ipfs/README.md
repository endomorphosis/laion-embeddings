# IPFS Integration Documentation

The LAION Embeddings project provides comprehensive integration with IPFS (InterPlanetary File System) for decentralized storage and retrieval of embeddings data. This documentation covers all IPFS-related functionality including content addressing, data formats, and storage workflows.

## Overview

The IPFS integration enables:

- **Content Addressing**: Generate and manage Content Identifiers (CIDs) for embeddings data
- **Decentralized Storage**: Store and retrieve embeddings across IPFS networks
- **Data Conversion**: Convert between data formats (Parquet ↔ CAR, folders ↔ datasets)
- **Multiformats Support**: Handle CID generation and content verification
- **Cluster Integration**: Work with IPFS clusters for redundancy and performance

## Core Components

### 1. IPFS Multiformats (`ipfs_multiformats.py`)

Handles CID generation and content addressing using IPFS multiformats standards.

```python
from ipfs_embeddings_py.ipfs_multiformats import ipfs_multiformats_py

# Initialize multiformats handler
multiformats = ipfs_multiformats_py()

# Generate CID from text content
text_content = "Sample embedding data"
cid = multiformats.get_cid(text_content)
print(f"Generated CID: {cid}")

# Generate CID from file
file_cid = multiformats.get_cid("/path/to/file.txt")
print(f"File CID: {file_cid}")

# Get SHA-256 hash of file
file_hash = multiformats.get_file_sha256("/path/to/file.txt")
print(f"File hash: {file_hash.hex()}")
```

#### Key Methods

- `get_cid(data)`: Generate CID for text or file data
- `get_file_sha256(file_path)`: Get SHA-256 hash of file
- `get_multihash_sha256(hash)`: Wrap hash in multihash format

### 2. Parquet to CAR Conversion (`ipfs_parquet_to_car.py`)

Converts Parquet files to CAR (Content Addressable aRchive) format for IPFS storage.

```python
from ipfs_embeddings_py.ipfs_parquet_to_car import ipfs_parquet_to_car_py

# Initialize converter
resources = {
    "ipfs_parquet_to_car.js": "/path/to/conversion/script.js"
}
metadata = {
    "name": "parquet_converter",
    "version": "1.0.0"
}

converter = ipfs_parquet_to_car_py(resources, metadata)

# Install Node.js dependencies
await converter.install()

# Convert single file
await converter.run("/path/to/input.parquet", "/path/to/output.car")

# Batch convert directory
await converter.run_batch("/input/directory", "/output/directory")
```

#### Configuration Options

```python
conversion_config = {
    "batch_size": 100,           # Files per batch
    "parallel_workers": 4,       # Parallel conversion processes
    "compression": "gzip",       # Compression method
    "validation": True           # Validate output files
}
```

### 3. Folder to Parquet Conversion (`ipfs_folder_to_parquet.py`)

Converts filesystem folders to Parquet datasets and vice versa.

```python
from ipfs_embeddings_py.ipfs_folder_to_parquet import ipfs_folder_to_parquet

# Initialize converter
converter = ipfs_folder_to_parquet(resources, metadata)

# Convert folder to Parquet
converter.folder_to_parquet("/input/folder", "/output/dataset.parquet")

# Convert Parquet to folder
converter.parquet_to_folder("/input/dataset.parquet", "/output/folder")

# Create HuggingFace dataset from folder
dataset = converter.folder_to_huggingface_dataset("/input/folder")
print(f"Created dataset with {len(dataset)} files")
```

#### Supported Operations

- **Folder → Parquet**: Convert directory contents to structured dataset
- **Parquet → Folder**: Extract files from dataset back to filesystem
- **HuggingFace Integration**: Create datasets compatible with HuggingFace libraries

## Content Addressing and CID Management

### CID Generation Workflow

```python
# Example: Generate CIDs for embedding chunks
def generate_embedding_cids(embeddings_data):
    """Generate CIDs for embedding data chunks"""
    
    multiformats = ipfs_multiformats_py()
    cid_mappings = {}
    
    for chunk_id, chunk_data in embeddings_data.items():
        # Serialize chunk data
        chunk_text = json.dumps(chunk_data, sort_keys=True)
        
        # Generate CID
        chunk_cid = multiformats.get_cid(chunk_text)
        cid_mappings[chunk_id] = chunk_cid
        
        print(f"Chunk {chunk_id}: {chunk_cid}")
    
    return cid_mappings

# Usage
embeddings = {
    "chunk_001": {
        "text": "Sample text content",
        "embedding": [0.1, 0.2, 0.3, ...],
        "metadata": {"model": "gte-small"}
    }
}

cids = generate_embedding_cids(embeddings)
```

### Content Verification

```python
def verify_content_integrity(file_path, expected_cid):
    """Verify file content matches expected CID"""
    
    multiformats = ipfs_multiformats_py()
    
    # Generate CID from current file
    actual_cid = multiformats.get_cid(file_path)
    
    # Compare with expected CID
    if actual_cid == expected_cid:
        print(f"✓ Content integrity verified for {file_path}")
        return True
    else:
        print(f"✗ Content mismatch: expected {expected_cid}, got {actual_cid}")
        return False
```

## IPFS Datasets Integration

### Loading IPFS-stored Datasets

```python
from ipfs_embeddings_py.ipfs_datasets import ipfs_datasets_py

# Initialize datasets handler
datasets_handler = ipfs_datasets_py(resources, metadata)

# Load dataset from IPFS
dataset = await datasets_handler.load_dataset("dataset_name", split="train")

# Load checkpoints from distributed storage
models = ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5"]
await datasets_handler.load_checkpoints(
    dataset="my_dataset",
    split="train", 
    dst_path="/storage/path",
    models=models
)
```

### Content Chunking with CIDs

```python
# Example: Chunk content and generate CIDs
async def chunk_and_index_content(content, chunker_config):
    """Chunk content and generate CIDs for each chunk"""
    
    multiformats = ipfs_multiformats_py()
    chunks_with_cids = []
    
    # Generate parent CID for original content
    parent_cid = multiformats.get_cid(content)
    
    # Chunk the content
    chunks = chunker.chunk_text(content, **chunker_config)
    
    # Generate CIDs for each chunk
    for i, chunk in enumerate(chunks):
        chunk_cid = multiformats.get_cid(chunk)
        
        chunk_data = {
            "cid": chunk_cid,
            "parent_cid": parent_cid,
            "index": i,
            "content": chunk,
            "metadata": {
                "chunk_size": len(chunk),
                "position": i
            }
        }
        
        chunks_with_cids.append(chunk_data)
    
    return chunks_with_cids

# Usage
chunker_config = {
    "chunk_size": 512,
    "overlap": 50,
    "method": "sentence"
}

chunks = await chunk_and_index_content("Long text content...", chunker_config)
```

## Data Format Conversions

### Parquet to CAR Pipeline

```python
async def embeddings_to_ipfs_pipeline(parquet_path, output_dir):
    """Convert embeddings from Parquet to IPFS-ready CAR format"""
    
    converter = ipfs_parquet_to_car_py(resources, metadata)
    
    # Ensure dependencies are installed
    await converter.install()
    
    # Convert to CAR format
    car_path = f"{output_dir}/embeddings.car"
    await converter.run(parquet_path, car_path)
    
    # Generate CID for the CAR file
    multiformats = ipfs_multiformats_py()
    car_cid = multiformats.get_cid(car_path)
    
    print(f"Generated CAR file: {car_path}")
    print(f"CAR file CID: {car_cid}")
    
    return car_path, car_cid
```

### Batch Processing

```python
async def batch_convert_embeddings(input_dir, output_dir):
    """Batch convert multiple Parquet files to CAR format"""
    
    converter = ipfs_parquet_to_car_py(resources, metadata)
    
    # Get all Parquet files
    parquet_files = glob.glob(f"{input_dir}/*.parquet")
    print(f"Found {len(parquet_files)} Parquet files")
    
    # Batch convert
    await converter.run_batch(input_dir, output_dir)
    
    # Generate manifest with CIDs
    manifest = {}
    multiformats = ipfs_multiformats_py()
    
    for car_file in glob.glob(f"{output_dir}/*.car"):
        car_cid = multiformats.get_cid(car_file)
        manifest[os.path.basename(car_file)] = car_cid
    
    # Save manifest
    with open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest
```

## Integration with Embedding Processing

### CID-based Embedding Storage

```python
class IPFSEmbeddingStorage:
    """Store and retrieve embeddings using IPFS content addressing"""
    
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.multiformats = ipfs_multiformats_py()
        self.cid_index = {}
    
    async def store_embedding(self, text, embedding, metadata=None):
        """Store embedding with content-addressed ID"""
        
        # Create embedding record
        record = {
            "text": text,
            "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        # Generate CID
        record_json = json.dumps(record, sort_keys=True)
        cid = self.multiformats.get_cid(record_json)
        
        # Store to filesystem (could be IPFS)
        storage_file = f"{self.storage_path}/{cid}.json"
        with open(storage_file, "w") as f:
            json.dump(record, f)
        
        # Update index
        self.cid_index[cid] = {
            "file": storage_file,
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "stored_at": time.time()
        }
        
        return cid
    
    async def retrieve_embedding(self, cid):
        """Retrieve embedding by CID"""
        
        if cid in self.cid_index:
            file_path = self.cid_index[cid]["file"]
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            raise KeyError(f"CID {cid} not found in storage")
    
    def export_index(self, output_path):
        """Export CID index to Parquet"""
        
        index_data = []
        for cid, info in self.cid_index.items():
            index_data.append({
                "cid": cid,
                "file_path": info["file"],
                "text_hash": info["text_hash"],
                "stored_at": info["stored_at"]
            })
        
        df = pd.DataFrame(index_data)
        df.to_parquet(output_path, index=False)
        
        return output_path

# Usage
storage = IPFSEmbeddingStorage("/storage/embeddings")

# Store embedding
cid = await storage.store_embedding(
    text="Sample text",
    embedding=np.array([0.1, 0.2, 0.3]),
    metadata={"model": "gte-small"}
)

# Retrieve embedding
record = await storage.retrieve_embedding(cid)
```

## Performance Optimization

### Efficient CID Generation

```python
class BatchCIDGenerator:
    """Generate CIDs efficiently for large datasets"""
    
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.multiformats = ipfs_multiformats_py()
    
    def generate_batch_cids(self, texts):
        """Generate CIDs for a batch of texts"""
        
        cids = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Process batch
            batch_cids = []
            for text in batch:
                cid = self.multiformats.get_cid(text)
                batch_cids.append(cid)
            
            cids.extend(batch_cids)
            
            # Progress update
            if i % (self.batch_size * 10) == 0:
                print(f"Processed {i + len(batch)}/{len(texts)} texts")
        
        return cids
    
    def deduplicate_by_cid(self, texts):
        """Remove duplicate texts using CID comparison"""
        
        unique_texts = []
        seen_cids = set()
        
        for text in texts:
            cid = self.multiformats.get_cid(text)
            if cid not in seen_cids:
                unique_texts.append(text)
                seen_cids.add(cid)
        
        return unique_texts, seen_cids

# Usage
generator = BatchCIDGenerator(batch_size=500)
texts = ["Sample text 1", "Sample text 2", ...]

# Generate CIDs efficiently
cids = generator.generate_batch_cids(texts)

# Remove duplicates
unique_texts, unique_cids = generator.deduplicate_by_cid(texts)
```

### Caching and Indexing

```python
class CIDCache:
    """Cache CIDs to avoid recomputation"""
    
    def __init__(self, cache_file="cid_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.multiformats = ipfs_multiformats_py()
    
    def _load_cache(self):
        """Load existing cache from disk"""
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
    
    def get_cid(self, content):
        """Get CID with caching"""
        
        # Use content hash as cache key
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        if content_hash in self.cache:
            return self.cache[content_hash]
        
        # Generate and cache CID
        cid = self.multiformats.get_cid(content)
        self.cache[content_hash] = cid
        
        # Periodically save cache
        if len(self.cache) % 100 == 0:
            self._save_cache()
        
        return cid
    
    def __del__(self):
        """Ensure cache is saved on cleanup"""
        self._save_cache()
```

## Error Handling and Recovery

### Content Validation

```python
def validate_ipfs_content(file_path, expected_cid):
    """Validate IPFS content integrity"""
    
    try:
        multiformats = ipfs_multiformats_py()
        actual_cid = multiformats.get_cid(file_path)
        
        if actual_cid == expected_cid:
            return True, "Content validation successful"
        else:
            return False, f"CID mismatch: expected {expected_cid}, got {actual_cid}"
            
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def recover_corrupted_content(corrupted_file, backup_sources):
    """Attempt to recover corrupted content from backup sources"""
    
    for source in backup_sources:
        try:
            # Attempt recovery from source
            if source["type"] == "ipfs":
                # Recover from IPFS network
                content = recover_from_ipfs(source["cid"])
            elif source["type"] == "file":
                # Recover from file backup
                with open(source["path"], "rb") as f:
                    content = f.read()
            
            # Validate recovered content
            is_valid, message = validate_ipfs_content(content, source["expected_cid"])
            
            if is_valid:
                # Restore corrupted file
                with open(corrupted_file, "wb") as f:
                    f.write(content)
                return True, f"Recovery successful from {source['type']}"
                
        except Exception as e:
            continue
    
    return False, "All recovery attempts failed"
```

## Configuration Examples

### Complete IPFS Configuration

```python
ipfs_config = {
    "multiformats": {
        "hash_algorithm": "sha2-256",
        "cid_version": 1,
        "multibase": "base32"
    },
    "conversion": {
        "parquet_to_car": {
            "batch_size": 100,
            "compression": "gzip",
            "validation": True
        },
        "folder_to_parquet": {
            "max_file_size": "100MB",
            "include_metadata": True
        }
    },
    "storage": {
        "cache_size": 1000,
        "cache_file": "/tmp/cid_cache.json",
        "temp_directory": "/tmp/ipfs_processing"
    },
    "performance": {
        "parallel_workers": 4,
        "batch_cid_generation": True,
        "enable_caching": True
    }
}
```

### Environment Variables

```bash
# IPFS Integration Settings
export IPFS_MULTIFORMATS_CACHE_SIZE=10000
export IPFS_CONVERSION_BATCH_SIZE=500
export IPFS_TEMP_DIR="/tmp/ipfs_temp"
export IPFS_VALIDATION_ENABLED=true

# Performance Settings  
export IPFS_PARALLEL_WORKERS=8
export IPFS_CID_CACHE_FILE="/storage/cid_cache.json"
export IPFS_ENABLE_COMPRESSION=true
```

## Dependencies

### Required Python Packages

```bash
pip install multiformats
pip install datasets
pip install pandas
pip install pyarrow
```

### Node.js Dependencies (for CAR conversion)

```bash
npm install ipfs_parquet_to_car_js
```

## Related Components

- [Create Embeddings](../components/create-embeddings.md): Generate embeddings with IPFS storage
- [IPFS Cluster Index](../components/ipfs-cluster-index.md): Index and manage IPFS cluster content
- [Storacha Clusters](../components/storacha-clusters.md): Decentralized storage integration
- [Configuration Guide](../configuration.md): IPFS configuration settings

## Troubleshooting

### Common Issues

1. **CID Generation Failures**
   - Ensure multiformats library is installed
   - Check file permissions for temporary files
   - Verify content encoding (UTF-8 vs binary)

2. **CAR Conversion Errors**
   - Install Node.js dependencies: `npm install ipfs_parquet_to_car_js`
   - Check input Parquet file format
   - Ensure sufficient disk space

3. **Performance Issues**
   - Enable CID caching for repeated operations
   - Use batch processing for large datasets
   - Consider parallel processing for independent operations

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test IPFS integration
multiformats = ipfs_multiformats_py()
test_cid = multiformats.get_cid("test content")
print(f"Test CID: {test_cid}")
```
