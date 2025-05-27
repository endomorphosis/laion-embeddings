# IPFS Cluster Index Component

The IPFS Cluster Index component manages the indexing and organization of content stored across IPFS clusters. It provides functionality for content discovery, metadata management, and efficient access to distributed embedding data stored on IPFS.

## Overview

The `ipfs_cluster_index` class provides:
- **Content Indexing**: Track and organize content across IPFS clusters
- **Metadata Management**: Store and retrieve content metadata and relationships
- **CID Management**: Manage Content Identifiers (CIDs) for efficient access
- **Cluster Coordination**: Coordinate between multiple IPFS cluster nodes

## Key Features

### Content Discovery
- **CID Enumeration**: List all pinned content in IPFS clusters
- **Content Classification**: Automatically classify content types
- **Metadata Extraction**: Extract and store content metadata
- **Relationship Mapping**: Track relationships between content pieces

### Distributed Storage Integration
- **IPFS Integration**: Direct integration with IPFS and IPFS Cluster
- **Storacha Support**: Integration with Storacha decentralized storage
- **Multi-Backend**: Support multiple storage backends simultaneously
- **Data Synchronization**: Keep indexes synchronized across nodes

### Performance Optimization
- **Efficient Indexing**: Optimized data structures for fast lookups
- **Caching**: Cache frequently accessed content information
- **Batch Operations**: Process multiple operations efficiently
- **Lazy Loading**: Load content information on demand

## Usage

### Basic Index Operations

```python
from ipfs_cluster_index import ipfs_cluster_index

# Configuration
metadata = {
    "cluster_endpoints": [
        "http://ipfs-cluster-1:9094",
        "http://ipfs-cluster-2:9094"
    ],
    "index_config": {
        "update_interval": 3600,  # 1 hour
        "cache_size": 10000,
        "batch_size": 1000
    }
}

resources = {
    "ipfs_gateway": "http://localhost:8080",
    "cluster_secret": "your-cluster-secret"
}

# Initialize component
indexer = ipfs_cluster_index(resources, metadata)

# Export CID list and metadata
cid_data = indexer.export_cid_list("/path/to/export")
```

### Content Type Detection

```python
# The component automatically detects content types
content_analysis = {
    "embedding_vectors": [],
    "datasets": [],
    "models": [],
    "configurations": [],
    "unknown": []
}

# Process all pinned content
pinset = indexer.ipfs_kit_py.ipfs_get_pinset()
for cid in pinset:
    content_type = indexer.classify_content(cid)
    content_analysis[content_type].append(cid)
```

### Advanced Indexing

```python
# Custom indexing with filters
index_config = {
    "content_filters": [
        {"type": "size", "min": 1024, "max": 100*1024*1024},  # 1KB to 100MB
        {"type": "format", "allowed": ["parquet", "json", "txt"]},
        {"type": "embedding_model", "models": ["gte-large", "gte-small"]}
    ],
    "metadata_extraction": {
        "deep_scan": True,
        "extract_schemas": True,
        "validate_integrity": True
    }
}

# Create filtered index
filtered_index = indexer.create_filtered_index(index_config)
```

## Configuration

### Metadata Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `cluster_endpoints` | list | IPFS cluster API endpoints | Yes |
| `index_config` | dict | Indexing configuration parameters | No |
| `storage_config` | dict | Storage backend configuration | No |

### Index Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `update_interval` | int | Index update frequency (seconds) | 3600 |
| `cache_size` | int | Maximum cached entries | 10000 |
| `batch_size` | int | Batch processing size | 1000 |
| `deep_scan` | bool | Enable deep content analysis | False |
| `validate_content` | bool | Validate content integrity | True |

### Storage Configuration

```python
storage_config = {
    "backends": [
        {
            "type": "ipfs",
            "gateway": "http://localhost:8080",
            "cluster_api": "http://localhost:9094"
        },
        {
            "type": "storacha", 
            "endpoint": "https://w3s.link",
            "auth_token": "your-auth-token"
        }
    ],
    "replication": {
        "min_replicas": 3,
        "preferred_replicas": 5
    }
}
```

## API Reference

### Class: `ipfs_cluster_index`

#### `__init__(resources, metadata)`
Initialize the IPFS cluster index component.

**Parameters:**
- `resources` (dict): IPFS and cluster configuration
- `metadata` (dict): Index configuration and parameters

#### `export_cid_list(dst_path)`
Export comprehensive CID information to specified path.

**Parameters:**
- `dst_path` (str): Destination path for exported data

**Returns:**
- Dictionary containing:
  - `cid_list`: List of all CIDs
  - `cid_set`: Set of unique CIDs
  - `metadata`: Content metadata for each CID
  - `content_types`: Classified content types

## Implementation Examples

### Complete Cluster Indexing

```python
# Index an entire IPFS cluster
def index_complete_cluster():
    """Create comprehensive index of cluster content"""
    
    indexer = ipfs_cluster_index(resources, metadata)
    
    # Get all pinned content
    pinset = indexer.ipfs_kit_py.ipfs_get_pinset()
    print(f"Found {len(pinset)} pinned objects")
    
    # Analyze each object
    cluster_analysis = {
        "total_objects": len(pinset),
        "content_types": {},
        "size_distribution": {},
        "embedding_models": set(),
        "datasets": set()
    }
    
    for cid in pinset:
        try:
            # Get content metadata
            content_data = indexer.ipfs_kit_py.ipfs_get(cid)
            content_type = type(content_data).__name__
            
            # Update statistics
            if content_type not in cluster_analysis["content_types"]:
                cluster_analysis["content_types"][content_type] = 0
            cluster_analysis["content_types"][content_type] += 1
            
            # Detect embedding-related content
            if indexer._is_embedding_content(content_data):
                model_info = indexer._extract_model_info(content_data)
                if model_info:
                    cluster_analysis["embedding_models"].add(model_info["model"])
                    cluster_analysis["datasets"].add(model_info["dataset"])
                    
        except Exception as e:
            print(f"Error processing CID {cid}: {e}")
    
    return cluster_analysis

# Execute indexing
analysis = index_complete_cluster()
print(f"Analysis complete: {analysis}")
```

### Content Migration and Backup

```python
# Migrate content between clusters with indexing
async def migrate_with_index(source_cluster, target_cluster):
    """Migrate content while maintaining comprehensive index"""
    
    # Source indexer
    source_indexer = ipfs_cluster_index(
        {"cluster_api": source_cluster}, 
        metadata
    )
    
    # Target indexer
    target_indexer = ipfs_cluster_index(
        {"cluster_api": target_cluster},
        metadata
    )
    
    # Export source index
    source_index = source_indexer.export_cid_list("/tmp/source_index")
    
    # Migrate content
    migration_report = {
        "migrated": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }
    
    for cid in source_index["cid_list"]:
        try:
            # Check if already exists in target
            if target_indexer.content_exists(cid):
                migration_report["skipped"] += 1
                continue
            
            # Fetch content from source
            content = source_indexer.ipfs_kit_py.ipfs_get(cid)
            
            # Pin to target cluster
            target_indexer.ipfs_kit_py.ipfs_pin(cid, content)
            migration_report["migrated"] += 1
            
        except Exception as e:
            migration_report["failed"] += 1
            migration_report["errors"].append({"cid": cid, "error": str(e)})
    
    # Create target index
    target_index = target_indexer.export_cid_list("/tmp/target_index")
    
    return migration_report, target_index

# Execute migration
report, index = await migrate_with_index(
    "http://source-cluster:9094",
    "http://target-cluster:9094"
)
```

### Content Validation and Integrity

```python
# Validate cluster content integrity
def validate_cluster_integrity(indexer):
    """Validate integrity of all cluster content"""
    
    validation_report = {
        "total_checked": 0,
        "valid": 0,
        "corrupted": 0,
        "missing": 0,
        "errors": []
    }
    
    pinset = indexer.ipfs_kit_py.ipfs_get_pinset()
    
    for cid in pinset:
        validation_report["total_checked"] += 1
        
        try:
            # Attempt to fetch content
            content = indexer.ipfs_kit_py.ipfs_get(cid)
            
            # Validate content hash
            if indexer._validate_content_hash(cid, content):
                validation_report["valid"] += 1
            else:
                validation_report["corrupted"] += 1
                validation_report["errors"].append({
                    "cid": cid,
                    "error": "Hash mismatch"
                })
                
        except FileNotFoundError:
            validation_report["missing"] += 1
            validation_report["errors"].append({
                "cid": cid,
                "error": "Content not found"
            })
        except Exception as e:
            validation_report["corrupted"] += 1
            validation_report["errors"].append({
                "cid": cid,
                "error": str(e)
            })
    
    return validation_report

# Run validation
integrity_report = validate_cluster_integrity(indexer)
print(f"Integrity check complete: {integrity_report}")
```

## Content Classification

### Automatic Type Detection

The component automatically classifies content into categories:

```python
content_types = {
    "embeddings": {
        "extensions": [".npy", ".h5", ".pkl"],
        "markers": ["embedding", "vector", "features"],
        "models": ["gte-large", "gte-small", "sentence-transformers"]
    },
    "datasets": {
        "extensions": [".parquet", ".json", ".jsonl", ".csv"],
        "markers": ["dataset", "corpus", "collection"],
        "formats": ["huggingface", "arrow", "pandas"]
    },
    "models": {
        "extensions": [".bin", ".safetensors", ".onnx"],
        "markers": ["model", "checkpoint", "weights"],
        "frameworks": ["pytorch", "tensorflow", "onnx"]
    },
    "configurations": {
        "extensions": [".yaml", ".json", ".toml"],
        "markers": ["config", "settings", "parameters"]
    }
}
```

### Custom Classification Rules

```python
# Define custom classification rules
custom_rules = {
    "legal_documents": {
        "content_patterns": [r"case\s+law", r"legal\s+precedent"],
        "metadata_fields": ["jurisdiction", "court", "case_number"],
        "file_patterns": ["*legal*", "*case*", "*court*"]
    },
    "scientific_papers": {
        "content_patterns": [r"abstract", r"references", r"doi:"],
        "metadata_fields": ["authors", "journal", "publication_date"],
        "file_patterns": ["*paper*", "*article*", "*research*"]
    }
}

# Apply custom classification
indexer.add_classification_rules(custom_rules)
classified_content = indexer.classify_all_content()
```

## Performance Optimization

### Batch Processing

```python
# Optimize for large clusters
batch_config = {
    "batch_size": 1000,           # Process 1000 CIDs at once
    "parallel_workers": 8,        # Use 8 worker threads
    "cache_results": True,        # Cache classification results
    "lazy_loading": True,         # Load content on demand
    "checkpoint_interval": 5000   # Save progress every 5000 items
}

indexer = ipfs_cluster_index(resources, {
    "index_config": batch_config,
    **metadata
})
```

### Memory Management

```python
# Configure memory usage
memory_config = {
    "max_cache_size": "2GB",      # Maximum cache size
    "content_cache_ttl": 3600,    # Cache TTL in seconds
    "metadata_cache_size": 50000, # Metadata entries to cache
    "cleanup_interval": 1800      # Cleanup interval
}

# Monitor memory usage
def monitor_memory_usage(indexer):
    import psutil
    process = psutil.Process()
    
    memory_info = {
        "rss": process.memory_info().rss,
        "vms": process.memory_info().vms,
        "cache_size": len(indexer._content_cache),
        "metadata_cache_size": len(indexer._metadata_cache)
    }
    
    return memory_info
```

## Integration with Storage Backends

### Multi-Backend Support

```python
# Configure multiple storage backends
multi_backend_config = {
    "backends": [
        {
            "name": "primary_ipfs",
            "type": "ipfs",
            "gateway": "http://localhost:8080",
            "priority": 1
        },
        {
            "name": "backup_cluster", 
            "type": "ipfs_cluster",
            "api": "http://backup-cluster:9094",
            "priority": 2
        },
        {
            "name": "cold_storage",
            "type": "storacha",
            "endpoint": "https://w3s.link",
            "priority": 3
        }
    ],
    "failover": True,
    "replication": 2
}

# Initialize with multiple backends
indexer = ipfs_cluster_index(multi_backend_config, metadata)
```

### Storacha Integration

```python
# Integrate with Storacha decentralized storage
storacha_config = {
    "storacha_endpoint": "https://w3s.link",
    "auth_token": "your-auth-token",
    "sync_interval": 7200,  # 2 hours
    "backup_to_storacha": True
}

# Sync content to Storacha
def sync_to_storacha(indexer):
    """Backup cluster content to Storacha"""
    
    pinset = indexer.ipfs_kit_py.ipfs_get_pinset()
    sync_report = {"synced": 0, "failed": 0, "errors": []}
    
    for cid in pinset:
        try:
            # Check if already in Storacha
            if not indexer.storacha_kit_py.content_exists(cid):
                # Upload to Storacha
                content = indexer.ipfs_kit_py.ipfs_get(cid)
                indexer.storacha_kit_py.upload(cid, content)
                sync_report["synced"] += 1
        except Exception as e:
            sync_report["failed"] += 1
            sync_report["errors"].append({"cid": cid, "error": str(e)})
    
    return sync_report
```

## Error Handling and Recovery

### Robust Error Handling

```python
class RobustIndexer(ipfs_cluster_index):
    def __init__(self, resources, metadata):
        super().__init__(resources, metadata)
        self.retry_config = {
            "max_retries": 3,
            "backoff_factor": 2,
            "timeout": 30
        }
    
    def robust_export_cid_list(self, dst_path):
        """Export with retry logic and error recovery"""
        
        for attempt in range(self.retry_config["max_retries"]):
            try:
                return self.export_cid_list(dst_path)
            except Exception as e:
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = self.retry_config["backoff_factor"] ** attempt
                    print(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    print(f"All attempts failed: {e}")
                    raise
```

## Dependencies

- `ipfs_kit_py`: Core IPFS functionality
- `datasets`: Dataset handling and processing
- `asyncio`: Asynchronous operations
- `json`: JSON data handling
- `pathlib`: Path management

## Related Components

- [Storacha Clusters](storacha-clusters.md): Storacha integration
- [Create Embeddings](create-embeddings.md): Content creation
- [Search Embeddings](search-embeddings.md): Content discovery
- [IPFS Integration](../ipfs/README.md): IPFS documentation
