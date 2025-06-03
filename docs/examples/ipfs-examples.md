# IPFS Vector Service Examples

This document provides practical examples for using the IPFS Vector Service for distributed vector storage and search. All examples are updated to work with the latest fixes and improvements.

## Basic Usage Examples

### Initialize the IPFS Vector Storage

```python
from services.ipfs_vector_service import IPFSVectorStorage

# Configure IPFS connection
ipfs_config = {
    'api_url': '/ip4/127.0.0.1/tcp/5001',  # Local IPFS node
    'chunk_size': 1048576,  # 1MB chunks (optional)
    'connection_timeout': 30,  # Connection timeout in seconds
    'retry_count': 3  # Number of retry attempts
}

# Initialize the storage
storage = IPFSVectorStorage(ipfs_config)
```

### Store and Retrieve Vector Shards

```python
import numpy as np

async def store_and_retrieve_example():
    # Create sample vectors (explicit float32 type for compatibility)
    vectors = np.random.rand(100, 768).astype(np.float32)
    
    # Create metadata in standardized format
    metadata = {
        'texts': [f"Document {i}" for i in range(100)],
        'attributes': [{"id": i, "source": "example", "timestamp": "2025-05-31"} for i in range(100)]
    }
    
    # Store the vectors with proper error handling
    try:
        shard_id = "example_shard_001"
        cid = await storage.store_vector_shard(vectors, metadata, shard_id)
        print(f"Stored vectors with CID: {cid}")
        
        # Optionally pin the shard for persistence
        await storage.client.pin.add(cid)
        print(f"Pinned shard with CID: {cid}")
        
        # Retrieve the vectors
        retrieved = await storage.retrieve_vector_shard(cid)
        
        # Verify the data
        assert retrieved['vectors'].shape == vectors.shape
        assert len(retrieved['metadata']['texts']) == len(metadata['texts'])
        
        print(f"Successfully retrieved {retrieved['vectors'].shape[0]} vectors")
        return retrieved
    except Exception as e:
        print(f"Error in store/retrieve operation: {e}")
        raise
```

## Distributed Index Examples

### Create a Distributed Vector Index

```python
from services.ipfs_vector_service import DistributedVectorIndex
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_example")

async def create_distributed_index_example():
    try:
        # Vector configuration with explicit parameters
        vector_config = {
            'dimension': 768,
            'metric': 'cosine',
            'normalize_vectors': True  # Normalize vectors for better search
        }
        
        # Initialize distributed index with shard size of 1000 vectors
        index = DistributedVectorIndex(vector_config, storage, shard_size=1000)
        
        # Generate sample data (5000 vectors)
        vectors = np.random.rand(5000, 768).astype(np.float32)
        texts = [f"Document {i}" for i in range(5000)]
        
        # Enhanced metadata with more attributes
        metadata = [
            {
                "id": i, 
                "category": i % 10,
                "timestamp": "2025-05-31",
                "source": "example_dataset",
                "language": "en"
            } for i in range(5000)
        ]
        
        logger.info(f"Adding {len(vectors)} vectors to distributed index")
        
        # Add vectors to the distributed index with progress tracking
        total_shards = (len(vectors) + index.shard_size - 1) // index.shard_size
        logger.info(f"Expected shard count: {total_shards}")
        
        manifest_hash = await index.add_vectors_distributed(vectors, texts, metadata)
        
        logger.info(f"Created distributed index with manifest: {manifest_hash}")
        logger.info(f"Index contains {len(index.shards)} shards")
        
        # Display shard information
        for shard_id, shard_info in index.shards.items():
            logger.info(f"Shard {shard_id}: {shard_info['vector_count']} vectors, IPFS hash: {shard_info['ipfs_hash']}")
        
        return manifest_hash
    except Exception as e:
        logger.error(f"Error creating distributed index: {e}")
        raise
```

### Search Across Distributed Shards

```python
async def search_distributed_example(manifest_hash):
    try:
        # Initialize distributed index with explicit parameters
        vector_config = {
            'dimension': 768,
            'metric': 'cosine',
            'normalize_vectors': True
        }
        index = DistributedVectorIndex(vector_config, storage)
        
        logger.info(f"Loading index from manifest: {manifest_hash}")
        
        # Load from manifest with validation
        manifest_data = await index.load_from_manifest(manifest_hash)
        logger.info(f"Loaded manifest with {manifest_data.get('total_vectors', 0)} total vectors")
        
        # Verify manifest contains expected fields
        if 'shards' not in manifest_data:
            raise ValueError("Invalid manifest: missing 'shards' key")
            
        # Create a query vector and normalize it
        query_vector = np.random.rand(768).astype(np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        logger.info("Searching across distributed shards...")
        
        # Search across all shards with timeout
        import asyncio
        try:
            # Set a reasonable timeout
            results = await asyncio.wait_for(
                index.search_distributed(query_vector, k=10),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error("Search timed out after 30 seconds")
            return []
        
        # Process and display results
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. Similarity: {result['similarity']:.4f}, " 
                  f"Text: {result['metadata'].get('text', 'N/A')}, "
                  f"Shard: {result['shard_id']}, "
                  f"IPFS Hash: {result['ipfs_hash']}")
        
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise
```

## Advanced Usage Examples

### Incremental Updates to Distributed Index

```python
async def incremental_update_example(manifest_hash):
    # Load existing index
    vector_config = {
        'dimension': 768,
        'metric': 'cosine'
    }
    index = DistributedVectorIndex(vector_config, storage)
    await index.load_from_manifest(manifest_hash)
    
    # Get current stats
    initial_stats = index.get_index_stats()
    print(f"Initial index: {initial_stats['total_vectors']} vectors in {initial_stats['shard_count']} shards")
    
    # Add new vectors
    new_vectors = np.random.rand(1000, 768).astype(np.float32)
    new_texts = [f"New Document {i}" for i in range(1000)]
    new_metadata = [{"id": i + 5000, "category": "new"} for i in range(1000)]
    
    # Update the index
    updated_manifest = await index.add_vectors_distributed(new_vectors, new_texts, new_metadata)
    
    # Get updated stats
    final_stats = index.get_index_stats()
    print(f"Updated index: {final_stats['total_vectors']} vectors in {final_stats['shard_count']} shards")
    
    return updated_manifest
```

### Working with Metadata Filters

```python
async def metadata_filter_example(manifest_hash):
    # Load existing index
    vector_config = {
        'dimension': 768,
        'metric': 'cosine'
    }
    index = DistributedVectorIndex(vector_config, storage)
    await index.load_from_manifest(manifest_hash)
    
    # Create a query vector
    query_vector = np.random.rand(768).astype(np.float32)
    
    # Define metadata filter function
    def filter_function(metadata):
        # Only include items from category 5
        return metadata.get('category') == 5
    
    # Search with filter
    results = await index.search_vectors(query_vector, k=10, filter_function=filter_function)
    
    print(f"Found {len(results)} results matching filter criteria")
    
    return results
```

### High-Level Service Usage

```python
from services.ipfs_vector_service import IPFSVectorService

async def service_example():
    # Vector and IPFS configurations
    vector_config = {
        'dimension': 768,
        'metric': 'cosine',
        'index_type': 'flat'
    }
    
    ipfs_config = {
        'api_url': '/ip4/127.0.0.1/tcp/5001'
    }
    
    # Initialize the service
    service = IPFSVectorService(vector_config, ipfs_config)
    
    # Add embeddings
    embeddings = np.random.rand(100, 768).astype(np.float32)
    texts = [f"Service Document {i}" for i in range(100)]
    
    # Add to both local and distributed storage
    result = await service.add_embeddings(embeddings, texts)
    print(f"Added embeddings with result: {result}")
    
    # Search
    query = "example search query"  # This would be converted to embedding by the service
    search_results = await service.search(query, k=5)
    
    print(f"Found {len(search_results)} results for query")
    
    return search_results
```

## Running the Examples

To run these examples, create a Python script with the necessary imports and async execution:

```python
import asyncio
from examples import *

async def run_all_examples():
    # Store and retrieve vectors
    retrieved_shard = await store_and_retrieve_example()
    
    # Create a distributed index
    manifest_hash = await create_distributed_index_example()
    
    # Search the index
    search_results = await search_distributed_example(manifest_hash)
    
    # Update the index
    updated_manifest = await incremental_update_example(manifest_hash)
    
    # Search with filters
    filtered_results = await metadata_filter_example(updated_manifest)
    
    # Use the high-level service
    service_results = await service_example()

if __name__ == "__main__":
    asyncio.run(run_all_examples())
```

## Important Notes

1. Make sure an IPFS node is running and accessible at the configured URL
2. Large vector operations may take significant time depending on your hardware
3. Pin important CIDs to ensure they're not garbage collected
4. Consider implementing retry logic for IPFS operations in production

## Advanced Configuration

The IPFS Vector Service can be configured with additional options for production use:

```python
ipfs_config = {
    'api_url': '/ip4/127.0.0.1/tcp/5001',
    'chunk_size': 2097152,           # 2MB chunks
    'connection_timeout': 30,        # 30 second timeout
    'retry_count': 3,                # Retry failed operations 3 times
    'pin_shards': True,              # Automatically pin shards
    'compression': 'zlib',           # Compress data (options: 'zlib', 'gzip', None)
    'queue_size': 100,               # Queue size for batch operations
}

# Advanced vector configuration 
vector_config = {
    'dimension': 768,                # Vector dimension
    'metric': 'cosine',              # Distance metric: 'cosine', 'l2', 'ip'
    'normalize_vectors': True,       # Apply vector normalization
    'precision': 'float32',          # Vector precision (float32 recommended)
    'max_vectors_per_batch': 5000,   # Maximum vectors per batch operation
    'timeout_seconds': 60,           # Operation timeout
    'verify_dimension': True,        # Enable dimension validation
}
```

### Using Environment Variables

For better deployment flexibility, you can configure the service using environment variables:

```python
import os

# Load configuration from environment variables
ipfs_config = {
    'api_url': os.environ.get('IPFS_API_URL', '/ip4/127.0.0.1/tcp/5001'),
    'connection_timeout': int(os.environ.get('IPFS_TIMEOUT', '30')),
    'pin_shards': os.environ.get('IPFS_PIN_SHARDS', 'true').lower() == 'true',
}

# Initialize storage with environment-based config
storage = IPFSVectorStorage(ipfs_config)
```

These settings can be adjusted based on your specific requirements and infrastructure.

## Testing in Isolated Environment

For testing IPFS functionality without a real IPFS node:

```python
from mock_ipfs import MockIPFSClient
import os

# Enable testing mode
os.environ['TESTING'] = 'true'

# Create a mock storage for testing
class MockIPFSVectorStorage(IPFSVectorStorage):
    def __init__(self, config=None):
        super().__init__(config)
        self.client = MockIPFSClient()

# Use the mock in tests
storage = MockIPFSVectorStorage()
index = DistributedVectorIndex(vector_config, storage)

# Run your tests with the mock
await index.add_vectors_distributed(test_vectors, test_texts, test_metadata)
```
