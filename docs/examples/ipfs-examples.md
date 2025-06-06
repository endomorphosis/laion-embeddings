# IPFS Vector Store Examples

This document provides practical examples for using the IPFS Vector Store in the unified vector database architecture.

## Basic Usage

### Connecting to IPFS and Creating an Index

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def basic_ipfs_example():
    # Create an IPFS vector store
    store = await create_vector_store(VectorDBType.IPFS)
    
    # Connect to IPFS
    await store.connect()
    
    # Check connection
    is_connected = await store.ping()
    print(f"IPFS connected: {is_connected}")
    
    # Create a vector index with dimension 384
    created = await store.create_index(
        dimension=384,
        overwrite=True  # Replace any existing index
    )
    print(f"Index created: {created}")
    
    # Get index stats
    stats = await store.get_stats()
    print(f"Index stats: {stats}")
    
    # Disconnect when done
    await store.disconnect()

# Run the example
asyncio.run(basic_ipfs_example())
```

### Adding and Searching Vectors

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def add_search_example():
    # Create and connect to store
    store = await create_vector_store(VectorDBType.IPFS)
    await store.connect()
    
    # Dimension for our vectors
    dim = 384
    
    # Create index
    await store.create_index(dimension=dim)
    
    # Create sample documents with random vectors
    num_docs = 100
    docs = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_docs):
        # Create a random vector and normalize it
        vec = np.random.random(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Create document
        doc = VectorDocument(
            id=f"doc-{i}",
            vector=vec.tolist(),
            metadata={
                "title": f"Document {i}",
                "category": f"cat-{i % 5}",
                "value": float(i)
            }
        )
        docs.append(doc)
    
    # Add documents to IPFS
    print(f"Adding {len(docs)} documents...")
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Check index stats
    stats = await store.get_stats()
    print(f"Index stats after adding: {stats}")
    
    # Create a query vector (use first document's vector)
    query_vector = docs[0].vector
    
    # Basic search
    query = SearchQuery(
        vector=query_vector,
        top_k=5
    )
    
    print("\nBasic search results:")
    results = await store.search(query)
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
    
    # Search with filter
    filter_query = SearchQuery(
        vector=query_vector,
        top_k=5,
        filter={"category": "cat-2"}  # Only cat-2 documents
    )
    
    print("\nFiltered search results (category=cat-2):")
    filtered_results = await store.search(filter_query)
    for i, match in enumerate(filtered_results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        print(f"   Category: {match.metadata.get('category')}")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(add_search_example())
```

## Advanced Usage

### Sharding with IPFS

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery
import numpy as np

async def sharding_example():
    # Create IPFS store with sharding configuration
    store = await create_vector_store(
        VectorDBType.IPFS,
        config_override={
            "sharding_enabled": True,
            "max_shard_size": 25,  # Small shard size for demonstration
            "dimension": 64
        }
    )
    
    await store.connect()
    
    # Create index with sharding
    await store.create_index(
        dimension=64,
        sharding=True,
        sharding_params={"max_shard_size": 25}
    )
    
    # Create 100 sample documents (to demonstrate sharding)
    docs = []
    np.random.seed(42)
    
    for i in range(100):
        vec = np.random.random(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        doc = VectorDocument(
            id=f"shard-doc-{i}",
            vector=vec.tolist(),
            metadata={"group": f"group-{i // 10}"}
        )
        docs.append(doc)
    
    # Add documents (will be automatically sharded)
    print("Adding documents with sharding...")
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Check stats to see sharding info
    stats = await store.get_stats()
    print(f"Shard count: {stats.get('shard_count', 0)}")
    print(f"Total vectors: {stats.get('total_vectors', 0)}")
    
    # Search across shards
    query_vector = docs[50].vector  # Use a random document's vector
    
    query = SearchQuery(
        vector=query_vector,
        top_k=5
    )
    
    print("\nCross-shard search results:")
    results = await store.search(query)
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(sharding_example())
```

### Vector Quantization with IPFS

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery
import numpy as np
import time

async def quantization_example():
    # Create store with quantization config
    store = await create_vector_store(
        VectorDBType.IPFS,
        config_override={
            "dimension": 128,
            "search": {
                "use_quantization": True
            }
        }
    )
    
    await store.connect()
    
    # Create index with quantization
    await store.create_index(
        dimension=128,
        quantization=True,
        quantization_params={
            "type": "scalar",
            "bits": 8
        }
    )
    
    # Create sample documents
    docs = []
    np.random.seed(42)
    
    for i in range(1000):
        vec = np.random.random(128).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        doc = VectorDocument(
            id=f"quant-doc-{i}",
            vector=vec.tolist(),
            metadata={"category": f"cat-{i % 5}"}
        )
        docs.append(doc)
    
    # Add documents
    print("Adding documents with quantization...")
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Query vector for searching
    query_vector = docs[0].vector
    
    # Compare search with and without quantization
    
    # 1. With quantization
    q_query = SearchQuery(
        vector=query_vector,
        top_k=5,
        use_quantization=True
    )
    
    start = time.time()
    q_results = await store.search(q_query)
    q_time = time.time() - start
    
    print(f"\nQuantized search ({q_time:.4f} seconds):")
    for i, match in enumerate(q_results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
    
    # 2. Without quantization (if supported)
    try:
        nq_query = SearchQuery(
            vector=query_vector,
            top_k=5,
            use_quantization=False
        )
        
        start = time.time()
        nq_results = await store.search(nq_query)
        nq_time = time.time() - start
        
        print(f"\nNon-quantized search ({nq_time:.4f} seconds):")
        for i, match in enumerate(nq_results.matches):
            print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        
        # Compare results
        q_ids = [m.id for m in q_results.matches]
        nq_ids = [m.id for m in nq_results.matches]
        overlap = set(q_ids).intersection(set(nq_ids))
        
        print(f"\nOverlap between methods: {len(overlap)}/{len(q_ids)}")
        print(f"Speed improvement: {nq_time/q_time:.2f}x")
        
    except Exception as e:
        print(f"\nNon-quantized search not supported: {e}")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(quantization_example())
```

### Working with IPFS CIDs

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def cid_example():
    # Create and connect
    store = await create_vector_store(VectorDBType.IPFS)
    await store.connect()
    
    # Get IPFS-specific functionality
    if hasattr(store, 'get_index_cid'):
        # Get the CID of the current index
        cid = await store.get_index_cid()
        print(f"Current index CID: {cid}")
        
        # Export index metadata
        metadata = await store.export_metadata()
        print(f"Index metadata: {metadata}")
        
        # Import from specific CID (hypothetical example)
        if cid:
            try:
                success = await store.import_from_cid(cid)
                print(f"Import from CID {cid}: {success}")
            except Exception as e:
                print(f"Import failed: {e}")
    else:
        print("This IPFS implementation doesn't expose CID operations")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(cid_example())
```

## Production Usage Patterns

### Error Handling and Reconnection

```python
import asyncio
import logging
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import ConnectionError, VectorStoreError

async def robust_ipfs_usage():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ipfs-example")
    
    store = None
    max_retries = 3
    retry_delay = 2  # seconds
    
    # Connection with retry logic
    for attempt in range(max_retries):
        try:
            logger.info(f"Connection attempt {attempt + 1}/{max_retries}")
            store = await create_vector_store(VectorDBType.IPFS)
            await store.connect()
            
            # Test connection
            if await store.ping():
                logger.info("Successfully connected to IPFS")
                break
            else:
                raise ConnectionError("IPFS ping failed")
                
        except (ConnectionError, VectorStoreError) as e:
            logger.error(f"Connection failed: {e}")
            if store:
                try:
                    await store.disconnect()
                except:
                    pass
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("All connection attempts failed")
                return
    
    # Now use the store with proper error handling
    try:
        # Check stats
        stats = await store.get_stats()
        logger.info(f"IPFS index stats: {stats}")
        
        # Perform operations...
        # [your operations here]
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
    finally:
        # Always clean up connection
        if store:
            logger.info("Disconnecting from IPFS")
            try:
                await store.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

# Run the example
asyncio.run(robust_ipfs_usage())
```

### Concurrent Operations

```python
import asyncio
import numpy as np
import time
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def concurrent_search(store, query_vector, top_k, filter_dict=None):
    """Perform a single search operation."""
    query = SearchQuery(
        vector=query_vector,
        top_k=top_k,
        filter=filter_dict
    )
    return await store.search(query)

async def concurrent_ipfs_example():
    # Create and connect
    store = await create_vector_store(
        VectorDBType.IPFS,
        config_override={
            "performance": {
                "concurrent_searches": 4  # Allow 4 concurrent searches
            }
        }
    )
    await store.connect()
    
    # Set up some test data
    dim = 128
    np.random.seed(42)
    
    # Create sample documents
    docs = []
    for i in range(1000):
        vec = np.random.random(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        doc = VectorDocument(
            id=f"doc-{i}",
            vector=vec.tolist(),
            metadata={"category": f"cat-{i % 5}"}
        )
        docs.append(doc)
    
    # Add documents
    print("Adding documents...")
    await store.create_index(dimension=dim)
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Prepare different query vectors
    query_vectors = [np.random.random(dim).astype(np.float32) for _ in range(10)]
    query_vectors = [vec / np.linalg.norm(vec) for vec in query_vectors]
    
    # Different filters for variety
    filters = [
        {"category": "cat-0"},
        {"category": "cat-1"},
        {"category": "cat-2"},
        {"category": "cat-3"},
        {"category": "cat-4"},
        None,
        None,
        None,
        None,
        None
    ]
    
    # Sequential search
    print("\nRunning 10 sequential searches...")
    start_time = time.time()
    
    sequential_results = []
    for i in range(10):
        result = await concurrent_search(store, query_vectors[i].tolist(), 5, filters[i])
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    print(f"Sequential search time: {sequential_time:.4f} seconds")
    
    # Concurrent search
    print("\nRunning 10 concurrent searches...")
    start_time = time.time()
    
    tasks = []
    for i in range(10):
        task = asyncio.create_task(
            concurrent_search(store, query_vectors[i].tolist(), 5, filters[i])
        )
        tasks.append(task)
    
    concurrent_results = await asyncio.gather(*tasks)
    
    concurrent_time = time.time() - start_time
    print(f"Concurrent search time: {concurrent_time:.4f} seconds")
    print(f"Speed improvement: {sequential_time/concurrent_time:.2f}x")
    
    # Verify results are the same
    match_count = 0
    for i, (seq_res, conc_res) in enumerate(zip(sequential_results, concurrent_results)):
        seq_ids = {m.id for m in seq_res.matches}
        conc_ids = {m.id for m in conc_res.matches}
        if seq_ids == conc_ids:
            match_count += 1
    
    print(f"Results matched: {match_count}/10")
    
    # Clean up
    await store.disconnect()

# Run the example
asyncio.run(concurrent_ipfs_example())
```

## For More Information

- [IPFS Vector Service Documentation](../ipfs-vector-service.md) - Complete guide to IPFS integration
- [Vector Stores Overview](../vector-stores.md) - Introduction to the vector store architecture
