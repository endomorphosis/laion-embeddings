# Examples

This directory contains practical examples for the LAION Embeddings search engine.

## Vector Store Examples

- [IPFS Examples](ipfs-examples.md) - Examples for using the IPFS vector store
- [DuckDB Examples](duckdb-examples.md) - Examples for using the DuckDB vector store

## Basic Usage Examples

### Creating and Searching Embeddings

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def basic_example():
    # Create a store (default is FAISS if not specified)
    store = await create_vector_store()
    
    # Connect
    await store.connect()
    
    # Create index with 384 dimensions
    await store.create_index(dimension=384)
    
    # Create random documents for example
    np.random.seed(42)  # For reproducibility
    docs = []
    
    for i in range(10):
        # Create normalized random vector
        vec = np.random.random(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Create document
        doc = VectorDocument(
            id=f"doc-{i}",
            vector=vec.tolist(),
            metadata={"text": f"This is document {i}"}
        )
        docs.append(doc)
    
    # Add to index
    count = await store.add_documents(docs)
    print(f"Added {count} documents")
    
    # Create a query vector
    query_vec = np.random.random(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    # Search
    query = SearchQuery(
        vector=query_vec.tolist(),
        top_k=3
    )
    
    results = await store.search(query)
    
    # Print results
    print(f"Found {len(results.matches)} matches:")
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        print(f"   Text: {match.metadata.get('text')}")
    
    # Clean up
    await store.disconnect()

# Run example
if __name__ == "__main__":
    asyncio.run(basic_example())
```

### Comparing Different Vector Stores

```python
import asyncio
import time
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType
from services.vector_store_base import VectorDocument, SearchQuery

async def compare_stores():
    # Vector stores to compare
    store_types = [
        VectorDBType.FAISS,
        VectorDBType.IPFS,
        VectorDBType.DUCKDB
    ]
    
    # Results tracking
    results = {}
    
    # Create test data
    np.random.seed(42)
    dimension = 128
    num_docs = 1000
    
    docs = []
    for i in range(num_docs):
        vec = np.random.random(dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        doc = VectorDocument(
            id=f"doc-{i}",
            vector=vec.tolist(),
            metadata={"text": f"Document {i}"}
        )
        docs.append(doc)
    
    # Query vector for searching
    query_vec = np.random.random(dimension).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    # Test each store
    for store_type in store_types:
        try:
            print(f"\nTesting {store_type.value}...")
            
            # Initialize metrics
            metrics = {
                "add_time": 0,
                "search_time": 0,
                "success": False
            }
            
            # Create store
            store = await create_vector_store(store_type)
            
            # Connect
            await store.connect()
            
            # Create index
            await store.create_index(dimension=dimension)
            
            # Add documents and measure time
            start = time.time()
            count = await store.add_documents(docs)
            metrics["add_time"] = time.time() - start
            
            # Search and measure time
            query = SearchQuery(
                vector=query_vec.tolist(),
                top_k=10
            )
            
            start = time.time()
            search_results = await store.search(query)
            metrics["search_time"] = time.time() - start
            
            # Record success and result count
            metrics["success"] = True
            metrics["result_count"] = len(search_results.matches)
            
            # Print results
            print(f"Added {count} documents in {metrics['add_time']:.4f}s")
            print(f"Found {metrics['result_count']} results in {metrics['search_time']:.4f}s")
            
            # Cleanup
            await store.disconnect()
            
        except Exception as e:
            print(f"Error with {store_type.value}: {e}")
            metrics = {
                "add_time": 0,
                "search_time": 0,
                "success": False,
                "error": str(e)
            }
        
        results[store_type.value] = metrics
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Store':<15} {'Add Time':<15} {'Search Time':<15} {'Success':<10}")
    print("-" * 55)
    
    for store, metrics in results.items():
        add_time = f"{metrics['add_time']:.4f}s" if metrics['success'] else "N/A"
        search_time = f"{metrics['search_time']:.4f}s" if metrics['success'] else "N/A"
        success = "✅" if metrics['success'] else "❌"
        
        print(f"{store:<15} {add_time:<15} {search_time:<15} {success:<10}")

# Run example
if __name__ == "__main__":
    asyncio.run(compare_stores())
```

## Advanced Examples

For more advanced examples, check out:

- [Vector Quantization Examples](../advanced/vector-quantization.md)
- [Sharding Examples](../advanced/sharding.md)
- [Production Deployment Examples](../advanced/production.md)

## API Integration Examples

See the [API Examples](../api/examples.md) for examples of using the REST API.
