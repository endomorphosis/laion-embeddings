# Vector Quantization

Vector quantization is a technique to reduce the dimensionality and storage requirements of embedding vectors while preserving their semantic similarity. This documentation explains how to use vector quantization with the LAION Embeddings system.

## Introduction

Vector quantization involves converting high-dimensional floating-point vectors into more compact representations by mapping them to a finite set of codes. For large-scale vector collections, this offers significant advantages:

- **Reduced Storage Requirements**: Quantized vectors require less storage space
- **Faster Search Operations**: Smaller vectors lead to faster similarity computations
- **Lower Bandwidth Usage**: Transmitting quantized vectors requires less network bandwidth
- **Memory Efficiency**: More vectors can be held in memory at once

## Supported Quantization Methods

The LAION Embeddings system supports several vector quantization methods:

1. **Scalar Quantization (SQ)**: Simple reduction of precision (e.g., 32-bit to 8-bit)
2. **Product Quantization (PQ)**: Splits vectors into subvectors and quantizes each separately
3. **Optimized Product Quantization (OPQ)**: Applies a rotation to the vectors before PQ
4. **Residual Quantization (RQ)**: Represents vectors as a sum of quantization centroids

## Configuration

To use quantization with any of the vector stores, including IPFS and DuckDB, add the quantization configuration to your vector store configuration:

```yaml
vector_databases:
  ipfs:
    # Other IPFS configuration...
    quantization:
      method: "pq"  # Options: "sq", "pq", "opq", "rq"
      bits: 8  # Bits per dimension for scalar quantization
      subquantizers: 8  # Number of subquantizers for PQ methods
      
  duckdb:
    # Other DuckDB configuration...
    quantization:
      method: "sq"
      bits: 8
```

## Using Quantization in Code

You can specify quantization parameters when creating a vector store:

```python
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType, QuantizationConfig

# Create a quantization config
quantization = QuantizationConfig(
    method="pq",  # product quantization
    subquantizers=8,
    bits=8
)

# Create a vector store with quantization
store = await create_vector_store(
    db_type=VectorDBType.IPFS,
    quantization=quantization
)
```

## Performance Considerations

Quantization involves a trade-off between accuracy and efficiency:

| Quantization Method | Storage Reduction | Search Speed | Accuracy | Best For |
|---------------------|-------------------|--------------|----------|----------|
| None (original)     | 1x                | Baseline     | 100%     | Highest accuracy needs |
| Scalar (SQ)         | 2-4x              | 1.5-2x       | 97-98%   | Balanced performance |
| Product (PQ)        | 4-32x             | 3-10x        | 90-95%   | Large-scale applications |
| Optimized PQ (OPQ)  | 4-32x             | 3-10x        | 92-96%   | Better accuracy than PQ |
| Residual (RQ)       | 8-64x             | 5-20x        | 88-93%   | Maximum compression |

## IPFS-Specific Considerations

When using quantization with IPFS vector store:

- **Storage Format**: Quantized vectors are stored as binary blobs within IPFS
- **Block Size Optimization**: Quantization helps optimize IPFS block sizes
- **Distributed Retrieval**: Quantized vectors can be more efficiently distributed

## DuckDB-Specific Considerations

When using quantization with DuckDB vector store:

- **Column Storage**: Quantized vectors are stored in columnar format for efficient filtering
- **Parquet Integration**: Quantized vectors integrate well with Parquet's compression
- **Query Push-down**: DuckDB optimizes queries with quantized vectors

## Example Implementation

Here's a complete example of using vector quantization with the IPFS vector store:

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType, QuantizationConfig
from services.vector_store_base import VectorDocument

async def quantization_example():
    # Configure quantization
    quantization = QuantizationConfig(
        method="pq",
        subquantizers=8,
        bits=8
    )
    
    # Create IPFS store with quantization
    store = await create_vector_store(
        db_type=VectorDBType.IPFS,
        quantization=quantization
    )
    
    await store.connect()
    await store.create_index(dimension=384)
    
    # Add documents (quantization happens automatically)
    docs = [...]  # your vector documents
    await store.add_documents(docs)
    
    # Search (automatically handles quantized vectors)
    results = await store.search(query)
    
    await store.disconnect()

# Run example
if __name__ == "__main__":
    asyncio.run(quantization_example())
```

## Conclusion

Vector quantization is a powerful technique for scaling vector search systems. By choosing the appropriate quantization method and parameters, you can significantly reduce storage requirements and improve search performance while maintaining acceptable accuracy for most applications.
