# Basic Usage Examples

This document provides examples of basic usage patterns for the LAION Embeddings system, with examples for all vector store types including IPFS and DuckDB.

## Initializing Different Vector Stores

### Default Vector Store

```python
import asyncio
from services.vector_store_factory import create_vector_store

async def init_default_store():
    # Creates the default vector store (based on configuration)
    store = await create_vector_store()
    await store.connect()
    
    print(f"Connected to: {store.__class__.__name__}")
    
    # Always close connections when done
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(init_default_store())
```

### IPFS Vector Store

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def init_ipfs_store():
    # Explicitly create IPFS vector store
    store = await create_vector_store(db_type=VectorDBType.IPFS)
    await store.connect()
    
    print(f"Connected to IPFS at: {store.multiaddr}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(init_ipfs_store())
```

### DuckDB Vector Store

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def init_duckdb_store():
    # Explicitly create DuckDB vector store
    store = await create_vector_store(db_type=VectorDBType.DUCKDB)
    await store.connect()
    
    print(f"Connected to DuckDB at: {store.database_path}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(init_duckdb_store())
```

## Creating and Managing Indices

### Creating an Index

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_config import VectorDBType

async def create_index_example():
    # Create the vector store
    store = await create_vector_store()
    await store.connect()
    
    # Create index with 384 dimensions (matching model output)
    index_params = {
        "dimension": 384,
        "metric": "cosine"  # Options: "cosine", "inner_product", "l2"
    }
    
    index_id = await store.create_index(**index_params)
    print(f"Created index: {index_id}")
    
    # Check if index exists
    exists = await store.index_exists()
    print(f"Index exists: {exists}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(create_index_example())
```

### Managing Multiple Indices

```python
import asyncio
from services.vector_store_factory import create_vector_store

async def multi_index_example():
    store = await create_vector_store()
    await store.connect()
    
    # Create multiple indices
    index1 = await store.create_index(name="product_embeddings", dimension=384)
    index2 = await store.create_index(name="image_embeddings", dimension=512)
    
    # List all indices
    indices = await store.list_indices()
    print(f"Available indices: {indices}")
    
    # Switch between indices
    await store.select_index("product_embeddings")
    print(f"Selected index: {await store.get_current_index()}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(multi_index_example())
```

## Working with Vectors and Documents

### Adding Documents

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_store_base import VectorDocument

async def add_documents_example():
    store = await create_vector_store()
    await store.connect()
    await store.create_index(dimension=384)
    
    # Create sample documents
    documents = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(10):
        # Create a random vector (normally you'd get this from an embedding model)
        vector = np.random.random(384).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # Normalize for cosine similarity
        
        # Create a document
        doc = VectorDocument(
            id=f"doc-{i}",
            vector=vector.tolist(),
            metadata={
                "title": f"Document {i}",
                "content": f"This is the content of document {i}",
                "tags": ["sample", f"tag-{i}"],
                "created_at": "2025-01-01"
            }
        )
        documents.append(doc)
    
    # Add documents to the store
    count = await store.add_documents(documents)
    print(f"Added {count} documents")
    
    # Get document count
    total = await store.get_document_count()
    print(f"Total documents: {total}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(add_documents_example())
```

### Retrieving Documents

```python
import asyncio
from services.vector_store_factory import create_vector_store

async def get_documents_example():
    store = await create_vector_store()
    await store.connect()
    
    # Get a single document by ID
    doc = await store.get_document("doc-1")
    if doc:
        print(f"Retrieved document: {doc.id}")
        print(f"Metadata: {doc.metadata}")
    else:
        print("Document not found")
    
    # Get multiple documents by ID
    docs = await store.get_documents(["doc-1", "doc-2", "doc-3"])
    print(f"Retrieved {len(docs)} documents")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(get_documents_example())
```

### Updating Documents

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.vector_store_base import VectorDocument

async def update_documents_example():
    store = await create_vector_store()
    await store.connect()
    
    # Get the document to update
    doc = await store.get_document("doc-1")
    if not doc:
        print("Document not found")
        return
    
    # Update metadata (vector remains the same)
    doc.metadata["updated"] = True
    doc.metadata["tags"].append("updated")
    
    # Update the document
    success = await store.update_document(doc)
    print(f"Document update successful: {success}")
    
    # Verify the update
    updated_doc = await store.get_document("doc-1")
    print(f"Updated document metadata: {updated_doc.metadata}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(update_documents_example())
```

### Deleting Documents

```python
import asyncio
from services.vector_store_factory import create_vector_store

async def delete_documents_example():
    store = await create_vector_store()
    await store.connect()
    
    # Delete a single document
    success = await store.delete_document("doc-1")
    print(f"Deleted document: {success}")
    
    # Delete multiple documents
    count = await store.delete_documents(["doc-2", "doc-3"])
    print(f"Deleted {count} documents")
    
    # Verify deletion
    doc = await store.get_document("doc-1")
    print(f"Document still exists: {doc is not None}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(delete_documents_example())
```

## Searching for Similar Vectors

### Basic Vector Search

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_store_base import SearchQuery

async def basic_search_example():
    store = await create_vector_store()
    await store.connect()
    
    # Create a query vector (normally from an embedding model)
    np.random.seed(100)
    query_vector = np.random.random(384).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Create search query
    query = SearchQuery(
        vector=query_vector.tolist(),
        top_k=5
    )
    
    # Search
    results = await store.search(query)
    
    # Print results
    print(f"Found {len(results.matches)} matches:")
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        print(f"   Title: {match.metadata.get('title')}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(basic_search_example())
```

### Filtered Vector Search

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_store_base import SearchQuery

async def filtered_search_example():
    store = await create_vector_store()
    await store.connect()
    
    # Create a query vector
    np.random.seed(100)
    query_vector = np.random.random(384).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Create search query with filters
    query = SearchQuery(
        vector=query_vector.tolist(),
        top_k=5,
        filter={
            "tags": {"$contains": "sample"},  # Documents must have "sample" tag
            "created_at": {"$gte": "2025-01-01"}  # Created on or after 2025-01-01
        }
    )
    
    # Search
    results = await store.search(query)
    
    # Print results
    print(f"Found {len(results.matches)} filtered matches:")
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        print(f"   Tags: {match.metadata.get('tags')}")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(filtered_search_example())
```

### Hybrid Search (Vector + Keyword)

```python
import asyncio
import numpy as np
from services.vector_store_factory import create_vector_store
from services.vector_store_base import SearchQuery

async def hybrid_search_example():
    store = await create_vector_store()
    await store.connect()
    
    # Create a query vector
    np.random.seed(100)
    query_vector = np.random.random(384).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Create hybrid search query
    query = SearchQuery(
        vector=query_vector.tolist(),
        top_k=10,
        hybrid_search={
            "text": "document content",
            "fields": ["title", "content"],
            "vector_weight": 0.7,
            "text_weight": 0.3
        }
    )
    
    # Search
    results = await store.search(query)
    
    # Print results
    print(f"Found {len(results.matches)} hybrid matches:")
    for i, match in enumerate(results.matches):
        print(f"{i+1}. ID: {match.id}, Score: {match.score:.4f}")
        print(f"   Title: {match.metadata.get('title')}")
        print(f"   Content: {match.metadata.get('content')[:50]}...")
    
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(hybrid_search_example())
```

## Working with Embeddings

### Creating Embeddings

```python
import asyncio
from services.embedding import get_embedding_model

async def create_embeddings_example():
    # Get the embedding model
    model = get_embedding_model("thenlper/gte-small")
    
    # Create embeddings for a single text
    text = "This is a sample document for embedding."
    embedding = await model.embed(text)
    
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First few values: {embedding[:5]}")
    
    # Create embeddings for multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Vector databases store and retrieve embeddings efficiently."
    ]
    
    embeddings = await model.embed_batch(texts)
    
    print(f"Created {len(embeddings)} embeddings")
    for i, emb in enumerate(embeddings):
        print(f"Embedding {i+1} dimension: {len(emb)}")

if __name__ == "__main__":
    asyncio.run(create_embeddings_example())
```

### End-to-End Example

```python
import asyncio
from services.vector_store_factory import create_vector_store
from services.embedding import get_embedding_model
from services.vector_store_base import VectorDocument, SearchQuery

async def end_to_end_example():
    # 1. Connect to vector store
    store = await create_vector_store()
    await store.connect()
    
    # 2. Get embedding model
    model = get_embedding_model("thenlper/gte-small")
    
    # 3. Create sample documents
    texts = [
        "Artificial intelligence is reshaping industries worldwide.",
        "Machine learning models require significant training data.",
        "Neural networks excel at pattern recognition tasks.",
        "Natural language processing has advanced rapidly in recent years.",
        "Computer vision systems can now identify objects with high accuracy."
    ]
    
    # 4. Create embeddings
    embeddings = await model.embed_batch(texts)
    
    # 5. Create vector documents
    documents = []
    for i, (text, vector) in enumerate(zip(texts, embeddings)):
        doc = VectorDocument(
            id=f"tech-{i+1}",
            vector=vector,
            metadata={"text": text, "category": "technology"}
        )
        documents.append(doc)
    
    # 6. Create index and add documents
    await store.create_index(dimension=len(embeddings[0]))
    count = await store.add_documents(documents)
    print(f"Added {count} documents")
    
    # 7. Create query text and embedding
    query_text = "How does machine learning work?"
    query_embedding = await model.embed(query_text)
    
    # 8. Search using the query embedding
    query = SearchQuery(
        vector=query_embedding,
        top_k=3
    )
    
    results = await store.search(query)
    
    # 9. Process results
    print(f"\nSearch query: '{query_text}'")
    print(f"Found {len(results.matches)} matches:")
    for i, match in enumerate(results.matches):
        print(f"{i+1}. Score: {match.score:.4f}")
        print(f"   Text: {match.metadata.get('text')}")
    
    # 10. Clean up
    await store.disconnect()

if __name__ == "__main__":
    asyncio.run(end_to_end_example())
```

## Using with FastAPI

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any
import asyncio

from services.vector_store_factory import create_vector_store
from services.vector_store_base import SearchQuery, VectorStoreBase
from services.embedding import get_embedding_model

app = FastAPI()

# Models for API
class EmbeddingRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    text: str
    top_k: int = 5
    filter: Dict[str, Any] = None

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]

# Dependency for vector store
async def get_vector_store():
    store = await create_vector_store()
    await store.connect()
    try:
        yield store
    finally:
        await store.disconnect()

# Dependency for embedding model
async def get_model():
    return get_embedding_model("thenlper/gte-small")

# Create embedding endpoint
@app.post("/embed")
async def embed_text(request: EmbeddingRequest, model = Depends(get_model)):
    embedding = await model.embed(request.text)
    return {"embedding": embedding, "dimension": len(embedding)}

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    store: VectorStoreBase = Depends(get_vector_store),
    model = Depends(get_model)
):
    # Create embedding from text
    embedding = await model.embed(request.text)
    
    # Create search query
    query = SearchQuery(
        vector=embedding,
        top_k=request.top_k,
        filter=request.filter
    )
    
    # Execute search
    results = await store.search(query)
    
    # Format response
    response = {
        "results": [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
        ]
    }
    
    return response
```

## Conclusion

These examples demonstrate the fundamental operations of the LAION Embeddings system. For more advanced usage, including IPFS-specific and DuckDB-specific examples, see the [IPFS Examples](ipfs-examples.md) and [DuckDB Examples](duckdb-examples.md) documentation.
