# Simple Search Example

This example demonstrates basic text similarity search using the LAION Embeddings API.

## Overview

You'll learn how to:
- Start the embeddings service
- Create embeddings for text
- Perform similarity searches
- Handle API responses

## Prerequisites

- LAION Embeddings service installed and configured
- Python 3.8+ with `requests` library
- At least one embedding model endpoint running

## Step 1: Start the Service

First, start the embeddings service:

```bash
# Start with default configuration
python main.py

# Or with custom configuration
python main.py --config config.yaml
```

The service will start on `http://localhost:8000` by default.

## Step 2: Verify Service Status

Check that the service is running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "endpoints": {
    "gte-small": "running",
    "gte-large-en-v1.5": "running"
  }
}
```

## Step 3: Create Embeddings

### Python Example

```python
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def create_embedding(text, model="gte-small"):
    """Create an embedding for a single text."""
    url = f"{BASE_URL}/create_embeddings/"
    
    payload = {
        "texts": [text],
        "model": model,
        "normalize": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()["embeddings"][0]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Example usage
text = "The quick brown fox jumps over the lazy dog"
embedding = create_embedding(text)

print(f"Created embedding with {len(embedding)} dimensions")
print(f"First 5 values: {embedding[:5]}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/create_embeddings/" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The quick brown fox jumps over the lazy dog"],
    "model": "gte-small",
    "normalize": true
  }'
```

## Step 4: Search Similar Texts

### Python Example

```python
def search_similar_texts(query_text, corpus_texts, model="gte-small", top_k=3):
    """Find most similar texts in a corpus."""
    
    # Create embedding for query
    query_embedding = create_embedding(query_text, model)
    
    # Create embeddings for corpus
    corpus_embeddings = []
    for text in corpus_texts:
        embedding = create_embedding(text, model)
        corpus_embeddings.append(embedding)
    
    # Search using the API
    url = f"{BASE_URL}/search_embeddings/"
    
    payload = {
        "query_embedding": query_embedding,
        "embeddings": corpus_embeddings,
        "texts": corpus_texts,
        "top_k": top_k,
        "metric": "cosine"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()["results"]
    else:
        raise Exception(f"Search Error: {response.status_code} - {response.text}")

# Example corpus
corpus = [
    "A dog is running in the park",
    "The cat sits on the windowsill",
    "Birds fly in the blue sky",
    "Fish swim in the ocean",
    "A fox hunts in the forest"
]

# Search query
query = "Animals in nature"

# Find similar texts
results = search_similar_texts(query, corpus)

print(f"Query: {query}")
print("\nMost similar texts:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['text']} (similarity: {result['similarity']:.3f})")
```

Expected output:
```
Query: Animals in nature
Most similar texts:
1. A fox hunts in the forest (similarity: 0.847)
2. A dog is running in the park (similarity: 0.823)
3. Birds fly in the blue sky (similarity: 0.801)
```

## Step 5: Batch Processing

For better performance with multiple texts:

```python
def batch_create_embeddings(texts, model="gte-small", batch_size=32):
    """Create embeddings for multiple texts efficiently."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        url = f"{BASE_URL}/create_embeddings/"
        payload = {
            "texts": batch,
            "model": model,
            "normalize": True
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            embeddings = response.json()["embeddings"]
            all_embeddings.extend(embeddings)
        else:
            raise Exception(f"Batch Error: {response.status_code} - {response.text}")
    
    return all_embeddings

# Process large text collection
large_corpus = ["Text " + str(i) for i in range(100)]
embeddings = batch_create_embeddings(large_corpus)
print(f"Created {len(embeddings)} embeddings")
```

## Step 6: Working with Different Models

```python
def compare_models(text):
    """Compare embeddings from different models."""
    models = ["gte-small", "gte-large-en-v1.5"]
    
    results = {}
    for model in models:
        try:
            embedding = create_embedding(text, model)
            results[model] = {
                "dimensions": len(embedding),
                "norm": sum(x*x for x in embedding) ** 0.5,
                "sample": embedding[:3]
            }
        except Exception as e:
            results[model] = {"error": str(e)}
    
    return results

# Compare models
text = "Machine learning is transforming technology"
comparison = compare_models(text)

for model, info in comparison.items():
    print(f"\n{model}:")
    if "error" in info:
        print(f"  Error: {info['error']}")
    else:
        print(f"  Dimensions: {info['dimensions']}")
        print(f"  Norm: {info['norm']:.3f}")
        print(f"  Sample: {info['sample']}")
```

## Error Handling

```python
def robust_create_embedding(text, model="gte-small", max_retries=3):
    """Create embedding with error handling and retries."""
    import time
    
    for attempt in range(max_retries):
        try:
            return create_embedding(text, model)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e

# Usage with error handling
try:
    embedding = robust_create_embedding("Sample text", "gte-small")
    print("Successfully created embedding")
except Exception as e:
    print(f"Failed to create embedding: {e}")
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""
Complete simple search example for LAION Embeddings.
"""

import requests
import json
import time

class EmbeddingsClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check service health."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def create_embedding(self, text, model="gte-small"):
        """Create embedding for single text."""
        url = f"{self.base_url}/create_embeddings/"
        payload = {
            "texts": [text],
            "model": model,
            "normalize": True
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()["embeddings"][0]
    
    def search_similar(self, query_text, corpus_texts, model="gte-small", top_k=3):
        """Search for similar texts."""
        # Create embeddings
        query_embedding = self.create_embedding(query_text, model)
        corpus_embeddings = [
            self.create_embedding(text, model) 
            for text in corpus_texts
        ]
        
        # Search
        url = f"{self.base_url}/search_embeddings/"
        payload = {
            "query_embedding": query_embedding,
            "embeddings": corpus_embeddings,
            "texts": corpus_texts,
            "top_k": top_k,
            "metric": "cosine"
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()["results"]

def main():
    # Initialize client
    client = EmbeddingsClient()
    
    # Check service health
    if not client.health_check():
        print("Error: Service is not healthy")
        return
    
    print("✓ Service is running")
    
    # Example data
    query = "What is artificial intelligence?"
    corpus = [
        "AI is a branch of computer science",
        "Machine learning uses algorithms to learn patterns",
        "Natural language processing helps computers understand text",
        "The weather is sunny today",
        "Cooking requires fresh ingredients"
    ]
    
    print(f"\nQuery: {query}")
    print("Searching in corpus...")
    
    # Perform search
    try:
        results = client.search_similar(query, corpus, top_k=3)
        
        print("\nTop results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text']}")
            print(f"   Similarity: {result['similarity']:.3f}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## Next Steps

- Try the [Batch Processing Example](./batch-processing.md) for handling larger datasets
- Explore [IPFS Integration](./ipfs-integration.md) for distributed storage
- Learn about [Custom Models](./custom-models.md) for specialized use cases

## Troubleshooting

If you encounter issues:
1. Verify the service is running: `curl http://localhost:8000/health`
2. Check logs for error messages
3. Ensure your model endpoints are configured correctly
4. See the [Troubleshooting Guide](../troubleshooting/README.md) for common issues
