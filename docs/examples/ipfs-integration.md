# IPFS Integration Example

This example demonstrates how to work with IPFS-stored embeddings, including uploading, downloading, and searching distributed datasets.

## Overview

You'll learn how to:
- Store embeddings on IPFS
- Retrieve embeddings from IPFS
- Search across distributed datasets
- Work with CAR files and CIDs
- Handle large datasets with sharding

## Prerequisites

- LAION Embeddings service running
- IPFS node running (`ipfs daemon`)
- Python with `ipfshttpclient`, `datasets`, `pandas`

## Setup

```bash
# Install dependencies
pip install ipfshttpclient datasets pandas pyarrow

# Start IPFS daemon
ipfs daemon --enable-gc &

# Verify IPFS connection
curl http://localhost:5001/api/v0/version
```

## Step 1: Basic IPFS Operations

```python
import ipfshttpclient
import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path

class IPFSEmbeddingsClient:
    def __init__(self, ipfs_api='/ip4/127.0.0.1/tcp/5001', 
                 embeddings_api='http://localhost:8000'):
        self.ipfs = ipfshttpclient.connect(ipfs_api)
        self.embeddings_api = embeddings_api
    
    def test_connections(self):
        """Test IPFS and embeddings service connections."""
        try:
            # Test IPFS
            ipfs_version = self.ipfs.version()
            print(f"✓ IPFS connected: {ipfs_version['Version']}")
            
            # Test embeddings service
            response = requests.get(f"{self.embeddings_api}/health")
            if response.status_code == 200:
                print("✓ Embeddings service connected")
                return True
            else:
                print("✗ Embeddings service not available")
                return False
                
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
    
    def create_and_store_embeddings(self, texts, model="gte-small"):
        """Create embeddings and store on IPFS."""
        
        # Create embeddings
        print(f"Creating embeddings for {len(texts)} texts...")
        payload = {
            "texts": texts,
            "model": model,
            "normalize": True
        }
        
        response = requests.post(
            f"{self.embeddings_api}/create_embeddings/",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Embeddings creation failed: {response.text}")
        
        embeddings = response.json()["embeddings"]
        
        # Prepare dataset
        dataset = {
            "texts": texts,
            "embeddings": embeddings,
            "model": model,
            "metadata": {
                "count": len(texts),
                "dimensions": len(embeddings[0]),
                "created_at": pd.Timestamp.now().isoformat()
            }
        }
        
        # Store on IPFS
        print("Storing on IPFS...")
        dataset_json = json.dumps(dataset)
        result = self.ipfs.add_json(dataset)
        
        print(f"✓ Stored on IPFS: {result}")
        return result
    
    def retrieve_embeddings(self, cid):
        """Retrieve embeddings from IPFS by CID."""
        try:
            dataset = self.ipfs.get_json(cid)
            print(f"✓ Retrieved dataset from IPFS")
            print(f"  Texts: {dataset['metadata']['count']}")
            print(f"  Dimensions: {dataset['metadata']['dimensions']}")
            print(f"  Model: {dataset['model']}")
            return dataset
        except Exception as e:
            print(f"✗ Failed to retrieve from IPFS: {e}")
            return None

# Basic usage example
def basic_ipfs_example():
    client = IPFSEmbeddingsClient()
    
    # Test connections
    if not client.test_connections():
        return
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "IPFS is a distributed file system",
        "Embeddings capture semantic meaning of text",
        "Python is a versatile programming language"
    ]
    
    # Create and store embeddings
    cid = client.create_and_store_embeddings(texts)
    
    # Retrieve embeddings
    dataset = client.retrieve_embeddings(cid)
    
    if dataset:
        print(f"Successfully stored and retrieved {len(dataset['texts'])} embeddings")

if __name__ == "__main__":
    basic_ipfs_example()
```

## Step 2: Working with Parquet Files

```python
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

class ParquetIPFSHandler:
    def __init__(self, ipfs_client, embeddings_client):
        self.ipfs = ipfs_client
        self.embeddings_client = embeddings_client
    
    def create_parquet_dataset(self, texts, model="gte-small", batch_size=100):
        """Create embeddings and save as Parquet on IPFS."""
        
        print(f"Processing {len(texts)} texts in batches...")
        
        all_data = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Create embeddings
            payload = {
                "texts": batch_texts,
                "model": model,
                "normalize": True
            }
            
            response = requests.post(
                f"{self.embeddings_client}/create_embeddings/",
                json=payload
            )
            
            if response.status_code == 200:
                embeddings = response.json()["embeddings"]
                
                # Add to dataset
                for text, embedding in zip(batch_texts, embeddings):
                    all_data.append({
                        "text": text,
                        "embedding": embedding,
                        "model": model,
                        "text_length": len(text)
                    })
                
                print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            else:
                print(f"Batch {i//batch_size + 1} failed: {response.text}")
        
        # Create Parquet file
        df = pd.DataFrame(all_data)
        
        # Save to local Parquet file
        parquet_file = f"embeddings_{model}_{len(texts)}.parquet"
        df.to_parquet(parquet_file, index=False)
        
        # Upload to IPFS
        print("Uploading Parquet file to IPFS...")
        with open(parquet_file, 'rb') as f:
            result = self.ipfs.add(f)
        
        cid = result['Hash']
        print(f"✓ Parquet file stored on IPFS: {cid}")
        
        # Clean up local file
        Path(parquet_file).unlink()
        
        return cid, df
    
    def load_parquet_from_ipfs(self, cid):
        """Load Parquet file from IPFS."""
        try:
            # Download from IPFS
            print(f"Downloading Parquet file from IPFS: {cid}")
            file_content = self.ipfs.cat(cid)
            
            # Write to temporary file
            temp_file = f"temp_{cid}.parquet"
            with open(temp_file, 'wb') as f:
                f.write(file_content)
            
            # Load as DataFrame
            df = pd.read_parquet(temp_file)
            
            # Clean up
            Path(temp_file).unlink()
            
            print(f"✓ Loaded {len(df)} records from IPFS")
            return df
            
        except Exception as e:
            print(f"✗ Failed to load Parquet from IPFS: {e}")
            return None
    
    def search_parquet_embeddings(self, cid, query_text, model="gte-small", top_k=5):
        """Search embeddings in Parquet file stored on IPFS."""
        
        # Load dataset from IPFS
        df = self.load_parquet_from_ipfs(cid)
        if df is None:
            return []
        
        # Create query embedding
        payload = {
            "texts": [query_text],
            "model": model,
            "normalize": True
        }
        
        response = requests.post(
            f"{self.embeddings_client}/create_embeddings/",
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Query embedding creation failed: {response.text}")
            return []
        
        query_embedding = response.json()["embeddings"][0]
        
        # Search using API
        corpus_embeddings = df['embedding'].tolist()
        corpus_texts = df['text'].tolist()
        
        search_payload = {
            "query_embedding": query_embedding,
            "embeddings": corpus_embeddings,
            "texts": corpus_texts,
            "top_k": top_k,
            "metric": "cosine"
        }
        
        response = requests.post(
            f"{self.embeddings_client}/search_embeddings/",
            json=search_payload
        )
        
        if response.status_code == 200:
            results = response.json()["results"]
            print(f"Found {len(results)} similar texts:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['text'][:100]}... (similarity: {result['similarity']:.3f})")
            return results
        else:
            print(f"Search failed: {response.text}")
            return []

# Example usage
def parquet_example():
    ipfs_client = ipfshttpclient.connect()
    embeddings_api = "http://localhost:8000"
    
    handler = ParquetIPFSHandler(ipfs_client, embeddings_api)
    
    # Create sample dataset
    texts = [
        "Artificial intelligence and machine learning",
        "Deep learning neural networks",
        "Natural language processing techniques",
        "Computer vision and image recognition",
        "Data science and analytics",
        "Cloud computing and distributed systems",
        "Blockchain and cryptocurrency",
        "Internet of Things (IoT) devices",
        "Cybersecurity and data protection",
        "Software development methodologies"
    ]
    
    # Create and store Parquet dataset
    cid, df = handler.create_parquet_dataset(texts)
    
    # Search the dataset
    query = "AI and machine learning"
    results = handler.search_parquet_embeddings(cid, query, top_k=3)
    
    return cid, results
```

## Step 3: CAR File Operations

```python
from ipfs_embeddings_py.ipfs_parquet_to_car import parquet_to_car
from ipfs_embeddings_py.ipfs_multiformats import generate_cid

class CARFileHandler:
    def __init__(self, ipfs_client):
        self.ipfs = ipfs_client
    
    def create_car_from_embeddings(self, embeddings_data, output_path="embeddings.car"):
        """Create CAR file from embeddings data."""
        
        # First save as Parquet
        df = pd.DataFrame(embeddings_data)
        parquet_path = "temp_embeddings.parquet"
        df.to_parquet(parquet_path, index=False)
        
        try:
            # Convert Parquet to CAR
            print(f"Converting Parquet to CAR format...")
            car_cid = parquet_to_car(parquet_path, output_path)
            print(f"✓ CAR file created: {output_path}")
            print(f"✓ CAR CID: {car_cid}")
            
            return car_cid, output_path
            
        finally:
            # Clean up temporary Parquet file
            if Path(parquet_path).exists():
                Path(parquet_path).unlink()
    
    def upload_car_to_ipfs(self, car_path):
        """Upload CAR file to IPFS."""
        try:
            with open(car_path, 'rb') as f:
                result = self.ipfs.add(f, pin=True)
            
            cid = result['Hash']
            print(f"✓ CAR file uploaded to IPFS: {cid}")
            return cid
            
        except Exception as e:
            print(f"✗ Failed to upload CAR file: {e}")
            return None
    
    def download_car_from_ipfs(self, cid, output_path):
        """Download CAR file from IPFS."""
        try:
            print(f"Downloading CAR file: {cid}")
            file_content = self.ipfs.cat(cid)
            
            with open(output_path, 'wb') as f:
                f.write(file_content)
            
            print(f"✓ CAR file downloaded: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Failed to download CAR file: {e}")
            return None

# Example usage
def car_file_example():
    ipfs_client = ipfshttpclient.connect()
    car_handler = CARFileHandler(ipfs_client)
    
    # Sample embeddings data
    embeddings_data = [
        {
            "text": "Sample text 1",
            "embedding": [0.1, 0.2, 0.3] * 128,  # 384-dim embedding
            "model": "gte-small"
        },
        {
            "text": "Sample text 2", 
            "embedding": [0.4, 0.5, 0.6] * 128,
            "model": "gte-small"
        }
    ]
    
    # Create CAR file
    car_cid, car_path = car_handler.create_car_from_embeddings(embeddings_data)
    
    # Upload to IPFS
    ipfs_cid = car_handler.upload_car_to_ipfs(car_path)
    
    # Download from IPFS
    if ipfs_cid:
        downloaded_path = car_handler.download_car_from_ipfs(ipfs_cid, "downloaded.car")
    
    return car_cid, ipfs_cid
```

## Step 4: Distributed Search

```python
class DistributedSearchEngine:
    def __init__(self, ipfs_client, embeddings_api):
        self.ipfs = ipfs_client
        self.embeddings_api = embeddings_api
        self.dataset_index = {}  # CID -> metadata
    
    def add_dataset(self, cid, metadata):
        """Add dataset to search index."""
        self.dataset_index[cid] = metadata
        print(f"Added dataset {cid} to index: {metadata.get('description', 'No description')}")
    
    def search_across_datasets(self, query_text, model="gte-small", top_k=10):
        """Search across all indexed datasets."""
        
        if not self.dataset_index:
            print("No datasets in index")
            return []
        
        # Create query embedding
        payload = {
            "texts": [query_text],
            "model": model,
            "normalize": True
        }
        
        response = requests.post(
            f"{self.embeddings_api}/create_embeddings/",
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Query embedding creation failed: {response.text}")
            return []
        
        query_embedding = response.json()["embeddings"][0]
        
        all_results = []
        
        # Search each dataset
        for cid, metadata in self.dataset_index.items():
            print(f"Searching dataset {cid}...")
            
            try:
                # Load dataset
                if metadata.get('format') == 'parquet':
                    df = self._load_parquet_dataset(cid)
                else:
                    df = self._load_json_dataset(cid)
                
                if df is None:
                    continue
                
                # Perform search
                dataset_results = self._search_dataset(
                    df, query_embedding, 
                    dataset_cid=cid,
                    top_k=top_k
                )
                
                all_results.extend(dataset_results)
                
            except Exception as e:
                print(f"Error searching dataset {cid}: {e}")
                continue
        
        # Sort by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        return all_results[:top_k]
    
    def _load_parquet_dataset(self, cid):
        """Load Parquet dataset from IPFS."""
        try:
            file_content = self.ipfs.cat(cid)
            temp_file = f"temp_{cid}.parquet"
            
            with open(temp_file, 'wb') as f:
                f.write(file_content)
            
            df = pd.read_parquet(temp_file)
            Path(temp_file).unlink()
            
            return df
        except Exception as e:
            print(f"Failed to load Parquet dataset {cid}: {e}")
            return None
    
    def _load_json_dataset(self, cid):
        """Load JSON dataset from IPFS."""
        try:
            dataset = self.ipfs.get_json(cid)
            df = pd.DataFrame({
                'text': dataset['texts'],
                'embedding': dataset['embeddings']
            })
            return df
        except Exception as e:
            print(f"Failed to load JSON dataset {cid}: {e}")
            return None
    
    def _search_dataset(self, df, query_embedding, dataset_cid, top_k=10):
        """Search within a single dataset."""
        
        corpus_embeddings = df['embedding'].tolist()
        corpus_texts = df['text'].tolist()
        
        # Calculate similarities
        similarities = []
        for embedding in corpus_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append(similarity)
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': corpus_texts[idx],
                'similarity': similarities[idx],
                'dataset_cid': dataset_cid,
                'index': int(idx)
            })
        
        return results

# Example usage
def distributed_search_example():
    ipfs_client = ipfshttpclient.connect()
    embeddings_api = "http://localhost:8000"
    
    search_engine = DistributedSearchEngine(ipfs_client, embeddings_api)
    
    # Add datasets to index (these would be real CIDs)
    search_engine.add_dataset(
        "QmYourDataset1CID",
        {
            "description": "Technology articles",
            "format": "parquet",
            "model": "gte-small",
            "count": 1000
        }
    )
    
    search_engine.add_dataset(
        "QmYourDataset2CID", 
        {
            "description": "Science papers",
            "format": "json",
            "model": "gte-small",
            "count": 500
        }
    )
    
    # Perform distributed search
    results = search_engine.search_across_datasets(
        "machine learning algorithms",
        top_k=5
    )
    
    print(f"Found {len(results)} results across datasets:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text'][:100]}...")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Dataset: {result['dataset_cid']}")
        print()
```

## Step 5: Complete IPFS Workflow

```python
#!/usr/bin/env python3
"""
Complete IPFS integration example.
"""

def complete_ipfs_workflow():
    print("=== IPFS Embeddings Workflow ===\n")
    
    # Initialize clients
    ipfs_client = ipfshttpclient.connect()
    embeddings_api = "http://localhost:8000"
    
    # Test connections
    print("1. Testing connections...")
    client = IPFSEmbeddingsClient()
    if not client.test_connections():
        return
    
    # Create sample dataset
    print("\n2. Creating sample dataset...")
    texts = [
        "Artificial intelligence revolutionizes technology",
        "Machine learning algorithms process big data",
        "Deep neural networks recognize patterns",
        "Natural language understanding improves",
        "Computer vision analyzes images effectively",
        "Robotics automates complex tasks",
        "Data science drives business insights",
        "Cloud computing scales applications",
        "Blockchain ensures data integrity",
        "IoT connects everyday devices"
    ]
    
    # Create embeddings and store as Parquet
    print("\n3. Creating embeddings and storing on IPFS...")
    parquet_handler = ParquetIPFSHandler(ipfs_client, embeddings_api)
    parquet_cid, df = parquet_handler.create_parquet_dataset(texts)
    
    # Convert to CAR format
    print("\n4. Converting to CAR format...")
    car_handler = CARFileHandler(ipfs_client)
    embeddings_data = df.to_dict('records')
    car_cid, car_path = car_handler.create_car_from_embeddings(embeddings_data)
    
    # Upload CAR to IPFS
    ipfs_car_cid = car_handler.upload_car_to_ipfs(car_path)
    
    # Set up distributed search
    print("\n5. Setting up distributed search...")
    search_engine = DistributedSearchEngine(ipfs_client, embeddings_api)
    search_engine.add_dataset(parquet_cid, {
        "description": "Sample AI/ML dataset",
        "format": "parquet",
        "model": "gte-small",
        "count": len(texts)
    })
    
    # Perform searches
    print("\n6. Performing searches...")
    queries = [
        "artificial intelligence",
        "data processing",
        "neural networks"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = parquet_handler.search_parquet_embeddings(
            parquet_cid, query, top_k=3
        )
    
    print(f"\n✓ Complete workflow finished!")
    print(f"  Parquet CID: {parquet_cid}")
    print(f"  CAR CID: {car_cid}")
    print(f"  IPFS CAR CID: {ipfs_car_cid}")
    
    # Clean up
    if Path(car_path).exists():
        Path(car_path).unlink()

if __name__ == "__main__":
    complete_ipfs_workflow()
```

## Performance Tips

1. **Batch Processing**: Process embeddings in batches for efficiency
2. **Compression**: Use Parquet compression for storage efficiency
3. **Sharding**: Split large datasets across multiple IPFS objects
4. **Caching**: Cache frequently accessed embeddings locally
5. **Pinning**: Pin important datasets to ensure availability

## Troubleshooting

Common IPFS issues:
- **Connection failures**: Check IPFS daemon is running
- **Upload timeouts**: Reduce batch sizes or check network
- **CID not found**: Ensure content is pinned or try different gateways
- **Memory issues**: Use streaming for large files

## Next Steps

- Explore [Batch Processing](./batch-processing.md) for large-scale operations
- Learn about [Custom Models](./custom-models.md) for specialized embeddings
- Check [Production Deployment](./production-deployment.md) for scaling IPFS integration
