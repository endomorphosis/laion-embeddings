# Python Client Example

This example shows how to build applications using the LAION Embeddings Python API, including client libraries, error handling, and integration patterns.

## Overview

You'll learn how to:
- Create a robust Python client
- Handle authentication and errors
- Implement async operations
- Build client applications
- Integrate with popular frameworks

## Prerequisites

- LAION Embeddings service running
- Python 3.8+ with `requests`, `asyncio`, `aiohttp`

```bash
pip install requests aiohttp asyncio pandas numpy tqdm
```

## Step 1: Basic Client Library

```python
import requests
import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from urllib.parse import urljoin

@dataclass
class EmbeddingResult:
    """Result from embedding creation."""
    embeddings: List[List[float]]
    model: str
    processing_time: float
    token_count: Optional[int] = None

@dataclass
class SearchResult:
    """Result from similarity search."""
    text: str
    similarity: float
    index: int
    metadata: Optional[Dict] = None

class EmbeddingsClientError(Exception):
    """Base exception for client errors."""
    pass

class EmbeddingsClient:
    """Synchronous client for LAION Embeddings API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        
        # Configure session
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> Dict:
        """Check service health."""
        try:
            response = self._request("GET", "/health")
            return response.json()
        except Exception as e:
            raise EmbeddingsClientError(f"Health check failed: {e}")
    
    def create_embeddings(
        self,
        texts: List[str],
        model: str = "gte-small",
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """Create embeddings for texts."""
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Use batch processing if batch_size specified
        if batch_size and len(texts) > batch_size:
            return self._create_embeddings_batched(texts, model, normalize, batch_size)
        
        start_time = time.time()
        
        payload = {
            "texts": texts,
            "model": model,
            "normalize": normalize
        }
        
        response = self._request("POST", "/create_embeddings/", json=payload)
        result = response.json()
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=result["embeddings"],
            model=model,
            processing_time=processing_time,
            token_count=result.get("token_count")
        )
    
    def search_embeddings(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        texts: List[str],
        top_k: int = 10,
        metric: str = "cosine",
        metadata: Optional[List[Dict]] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        
        payload = {
            "query_embedding": query_embedding,
            "embeddings": embeddings,
            "texts": texts,
            "top_k": top_k,
            "metric": metric
        }
        
        response = self._request("POST", "/search_embeddings/", json=payload)
        results = response.json()["results"]
        
        search_results = []
        for i, result in enumerate(results):
            search_results.append(SearchResult(
                text=result["text"],
                similarity=result["similarity"],
                index=result["index"],
                metadata=metadata[result["index"]] if metadata else None
            ))
        
        return search_results
    
    def search_by_text(
        self,
        query_text: str,
        corpus_texts: List[str],
        model: str = "gte-small",
        top_k: int = 10,
        metric: str = "cosine"
    ) -> List[SearchResult]:
        """Search by text query (convenience method)."""
        
        # Create embeddings for query and corpus
        all_texts = [query_text] + corpus_texts
        result = self.create_embeddings(all_texts, model)
        
        query_embedding = result.embeddings[0]
        corpus_embeddings = result.embeddings[1:]
        
        return self.search_embeddings(
            query_embedding=query_embedding,
            embeddings=corpus_embeddings,
            texts=corpus_texts,
            top_k=top_k,
            metric=metric
        )
    
    def _create_embeddings_batched(
        self,
        texts: List[str],
        model: str,
        normalize: bool,
        batch_size: int
    ) -> EmbeddingResult:
        """Create embeddings in batches."""
        
        all_embeddings = []
        total_time = 0
        
        self.logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_result = self.create_embeddings(batch, model, normalize)
            
            all_embeddings.extend(batch_result.embeddings)
            total_time += batch_result.processing_time
            
            self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            processing_time=total_time
        )
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retries."""
        
        url = urljoin(self.base_url, endpoint)
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise EmbeddingsClientError(f"Request failed: {last_error}")
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Usage example
def basic_client_example():
    with EmbeddingsClient() as client:
        # Health check
        health = client.health_check()
        print(f"Service status: {health['status']}")
        
        # Create embeddings
        texts = ["Hello world", "Python programming", "Machine learning"]
        result = client.create_embeddings(texts)
        print(f"Created {len(result.embeddings)} embeddings in {result.processing_time:.2f}s")
        
        # Search
        search_results = client.search_by_text(
            "programming languages",
            ["Python is great", "Java is verbose", "C++ is fast"],
            top_k=2
        )
        
        for result in search_results:
            print(f"{result.text} (similarity: {result.similarity:.3f})")

if __name__ == "__main__":
    basic_client_example()
```

## Step 2: Async Client

```python
import aiohttp
import asyncio
from typing import AsyncGenerator

class AsyncEmbeddingsClient:
    """Asynchronous client for LAION Embeddings API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        max_connections: int = 100
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.api_key = api_key
        
        # Headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Connection limits
        connector = aiohttp.TCPConnector(limit=max_connections)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            timeout=self.timeout
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def health_check(self) -> Dict:
        """Check service health."""
        async with self.session.get(f"{self.base_url}/health") as response:
            response.raise_for_status()
            return await response.json()
    
    async def create_embeddings(
        self,
        texts: List[str],
        model: str = "gte-small",
        normalize: bool = True
    ) -> EmbeddingResult:
        """Create embeddings for texts."""
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        start_time = time.time()
        
        payload = {
            "texts": texts,
            "model": model,
            "normalize": normalize
        }
        
        async with self.session.post(
            f"{self.base_url}/create_embeddings/",
            json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=result["embeddings"],
            model=model,
            processing_time=processing_time,
            token_count=result.get("token_count")
        )
    
    async def create_embeddings_concurrent(
        self,
        texts: List[str],
        model: str = "gte-small",
        batch_size: int = 32,
        max_concurrent: int = 5
    ) -> EmbeddingResult:
        """Create embeddings with concurrent processing."""
        
        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await self.create_embeddings(batch, model)
        
        # Process batches concurrently
        start_time = time.time()
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Combine results
        all_embeddings = []
        for result in batch_results:
            all_embeddings.extend(result.embeddings)
        
        total_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            processing_time=total_time
        )
    
    async def search_embeddings(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        texts: List[str],
        top_k: int = 10,
        metric: str = "cosine"
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        
        payload = {
            "query_embedding": query_embedding,
            "embeddings": embeddings,
            "texts": texts,
            "top_k": top_k,
            "metric": metric
        }
        
        async with self.session.post(
            f"{self.base_url}/search_embeddings/",
            json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()
        
        search_results = []
        for result_item in result["results"]:
            search_results.append(SearchResult(
                text=result_item["text"],
                similarity=result_item["similarity"],
                index=result_item["index"]
            ))
        
        return search_results
    
    async def stream_embeddings(
        self,
        texts: AsyncGenerator[str, None],
        model: str = "gte-small",
        batch_size: int = 32
    ) -> AsyncGenerator[EmbeddingResult, None]:
        """Stream embeddings for continuous processing."""
        
        batch = []
        async for text in texts:
            batch.append(text)
            
            if len(batch) >= batch_size:
                result = await self.create_embeddings(batch, model)
                yield result
                batch = []
        
        # Process remaining texts
        if batch:
            result = await self.create_embeddings(batch, model)
            yield result
    
    async def close(self):
        """Close the session."""
        await self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Async usage examples
async def async_client_example():
    async with AsyncEmbeddingsClient() as client:
        # Health check
        health = await client.health_check()
        print(f"Service status: {health['status']}")
        
        # Create embeddings
        texts = ["Hello world", "Python programming", "Machine learning"]
        result = await client.create_embeddings(texts)
        print(f"Created {len(result.embeddings)} embeddings in {result.processing_time:.2f}s")
        
        # Concurrent processing
        large_texts = [f"Sample text {i}" for i in range(100)]
        result = await client.create_embeddings_concurrent(
            large_texts,
            batch_size=20,
            max_concurrent=3
        )
        print(f"Processed {len(result.embeddings)} embeddings concurrently in {result.processing_time:.2f}s")

async def streaming_example():
    async def text_generator():
        for i in range(100):
            yield f"This is sample text number {i}"
    
    async with AsyncEmbeddingsClient() as client:
        async for result in client.stream_embeddings(text_generator(), batch_size=10):
            print(f"Processed batch of {len(result.embeddings)} embeddings")

if __name__ == "__main__":
    asyncio.run(async_client_example())
```

## Step 3: Client with Caching

```python
import hashlib
import pickle
import os
from datetime import datetime, timedelta

class CachedEmbeddingsClient(EmbeddingsClient):
    """Client with local caching capabilities."""
    
    def __init__(self, cache_dir: str = "./cache", cache_ttl: int = 3600, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def create_embeddings(self, texts: List[str], model: str = "gte-small", 
                         normalize: bool = True, use_cache: bool = True) -> EmbeddingResult:
        """Create embeddings with caching."""
        
        if not use_cache:
            return super().create_embeddings(texts, model, normalize)
        
        # Generate cache key
        cache_key = self._generate_cache_key(texts, model, normalize)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid
                if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                    self.logger.info(f"Cache hit for {len(texts)} texts")
                    return cached_data['result']
                else:
                    self.logger.info("Cache expired, removing old cache")
                    os.remove(cache_path)
            
            except Exception as e:
                self.logger.warning(f"Cache read error: {e}")
        
        # Create embeddings
        result = super().create_embeddings(texts, model, normalize)
        
        # Save to cache
        try:
            cached_data = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            self.logger.info(f"Cached embeddings for {len(texts)} texts")
        
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
        
        return result
    
    def _generate_cache_key(self, texts: List[str], model: str, normalize: bool) -> str:
        """Generate cache key from inputs."""
        content = f"{model}:{normalize}:" + "|".join(texts)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))
        self.logger.info("Cache cleared")
    
    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in cache_files
        )
        
        return {
            'files': len(cache_files),
            'total_size_mb': total_size / 1024 / 1024,
            'cache_dir': self.cache_dir
        }

# Caching example
def caching_example():
    with CachedEmbeddingsClient(cache_ttl=1800) as client:  # 30 min cache
        texts = ["Python programming", "Machine learning", "Data science"]
        
        # First call - creates embeddings
        print("First call (no cache):")
        start = time.time()
        result1 = client.create_embeddings(texts)
        print(f"Time: {time.time() - start:.2f}s")
        
        # Second call - uses cache
        print("\nSecond call (with cache):")
        start = time.time()
        result2 = client.create_embeddings(texts)
        print(f"Time: {time.time() - start:.2f}s")
        
        # Cache stats
        stats = client.cache_stats()
        print(f"\nCache stats: {stats}")
```

## Step 4: Application Examples

### Document Search Application

```python
import pandas as pd
from pathlib import Path

class DocumentSearchApp:
    """Document search application using embeddings."""
    
    def __init__(self, client: EmbeddingsClient):
        self.client = client
        self.documents = []
        self.embeddings = []
        self.index_built = False
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the search index."""
        
        texts = [doc['content'] for doc in documents]
        
        print(f"Creating embeddings for {len(documents)} documents...")
        result = self.client.create_embeddings(texts)
        
        self.documents.extend(documents)
        self.embeddings.extend(result.embeddings)
        self.index_built = True
        
        print(f"Added {len(documents)} documents to index")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents by text query."""
        
        if not self.index_built:
            raise ValueError("No documents in index")
        
        # Create query embedding
        query_result = self.client.create_embeddings([query])
        query_embedding = query_result.embeddings[0]
        
        # Search
        texts = [doc['content'] for doc in self.documents]
        search_results = self.client.search_embeddings(
            query_embedding=query_embedding,
            embeddings=self.embeddings,
            texts=texts,
            top_k=top_k
        )
        
        # Combine with document metadata
        results = []
        for result in search_results:
            doc = self.documents[result.index]
            results.append({
                'title': doc.get('title', 'Untitled'),
                'content': result.text,
                'similarity': result.similarity,
                'metadata': doc.get('metadata', {})
            })
        
        return results
    
    def save_index(self, filepath: str):
        """Save search index to file."""
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load search index from file."""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data['documents']
        self.embeddings = index_data['embeddings']
        self.index_built = True
        
        print(f"Index loaded from {filepath}")

# Document search example
def document_search_example():
    # Sample documents
    documents = [
        {
            'title': 'Introduction to Python',
            'content': 'Python is a high-level programming language known for its simplicity',
            'metadata': {'category': 'programming', 'author': 'John Doe'}
        },
        {
            'title': 'Machine Learning Basics',
            'content': 'Machine learning algorithms can learn patterns from data automatically',
            'metadata': {'category': 'ai', 'author': 'Jane Smith'}
        },
        {
            'title': 'Web Development',
            'content': 'Building web applications requires knowledge of HTML, CSS, and JavaScript',
            'metadata': {'category': 'web', 'author': 'Bob Johnson'}
        }
    ]
    
    # Create search app
    with EmbeddingsClient() as client:
        app = DocumentSearchApp(client)
        
        # Build index
        app.add_documents(documents)
        
        # Search
        results = app.search("programming languages", top_k=2)
        
        print("Search results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
            print(f"   {result['content'][:100]}...")
            print(f"   Category: {result['metadata']['category']}")
            print()
```

### Semantic Clustering Application

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

class SemanticClusteringApp:
    """Application for semantic clustering of texts."""
    
    def __init__(self, client: EmbeddingsClient):
        self.client = client
        self.texts = []
        self.embeddings = []
        self.clusters = None
    
    def add_texts(self, texts: List[str]):
        """Add texts and create embeddings."""
        
        print(f"Creating embeddings for {len(texts)} texts...")
        result = self.client.create_embeddings(texts)
        
        self.texts.extend(texts)
        self.embeddings.extend(result.embeddings)
        
        print(f"Added {len(texts)} texts")
    
    def cluster(self, n_clusters: int = 5) -> Dict:
        """Perform semantic clustering."""
        
        if not self.embeddings:
            raise ValueError("No texts added")
        
        # Perform clustering
        embeddings_array = np.array(self.embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Organize results
        clusters = {}
        for i, (text, label) in enumerate(zip(self.texts, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'text': text,
                'index': i
            })
        
        self.clusters = clusters
        
        return {
            'n_clusters': n_clusters,
            'cluster_sizes': [len(cluster) for cluster in clusters.values()],
            'silhouette_score': self._calculate_silhouette_score(embeddings_array, cluster_labels)
        }
    
    def visualize_clusters(self, save_path: Optional[str] = None):
        """Visualize clusters using PCA."""
        
        if self.clusters is None:
            raise ValueError("No clustering performed")
        
        # Reduce dimensions for visualization
        embeddings_array = np.array(self.embeddings)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
        
        for cluster_id, color in zip(self.clusters.keys(), colors):
            cluster_points = self.clusters[cluster_id]
            indices = [point['index'] for point in cluster_points]
            
            plt.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=[color],
                label=f'Cluster {cluster_id} ({len(indices)} texts)',
                alpha=0.7
            )
        
        plt.title('Semantic Clustering Visualization')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_cluster_summaries(self) -> Dict:
        """Get representative texts for each cluster."""
        
        if self.clusters is None:
            raise ValueError("No clustering performed")
        
        summaries = {}
        
        for cluster_id, cluster_texts in self.clusters.items():
            # Get cluster center
            cluster_indices = [item['index'] for item in cluster_texts]
            cluster_embeddings = np.array([self.embeddings[i] for i in cluster_indices])
            center = np.mean(cluster_embeddings, axis=0)
            
            # Find closest text to center
            distances = [
                np.linalg.norm(embedding - center)
                for embedding in cluster_embeddings
            ]
            
            closest_idx = np.argmin(distances)
            representative_text = cluster_texts[closest_idx]['text']
            
            summaries[cluster_id] = {
                'size': len(cluster_texts),
                'representative': representative_text,
                'texts': [item['text'] for item in cluster_texts[:5]]  # Top 5
            }
        
        return summaries
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(embeddings, labels)
        except ImportError:
            return 0.0  # sklearn not available

# Clustering example
def clustering_example():
    # Sample texts from different topics
    texts = [
        # Programming
        "Python is a versatile programming language",
        "JavaScript runs in web browsers",
        "C++ is used for system programming",
        
        # Science
        "Physics studies matter and energy",
        "Chemistry involves molecular interactions",
        "Biology examines living organisms",
        
        # Food
        "Pizza is a popular Italian dish",
        "Sushi originated in Japan",
        "Tacos are traditional Mexican food",
        
        # Sports
        "Football is played with a round ball",
        "Basketball involves shooting hoops",
        "Tennis requires a racket and ball"
    ]
    
    with EmbeddingsClient() as client:
        app = SemanticClusteringApp(client)
        
        # Add texts and cluster
        app.add_texts(texts)
        stats = app.cluster(n_clusters=4)
        
        print(f"Clustering results:")
        print(f"  Clusters: {stats['n_clusters']}")
        print(f"  Silhouette score: {stats['silhouette_score']:.3f}")
        
        # Get summaries
        summaries = app.get_cluster_summaries()
        
        for cluster_id, summary in summaries.items():
            print(f"\nCluster {cluster_id} ({summary['size']} texts):")
            print(f"  Representative: {summary['representative']}")
            print("  Sample texts:")
            for text in summary['texts'][:3]:
                print(f"    - {text}")
        
        # Visualize (requires matplotlib)
        try:
            app.visualize_clusters()
        except Exception as e:
            print(f"Visualization not available: {e}")
```

## Step 5: Error Handling and Monitoring

```python
import logging
from datetime import datetime
from contextlib import contextmanager

class MonitoredEmbeddingsClient(EmbeddingsClient):
    """Client with comprehensive monitoring and error handling."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Metrics
        self.metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'total_processing_time': 0,
            'total_texts_processed': 0
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup detailed logging."""
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @contextmanager
    def _monitor_request(self, operation: str):
        """Context manager for monitoring requests."""
        
        start_time = time.time()
        self.metrics['requests_total'] += 1
        
        self.logger.info(f"Starting {operation}")
        
        try:
            yield
            
            # Success
            duration = time.time() - start_time
            self.metrics['requests_successful'] += 1
            self.metrics['total_processing_time'] += duration
            
            self.logger.info(f"Completed {operation} in {duration:.2f}s")
            
        except Exception as e:
            # Failure
            self.metrics['requests_failed'] += 1
            self.logger.error(f"Failed {operation}: {e}")
            raise
    
    def create_embeddings(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Create embeddings with monitoring."""
        
        with self._monitor_request(f"create_embeddings(texts={len(texts)})"):
            result = super().create_embeddings(texts, **kwargs)
            self.metrics['total_texts_processed'] += len(texts)
            return result
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics['requests_total'] > 0:
            metrics['success_rate'] = metrics['requests_successful'] / metrics['requests_total']
            metrics['failure_rate'] = metrics['requests_failed'] / metrics['requests_total']
        
        if metrics['total_texts_processed'] > 0:
            metrics['avg_processing_time_per_text'] = (
                metrics['total_processing_time'] / metrics['total_texts_processed']
            )
        
        if metrics['requests_successful'] > 0:
            metrics['avg_processing_time_per_request'] = (
                metrics['total_processing_time'] / metrics['requests_successful']
            )
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {key: 0 for key in self.metrics}

# Monitoring example
def monitoring_example():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    with MonitoredEmbeddingsClient() as client:
        try:
            # Successful operations
            result1 = client.create_embeddings(["Hello world"])
            result2 = client.create_embeddings(["Python programming", "Data science"])
            
            # This might fail
            result3 = client.create_embeddings(["Text"] * 1000)  # Large batch
            
        except Exception as e:
            print(f"Operation failed: {e}")
        
        # Check metrics
        metrics = client.get_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
```

## Performance Tips

1. **Use batching** for multiple texts
2. **Enable caching** for repeated requests
3. **Use async client** for high-throughput applications
4. **Monitor metrics** to identify bottlenecks
5. **Implement retries** for reliability

## Next Steps

- Explore [Batch Processing](./batch-processing.md) for large-scale operations
- Learn about [IPFS Integration](./ipfs-integration.md) for distributed storage
- Check [Production Deployment](./production-deployment.md) for scaling applications

This comprehensive Python client example provides a solid foundation for building applications with the LAION Embeddings API.
