# Evaluation and Benchmarking

The LAION Embeddings project includes comprehensive evaluation frameworks for assessing embedding quality, retrieval performance, and system effectiveness using industry-standard benchmarks.

## Table of Contents

- [Overview](#overview)
- [Evaluation Frameworks](#evaluation-frameworks)
- [Benchmark Datasets](#benchmark-datasets)
- [Metrics and Scoring](#metrics-and-scoring)
- [Running Evaluations](#running-evaluations)
- [Custom Evaluations](#custom-evaluations)
- [Performance Analysis](#performance-analysis)
- [Integration Testing](#integration-testing)
- [Best Practices](#best-practices)

## Overview

The evaluation system supports multiple evaluation methodologies:

- **Information Retrieval Benchmarks** - BEIR, HotpotQA, and custom IR tasks
- **Quality Assessment** - Answer consistency, retrieval precision, augmentation accuracy
- **Performance Benchmarking** - Throughput, latency, and resource utilization
- **Validation Frameworks** - Tonic Validate for comprehensive RAG evaluation

### Supported Evaluation Types

| Type | Framework | Purpose | Metrics |
|------|-----------|---------|---------|
| Information Retrieval | BEIR | Standard IR benchmarks | NDCG, MAP, Recall, Precision |
| Question Answering | HotpotQA | Multi-hop reasoning | Exact Match, F1 Score |
| RAG Quality | Tonic Validate | End-to-end RAG evaluation | Answer Consistency, Retrieval Precision |
| Performance | Custom | System benchmarking | Throughput, Latency, Resource Usage |

## Evaluation Frameworks

### BEIR (Benchmarking Information Retrieval)

BEIR provides standardized evaluation across multiple information retrieval datasets.

#### Supported BEIR Datasets

```python
# Available BEIR datasets
BEIR_DATASETS = [
    "nfcorpus",     # Medical information retrieval
    "scifact",      # Scientific fact verification
    "arguana",      # Argument retrieval
    "scidocs",      # Scientific document retrieval
    "fiqa",         # Financial question answering
    "trec-covid",   # COVID-19 research
    "nq",           # Natural Questions
    "msmarco",      # Microsoft MARCO
    "hotpotqa",     # Multi-hop QA
    "climate-fever" # Climate fact verification
]
```

#### Running BEIR Evaluation

```python
from ipfs_embeddings_py.llama_index.core.evaluation.benchmarks import BeirEvaluator
from ipfs_embeddings_py import ipfs_embeddings_py

# Initialize evaluator
evaluator = BeirEvaluator()

# Create retriever function
def create_retriever(documents):
    """Create retriever from documents for evaluation"""
    # Initialize embeddings client
    resources = {
        "local_endpoints": [["thenlper/gte-small", "cpu", 512]]
    }
    client = ipfs_embeddings_py(resources, {})
    
    # Build retriever with embeddings
    return client.create_retriever(documents)

# Run evaluation
evaluator.run(
    create_retriever=create_retriever,
    datasets=["nfcorpus", "scifact"],
    metrics_k_values=[3, 10, 20],
    node_postprocessors=None
)
```

### HotpotQA Evaluation

HotpotQA evaluates multi-hop reasoning capabilities in question answering.

```python
from ipfs_embeddings_py.llama_index.core.evaluation.benchmarks import HotpotQAEvaluator

# Initialize evaluator
evaluator = HotpotQAEvaluator()

# Run evaluation with query engine
evaluator.run(
    query_engine=query_engine,
    queries=100,  # Number of queries to evaluate
    queries_fraction=0.1,  # Or fraction of dataset
    show_result=True  # Show individual results
)
```

### Tonic Validate Framework

Comprehensive RAG evaluation using Tonic Validate metrics.

```python
from ipfs_embeddings_py.llama_index.legacy.evaluation.tonic_validate import (
    TonicValidateEvaluator,
    AnswerConsistencyEvaluator,
    RetrievalPrecisionEvaluator,
    AugmentationAccuracyEvaluator
)

# Individual evaluators
consistency_evaluator = AnswerConsistencyEvaluator()
precision_evaluator = RetrievalPrecisionEvaluator()
accuracy_evaluator = AugmentationAccuracyEvaluator()

# Comprehensive evaluation
tonic_evaluator = TonicValidateEvaluator(
    model_evaluator="gpt-4"  # LLM for evaluation
)

# Evaluate single query-response pair
result = await tonic_evaluator.aevaluate(
    query="What is the capital of France?",
    response="The capital of France is Paris.",
    contexts=["France is a country in Europe...", "Paris is located..."],
    reference_response="Paris is the capital of France."
)

print(f"Score: {result.score}")
print(f"Detailed scores: {result.score_dict}")
```

## Benchmark Datasets

### BEIR Dataset Details

#### NFCorpus (Medical)
- **Domain**: Medical information retrieval
- **Size**: 3,633 queries, 3.6M passages
- **Task**: Medical question answering
- **Language**: English

#### SciFact (Scientific)
- **Domain**: Scientific fact verification
- **Size**: 300 queries, 5,183 passages
- **Task**: Scientific claim verification
- **Language**: English

#### MS MARCO (General)
- **Domain**: General web search
- **Size**: 6,980 queries, 8.8M passages
- **Task**: Web passage retrieval
- **Language**: English

#### Example Dataset Loading

```python
def evaluate_on_custom_dataset():
    """Evaluate on custom dataset format"""
    
    # Load custom dataset
    documents = [
        {"id": "doc1", "text": "Document text 1", "title": "Title 1"},
        {"id": "doc2", "text": "Document text 2", "title": "Title 2"},
        # ... more documents
    ]
    
    queries = [
        {"id": "q1", "text": "Query text 1"},
        {"id": "q2", "text": "Query text 2"},
        # ... more queries
    ]
    
    # Ground truth relevance judgments
    qrels = {
        "q1": {"doc1": 1, "doc2": 0},  # 1 = relevant, 0 = not relevant
        "q2": {"doc1": 0, "doc2": 1},
        # ... more relevance judgments
    }
    
    return documents, queries, qrels
```

## Metrics and Scoring

### Information Retrieval Metrics

#### NDCG (Normalized Discounted Cumulative Gain)
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Measures ranking quality with position discount
- **Interpretation**: How well relevant documents are ranked

#### MAP (Mean Average Precision)
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Average precision across all queries
- **Interpretation**: Overall retrieval precision

#### Recall@K
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Fraction of relevant documents retrieved in top K
- **Interpretation**: Coverage of relevant results

#### Precision@K
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Fraction of retrieved documents that are relevant
- **Interpretation**: Accuracy of top K results

### Question Answering Metrics

#### Exact Match (EM)
- **Range**: 0.0 to 1.0
- **Purpose**: Binary match between prediction and ground truth
- **Implementation**: Normalized string comparison

```python
def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match score"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison"""
    # Remove articles (a, an, the)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Fix whitespace
    text = ' '.join(text.split())
    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    # Convert to lowercase
    return text.lower()
```

#### F1 Score
- **Range**: 0.0 to 1.0
- **Purpose**: Token-level overlap between prediction and ground truth
- **Implementation**: Precision and recall of word tokens

```python
def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Calculate F1 score between prediction and ground truth"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Calculate token overlap
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1, precision, recall
```

### Tonic Validate Metrics

#### Answer Consistency
- **Range**: 0.0 to 1.0
- **Purpose**: Measures consistency between answers and context
- **Method**: LLM-based evaluation

#### Retrieval Precision
- **Range**: 0.0 to 1.0
- **Purpose**: Evaluates relevance of retrieved contexts
- **Method**: LLM judges context relevance

#### Augmentation Accuracy
- **Range**: 0.0 to 1.0
- **Purpose**: Measures accuracy of context augmentation
- **Method**: Evaluates whether context improves answers

## Running Evaluations

### Complete Evaluation Pipeline

```python
import asyncio
from pathlib import Path
import json

class EvaluationPipeline:
    def __init__(self, embeddings_client, output_dir="./evaluation_results"):
        self.client = embeddings_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    async def run_full_evaluation(self):
        """Run comprehensive evaluation suite"""
        
        results = {
            "beir_results": await self.run_beir_evaluation(),
            "hotpotqa_results": await self.run_hotpotqa_evaluation(),
            "tonic_results": await self.run_tonic_evaluation(),
            "performance_results": await self.run_performance_evaluation()
        }
        
        # Save results
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    async def run_beir_evaluation(self):
        """Run BEIR benchmark evaluation"""
        evaluator = BeirEvaluator()
        
        def create_retriever(documents):
            return self.client.create_retriever(documents)
        
        # Test on multiple datasets
        datasets = ["nfcorpus", "scifact", "arguana"]
        results = {}
        
        for dataset in datasets:
            print(f"Evaluating on {dataset}...")
            result = evaluator.run(
                create_retriever=create_retriever,
                datasets=[dataset],
                metrics_k_values=[1, 3, 5, 10, 20]
            )
            results[dataset] = result
            
        return results
    
    async def run_hotpotqa_evaluation(self):
        """Run HotpotQA evaluation"""
        evaluator = HotpotQAEvaluator()
        
        # Create query engine from embeddings client
        query_engine = self.client.create_query_engine()
        
        return evaluator.run(
            query_engine=query_engine,
            queries=100,
            show_result=False
        )
    
    async def run_tonic_evaluation(self):
        """Run Tonic Validate evaluation"""
        evaluator = TonicValidateEvaluator()
        
        # Sample evaluation data
        test_cases = [
            {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence...",
                "contexts": ["AI context 1", "ML context 2"],
                "reference": "Machine learning is an AI technique..."
            },
            # Add more test cases
        ]
        
        results = []
        for case in test_cases:
            result = await evaluator.aevaluate(**case)
            results.append({
                "query": case["query"],
                "score": result.score,
                "detailed_scores": result.score_dict
            })
            
        return results
    
    async def run_performance_evaluation(self):
        """Run performance benchmarks"""
        import time
        import psutil
        
        # Test data
        test_texts = [f"Sample text {i}" for i in range(1000)]
        
        # Measure embedding generation performance
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        embeddings = await self.client.embed_texts(
            texts=test_texts,
            model="thenlper/gte-small",
            batch_size=32
        )
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        return {
            "total_time": end_time - start_time,
            "throughput": len(test_texts) / (end_time - start_time),
            "memory_used": end_memory - start_memory,
            "embeddings_generated": len(embeddings)
        }

# Usage
async def main():
    # Initialize embeddings client
    resources = {
        "local_endpoints": [["thenlper/gte-small", "cpu", 512]]
    }
    client = ipfs_embeddings_py(resources, {})
    
    # Run evaluation
    pipeline = EvaluationPipeline(client)
    results = await pipeline.run_full_evaluation()
    
    print("Evaluation complete!")
    print(f"Results saved to {pipeline.output_dir}")

# Run evaluation
asyncio.run(main())
```

### Quick Evaluation Script

```python
#!/usr/bin/env python3
"""Quick evaluation script for LAION Embeddings"""

import argparse
import asyncio
from ipfs_embeddings_py import ipfs_embeddings_py

async def quick_evaluation():
    """Run quick evaluation on sample data"""
    
    # Simple configuration
    resources = {
        "local_endpoints": [["thenlper/gte-small", "cpu", 512]]
    }
    
    client = ipfs_embeddings_py(resources, {})
    
    # Sample evaluation
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?"
    ]
    
    test_documents = [
        "AI is the simulation of human intelligence in machines.",
        "Machine learning uses algorithms to analyze data.",
        "Neural networks are computing systems inspired by biological neural networks."
    ]
    
    print("Running quick evaluation...")
    
    # Generate embeddings
    query_embeddings = await client.embed_texts(test_queries)
    doc_embeddings = await client.embed_texts(test_documents)
    
    # Simple similarity evaluation
    similarities = []
    for i, query_emb in enumerate(query_embeddings):
        for j, doc_emb in enumerate(doc_embeddings):
            similarity = cosine_similarity(query_emb, doc_emb)
            similarities.append({
                "query": test_queries[i],
                "document": test_documents[j],
                "similarity": similarity
            })
    
    # Print top matches
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    print("\nTop matches:")
    for match in similarities[:3]:
        print(f"Query: {match['query']}")
        print(f"Document: {match['document']}")
        print(f"Similarity: {match['similarity']:.4f}")
        print("-" * 50)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    asyncio.run(quick_evaluation())
```

## Custom Evaluations

### Creating Custom Evaluators

```python
from ipfs_embeddings_py.llama_index.core.evaluation.base import BaseEvaluator
from ipfs_embeddings_py.llama_index.core.evaluation.base import EvaluationResult

class CustomSemanticEvaluator(BaseEvaluator):
    """Custom evaluator for semantic similarity"""
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
    
    async def aevaluate(
        self,
        query: str = None,
        response: str = None,
        contexts: list = None,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate semantic similarity between query and response"""
        
        # Generate embeddings
        query_embedding = await self.embed_text(query)
        response_embedding = await self.embed_text(response)
        
        # Calculate similarity
        similarity = cosine_similarity(query_embedding, response_embedding)
        
        # Score based on threshold
        score = 1.0 if similarity >= self.threshold else similarity
        
        # Determine if passing
        passing = similarity >= self.threshold
        
        return EvaluationResult(
            query=query,
            response=response,
            score=score,
            passing=passing,
            feedback=f"Semantic similarity: {similarity:.4f}"
        )
    
    async def embed_text(self, text):
        """Generate embedding for text"""
        # Implementation depends on your embedding client
        pass

# Usage
evaluator = CustomSemanticEvaluator(threshold=0.8)
result = await evaluator.aevaluate(
    query="What is AI?",
    response="Artificial intelligence is machine intelligence."
)
```

### Domain-Specific Evaluation

```python
class DomainSpecificEvaluator:
    """Evaluate embeddings for specific domains"""
    
    def __init__(self, domain_datasets):
        self.domain_datasets = domain_datasets
    
    async def evaluate_domain_performance(self, embeddings_client, domain):
        """Evaluate performance on domain-specific data"""
        
        if domain not in self.domain_datasets:
            raise ValueError(f"Domain {domain} not supported")
        
        dataset = self.domain_datasets[domain]
        
        # Load domain data
        queries = dataset["queries"]
        documents = dataset["documents"]
        ground_truth = dataset["ground_truth"]
        
        # Generate embeddings
        query_embeddings = await embeddings_client.embed_texts(queries)
        doc_embeddings = await embeddings_client.embed_texts(documents)
        
        # Calculate metrics
        results = self.calculate_domain_metrics(
            query_embeddings, 
            doc_embeddings, 
            ground_truth
        )
        
        return results
    
    def calculate_domain_metrics(self, query_embs, doc_embs, ground_truth):
        """Calculate domain-specific metrics"""
        # Implementation specific to domain requirements
        pass

# Usage for medical domain
medical_evaluator = DomainSpecificEvaluator({
    "medical": {
        "queries": medical_queries,
        "documents": medical_documents,
        "ground_truth": medical_ground_truth
    }
})

results = await medical_evaluator.evaluate_domain_performance(
    client, "medical"
)
```

## Performance Analysis

### Embedding Quality Analysis

```python
class EmbeddingQualityAnalyzer:
    """Analyze embedding quality and characteristics"""
    
    def __init__(self, embeddings_client):
        self.client = embeddings_client
    
    async def analyze_embedding_quality(self, texts, labels=None):
        """Comprehensive embedding quality analysis"""
        
        # Generate embeddings
        embeddings = await self.client.embed_texts(texts)
        
        analysis = {
            "dimensionality": len(embeddings[0]),
            "statistics": self.calculate_embedding_statistics(embeddings),
            "clustering": self.analyze_clustering(embeddings, labels),
            "diversity": self.calculate_diversity(embeddings),
            "stability": await self.test_stability(texts[:10])  # Sample for stability
        }
        
        return analysis
    
    def calculate_embedding_statistics(self, embeddings):
        """Calculate basic embedding statistics"""
        import numpy as np
        
        embeddings_array = np.array(embeddings)
        
        return {
            "mean_norm": np.mean(np.linalg.norm(embeddings_array, axis=1)),
            "std_norm": np.std(np.linalg.norm(embeddings_array, axis=1)),
            "mean_values": np.mean(embeddings_array, axis=0).tolist(),
            "std_values": np.std(embeddings_array, axis=0).tolist()
        }
    
    def analyze_clustering(self, embeddings, labels=None):
        """Analyze clustering properties"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np
        
        embeddings_array = np.array(embeddings)
        
        # Find optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(10, len(embeddings)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            score = silhouette_score(embeddings_array, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            "optimal_clusters": optimal_k,
            "silhouette_scores": silhouette_scores,
            "cluster_analysis": self.detailed_cluster_analysis(
                embeddings_array, optimal_k
            )
        }
    
    def calculate_diversity(self, embeddings):
        """Calculate embedding diversity metrics"""
        import numpy as np
        from itertools import combinations
        
        # Pairwise similarities
        similarities = []
        for i, j in combinations(range(len(embeddings)), 2):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
        
        return {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities)
        }
    
    async def test_stability(self, texts):
        """Test embedding stability across multiple generations"""
        
        # Generate embeddings multiple times
        embedding_sets = []
        for _ in range(3):
            embeddings = await self.client.embed_texts(texts)
            embedding_sets.append(embeddings)
        
        # Calculate stability metrics
        stabilities = []
        for i in range(len(texts)):
            text_embeddings = [emb_set[i] for emb_set in embedding_sets]
            stability = self.calculate_pairwise_stability(text_embeddings)
            stabilities.append(stability)
        
        return {
            "mean_stability": np.mean(stabilities),
            "std_stability": np.std(stabilities),
            "per_text_stability": stabilities
        }
    
    def calculate_pairwise_stability(self, embeddings):
        """Calculate stability between multiple embeddings of same text"""
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        return np.mean(similarities)

# Usage
analyzer = EmbeddingQualityAnalyzer(embeddings_client)
quality_report = await analyzer.analyze_embedding_quality(
    texts=sample_texts,
    labels=sample_labels
)
```

### Performance Benchmarking

```python
class PerformanceBenchmark:
    """Benchmark system performance across different configurations"""
    
    def __init__(self):
        self.results = {}
    
    async def benchmark_throughput(self, embeddings_client, test_sizes=[100, 500, 1000]):
        """Benchmark embedding generation throughput"""
        
        results = {}
        
        for size in test_sizes:
            test_texts = [f"Test text {i}" for i in range(size)]
            
            # Measure time
            start_time = time.time()
            embeddings = await embeddings_client.embed_texts(test_texts)
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = size / duration
            
            results[f"size_{size}"] = {
                "texts": size,
                "duration": duration,
                "throughput": throughput,
                "embeddings_generated": len(embeddings)
            }
        
        return results
    
    async def benchmark_batch_sizes(self, embeddings_client, batch_sizes=[1, 8, 16, 32, 64]):
        """Benchmark different batch sizes"""
        
        test_texts = [f"Test text {i}" for i in range(256)]
        results = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            embeddings = await embeddings_client.embed_texts(
                texts=test_texts,
                batch_size=batch_size
            )
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = len(test_texts) / duration
            
            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "duration": duration,
                "throughput": throughput
            }
        
        return results
    
    async def benchmark_models(self, models, test_texts):
        """Benchmark different models"""
        
        results = {}
        
        for model in models:
            # Configure client for specific model
            resources = {
                "local_endpoints": [[model, "cpu", 512]]
            }
            client = ipfs_embeddings_py(resources, {})
            
            # Benchmark model
            start_time = time.time()
            embeddings = await client.embed_texts(test_texts)
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = len(test_texts) / duration
            
            results[model] = {
                "model": model,
                "duration": duration,
                "throughput": throughput,
                "embedding_dim": len(embeddings[0]) if embeddings else 0
            }
        
        return results

# Usage
benchmark = PerformanceBenchmark()

# Benchmark throughput
throughput_results = await benchmark.benchmark_throughput(client)

# Benchmark batch sizes
batch_results = await benchmark.benchmark_batch_sizes(client)

# Benchmark models
model_results = await benchmark.benchmark_models(
    models=["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5"],
    test_texts=sample_texts
)
```

## Integration Testing

### End-to-End Integration Tests

```python
import pytest
import asyncio

class TestEmbeddingIntegration:
    """Integration tests for embedding pipeline"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete pipeline from text to retrieval"""
        
        # Initialize client
        resources = {
            "local_endpoints": [["thenlper/gte-small", "cpu", 512]]
        }
        client = ipfs_embeddings_py(resources, {})
        
        # Test data
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text understanding."
        ]
        
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What does NLP do?"
        ]
        
        # Step 1: Create embeddings
        doc_embeddings = await client.embed_texts(documents)
        query_embeddings = await client.embed_texts(queries)
        
        assert len(doc_embeddings) == len(documents)
        assert len(query_embeddings) == len(queries)
        
        # Step 2: Test retrieval
        for i, query_emb in enumerate(query_embeddings):
            similarities = []
            for j, doc_emb in enumerate(doc_embeddings):
                sim = cosine_similarity(query_emb, doc_emb)
                similarities.append((j, sim))
            
            # Check that most similar document makes sense
            best_match = max(similarities, key=lambda x: x[1])
            assert best_match[1] > 0.5  # Reasonable similarity threshold
        
        # Step 3: Test IPFS storage (if available)
        try:
            stored_cids = await client.store_embeddings(documents)
            assert len(stored_cids) == len(documents)
            
            # Test retrieval from storage
            retrieved = await client.retrieve_embeddings(stored_cids[0])
            assert retrieved is not None
            
        except Exception as e:
            pytest.skip(f"IPFS storage not available: {e}")
    
    @pytest.mark.asyncio
    async def test_evaluation_integration(self):
        """Test integration with evaluation frameworks"""
        
        # Initialize components
        resources = {
            "local_endpoints": [["thenlper/gte-small", "cpu", 512]]
        }
        client = ipfs_embeddings_py(resources, {})
        
        # Test BEIR integration
        evaluator = BeirEvaluator()
        
        def create_retriever(documents):
            return client.create_retriever(documents)
        
        # Run small evaluation
        try:
            evaluator.run(
                create_retriever=create_retriever,
                datasets=["nfcorpus"],
                metrics_k_values=[3, 10]
            )
        except Exception as e:
            # BEIR dataset might not be available in test environment
            pytest.skip(f"BEIR evaluation not available: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_integration(self):
        """Test performance meets minimum requirements"""
        
        resources = {
            "local_endpoints": [["thenlper/gte-small", "cpu", 512]]
        }
        client = ipfs_embeddings_py(resources, {})
        
        # Performance test
        test_texts = [f"Performance test text {i}" for i in range(100)]
        
        start_time = time.time()
        embeddings = await client.embed_texts(test_texts, batch_size=16)
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = len(test_texts) / duration
        
        # Assert minimum performance requirements
        assert len(embeddings) == len(test_texts)
        assert throughput > 10  # At least 10 texts per second
        assert duration < 60    # Complete within 1 minute

# Run tests
pytest.main([__file__, "-v"])
```

### System Health Evaluation

```python
class SystemHealthEvaluator:
    """Evaluate overall system health and reliability"""
    
    def __init__(self, embeddings_client):
        self.client = embeddings_client
    
    async def comprehensive_health_check(self):
        """Run comprehensive system health evaluation"""
        
        health_report = {
            "endpoint_health": await self.check_endpoint_health(),
            "embedding_quality": await self.check_embedding_quality(),
            "performance_health": await self.check_performance_health(),
            "stability_health": await self.check_stability_health(),
            "resource_health": self.check_resource_health()
        }
        
        # Calculate overall health score
        health_report["overall_score"] = self.calculate_health_score(health_report)
        
        return health_report
    
    async def check_endpoint_health(self):
        """Check health of all endpoints"""
        
        endpoints_status = {}
        
        # Test each endpoint type
        test_text = ["Health check test"]
        
        try:
            embeddings = await self.client.embed_texts(test_text)
            endpoints_status["embedding_generation"] = {
                "status": "healthy",
                "response_time": "fast",
                "output_quality": "good" if len(embeddings[0]) > 0 else "poor"
            }
        except Exception as e:
            endpoints_status["embedding_generation"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return endpoints_status
    
    async def check_embedding_quality(self):
        """Check embedding quality indicators"""
        
        # Test texts with known relationships
        similar_texts = [
            "The cat sat on the mat",
            "A cat was sitting on the mat"
        ]
        
        different_texts = [
            "The cat sat on the mat",
            "Quantum physics describes subatomic particles"
        ]
        
        # Generate embeddings
        similar_embs = await self.client.embed_texts(similar_texts)
        different_embs = await self.client.embed_texts(different_texts)
        
        # Check similarity relationships
        similar_sim = cosine_similarity(similar_embs[0], similar_embs[1])
        different_sim = cosine_similarity(different_embs[0], different_embs[1])
        
        quality_score = 1.0 if similar_sim > different_sim else 0.0
        
        return {
            "similar_similarity": similar_sim,
            "different_similarity": different_sim,
            "relationship_preserved": similar_sim > different_sim,
            "quality_score": quality_score
        }
    
    async def check_performance_health(self):
        """Check performance indicators"""
        
        test_texts = [f"Performance test {i}" for i in range(50)]
        
        start_time = time.time()
        embeddings = await self.client.embed_texts(test_texts)
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = len(test_texts) / duration
        
        # Performance thresholds
        performance_score = 1.0
        if throughput < 5:  # Less than 5 texts/second
            performance_score *= 0.5
        if duration > 30:  # Takes more than 30 seconds
            performance_score *= 0.5
        
        return {
            "duration": duration,
            "throughput": throughput,
            "performance_score": performance_score
        }
    
    async def check_stability_health(self):
        """Check system stability"""
        
        # Test repeated operations
        test_text = ["Stability test text"]
        embeddings = []
        
        for _ in range(3):
            emb = await self.client.embed_texts(test_text)
            embeddings.append(emb[0])
        
        # Check consistency
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        stability_score = np.mean(similarities)
        
        return {
            "consistency_scores": similarities,
            "stability_score": stability_score,
            "is_stable": stability_score > 0.99
        }
    
    def check_resource_health(self):
        """Check system resource usage"""
        import psutil
        
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        resource_score = 1.0
        if memory.percent > 80:  # High memory usage
            resource_score *= 0.7
        if cpu > 80:  # High CPU usage
            resource_score *= 0.7
        
        return {
            "memory_percent": memory.percent,
            "cpu_percent": cpu,
            "resource_score": resource_score
        }
    
    def calculate_health_score(self, health_report):
        """Calculate overall health score"""
        
        scores = []
        
        # Extract individual scores
        if "embedding_generation" in health_report["endpoint_health"]:
            endpoint_score = 1.0 if health_report["endpoint_health"]["embedding_generation"]["status"] == "healthy" else 0.0
            scores.append(endpoint_score)
        
        scores.append(health_report["embedding_quality"]["quality_score"])
        scores.append(health_report["performance_health"]["performance_score"])
        scores.append(health_report["stability_health"]["stability_score"])
        scores.append(health_report["resource_health"]["resource_score"])
        
        return np.mean(scores)

# Usage
health_evaluator = SystemHealthEvaluator(embeddings_client)
health_report = await health_evaluator.comprehensive_health_check()

print(f"Overall Health Score: {health_report['overall_score']:.2f}")
```

## Best Practices

### Evaluation Best Practices

1. **Use Multiple Metrics**
   - Combine different evaluation approaches
   - Don't rely on a single metric
   - Consider domain-specific requirements

2. **Establish Baselines**
   - Create baseline performance benchmarks
   - Compare against established models
   - Track performance over time

3. **Regular Evaluation**
   - Set up automated evaluation pipelines
   - Monitor performance degradation
   - Evaluate on representative data

4. **Cross-Validation**
   - Use multiple datasets for validation
   - Test on out-of-domain data
   - Validate across different text types

### Evaluation Pipeline Setup

```python
# evaluation_config.yaml
evaluation:
  schedule:
    - trigger: "daily"
      datasets: ["nfcorpus", "scifact"]
      metrics: ["ndcg@10", "map@10", "recall@10"]
    
    - trigger: "weekly"
      datasets: ["hotpotqa"]
      metrics: ["exact_match", "f1_score"]
    
    - trigger: "monthly"
      datasets: ["all_beir"]
      metrics: ["comprehensive"]
  
  thresholds:
    ndcg_10: 0.3
    map_10: 0.25
    exact_match: 0.4
    f1_score: 0.5
  
  alerts:
    - condition: "ndcg_10 < 0.25"
      action: "email_alert"
    - condition: "performance_degradation > 10%"
      action: "slack_notification"
```

### Continuous Evaluation

```python
class ContinuousEvaluator:
    """Continuous evaluation and monitoring system"""
    
    def __init__(self, config_path="evaluation_config.yaml"):
        self.config = self.load_config(config_path)
        self.baseline_results = self.load_baseline_results()
    
    async def run_scheduled_evaluation(self):
        """Run evaluation based on schedule"""
        
        for schedule in self.config["schedule"]:
            if self.should_run_evaluation(schedule):
                results = await self.run_evaluation(schedule)
                
                # Check thresholds
                self.check_thresholds(results)
                
                # Update baselines if needed
                self.update_baselines(results)
    
    def should_run_evaluation(self, schedule):
        """Check if evaluation should run based on schedule"""
        # Implementation based on trigger timing
        pass
    
    async def run_evaluation(self, schedule):
        """Run evaluation for specific schedule"""
        # Implementation based on schedule configuration
        pass
    
    def check_thresholds(self, results):
        """Check if results meet quality thresholds"""
        for metric, threshold in self.config["thresholds"].items():
            if metric in results and results[metric] < threshold:
                self.trigger_alert(metric, results[metric], threshold)
    
    def trigger_alert(self, metric, value, threshold):
        """Trigger alert for threshold violation"""
        alert_config = self.config["alerts"]
        # Implementation based on alert configuration
        pass

# Setup continuous evaluation
evaluator = ContinuousEvaluator()
await evaluator.run_scheduled_evaluation()
```

## Related Documentation

- [Models Documentation](../models/README.md) - Supported embedding models
- [Performance Benchmarking](../models/custom-models.md#performance-benchmarking) - Model performance analysis
- [Troubleshooting](../troubleshooting/README.md) - Debugging evaluation issues
- [Examples](../examples/README.md) - Practical evaluation examples
- [Development Guide](../development.md) - Testing and validation procedures

---

This evaluation framework provides comprehensive assessment capabilities for the LAION Embeddings project, enabling thorough quality assurance and performance monitoring across all system components.
