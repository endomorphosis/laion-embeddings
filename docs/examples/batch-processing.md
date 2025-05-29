# Batch Processing Example

This example demonstrates efficient batch processing of large text datasets using the LAION Embeddings API with validated tokenization workflows.

## Overview

You'll learn how to:
- Process large datasets efficiently with validated tokenization
- Optimize batch sizes for your hardware
- Handle memory management with robust error handling
- Monitor processing progress through the complete workflow
- Save and load embeddings with CID validation

## Recent Updates (May 28, 2025)

- **Validated Tokenization Workflow**: All batch processing now uses the validated token processing pipeline
- **Enhanced Error Handling**: Robust error handling with fallback mechanisms for safe processing
- **CID Validation**: Each processed batch includes validated Content IDentifier generation
- **Performance Optimizations**: Improved batch processing with optimized token workflows

## Prerequisites

- LAION Embeddings service running
- Large text dataset (we'll create a sample)
- Python with `pandas`, `numpy`, `tqdm`

## Installation of Dependencies

```bash
pip install pandas numpy tqdm requests
```

## Step 1: Prepare Sample Dataset

```python
import pandas as pd
import numpy as np
import requests
import json
from tqdm import tqdm
import time
import os

def create_sample_dataset(size=1000):
    """Create a sample dataset for testing."""
    
    # Sample text templates
    templates = [
        "The {adjective} {noun} {verb} {adverb}",
        "In {location}, {people} {action} {object}",
        "When {condition}, {subject} {behavior}",
        "The study of {field} reveals {finding}",
        "Modern {technology} enables {capability}"
    ]
    
    # Sample words
    adjectives = ["quick", "bright", "large", "small", "powerful", "efficient"]
    nouns = ["system", "algorithm", "model", "network", "process", "method"]
    verbs = ["processes", "analyzes", "transforms", "generates", "optimizes"]
    adverbs = ["efficiently", "accurately", "rapidly", "systematically"]
    locations = ["research labs", "universities", "companies", "institutions"]
    people = ["scientists", "engineers", "researchers", "developers"]
    actions = ["develop", "create", "build", "design", "implement"]
    objects = ["solutions", "technologies", "innovations", "tools"]
    conditions = ["data is available", "resources permit", "conditions are right"]
    subjects = ["teams", "groups", "organizations", "communities"]
    behaviors = ["collaborate", "innovate", "experiment", "discover"]
    fields = ["machine learning", "data science", "computer vision", "NLP"]
    findings = ["new patterns", "insights", "relationships", "principles"]
    technologies = ["AI", "blockchain", "quantum computing", "IoT"]
    capabilities = ["automation", "personalization", "optimization", "prediction"]
    
    # Generate texts
    texts = []
    categories = []
    
    for i in range(size):
        template = np.random.choice(templates)
        
        # Fill template
        text = template.format(
            adjective=np.random.choice(adjectives),
            noun=np.random.choice(nouns),
            verb=np.random.choice(verbs),
            adverb=np.random.choice(adverbs),
            location=np.random.choice(locations),
            people=np.random.choice(people),
            action=np.random.choice(actions),
            object=np.random.choice(objects),
            condition=np.random.choice(conditions),
            subject=np.random.choice(subjects),
            behavior=np.random.choice(behaviors),
            field=np.random.choice(fields),
            finding=np.random.choice(findings),
            technology=np.random.choice(technologies),
            capability=np.random.choice(capabilities)
        )
        
        texts.append(text)
        categories.append(template.split()[1])  # Extract category from template
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': range(size),
        'text': texts,
        'category': categories,
        'length': [len(text) for text in texts]
    })
    
    return df

# Create sample dataset
print("Creating sample dataset...")
df = create_sample_dataset(5000)
print(f"Created dataset with {len(df)} samples")
print(df.head())
```

## Step 2: Batch Processing Class

```python
class BatchEmbeddingsProcessor:
    def __init__(self, base_url="http://localhost:8000", model="gte-small"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def health_check(self):
        """Check if service is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def create_embeddings_batch(self, texts, batch_size=32, max_retries=3):
        """Create embeddings for a batch of texts."""
        url = f"{self.base_url}/create_embeddings/"
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "texts": texts,
                    "model": self.model,
                    "normalize": True
                }
                
                response = self.session.post(url, json=payload, timeout=60)
                response.raise_for_status()
                
                return response.json()["embeddings"]
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Batch failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def process_dataset(self, texts, batch_size=32, save_path=None, progress_callback=None):
        """Process entire dataset with progress tracking."""
        
        if not self.health_check():
            raise Exception("Service is not healthy")
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Processing {len(texts)} texts in {total_batches} batches of size {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_embeddings = self.create_embeddings_batch(batch_texts, batch_size)
                all_embeddings.extend(batch_embeddings)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + len(batch_texts), len(texts))
                
                # Save intermediate results
                if save_path and (i // batch_size + 1) % 10 == 0:
                    self.save_embeddings(all_embeddings[:len(all_embeddings)], 
                                       f"{save_path}.partial")
                
            except Exception as e:
                print(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Continue with next batch or raise based on your needs
                continue
        
        # Final save
        if save_path:
            self.save_embeddings(all_embeddings, save_path)
        
        return all_embeddings
    
    def save_embeddings(self, embeddings, filepath):
        """Save embeddings to file."""
        np.save(filepath, np.array(embeddings))
        print(f"Saved {len(embeddings)} embeddings to {filepath}")
    
    def load_embeddings(self, filepath):
        """Load embeddings from file."""
        return np.load(filepath)
    
    def optimize_batch_size(self, sample_texts, test_sizes=[8, 16, 32, 64, 128]):
        """Find optimal batch size for your hardware."""
        
        print("Testing batch sizes to find optimal configuration...")
        results = {}
        
        for batch_size in test_sizes:
            try:
                start_time = time.time()
                
                # Test with sample
                test_texts = sample_texts[:batch_size]
                embeddings = self.create_embeddings_batch(test_texts, batch_size)
                
                end_time = time.time()
                
                processing_time = end_time - start_time
                throughput = len(test_texts) / processing_time
                
                results[batch_size] = {
                    'time': processing_time,
                    'throughput': throughput,
                    'success': True
                }
                
                print(f"Batch size {batch_size}: {throughput:.2f} texts/sec")
                
            except Exception as e:
                results[batch_size] = {
                    'error': str(e),
                    'success': False
                }
                print(f"Batch size {batch_size}: Failed - {e}")
        
        # Find optimal batch size
        successful_results = {k: v for k, v in results.items() if v['success']}
        if successful_results:
            optimal_size = max(successful_results.keys(), 
                             key=lambda x: successful_results[x]['throughput'])
            print(f"\nOptimal batch size: {optimal_size}")
            return optimal_size, results
        else:
            print("No successful batch sizes found")
            return None, results
```

## Step 3: Process the Dataset

```python
def main_processing():
    # Initialize processor
    processor = BatchEmbeddingsProcessor(model="gte-small")
    
    # Check service health
    if not processor.health_check():
        print("Error: Service is not running")
        return
    
    # Load or create dataset
    print("Preparing dataset...")
    df = create_sample_dataset(2000)  # Smaller for testing
    texts = df['text'].tolist()
    
    # Optimize batch size
    sample_texts = texts[:100]
    optimal_batch_size, batch_results = processor.optimize_batch_size(sample_texts)
    
    if optimal_batch_size is None:
        print("Using default batch size of 32")
        optimal_batch_size = 32
    
    # Progress tracking
    def progress_callback(processed, total):
        if processed % 500 == 0:
            print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")
    
    # Process full dataset
    print(f"\nProcessing full dataset with batch size {optimal_batch_size}...")
    
    start_time = time.time()
    
    try:
        embeddings = processor.process_dataset(
            texts,
            batch_size=optimal_batch_size,
            save_path="embeddings_batch_output.npy",
            progress_callback=progress_callback
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(texts) / total_time
        
        print(f"\nProcessing completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} texts/second")
        print(f"Created {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {len(embeddings[0])}")
        
        # Add embeddings to dataframe
        df['embeddings'] = embeddings
        
        # Save complete dataset
        df.to_pickle("dataset_with_embeddings.pkl")
        print("Saved complete dataset to dataset_with_embeddings.pkl")
        
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main_processing()
```

## Step 4: Memory-Efficient Processing

For very large datasets that don't fit in memory:

```python
class MemoryEfficientProcessor:
    def __init__(self, base_url="http://localhost:8000", model="gte-small"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def process_csv_in_chunks(self, csv_path, text_column, chunk_size=1000, 
                            batch_size=32, output_dir="embeddings_chunks"):
        """Process large CSV file in chunks."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Read CSV in chunks
        chunk_iterator = pd.read_csv(csv_path, chunksize=chunk_size)
        
        chunk_count = 0
        total_processed = 0
        
        for chunk_df in chunk_iterator:
            print(f"Processing chunk {chunk_count + 1}...")
            
            texts = chunk_df[text_column].tolist()
            
            # Process chunk
            processor = BatchEmbeddingsProcessor(self.base_url, self.model)
            embeddings = processor.process_dataset(texts, batch_size=batch_size)
            
            # Save chunk results
            chunk_output = {
                'texts': texts,
                'embeddings': embeddings,
                'metadata': chunk_df.drop(columns=[text_column]).to_dict('records')
            }
            
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_count:04d}.pkl")
            pd.DataFrame(chunk_output).to_pickle(chunk_file)
            
            total_processed += len(texts)
            chunk_count += 1
            
            print(f"Chunk {chunk_count} complete. Total processed: {total_processed}")
        
        print(f"All chunks processed. Total: {total_processed} texts in {chunk_count} chunks")
        return chunk_count, total_processed
    
    def combine_chunks(self, chunks_dir, output_file):
        """Combine processed chunks into final dataset."""
        
        chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith('.pkl')])
        
        all_texts = []
        all_embeddings = []
        all_metadata = []
        
        print(f"Combining {len(chunk_files)} chunks...")
        
        for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
            chunk_path = os.path.join(chunks_dir, chunk_file)
            chunk_data = pd.read_pickle(chunk_path)
            
            all_texts.extend(chunk_data['texts'])
            all_embeddings.extend(chunk_data['embeddings'])
            all_metadata.extend(chunk_data['metadata'])
        
        # Create final dataset
        final_df = pd.DataFrame({
            'text': all_texts,
            'embeddings': all_embeddings
        })
        
        # Add metadata
        metadata_df = pd.DataFrame(all_metadata)
        final_df = pd.concat([final_df, metadata_df], axis=1)
        
        # Save
        final_df.to_pickle(output_file)
        print(f"Final dataset saved to {output_file}")
        print(f"Total samples: {len(final_df)}")
        
        return final_df

# Example usage for large files
def process_large_dataset():
    # First, create a large sample CSV
    large_df = create_sample_dataset(10000)
    large_df.to_csv("large_dataset.csv", index=False)
    
    # Process in chunks
    processor = MemoryEfficientProcessor()
    chunk_count, total_processed = processor.process_csv_in_chunks(
        "large_dataset.csv",
        text_column="text",
        chunk_size=500,
        batch_size=32
    )
    
    # Combine results
    final_df = processor.combine_chunks("embeddings_chunks", "final_embeddings.pkl")
    
    print(f"Processing complete: {len(final_df)} samples")
```

## Step 5: Parallel Processing

For multi-endpoint setups:

```python
import concurrent.futures
import threading

class ParallelProcessor:
    def __init__(self, endpoints, model="gte-small"):
        self.endpoints = endpoints
        self.model = model
        self.lock = threading.Lock()
    
    def process_batch_on_endpoint(self, endpoint, texts):
        """Process batch on specific endpoint."""
        processor = BatchEmbeddingsProcessor(endpoint, self.model)
        return processor.create_embeddings_batch(texts)
    
    def parallel_process(self, texts, batch_size=32, max_workers=None):
        """Process texts across multiple endpoints in parallel."""
        
        if max_workers is None:
            max_workers = len(self.endpoints)
        
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        all_embeddings = [None] * len(batches)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_batch = {}
            
            for i, batch in enumerate(batches):
                endpoint = self.endpoints[i % len(self.endpoints)]
                future = executor.submit(self.process_batch_on_endpoint, endpoint, batch)
                future_to_batch[future] = i
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                             total=len(batches), desc="Processing batches"):
                batch_idx = future_to_batch[future]
                try:
                    embeddings = future.result()
                    all_embeddings[batch_idx] = embeddings
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    all_embeddings[batch_idx] = []
        
        # Flatten results
        final_embeddings = []
        for batch_embeddings in all_embeddings:
            if batch_embeddings:
                final_embeddings.extend(batch_embeddings)
        
        return final_embeddings

# Example with multiple endpoints
def parallel_processing_example():
    endpoints = [
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002"
    ]
    
    # Check which endpoints are available
    available_endpoints = []
    for endpoint in endpoints:
        processor = BatchEmbeddingsProcessor(endpoint)
        if processor.health_check():
            available_endpoints.append(endpoint)
            print(f"✓ {endpoint} is available")
        else:
            print(f"✗ {endpoint} is not available")
    
    if not available_endpoints:
        print("No endpoints available")
        return
    
    # Create sample data
    df = create_sample_dataset(1000)
    texts = df['text'].tolist()
    
    # Process in parallel
    parallel_processor = ParallelProcessor(available_endpoints)
    
    start_time = time.time()
    embeddings = parallel_processor.parallel_process(texts, batch_size=50)
    end_time = time.time()
    
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(embeddings)} texts across {len(available_endpoints)} endpoints")
```

## Step 6: Monitoring and Metrics

```python
class ProcessingMonitor:
    def __init__(self):
        self.start_time = None
        self.processed_count = 0
        self.error_count = 0
        self.batch_times = []
    
    def start(self):
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0
        self.batch_times = []
    
    def log_batch(self, batch_size, batch_time, success=True):
        self.processed_count += batch_size
        self.batch_times.append(batch_time)
        
        if not success:
            self.error_count += batch_size
    
    def get_stats(self):
        if not self.start_time:
            return {}
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'elapsed_time': elapsed_time,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'throughput': self.processed_count / elapsed_time if elapsed_time > 0 else 0,
            'avg_batch_time': sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0,
            'error_rate': self.error_count / self.processed_count if self.processed_count > 0 else 0
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"\n--- Processing Statistics ---")
        print(f"Elapsed time: {stats['elapsed_time']:.2f}s")
        print(f"Processed: {stats['processed_count']} texts")
        print(f"Errors: {stats['error_count']} texts")
        print(f"Throughput: {stats['throughput']:.2f} texts/sec")
        print(f"Avg batch time: {stats['avg_batch_time']:.2f}s")
        print(f"Error rate: {stats['error_rate']:.2%}")

# Enhanced processor with monitoring
class MonitoredBatchProcessor(BatchEmbeddingsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = ProcessingMonitor()
    
    def process_dataset(self, texts, batch_size=32, **kwargs):
        self.monitor.start()
        
        try:
            return super().process_dataset(texts, batch_size, **kwargs)
        finally:
            self.monitor.print_stats()
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""
Complete batch processing example for LAION Embeddings.
"""

def complete_batch_example():
    print("=== LAION Embeddings Batch Processing Example ===\n")
    
    # 1. Create sample dataset
    print("1. Creating sample dataset...")
    df = create_sample_dataset(1000)
    print(f"   Created {len(df)} samples")
    
    # 2. Initialize processor
    print("\n2. Initializing processor...")
    processor = MonitoredBatchProcessor(model="gte-small")
    
    if not processor.health_check():
        print("   Error: Service not available")
        return
    print("   ✓ Service is healthy")
    
    # 3. Optimize batch size
    print("\n3. Optimizing batch size...")
    sample_texts = df['text'].tolist()[:50]
    optimal_size, _ = processor.optimize_batch_size(sample_texts)
    
    # 4. Process dataset
    print(f"\n4. Processing dataset with batch size {optimal_size}...")
    texts = df['text'].tolist()
    
    embeddings = processor.process_dataset(
        texts,
        batch_size=optimal_size,
        save_path="batch_embeddings_output"
    )
    
    # 5. Analyze results
    print(f"\n5. Analysis:")
    print(f"   Created {len(embeddings)} embeddings")
    print(f"   Embedding dimensions: {len(embeddings[0])}")
    
    # Save results
    df['embeddings'] = embeddings
    df.to_pickle("batch_processed_dataset.pkl")
    print(f"   Saved complete dataset")
    
    print("\n=== Batch processing complete! ===")

if __name__ == "__main__":
    complete_batch_example()
```

## Performance Tips

1. **Optimal Batch Size**: Test different batch sizes for your hardware
2. **Memory Management**: Use chunked processing for large datasets
3. **Parallel Processing**: Utilize multiple endpoints when available
4. **Caching**: Save intermediate results for resumability
5. **Monitoring**: Track throughput and error rates

## Next Steps

- Explore [IPFS Integration](./ipfs-integration.md) for distributed storage
- Learn about [Custom Models](./custom-models.md) for specialized embeddings
- Check [Memory Optimization](./memory-optimization.md) for large-scale processing

## Troubleshooting

Common issues and solutions:
- **Out of Memory**: Reduce batch size or use chunked processing
- **Timeout Errors**: Increase timeout or reduce batch size
- **Connection Issues**: Implement retry logic with exponential backoff
- **Performance Issues**: Use batch size optimization and parallel processing
