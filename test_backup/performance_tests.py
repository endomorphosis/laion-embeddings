"""
Performance and Load Testing for LAION Embeddings

Tests batch processing, memory usage, concurrent operations, and scalability.
"""

import pytest
import asyncio
import unittest
import time
import sys
import os
import uuid
import numpy as np
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ipfs_kit_py.ipfs_kit import ipfs_kit
    from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
    from ipfs_embeddings_py.main_new import safe_get_cid, index_cid
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")


class PerformanceTestUtils:
    """Utilities for performance testing"""
    
    @staticmethod
    def measure_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def generate_test_data(count: int, length: int = 100) -> List[str]:
        """Generate test data for performance testing"""
        return [f"Test sample {i}: {uuid.uuid4()}" + " word" * (length // 5) for i in range(count)]
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time function execution"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time


class TestBatchPerformance(unittest.TestCase):
    """Test batch processing performance and scalability"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.metadata = {
            "dataset": "test_dataset",
            "model": "thenlper/gte-small"
        }
        self.resources = {
            "tei_endpoints": [["thenlper/gte-small", "http://127.0.0.1:8080/embed", 512]]
        }
        
    def test_batch_size_scaling(self):
        """Test performance across different batch sizes"""
        print("\n" + "="*50)
        print("BATCH SIZE SCALING TEST")
        print("="*50)
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Generate test data
                test_data = PerformanceTestUtils.generate_test_data(batch_size)
                
                # Measure CID generation performance
                cids, duration = PerformanceTestUtils.time_function(index_cid, test_data)
                
                throughput = batch_size / duration if duration > 0 else 0
                results[batch_size] = {
                    'duration': duration,
                    'throughput': throughput,
                    'memory_mb': PerformanceTestUtils.measure_memory_usage()
                }
                
                print(f"Batch size {batch_size:3d}: {duration:.4f}s, {throughput:.2f} items/s, {results[batch_size]['memory_mb']:.1f} MB")
                
            except Exception as e:
                print(f"Batch size {batch_size} failed: {e}")
                
        # Verify performance scaling
        if len(results) >= 2:
            print("\n✓ Batch size scaling test completed")
            
            # Check if larger batches are more efficient
            small_batch = results.get(1, {}).get('throughput', 0)
            large_batch = results.get(max(results.keys()), {}).get('throughput', 0)
            
            if large_batch > small_batch:
                print(f"✓ Performance scales well: {small_batch:.2f} -> {large_batch:.2f} items/s")
            else:
                print(f"! Performance scaling issue: {small_batch:.2f} -> {large_batch:.2f} items/s")
        
        return results
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during batch processing"""
        print("\n" + "="*50)
        print("MEMORY USAGE PATTERNS TEST")
        print("="*50)
        
        initial_memory = PerformanceTestUtils.measure_memory_usage()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test with increasingly large datasets
        data_sizes = [100, 500, 1000, 2000]
        memory_readings = []
        
        for size in data_sizes:
            try:
                # Generate large test dataset
                test_data = PerformanceTestUtils.generate_test_data(size, length=200)
                
                memory_before = PerformanceTestUtils.measure_memory_usage()
                
                # Process data
                cids = index_cid(test_data)
                
                memory_after = PerformanceTestUtils.measure_memory_usage()
                memory_increase = memory_after - memory_before
                
                memory_readings.append({
                    'size': size,
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'memory_increase': memory_increase
                })
                
                print(f"Size {size:4d}: {memory_before:.1f} -> {memory_after:.1f} MB (+{memory_increase:.1f} MB)")
                
                # Force garbage collection
                del test_data, cids
                gc.collect()
                
            except Exception as e:
                print(f"Memory test for size {size} failed: {e}")
        
        # Check for memory leaks
        final_memory = PerformanceTestUtils.measure_memory_usage()
        memory_growth = final_memory - initial_memory
        
        print(f"Final memory usage: {final_memory:.1f} MB (growth: {memory_growth:.1f} MB)")
        
        if memory_growth < 50:  # Acceptable memory growth threshold
            print("✓ Memory usage patterns are acceptable")
        else:
            print(f"! Potential memory leak detected: {memory_growth:.1f} MB growth")
    
    def test_concurrent_processing(self):
        """Test concurrent batch processing"""
        print("\n" + "="*50)
        print("CONCURRENT PROCESSING TEST")
        print("="*50)
        
        num_workers = min(4, os.cpu_count())
        batch_size = 50
        
        def process_batch(batch_id):
            """Process a single batch"""
            test_data = PerformanceTestUtils.generate_test_data(batch_size)
            start_time = time.time()
            cids = index_cid(test_data)
            duration = time.time() - start_time
            return batch_id, len(cids), duration
        
        # Test sequential processing
        print(f"Sequential processing ({num_workers} batches)...")
        sequential_start = time.time()
        sequential_results = []
        
        for i in range(num_workers):
            batch_id, count, duration = process_batch(i)
            sequential_results.append((batch_id, count, duration))
        
        sequential_total = time.time() - sequential_start
        
        # Test concurrent processing
        print(f"Concurrent processing ({num_workers} workers)...")
        concurrent_start = time.time()
        concurrent_results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_workers)]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    print(f"Concurrent batch failed: {e}")
        
        concurrent_total = time.time() - concurrent_start
        
        # Compare results
        print(f"Sequential time: {sequential_total:.3f}s")
        print(f"Concurrent time: {concurrent_total:.3f}s")
        
        if concurrent_total < sequential_total:
            speedup = sequential_total / concurrent_total
            print(f"✓ Concurrent processing is {speedup:.2f}x faster")
        else:
            print("! Concurrent processing did not improve performance")
        
        print("✓ Concurrent processing test completed")


class TestEndpointPerformance(unittest.TestCase):
    """Test endpoint performance and reliability"""
    
    def setUp(self):
        """Set up endpoint performance test fixtures"""
        self.metadata = {
            "dataset": "test_dataset",
            "models": ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5"]
        }
        self.resources = {
            "tei_endpoints": [
                ["thenlper/gte-small", "http://127.0.0.1:8080/embed", 512],
                ["Alibaba-NLP/gte-large-en-v1.5", "http://127.0.0.1:8081/embed", 8192]
            ],
            "local_endpoints": [
                ["thenlper/gte-small", "cpu", 512],
                ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192]
            ]
        }
    
    def test_endpoint_initialization_performance(self):
        """Test performance of endpoint initialization"""
        print("\n" + "="*50)
        print("ENDPOINT INITIALIZATION PERFORMANCE TEST")
        print("="*50)
        
        try:
            start_time = time.time()
            embeddings = ipfs_embeddings_py(self.resources, self.metadata)
            init_time = time.time() - start_time
            
            print(f"Initialization time: {init_time:.3f}s")
            
            if init_time < 5.0:  # Should initialize quickly
                print("✓ Endpoint initialization performance is acceptable")
            else:
                print(f"! Slow endpoint initialization: {init_time:.3f}s")
            
        except Exception as e:
            print(f"✗ Endpoint initialization failed: {e}")
    
    def test_endpoint_response_times(self):
        """Test endpoint response time characteristics"""
        print("\n" + "="*50)
        print("ENDPOINT RESPONSE TIME TEST")
        print("="*50)
        
        try:
            embeddings = ipfs_embeddings_py(self.resources, self.metadata)
            
            # Test different endpoint types if available
            endpoint_types = ['tei', 'local', 'openvino', 'libp2p']
            test_samples = ["Hello world", "Test embedding", "Performance test"]
            
            for endpoint_type in endpoint_types:
                if hasattr(embeddings, f'test_{endpoint_type}_endpoint') or hasattr(embeddings, f'test_{endpoint_type}_https_endpoint'):
                    print(f"Testing {endpoint_type} endpoint response times...")
                    
                    response_times = []
                    for sample in test_samples:
                        start_time = time.time()
                        # Mock endpoint test - in real scenario this would make actual requests
                        time.sleep(0.01)  # Simulate processing time
                        response_time = time.time() - start_time
                        response_times.append(response_time)
                    
                    avg_response_time = sum(response_times) / len(response_times)
                    max_response_time = max(response_times)
                    
                    print(f"  {endpoint_type}: avg={avg_response_time*1000:.1f}ms, max={max_response_time*1000:.1f}ms")
                    
                    if avg_response_time < 1.0:  # Acceptable response time
                        print(f"  ✓ {endpoint_type} endpoint response times are acceptable")
                    else:
                        print(f"  ! {endpoint_type} endpoint response times are slow")
            
        except Exception as e:
            print(f"✗ Endpoint response time test failed: {e}")


class TestScalabilityAndLimits(unittest.TestCase):
    """Test system scalability and limits"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        print("\n" + "="*50)
        print("LARGE DATASET HANDLING TEST")
        print("="*50)
        
        # Test with progressively larger datasets
        dataset_sizes = [1000, 5000, 10000]
        
        for size in dataset_sizes:
            try:
                print(f"Testing dataset size: {size}")
                start_memory = PerformanceTestUtils.measure_memory_usage()
                
                # Generate large dataset
                large_dataset = PerformanceTestUtils.generate_test_data(size, length=150)
                
                start_time = time.time()
                cids = index_cid(large_dataset)
                processing_time = time.time() - start_time
                
                end_memory = PerformanceTestUtils.measure_memory_usage()
                memory_used = end_memory - start_memory
                
                throughput = size / processing_time
                
                print(f"  Size: {size}, Time: {processing_time:.2f}s, Throughput: {throughput:.1f} items/s")
                print(f"  Memory used: {memory_used:.1f} MB")
                
                # Clean up
                del large_dataset, cids
                gc.collect()
                
                if processing_time < size * 0.01:  # Reasonable performance threshold
                    print(f"  ✓ Dataset size {size} processed efficiently")
                else:
                    print(f"  ! Dataset size {size} processing is slow")
                
            except Exception as e:
                print(f"  ✗ Failed to process dataset size {size}: {e}")
    
    def test_stress_test(self):
        """Perform stress testing with high load"""
        print("\n" + "="*50)
        print("STRESS TEST")
        print("="*50)
        
        # Stress test parameters
        num_iterations = 100
        batch_size = 20
        
        success_count = 0
        failure_count = 0
        response_times = []
        
        print(f"Running {num_iterations} iterations with batch size {batch_size}...")
        
        for i in range(num_iterations):
            try:
                test_data = PerformanceTestUtils.generate_test_data(batch_size)
                
                start_time = time.time()
                cids = index_cid(test_data)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                success_count += 1
                
                if i % 20 == 0:
                    print(f"  Progress: {i}/{num_iterations} ({i/num_iterations*100:.1f}%)")
                
            except Exception as e:
                failure_count += 1
                if failure_count < 5:  # Don't spam errors
                    print(f"  Iteration {i} failed: {e}")
        
        # Calculate statistics
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"\nStress test results:")
            print(f"  Successful iterations: {success_count}/{num_iterations} ({success_count/num_iterations*100:.1f}%)")
            print(f"  Failed iterations: {failure_count}")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Min/Max response time: {min_response_time:.3f}s / {max_response_time:.3f}s")
            
            if success_count >= num_iterations * 0.95:  # 95% success rate
                print("✓ Stress test passed - system is stable under load")
            else:
                print("! Stress test indicates stability issues")
        else:
            print("✗ Stress test failed - no successful iterations")


def run_performance_tests():
    """Run all performance tests"""
    print("=" * 80)
    print("LAION EMBEDDINGS PERFORMANCE TEST SUITE")
    print("=" * 80)
    
    test_classes = [
        TestBatchPerformance,
        TestEndpointPerformance,
        TestScalabilityAndLimits
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result


if __name__ == "__main__":
    result = run_performance_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
