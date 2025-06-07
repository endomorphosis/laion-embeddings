#!/usr/bin/env python3
"""
Vector Store Benchmark Test Script

This script performs comprehensive benchmarking and stress testing for all vector
store providers, measuring performance, scalability, and resource usage.
"""

import asyncio
import time
import logging
import argparse
import sys
import psutil
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from services.vector_store_factory import get_vector_store_factory, VectorDBType, reset_factory
from services.vector_store_base import BaseVectorStore, VectorDocument, SearchQuery, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("benchmark_tests")

# Dependency checks
FAISS_AVAILABLE = True
try:
    import faiss
except ImportError:
    FAISS_AVAILABLE = False

IPFS_AVAILABLE = True
try:
    import ipfshttpclient
except ImportError:
    IPFS_AVAILABLE = False

DUCKDB_FULL_AVAILABLE = True
try:
    import duckdb
    from duckdb_engine import DuckDBVector
except ImportError:
    DUCKDB_FULL_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Stores benchmark test results."""
    operation: str
    provider: str
    data_size: int
    duration: float
    throughput: float
    memory_used: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class StressTestResult:
    """Stores stress test results."""
    provider: str
    concurrent_operations: int
    total_operations: int
    success_rate: float
    avg_latency: float
    max_latency: float
    min_latency: float
    memory_peak: float
    cpu_peak: float


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.start_cpu = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
        
    def get_current_usage(self) -> Tuple[float, float]:
        """Get current memory and CPU usage."""
        memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu = self.process.cpu_percent()
        return memory, cpu
        
    def get_memory_delta(self) -> float:
        """Get memory usage delta since start."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        return current_memory - self.start_memory


async def create_test_vectors(count: int, dimension: int = 128) -> List[VectorDocument]:
    """Create test vectors for benchmarking."""
    import random
    
    vectors = []
    for i in range(count):
        vector = [random.random() for _ in range(dimension)]
        vectors.append(VectorDocument(
            id=f"bench-{i}",
            vector=vector,
            text=f"Benchmark document {i}",
            metadata={"batch": i // 100, "index": i, "category": f"cat-{i % 10}"}
        ))
    
    return vectors


async def benchmark_insertion(store: BaseVectorStore, vectors: List[VectorDocument], 
                            index_name: str) -> BenchmarkResult:
    """Benchmark vector insertion performance."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        # Create index
        await store.create_index(index_name, len(vectors[0].vector))
        
        # Insert vectors
        result = await store.add_vectors(vectors, index_name)
        success = result
        
        end_time = time.time()
        duration = end_time - start_time
        
        memory_used = monitor.get_memory_delta()
        _, cpu_percent = monitor.get_current_usage()
        
        throughput = len(vectors) / duration if duration > 0 else 0
        
        logger.info(f"Insertion benchmark: {len(vectors)} vectors in {duration:.2f}s "
                   f"({throughput:.1f} vectors/sec)")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_msg = str(e)
        memory_used = monitor.get_memory_delta()
        _, cpu_percent = monitor.get_current_usage()
        throughput = 0
        logger.error(f"Insertion benchmark failed: {e}")
    
    return BenchmarkResult(
        operation="insertion",
        provider=store.__class__.__name__,
        data_size=len(vectors),
        duration=duration,
        throughput=throughput,
        memory_used=memory_used,
        cpu_percent=cpu_percent,
        success=success,
        error_message=error_msg
    )


async def benchmark_search(store: BaseVectorStore, query_vectors: List[List[float]], 
                         index_name: str, limit: int = 10) -> BenchmarkResult:
    """Benchmark search performance."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    start_time = time.time()
    success = False
    error_msg = None
    total_results = 0
    
    try:
        # Perform searches
        for query_vector in query_vectors:
            query = SearchQuery(vector=query_vector, limit=limit)
            results = await store.search(query, index_name)
            if results:
                total_results += len(results)
        
        success = True
        end_time = time.time()
        duration = end_time - start_time
        
        memory_used = monitor.get_memory_delta()
        _, cpu_percent = monitor.get_current_usage()
        
        throughput = len(query_vectors) / duration if duration > 0 else 0
        
        logger.info(f"Search benchmark: {len(query_vectors)} queries in {duration:.2f}s "
                   f"({throughput:.1f} queries/sec), {total_results} total results")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_msg = str(e)
        memory_used = monitor.get_memory_delta()
        _, cpu_percent = monitor.get_current_usage()
        throughput = 0
        logger.error(f"Search benchmark failed: {e}")
    
    return BenchmarkResult(
        operation="search",
        provider=store.__class__.__name__,
        data_size=len(query_vectors),
        duration=duration,
        throughput=throughput,
        memory_used=memory_used,
        cpu_percent=cpu_percent,
        success=success,
        error_message=error_msg
    )


async def stress_test_concurrent_operations(store: BaseVectorStore, index_name: str,
                                          concurrent_ops: int = 10, 
                                          ops_per_worker: int = 10) -> StressTestResult:
    """Perform concurrent operations stress test."""
    import random
    
    dimension = 64
    latencies = []
    successes = 0
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    peak_memory = 0
    peak_cpu = 0
    
    async def worker(worker_id: int):
        nonlocal successes, peak_memory, peak_cpu
        
        for i in range(ops_per_worker):
            start = time.time()
            try:
                # Mix of operations
                if i % 3 == 0:
                    # Search operation
                    query_vector = [random.random() for _ in range(dimension)]
                    query = SearchQuery(vector=query_vector, limit=5)
                    await store.search(query, index_name)
                elif i % 3 == 1:
                    # Add single vector
                    vector = VectorDocument(
                        id=f"stress-{worker_id}-{i}",
                        vector=[random.random() for _ in range(dimension)],
                        metadata={"worker": worker_id, "op": i}
                    )
                    await store.add_vectors([vector], index_name)
                else:
                    # Get stats
                    await store.get_index_stats(index_name)
                
                successes += 1
                
                # Monitor resources
                memory, cpu = monitor.get_current_usage()
                peak_memory = max(peak_memory, memory)
                peak_cpu = max(peak_cpu, cpu)
                
            except Exception as e:
                logger.debug(f"Worker {worker_id} operation {i} failed: {e}")
            
            latency = time.time() - start
            latencies.append(latency)
    
    # Create some initial data
    try:
        await store.create_index(index_name, dimension)
        initial_vectors = await create_test_vectors(50, dimension)
        await store.add_vectors(initial_vectors, index_name)
    except Exception as e:
        logger.warning(f"Failed to create initial data: {e}")
    
    # Run concurrent workers
    start_time = time.time()
    tasks = [worker(i) for i in range(concurrent_ops)]
    await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    total_operations = concurrent_ops * ops_per_worker
    success_rate = successes / total_operations if total_operations > 0 else 0
    
    avg_latency = statistics.mean(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    
    logger.info(f"Stress test: {concurrent_ops} workers × {ops_per_worker} ops = "
               f"{total_operations} total ops, {success_rate:.1%} success rate")
    
    return StressTestResult(
        provider=store.__class__.__name__,
        concurrent_operations=concurrent_ops,
        total_operations=total_operations,
        success_rate=success_rate,
        avg_latency=avg_latency,
        max_latency=max_latency,
        min_latency=min_latency,
        memory_peak=peak_memory,
        cpu_peak=peak_cpu
    )


async def run_scalability_test(store: BaseVectorStore, sizes: List[int]) -> List[BenchmarkResult]:
    """Test scalability with increasing dataset sizes."""
    results = []
    dimension = 128
    
    for size in sizes:
        logger.info(f"Running scalability test with {size} vectors...")
        
        index_name = f"scale_test_{size}"
        
        try:
            # Create test data
            vectors = await create_test_vectors(size, dimension)
            
            # Benchmark insertion
            insert_result = await benchmark_insertion(store, vectors, index_name)
            results.append(insert_result)
            
            if insert_result.success:
                # Benchmark search with subset of data as queries
                query_size = min(size // 10, 100)
                query_vectors = [v.vector for v in vectors[:query_size]]
                search_result = await benchmark_search(store, query_vectors, index_name)
                results.append(search_result)
            
            # Cleanup
            try:
                await store.delete_index(index_name)
            except Exception as e:
                logger.warning(f"Failed to cleanup index {index_name}: {e}")
                
        except Exception as e:
            logger.error(f"Scalability test with {size} vectors failed: {e}")
    
    return results


async def benchmark_store(db_type: VectorDBType) -> Dict[str, List]:
    """Run comprehensive benchmarks on a single store."""
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARKING {db_type.value.upper()}")
    logger.info(f"{'='*60}")
    
    results = {
        "benchmarks": [],
        "stress_tests": [],
        "scalability": []
    }
    
    try:
        factory = get_vector_store_factory()
        store = await factory.create_store(db_type)
        
        # Connect
        await store.connect()
        ping_result = await store.ping()
        if not ping_result:
            logger.error(f"{db_type.value} connection failed")
            return results
        
        logger.info(f"Connected to {db_type.value} store")
        
        # Basic benchmarks
        logger.info("Running basic benchmarks...")
        test_vectors = await create_test_vectors(1000, 128)
        
        insert_bench = await benchmark_insertion(store, test_vectors, "benchmark_basic")
        results["benchmarks"].append(insert_bench)
        
        if insert_bench.success:
            query_vectors = [v.vector for v in test_vectors[:100]]
            search_bench = await benchmark_search(store, query_vectors, "benchmark_basic", 10)
            results["benchmarks"].append(search_bench)
        
        # Stress test
        logger.info("Running stress test...")
        stress_result = await stress_test_concurrent_operations(store, "stress_test", 5, 20)
        results["stress_tests"].append(stress_result)
        
        # Scalability test
        logger.info("Running scalability test...")
        scale_sizes = [100, 500, 1000, 2000] if db_type == VectorDBType.FAISS else [100, 500]
        scale_results = await run_scalability_test(store, scale_sizes)
        results["scalability"].extend(scale_results)
        
        # Cleanup
        try:
            await store.delete_index("benchmark_basic")
            await store.delete_index("stress_test")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        await store.disconnect()
        
    except Exception as e:
        logger.error(f"Benchmark failed for {db_type.value}: {e}")
    
    return results


def print_benchmark_summary(all_results: Dict[str, Dict]):
    """Print a comprehensive summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for provider, results in all_results.items():
        print(f"\n{provider.upper()}")
        print("-" * 40)
        
        # Basic benchmarks
        benchmarks = results.get("benchmarks", [])
        if benchmarks:
            print("Basic Performance:")
            for bench in benchmarks:
                status = "✓" if bench.success else "✗"
                print(f"  {status} {bench.operation}: {bench.throughput:.1f} ops/sec "
                      f"({bench.duration:.2f}s, {bench.memory_used:.1f}MB)")
        
        # Stress tests
        stress_tests = results.get("stress_tests", [])
        if stress_tests:
            print("Stress Test:")
            for stress in stress_tests:
                print(f"  • {stress.concurrent_operations} workers: {stress.success_rate:.1%} success "
                      f"(avg: {stress.avg_latency*1000:.1f}ms, peak memory: {stress.memory_peak:.1f}MB)")
        
        # Scalability
        scalability = results.get("scalability", [])
        if scalability:
            print("Scalability:")
            insert_results = [s for s in scalability if s.operation == "insertion" and s.success]
            if insert_results:
                sizes_throughput = [(s.data_size, s.throughput) for s in insert_results]
                print(f"  Insert throughput: {sizes_throughput}")


async def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Vector store benchmark tests")
    parser.add_argument("--store", "-s", 
                        help="Specific store to benchmark (faiss, ipfs, duckdb)",
                        default=None)
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run quick benchmarks only")
    args = parser.parse_args()
    
    # Reset factory
    reset_factory()
    
    # Determine which stores to test
    if args.store:
        try:
            db_type = VectorDBType(args.store)
            stores_to_test = [db_type]
        except ValueError:
            logger.error(f"Unknown store type: {args.store}")
            sys.exit(1)
    else:
        # Test available stores
        stores_to_test = []
        if FAISS_AVAILABLE:
            stores_to_test.append(VectorDBType.FAISS)
        if IPFS_AVAILABLE:
            stores_to_test.append(VectorDBType.IPFS)
        if DUCKDB_FULL_AVAILABLE:
            stores_to_test.append(VectorDBType.DUCKDB)
    
    if not stores_to_test:
        logger.error("No available stores to benchmark")
        sys.exit(1)
    
    logger.info(f"Benchmarking {len(stores_to_test)} stores: {[s.value for s in stores_to_test]}")
    
    # Run benchmarks
    all_results = {}
    start_time = time.time()
    
    for db_type in stores_to_test:
        results = await benchmark_store(db_type)
        all_results[db_type.value] = results
    
    total_time = time.time() - start_time
    
    # Print summary
    print_benchmark_summary(all_results)
    print(f"\nTotal benchmark time: {total_time:.1f} seconds")
    
    # Check if any benchmarks succeeded
    any_success = any(
        any(b.success for b in results.get("benchmarks", []))
        for results in all_results.values()
    )
    
    sys.exit(0 if any_success else 1)


if __name__ == "__main__":
    asyncio.run(main())
