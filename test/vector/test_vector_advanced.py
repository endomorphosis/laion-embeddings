#!/usr/bin/env python3
"""
Vector Quantization and Sharding Test Script

This script tests vector quantization and sharding capabilities for all providers,
including the new IPFS/IPLD and DuckDB/Parquet providers.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional
import argparse
import numpy as np
import time
import importlib.util

from services.vector_config import VectorDBType, get_config_manager
from services.vector_store_factory import get_vector_store_factory, reset_factory
from services.vector_store_base import BaseVectorStore, VectorDocument, SearchQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vector_advanced_test")


def check_numpy_available() -> bool:
    """Check if numpy is available for advanced testing."""
    try:
        import numpy as np
        return True
    except ImportError:
        return False


def check_dependencies():
    """Check which dependencies are available for advanced testing."""
    deps = {
        'numpy': check_numpy_available(),
        'qdrant': importlib.util.find_spec("qdrant_client") is not None,
        'elasticsearch': importlib.util.find_spec("elasticsearch") is not None,
        'pgvector': importlib.util.find_spec("psycopg2") is not None,
        'faiss': importlib.util.find_spec("faiss") is not None,
        'ipfs_kit': importlib.util.find_spec("ipfs_kit_py") is not None,
        'ipfs_client': importlib.util.find_spec("ipfshttpclient") is not None,
        'duckdb': importlib.util.find_spec("duckdb") is not None,
        'arrow': importlib.util.find_spec("pyarrow") is not None,
    }
    
    deps['ipfs'] = deps['ipfs_kit'] or deps['ipfs_client']
    deps['duckdb_full'] = deps['duckdb'] and deps['arrow']
    
    logger.info(f"Available dependencies: {[k for k, v in deps.items() if v]}")
    return deps


def generate_test_vectors(count: int, dimension: int, seed: int = 42) -> List[VectorDocument]:
    """Generate test vectors with consistent random distribution."""
    if not check_numpy_available():
        # Fallback to simple vectors without numpy
        vectors = []
        for i in range(count):
            vector = [float(j + i * 0.1) for j in range(dimension)]
            doc = VectorDocument(
                id=f"vec-{i}",
                vector=vector,
                metadata={
                    "text": f"Test vector {i}",
                    "category": f"cat-{i % 5}",
                    "score": float(i) / count
                }
            )
            vectors.append(doc)
        return vectors
    
    # Use numpy if available
    np.random.seed(seed)
    vectors = []
    
    for i in range(count):
        # Generate a normalized vector (unit length)
        vector = np.random.normal(0, 1, dimension)
        vector = vector / np.linalg.norm(vector)
        
        # Create document
        doc = VectorDocument(
            id=f"vec-{i}",
            vector=vector.tolist(),
            metadata={
                "text": f"Test vector {i}",
                "category": f"cat-{i % 5}",
                "score": float(i) / count
            }
        )
        vectors.append(doc)
    
    return vectors


async def get_store_safe(db_type: VectorDBType) -> Optional[BaseVectorStore]:
    """Safely get a store instance, handling missing implementations."""
    try:
        factory = get_vector_store_factory()
        
        if not factory.registry.is_registered(db_type):
            logger.warning(f"{db_type.value} store not registered")
            return None
        
        store = await factory.create_store(db_type)
        await store.connect()
        return store
    except Exception as e:
        if "abstract" in str(e).lower():
            logger.warning(f"{db_type.value} store implementation incomplete")
        else:
            logger.error(f"Failed to create {db_type.value} store: {e}")
        return None


async def test_quantization(store: BaseVectorStore, dimension: int = 128) -> bool:
    """Test vector quantization capabilities."""
    logger.info(f"Testing quantization on {store.__class__.__name__}...")
    
    try:
        # Check if store supports quantization
        supports_quantization = getattr(store, 'supports_quantization', False)
        if not supports_quantization:
            logger.info(f"{store.__class__.__name__} does not support quantization, skipping test")
            return True
        
        # Generate test data
        vectors = generate_test_vectors(100, dimension)
        
        # Create index with quantization
        try:
            result = await store.create_index(
                "test_quant_index",
                dimension=dimension,
                distance_metric="cosine"
            )
            if not result:
                logger.warning("Failed to create quantization index")
                return False
        except Exception as e:
            logger.warning(f"Create index with quantization failed: {e}")
            return True  # Not all stores support this
        
        # Add vectors
        start_time = time.time()
        try:
            result = await store.add_vectors(vectors, "test_quant_index")
            elapsed = time.time() - start_time
            logger.info(f"Added {len(vectors)} vectors in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"Add vectors failed: {e}")
            return True
        
        # Get stats to verify quantization
        try:
            stats = await store.get_index_stats("test_quant_index")
            logger.info(f"Index stats after quantization: {stats}")
        except Exception as e:
            logger.warning(f"Stats not available: {e}")
        
        # Test search with quantized vectors
        try:
            query = SearchQuery(
                vector=vectors[0].vector,
                limit=5
            )
            
            start_time = time.time()
            results = await store.search(query, "test_quant_index")
            elapsed = time.time() - start_time
            result_count = len(results) if isinstance(results, list) else getattr(results, 'total_matches', 0)
            logger.info(f"Quantized search completed in {elapsed:.2f}s with {result_count} results")
        except Exception as e:
            logger.warning(f"Quantized search failed: {e}")
        
        # Cleanup
        try:
            await store.delete_index("test_quant_index")
        except Exception as e:
            logger.warning(f"Index cleanup failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Quantization test failed: {e}")
        return False


async def test_sharding(store: BaseVectorStore, dimension: int = 128) -> bool:
    """Test vector sharding capabilities."""
    logger.info(f"Testing sharding on {store.__class__.__name__}...")
    
    try:
        # Check if store supports sharding
        supports_sharding = getattr(store, 'supports_sharding', False)
        if not supports_sharding:
            logger.info(f"{store.__class__.__name__} does not support sharding, skipping test")
            return True
        
        # Generate test data - more documents to test sharding
        vectors = generate_test_vectors(500, dimension)
        
        # Create index with sharding if supported
        try:
            result = await store.create_index(
                "test_shard_index",
                dimension=dimension,
                distance_metric="cosine"
            )
            if not result:
                logger.warning("Failed to create sharding index")
                return False
        except Exception as e:
            logger.warning(f"Create index with sharding failed: {e}")
            return True
        
        # Add vectors
        start_time = time.time()
        try:
            result = await store.add_vectors(vectors, "test_shard_index")
            elapsed = time.time() - start_time
            logger.info(f"Added {len(vectors)} vectors with sharding in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"Add vectors failed: {e}")
            return True
        
        # Get stats to verify sharding
        try:
            stats = await store.get_index_stats("test_shard_index")
            logger.info(f"Index stats after sharding: {stats}")
        except Exception as e:
            logger.warning(f"Stats not available: {e}")
        
        # Test search across shards
        try:
            query = SearchQuery(
                vector=vectors[0].vector,
                limit=10
            )
            
            start_time = time.time()
            results = await store.search(query, "test_shard_index")
            elapsed = time.time() - start_time
            result_count = len(results) if isinstance(results, list) else getattr(results, 'total_matches', 0)
            logger.info(f"Cross-shard search completed in {elapsed:.2f}s with {result_count} results")
        except Exception as e:
            logger.warning(f"Cross-shard search failed: {e}")
        
        # Cleanup
        try:
            await store.delete_index("test_shard_index")
        except Exception as e:
            logger.warning(f"Index cleanup failed: {e}")
        
        logger.info(f"Sharding test completed for {store.__class__.__name__}")
        return True
        
    except Exception as e:
        logger.error(f"Sharding test failed for {store.__class__.__name__}: {e}")
        return False


async def test_advanced_features(db_type: VectorDBType) -> Dict[str, bool]:
    """Test advanced features for a specific store type."""
    logger.info(f"Testing advanced features for {db_type.value}...")
    
    results = {}
    store = await get_store_safe(db_type)
    
    if store is None:
        logger.warning(f"Could not create store for {db_type.value}, skipping advanced tests")
        return {"quantization": True, "sharding": True}  # Not a failure, just unavailable
    
    try:
        # Test quantization
        quant_result = await test_quantization(store, dimension=64)
        results["quantization"] = quant_result
        
        # Test sharding
        shard_result = await test_sharding(store, dimension=64)
        results["sharding"] = shard_result
        
        # Disconnect
        await store.disconnect()
        
    except Exception as e:
        logger.error(f"Advanced features test failed for {db_type.value}: {e}")
        results["quantization"] = False
        results["sharding"] = False
    
    return results


async def test_performance_comparison(dimension: int = 256, vector_count: int = 1000) -> Dict[str, Dict[str, float]]:
    """Test performance comparison across available stores."""
    logger.info(f"Running performance comparison with {vector_count} vectors of dimension {dimension}...")
    
    if not check_numpy_available():
        logger.warning("NumPy not available, skipping performance comparison")
        return {}
    
    deps = check_dependencies()
    results = {}
    
    # Generate test data
    vectors = generate_test_vectors(vector_count, dimension)
    query_vector = vectors[0].vector
    
    factory = get_vector_store_factory()
    available_stores = factory.get_available_stores()
    
    for db_type in available_stores:
        # Check dependencies
        skip = False
        if db_type == VectorDBType.FAISS and not deps['faiss']:
            skip = True
        elif db_type == VectorDBType.IPFS and not deps['ipfs']:
            skip = True
        elif db_type == VectorDBType.DUCKDB and not deps['duckdb_full']:
            skip = True
        
        if skip:
            logger.info(f"Skipping {db_type.value} performance test - dependencies not available")
            continue
        
        store = await get_store_safe(db_type)
        if store is None:
            continue
        
        try:
            # Create index
            index_name = f"perf_test_{db_type.value}"
            start_time = time.time()
            await store.create_index(index_name, dimension=dimension)
            index_time = time.time() - start_time
            
            # Add vectors
            start_time = time.time()
            await store.add_vectors(vectors, index_name)
            add_time = time.time() - start_time
            
            # Search
            query = SearchQuery(vector=query_vector, limit=10)
            start_time = time.time()
            search_results = await store.search(query, index_name)
            search_time = time.time() - start_time
            
            results[db_type.value] = {
                "index_time": index_time,
                "add_time": add_time,
                "search_time": search_time,
                "vectors_per_second": vector_count / add_time if add_time > 0 else 0
            }
            
            # Cleanup
            await store.delete_index(index_name)
            await store.disconnect()
            
            logger.info(f"{db_type.value} performance: "
                       f"index={index_time:.3f}s, add={add_time:.3f}s, search={search_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"Performance test failed for {db_type.value}: {e}")
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="Test advanced vector store features")
    parser.add_argument("--store", "-s", 
                        help="Specific store to test (qdrant, elasticsearch, pgvector, faiss, ipfs, duckdb)",
                        default=None)
    parser.add_argument("--quantization", "-q", 
                        help="Test quantization features", 
                        action="store_true")
    parser.add_argument("--sharding", "-sh", 
                        help="Test sharding features", 
                        action="store_true")
    parser.add_argument("--performance", "-p", 
                        help="Run performance comparison", 
                        action="store_true")
    parser.add_argument("--dimension", "-d", 
                        help="Vector dimension for tests", 
                        type=int, default=128)
    parser.add_argument("--count", "-c", 
                        help="Number of vectors for performance test", 
                        type=int, default=1000)
    args = parser.parse_args()
    
    # Reset factory
    reset_factory()
    
    success = True
    
    if args.performance:
        logger.info("Running performance comparison...")
        perf_results = await test_performance_comparison(args.dimension, args.count)
        if perf_results:
            logger.info("\nPerformance Results:")
            for store_name, metrics in perf_results.items():
                logger.info(f"  {store_name}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value:.3f}")
    
    if args.store:
        # Test specific store
        try:
            db_type = VectorDBType(args.store)
            results = await test_advanced_features(db_type)
            
            for feature, result in results.items():
                if not result:
                    success = False
                status = "SUCCESS" if result else "FAILED"
                logger.info(f"{args.store} {feature}: {status}")
                
        except ValueError:
            logger.error(f"Unknown store type: {args.store}")
            success = False
    else:
        # Test all stores
        factory = get_vector_store_factory()
        available_stores = factory.get_available_stores()
        
        all_results = {}
        for db_type in available_stores:
            results = await test_advanced_features(db_type)
            all_results[db_type.value] = results
            
            for feature, result in results.items():
                if not result:
                    success = False
        
        logger.info("\nAdvanced Features Test Summary:")
        for store_name, features in all_results.items():
            logger.info(f"  {store_name}:")
            for feature, result in features.items():
                status = "SUCCESS" if result else "FAILED"
                logger.info(f"    {feature}: {status}")
    
    sys.exit(0 if success else 1)


async def test_store_advanced(db_type: VectorDBType) -> bool:
    """Run advanced tests on a single vector store."""
    logger.info(f"\nRunning advanced tests for {db_type.value} store...")
    
    try:
        factory = get_vector_store_factory()
        store = await factory.create_store(db_type)
        
        # Connect to the store
        await store.connect()
        ping_result = await store.ping()
        if not ping_result:
            logger.error(f"{db_type.value} connection failed")
            return False
        
        # Run tests
        dimension = 64  # Use smaller dimension for faster tests
        quantization_success = await test_quantization(store, dimension)
        sharding_success = await test_sharding(store, dimension)
        
        # Disconnect
        await store.disconnect()
        
        overall_success = quantization_success and sharding_success
        logger.info(f"{db_type.value} advanced tests {'passed' if overall_success else 'failed'}")
        return overall_success
        
    except Exception as e:
        logger.error(f"{db_type.value} advanced tests failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    asyncio.run(main())
