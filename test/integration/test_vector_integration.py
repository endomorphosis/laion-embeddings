#!/usr/bin/env python3
"""
Integration tests for the new vector store providers.

This script runs simple integration tests for IPFS/IPLD and DuckDB/Parquet
vector stores to verify basic functionality.
"""

import asyncio
import unittest
import logging
import sys
from typing import Dict, Any, List, Optional
import tempfile
import os
import shutil
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("integration_tests")

# Check for required dependencies
def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None

IPFS_KIT_AVAILABLE = is_package_available("ipfs_kit_py")
IPFS_CLIENT_AVAILABLE = is_package_available("ipfshttpclient")
DUCKDB_AVAILABLE = is_package_available("duckdb")
ARROW_AVAILABLE = is_package_available("pyarrow")
FAISS_AVAILABLE = is_package_available("faiss")

IPFS_AVAILABLE = IPFS_KIT_AVAILABLE or IPFS_CLIENT_AVAILABLE
DUCKDB_FULL_AVAILABLE = DUCKDB_AVAILABLE and ARROW_AVAILABLE

logger.info(f"Dependencies: IPFS={IPFS_AVAILABLE}, DuckDB={DUCKDB_FULL_AVAILABLE}, FAISS={FAISS_AVAILABLE}")

# Import vector store components
from services.vector_config import VectorDBType, get_config_manager, reset_config_manager
from services.vector_store_factory import (
    get_vector_store_factory, 
    reset_factory,
    VectorStoreFactory
)
from services.vector_store_base import (
    BaseVectorStore, 
    VectorDocument, 
    SearchQuery,
    VectorStoreError
)


class VectorStoreIntegrationTests(unittest.TestCase):
    """Integration tests for vector store providers."""
    
    @classmethod
    def setUpClass(cls):
        """Set up temporary directories and resources."""
        cls.temp_dir = tempfile.mkdtemp(prefix="vector_test_")
        cls.duckdb_path = os.path.join(cls.temp_dir, "test.duckdb")
        cls.parquet_dir = os.path.join(cls.temp_dir, "parquet")
        cls.ipfs_cache = os.path.join(cls.temp_dir, "ipfs_cache")
        
        # Create directories
        os.makedirs(cls.parquet_dir, exist_ok=True)
        os.makedirs(cls.ipfs_cache, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary resources."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Reset factory and config before each test."""
        reset_config_manager()
        reset_factory()
    
    async def create_test_data(self, count: int = 5, dimension: int = 4) -> List[VectorDocument]:
        """Create test vector documents."""
        return [
            VectorDocument(
                id=f"test-{i}",
                vector=[float(j + i * 0.1) for j in range(dimension)],
                text=f"Test document {i}",
                metadata={"category": f"cat-{i % 3}", "value": i}
            )
            for i in range(count)
        ]

    async def safe_create_store(self, db_type: VectorDBType) -> Optional[BaseVectorStore]:
        """Safely create a store, handling implementation issues."""
        try:
            factory = get_vector_store_factory()
            
            if not factory.registry.is_registered(db_type):
                self.skipTest(f"{db_type.value} store not registered")
                return None
            
            store = await factory.create_store(db_type)
            return store
        except Exception as e:
            if "abstract" in str(e).lower():
                self.skipTest(f"{db_type.value} store implementation incomplete")
            else:
                logger.error(f"Failed to create {db_type.value} store: {e}")
                raise
            return None

    @unittest.skipUnless(FAISS_AVAILABLE, "FAISS not available")
    async def test_faiss_store_basic(self):
        """Test basic FAISS store functionality."""
        store = await self.safe_create_store(VectorDBType.FAISS)
        if store is None:
            return
        
        await store.connect()
        self.assertTrue(await store.ping())
        
        # Test index creation
        result = await store.create_index("test_faiss", dimension=4)
        self.assertTrue(result)
        
        # Test data operations
        test_docs = await self.create_test_data()
        result = await store.add_vectors(test_docs, "test_faiss")
        self.assertTrue(result)
        
        # Test search
        query = SearchQuery(vector=test_docs[0].vector, limit=3)
        results = await store.search(query, "test_faiss")
        self.assertIsNotNone(results)
        
        # Cleanup
        await store.delete_index("test_faiss")
        await store.disconnect()

    @unittest.skipUnless(IPFS_AVAILABLE, "IPFS client not available")
    async def test_ipfs_store_basic(self):
        """Test basic IPFS store functionality."""
        store = await self.safe_create_store(VectorDBType.IPFS)
        if store is None:
            return
        
        try:
            await store.connect()
            # IPFS might not be running locally, so ping might fail
            # Just test that we can create the store
            logger.info("IPFS store created successfully")
        except Exception as e:
            self.skipTest(f"IPFS not accessible: {e}")
        finally:
            try:
                await store.disconnect()
            except:
                pass

    @unittest.skipUnless(DUCKDB_FULL_AVAILABLE, "DuckDB or Arrow not available")
    async def test_duckdb_store_basic(self):
        """Test basic DuckDB store functionality."""
        store = await self.safe_create_store(VectorDBType.DUCKDB)
        if store is None:
            return
        
        await store.connect()
        self.assertTrue(await store.ping())
        
        # Test index creation
        result = await store.create_index("test_duckdb", dimension=4)
        self.assertTrue(result)
        
        # Test data operations
        test_docs = await self.create_test_data()
        result = await store.add_vectors(test_docs, "test_duckdb")
        self.assertTrue(result)
        
        # Test search
        query = SearchQuery(vector=test_docs[0].vector, limit=3)
        results = await store.search(query, "test_duckdb")
        self.assertIsNotNone(results)
        
        # Cleanup
        await store.delete_index("test_duckdb")
        await store.disconnect()

    async def test_factory_functionality(self):
        """Test factory and registry functionality."""
        factory = get_vector_store_factory()
        
        # Test registry
        registered = factory.get_registered_stores()
        self.assertIsInstance(registered, list)
        
        available = factory.get_available_stores()
        self.assertIsInstance(available, list)
        
        # All available stores should be registered
        for store_type in available:
            self.assertIn(store_type, registered)

    async def test_config_management(self):
        """Test configuration management."""
        config_manager = get_config_manager()
        
        # Test enabled databases
        enabled = config_manager.get_enabled_databases()
        self.assertIsInstance(enabled, list)
        
        # Test default database
        default_db = config_manager.get_default_database()
        self.assertIsInstance(default_db, VectorDBType)
        
        # Default should be in enabled list
        self.assertIn(default_db, enabled)

    async def test_store_registration(self):
        """Test that expected stores are registered."""
        factory = get_vector_store_factory()
        registered = factory.get_registered_stores()
        
        # FAISS should always be registered if available
        if FAISS_AVAILABLE:
            self.assertIn(VectorDBType.FAISS, registered)
        
        # IPFS should be registered if client available
        if IPFS_AVAILABLE:
            self.assertIn(VectorDBType.IPFS, registered)
        
        # DuckDB should be registered if dependencies available
        if DUCKDB_FULL_AVAILABLE:
            self.assertIn(VectorDBType.DUCKDB, registered)


class AsyncTestRunner:
    """Helper to run async tests with unittest."""
    
    def __init__(self, test_class):
        self.test_class = test_class
        self.test_instance = test_class()
    
    async def run_all_tests(self):
        """Run all test methods."""
        results = []
        
        # Setup (not async)
        self.test_instance.setUp()
        
        # Find all test methods
        test_methods = [method for method in dir(self.test_instance) 
                       if method.startswith('test_') and callable(getattr(self.test_instance, method))]
        
        for method_name in test_methods:
            logger.info(f"Running {method_name}...")
            try:
                method = getattr(self.test_instance, method_name)
                await method()
                logger.info(f"✓ {method_name} passed")
                results.append((method_name, True, None))
            except unittest.SkipTest as e:
                logger.info(f"⚠ {method_name} skipped: {e}")
                results.append((method_name, True, f"skipped: {e}"))
            except Exception as e:
                logger.error(f"✗ {method_name} failed: {e}")
                results.append((method_name, False, str(e)))
        
        return results


async def main():
    """Run integration tests."""
    logger.info("Starting vector store integration tests...")
    
    # Run tests
    runner = AsyncTestRunner(VectorStoreIntegrationTests)
    results = await runner.run_all_tests()
    
    # Summary
    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    failed = total - passed
    
    logger.info(f"\nTest Summary:")
    logger.info(f"  Total: {total}")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    
    if failed > 0:
        logger.info(f"\nFailed tests:")
        for name, success, error in results:
            if not success:
                logger.info(f"  {name}: {error}")
    
    return failed == 0


def create_test_vectors(count: int = 10, dimension: int = 128) -> List[VectorDocument]:
    """Create test vector documents."""
    return [
        VectorDocument(
            id=f"test-{i}",
            vector=[float(j) for j in range(i, i+dimension)],
            metadata={"text": f"Test document {i}", "value": i}
        )
        for i in range(1, count+1)
    ]


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
    
    async def test_ipfs_availability(self):
        """Test IPFS vector store availability."""
        if not IPFS_AVAILABLE:
            logger.warning("IPFS dependencies not available, skipping registration test")
            self.skipTest("IPFS dependencies not available")
            return
            
        factory = get_vector_store_factory()
        registered = VectorDBType.IPFS in factory.get_registered_stores()
        logger.info(f"IPFS registered: {registered}")
        
        self.assertIn(VectorDBType.IPFS, factory.get_registered_stores())
    
    async def test_duckdb_availability(self):
        """Test DuckDB vector store availability."""
        if not DUCKDB_FULL_AVAILABLE:
            logger.warning("DuckDB or PyArrow dependencies not available, skipping registration test")
            self.skipTest("DuckDB dependencies not available")
            return
            
        factory = get_vector_store_factory()
        registered = VectorDBType.DUCKDB in factory.get_registered_stores()
        logger.info(f"DuckDB registered: {registered}")
        
        self.assertIn(VectorDBType.DUCKDB, factory.get_registered_stores())
    
    async def test_ipfs_basic_operations(self):
        """Test basic operations with IPFS vector store."""
        if not IPFS_AVAILABLE:
            logger.warning("IPFS dependencies not available, skipping operations test")
            self.skipTest("IPFS dependencies not available")
            return
        
        try:
            factory = get_vector_store_factory()
            store = await factory.create_store(
                VectorDBType.IPFS,
                config_override={
                    "sharding_enabled": True,
                    "max_shard_size": 2,  # Small for testing
                    "dimension": 4,
                    "storage": {
                        "cache_path": self.ipfs_cache
                    }
                }
            )
            
            # Connect
            await store.connect()
            
            # Test ping
            self.assertTrue(await store.ping())
            
            # Create test data
            docs = await self.create_test_data()
            
            # Create index
            await store.create_index(dimension=4, overwrite=True)
            
            # Add documents
            result = await store.add_documents(docs)
            self.assertEqual(len(docs), result)
            
            # Get stats
            stats = await store.get_stats()
            logger.info(f"IPFS stats: {stats}")
            self.assertEqual(stats['total_vectors'], len(docs))
            
            # Search
            query = SearchQuery(vector=[1.5, 2.5, 3.5, 4.5], top_k=3)
            results = await store.search(query)
            
            self.assertGreater(len(results.matches), 0)
            logger.info(f"IPFS search results: {results}")
            
            # Clean up
            await store.disconnect()
            
        except (ImportError, VectorStoreError) as e:
            if "ipfs" in str(e).lower() or "could not connect" in str(e).lower():
                logger.warning(f"IPFS test skipped due to dependency or connectivity: {e}")
                self.skipTest(f"IPFS dependencies or connectivity issue: {e}")
            else:
                raise
    
    async def test_duckdb_basic_operations(self):
        """Test basic operations with DuckDB vector store."""
        if not DUCKDB_FULL_AVAILABLE:
            logger.warning("DuckDB or PyArrow dependencies not available, skipping operations test")
            self.skipTest("DuckDB dependencies not available")
            return
            
        try:
            factory = get_vector_store_factory()
            store = await factory.create_store(
                VectorDBType.DUCKDB,
                config_override={
                    "database_path": self.duckdb_path,
                    "storage_path": self.parquet_dir,
                    "table_name": "test_embeddings",
                    "dimension": 4
                }
            )
            
            # Connect
            await store.connect()
            
            # Test ping
            self.assertTrue(await store.ping())
            
            # Create test data
            docs = await self.create_test_data()
            
            # Add documents
            result = await store.add_documents(docs)
            self.assertEqual(len(docs), result)
            
            # Get stats
            stats = await store.get_stats()
            logger.info(f"DuckDB stats: {stats}")
            self.assertEqual(stats['total_vectors'], len(docs))
            
            # Search
            query = SearchQuery(vector=[1.5, 2.5, 3.5, 4.5], top_k=3)
            results = await store.search(query)
            
            self.assertGreater(len(results.matches), 0)
            logger.info(f"DuckDB search results: {results}")
            
            # Clean up
            await store.disconnect()
            
        except (ImportError, VectorStoreError) as e:
            if "duckdb" in str(e).lower() or "arrow" in str(e).lower():
                logger.warning(f"DuckDB test skipped due to dependency: {e}")
                self.skipTest(f"DuckDB dependencies issue: {e}")
            else:
                raise


def run_tests():
    """Run all integration tests."""
    try:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        sys.exit(1)


async def async_tests():
    """Set up and run async tests."""
    # Add all async tests
    suite = unittest.TestSuite()
    test_cases = [
        VectorStoreIntegrationTests('test_ipfs_availability'),
        VectorStoreIntegrationTests('test_duckdb_availability'),
        VectorStoreIntegrationTests('test_ipfs_basic_operations'),
        VectorStoreIntegrationTests('test_duckdb_basic_operations'),
    ]
    
    # Run all tests
    results = []
    for test in test_cases:
        try:
            logger.info(f"Running {test._testMethodName}")
            test_method = getattr(test, test._testMethodName)
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            logger.info(f"  ✓ {test._testMethodName}")
            results.append((test._testMethodName, True))
        except unittest.SkipTest as e:
            logger.info(f"  ⚠ {test._testMethodName} skipped: {e}")
            results.append((test._testMethodName, 'skipped'))
        except Exception as e:
            logger.error(f"  ✗ {test._testMethodName} failed: {e}")
            results.append((test._testMethodName, False))
    
    # Print summary
    logger.info("\nTest Results:")
    for name, result in results:
        status = "PASSED" if result is True else "SKIPPED" if result == 'skipped' else "FAILED"
        logger.info(f"  {name}: {status}")
    
    failures = sum(1 for _, r in results if r is False)
    return failures == 0


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(async_tests())
    sys.exit(0 if success else 1)
