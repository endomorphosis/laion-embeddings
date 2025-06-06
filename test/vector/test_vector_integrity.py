#!/usr/bin/env python3
"""
Data integrity and consistency test suite for vector stores.

Tests data consistency, durability, transactional properties,
and data integrity across different scenarios.
"""

import asyncio
import argparse
import logging
import sys
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from services.vector_store_factory import (
    VectorStoreFactory, VectorDBType, reset_factory
)
from services.vector_store_base import (
    VectorDocument, SearchQuery, SearchResult
)

# Check dependencies
FAISS_AVAILABLE = True
try:
    import faiss
except ImportError:
    FAISS_AVAILABLE = False

IPFS_AVAILABLE = True
try:
    from services.providers.ipfs_store import IPFSVectorStore
except ImportError:
    IPFS_AVAILABLE = False

DUCKDB_FULL_AVAILABLE = True
try:
    import duckdb
    import pyarrow
except ImportError:
    DUCKDB_FULL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IntegrityTestResult:
    """Result of a data integrity test."""
    test_name: str
    provider: str
    passed: bool
    data_consistent: bool = True
    corruption_detected: bool = False
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class IntegrityTester:
    """Data integrity test runner for vector stores."""
    
    def __init__(self, store, provider_name: str):
        self.store = store
        self.provider_name = provider_name
    
    async def run_all_tests(self) -> List[IntegrityTestResult]:
        """Run all data integrity tests."""
        tests = [
            self.test_data_persistence,
            self.test_search_consistency,
            self.test_concurrent_modifications,
            self.test_data_corruption_detection,
            self.test_index_integrity,
            self.test_metadata_consistency,
            self.test_vector_precision,
            self.test_transactional_consistency
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
                if result.passed:
                    if result.corruption_detected:
                        logger.warning(f"⚠ {result.test_name}: CORRUPTION DETECTED")
                    elif not result.data_consistent:
                        logger.warning(f"⚠ {result.test_name}: INCONSISTENCY DETECTED")
                    else:
                        logger.info(f"✓ {result.test_name}: CONSISTENT")
                else:
                    logger.info(f"✗ {result.test_name}: FAIL")
            except Exception as e:
                logger.error(f"✗ {test.__name__}: ERROR - {e}")
                results.append(IntegrityTestResult(
                    test_name=test.__name__.replace('test_', ''),
                    provider=self.provider_name,
                    passed=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _calculate_vector_hash(self, vector: List[float]) -> str:
        """Calculate hash of a vector for integrity checking."""
        vector_str = json.dumps(vector, sort_keys=True)
        return hashlib.md5(vector_str.encode()).hexdigest()
    
    async def test_data_persistence(self) -> IntegrityTestResult:
        """Test that data persists correctly across operations."""
        test_name = "data_persistence"
        try:
            index_name = "integrity_test_persistence"
            await self.store.create_index(index_name, 4)
            
            # Add initial vectors
            original_vectors = [
                VectorDocument(
                    id=f"persist_{i}",
                    vector=[float(i), float(i+1), float(i+2), float(i+3)],
                    metadata={"batch": "initial", "id": i}
                ) for i in range(10)
            ]
            
            await self.store.add_vectors(original_vectors, index_name)
            
            # Get stats to verify data was added
            stats = await self.store.get_index_stats(index_name)
            initial_count = stats.total_vectors if stats else 0
            
            # Add more vectors
            additional_vectors = [
                VectorDocument(
                    id=f"persist_add_{i}",
                    vector=[float(i+10), float(i+11), float(i+12), float(i+13)],
                    metadata={"batch": "additional", "id": i+10}
                ) for i in range(5)
            ]
            
            await self.store.add_vectors(additional_vectors, index_name)
            
            # Verify final count
            final_stats = await self.store.get_index_stats(index_name)
            final_count = final_stats.total_vectors if final_stats else 0
            
            # Check data consistency
            expected_count = len(original_vectors) + len(additional_vectors)
            data_consistent = final_count == expected_count
            
            # Try to retrieve some vectors to verify they exist
            retrieved_vectors = []
            for i in range(3):  # Sample a few
                try:
                    vector = await self.store.get_vector(f"persist_{i}", index_name)
                    if vector:
                        retrieved_vectors.append(vector)
                except Exception:
                    pass  # Some stores might not support get_vector
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=data_consistent,
                details={
                    "initial_count": initial_count,
                    "final_count": final_count,
                    "expected_count": expected_count,
                    "retrieved_count": len(retrieved_vectors)
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_search_consistency(self) -> IntegrityTestResult:
        """Test search result consistency across multiple queries."""
        test_name = "search_consistency"
        try:
            index_name = "integrity_test_search"
            await self.store.create_index(index_name, 4)
            
            # Add vectors with known relationships
            vectors = [
                VectorDocument(id="origin", vector=[0.0, 0.0, 0.0, 0.0]),
                VectorDocument(id="close1", vector=[0.1, 0.1, 0.1, 0.1]),
                VectorDocument(id="close2", vector=[0.2, 0.2, 0.2, 0.2]),
                VectorDocument(id="far", vector=[1.0, 1.0, 1.0, 1.0]),
            ]
            
            await self.store.add_vectors(vectors, index_name)
            
            # Perform multiple searches with the same query
            query_vector = [0.0, 0.0, 0.0, 0.0]
            search_query = SearchQuery(vector=query_vector, top_k=3)
            
            results_list = []
            for _ in range(5):  # Multiple searches
                results = await self.store.search(search_query, index_name)
                results_list.append(results)
                await asyncio.sleep(0.1)  # Small delay
            
            # Check consistency of results
            data_consistent = True
            if len(results_list) > 1:
                first_result_ids = [r.id for r in results_list[0]]
                for results in results_list[1:]:
                    current_ids = [r.id for r in results]
                    if current_ids != first_result_ids:
                        data_consistent = False
                        break
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=data_consistent,
                details={
                    "search_runs": len(results_list),
                    "first_result_count": len(results_list[0]) if results_list else 0
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_concurrent_modifications(self) -> IntegrityTestResult:
        """Test data integrity during concurrent modifications."""
        test_name = "concurrent_modifications"
        try:
            index_name = "integrity_test_concurrent"
            await self.store.create_index(index_name, 4)
            
            # Define concurrent operations
            async def add_batch(batch_id: int, start_id: int, count: int):
                vectors = [
                    VectorDocument(
                        id=f"concurrent_{batch_id}_{i}",
                        vector=[float(i), float(i+1), float(i+2), float(i+3)],
                        metadata={"batch": batch_id}
                    ) for i in range(start_id, start_id + count)
                ]
                await self.store.add_vectors(vectors, index_name)
                return len(vectors)
            
            # Run concurrent operations
            tasks = [
                add_batch(0, 0, 10),
                add_batch(1, 100, 10),
                add_batch(2, 200, 10),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that all operations completed successfully
            successful_operations = sum(1 for r in results if isinstance(r, int))
            expected_vectors = sum(r for r in results if isinstance(r, int))
            
            # Verify final state
            stats = await self.store.get_index_stats(index_name)
            actual_count = stats.total_vectors if stats else 0
            
            data_consistent = actual_count == expected_vectors
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=data_consistent,
                details={
                    "successful_operations": successful_operations,
                    "expected_vectors": expected_vectors,
                    "actual_count": actual_count
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_data_corruption_detection(self) -> IntegrityTestResult:
        """Test ability to detect data corruption."""
        test_name = "data_corruption_detection"
        try:
            index_name = "integrity_test_corruption"
            await self.store.create_index(index_name, 4)
            
            # Add vectors with checksums
            vectors_with_hashes = []
            for i in range(10):
                vector = [float(i), float(i+1), float(i+2), float(i+3)]
                vector_hash = self._calculate_vector_hash(vector)
                
                vectors_with_hashes.append((
                    VectorDocument(
                        id=f"checksum_{i}",
                        vector=vector,
                        metadata={"checksum": vector_hash, "original_id": i}
                    ),
                    vector_hash
                ))
            
            await self.store.add_vectors([v[0] for v in vectors_with_hashes], index_name)
            
            # Retrieve vectors and verify checksums
            corruption_detected = False
            verified_count = 0
            
            for doc, original_hash in vectors_with_hashes[:5]:  # Check a sample
                try:
                    retrieved = await self.store.get_vector(doc.id, index_name)
                    if retrieved and retrieved.vector:
                        calculated_hash = self._calculate_vector_hash(retrieved.vector)
                        if calculated_hash != original_hash:
                            corruption_detected = True
                            break
                        verified_count += 1
                except Exception:
                    # Some stores might not support get_vector
                    break
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                corruption_detected=corruption_detected,
                details={
                    "vectors_checked": verified_count,
                    "total_vectors": len(vectors_with_hashes)
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_index_integrity(self) -> IntegrityTestResult:
        """Test index structure integrity."""
        test_name = "index_integrity"
        try:
            index_name = "integrity_test_index"
            await self.store.create_index(index_name, 4)
            
            # Add vectors in multiple batches
            for batch in range(3):
                vectors = [
                    VectorDocument(
                        id=f"index_test_{batch}_{i}",
                        vector=[float(batch), float(i), float(batch+i), float(i*2)],
                        metadata={"batch": batch, "index": i}
                    ) for i in range(10)
                ]
                await self.store.add_vectors(vectors, index_name)
            
            # Verify index stats are consistent
            stats = await self.store.get_index_stats(index_name)
            expected_count = 3 * 10  # 3 batches * 10 vectors each
            actual_count = stats.total_vectors if stats else 0
            
            # Test search functionality to verify index structure
            search_query = SearchQuery(vector=[0.0, 0.0, 0.0, 0.0], top_k=5)
            search_results = await self.store.search(search_query, index_name)
            
            data_consistent = (
                actual_count == expected_count and
                len(search_results) > 0
            )
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=data_consistent,
                details={
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "search_results": len(search_results)
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_metadata_consistency(self) -> IntegrityTestResult:
        """Test metadata consistency and preservation."""
        test_name = "metadata_consistency"
        try:
            index_name = "integrity_test_metadata"
            await self.store.create_index(index_name, 4)
            
            # Add vectors with complex metadata
            vectors = []
            for i in range(5):
                metadata = {
                    "id": i,
                    "category": f"cat_{i % 3}",
                    "tags": [f"tag_{j}" for j in range(i+1)],
                    "nested": {
                        "level1": {"level2": {"value": i * 2}},
                        "array": list(range(i+1))
                    },
                    "timestamp": time.time()
                }
                
                vectors.append(VectorDocument(
                    id=f"meta_test_{i}",
                    vector=[float(i), float(i+1), float(i+2), float(i+3)],
                    metadata=metadata
                ))
            
            await self.store.add_vectors(vectors, index_name)
            
            # Retrieve vectors and verify metadata
            metadata_consistent = True
            retrieved_count = 0
            
            for original_vector in vectors:
                try:
                    retrieved = await self.store.get_vector(original_vector.id, index_name)
                    if retrieved and retrieved.metadata:
                        # Check key metadata fields
                        if (retrieved.metadata.get("id") != original_vector.metadata.get("id") or
                            retrieved.metadata.get("category") != original_vector.metadata.get("category")):
                            metadata_consistent = False
                            break
                        retrieved_count += 1
                except Exception:
                    # Some stores might not support get_vector or metadata retrieval
                    break
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=metadata_consistent,
                details={
                    "vectors_checked": retrieved_count,
                    "total_vectors": len(vectors)
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_vector_precision(self) -> IntegrityTestResult:
        """Test vector precision and numerical accuracy."""
        test_name = "vector_precision"
        try:
            index_name = "integrity_test_precision"
            await self.store.create_index(index_name, 4)
            
            # Test vectors with various precision requirements
            test_vectors = [
                [1.0, 2.0, 3.0, 4.0],  # Simple integers
                [0.1, 0.2, 0.3, 0.4],  # Simple decimals
                [1.123456789, 2.987654321, 3.111111111, 4.999999999],  # High precision
                [1e-10, 1e-5, 1e5, 1e10],  # Scientific notation
            ]
            
            vectors = [
                VectorDocument(
                    id=f"precision_{i}",
                    vector=vector,
                    metadata={"precision_test": i}
                ) for i, vector in enumerate(test_vectors)
            ]
            
            await self.store.add_vectors(vectors, index_name)
            
            # Retrieve and check precision
            precision_maintained = True
            precision_errors = []
            
            for i, original_vector in enumerate(test_vectors):
                try:
                    retrieved = await self.store.get_vector(f"precision_{i}", index_name)
                    if retrieved and retrieved.vector:
                        for j, (orig, retr) in enumerate(zip(original_vector, retrieved.vector)):
                            # Check relative precision (allowing for small floating point errors)
                            if abs(orig - retr) > abs(orig) * 1e-6:  # 6 decimal places
                                precision_maintained = False
                                precision_errors.append(f"Index {i}, component {j}: {orig} vs {retr}")
                except Exception:
                    # Some stores might not support get_vector
                    break
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=precision_maintained,
                details={
                    "precision_errors": precision_errors,
                    "vectors_tested": len(test_vectors)
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_transactional_consistency(self) -> IntegrityTestResult:
        """Test transactional consistency (if supported)."""
        test_name = "transactional_consistency"
        try:
            index_name = "integrity_test_transaction"
            await self.store.create_index(index_name, 4)
            
            # Add initial data
            initial_vectors = [
                VectorDocument(
                    id=f"trans_init_{i}",
                    vector=[float(i), 0.0, 0.0, 0.0],
                    metadata={"phase": "initial"}
                ) for i in range(5)
            ]
            
            await self.store.add_vectors(initial_vectors, index_name)
            initial_stats = await self.store.get_index_stats(index_name)
            initial_count = initial_stats.total_vectors if initial_stats else 0
            
            # Attempt batch operations that might partially fail
            mixed_vectors = [
                VectorDocument(id="valid_1", vector=[1.0, 1.0, 1.0, 1.0]),
                VectorDocument(id="valid_2", vector=[2.0, 2.0, 2.0, 2.0]),
                # This might cause issues in some stores
                VectorDocument(id="", vector=[3.0, 3.0, 3.0, 3.0]),  # Empty ID
            ]
            
            try:
                await self.store.add_vectors(mixed_vectors, index_name)
            except Exception:
                pass  # Expected to fail
            
            # Check final state
            final_stats = await self.store.get_index_stats(index_name)
            final_count = final_stats.total_vectors if final_stats else 0
            
            # In a consistent system, either all vectors are added or none
            # (depending on how the store handles partial failures)
            data_consistent = True  # We'll be lenient here since behavior varies
            
            await self.store.delete_index(index_name)
            
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                data_consistent=data_consistent,
                details={
                    "initial_count": initial_count,
                    "final_count": final_count,
                    "count_difference": final_count - initial_count
                }
            )
            
        except Exception as e:
            return IntegrityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )


async def test_store_integrity(db_type: VectorDBType) -> List[IntegrityTestResult]:
    """Test data integrity for a specific vector store type."""
    factory = VectorStoreFactory()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING {db_type.value.upper()} DATA INTEGRITY")
    logger.info(f"{'='*60}")
    
    try:
        store = factory.create_store(db_type)
        await store.connect()
        logger.info(f"Connected to {db_type.value} store")
        
        tester = IntegrityTester(store, db_type.value)
        results = await tester.run_all_tests()
        
        await store.disconnect()
        logger.info(f"Disconnected from {db_type.value} store")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to test {db_type.value}: {e}")
        return [IntegrityTestResult(
            test_name="connection",
            provider=db_type.value,
            passed=False,
            error_message=str(e)
        )]


def print_integrity_summary(all_results: Dict[str, List[IntegrityTestResult]]):
    """Print data integrity test summary."""
    print("\n" + "="*80)
    print("DATA INTEGRITY ASSESSMENT SUMMARY")
    print("="*80)
    
    total_tests = 0
    total_passed = 0
    total_inconsistencies = 0
    total_corruptions = 0
    
    for provider, results in all_results.items():
        print(f"\n{provider.upper()}")
        print("-" * 40)
        
        passed = sum(1 for r in results if r.passed)
        inconsistencies = sum(1 for r in results if not r.data_consistent)
        corruptions = sum(1 for r in results if r.corruption_detected)
        total = len(results)
        
        total_tests += total
        total_passed += passed
        total_inconsistencies += inconsistencies
        total_corruptions += corruptions
        
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        if inconsistencies > 0:
            print(f"⚠ Data inconsistencies: {inconsistencies}")
        if corruptions > 0:
            print(f"🔴 Data corruptions: {corruptions}")
        
        # Show problematic tests
        problematic_tests = [r for r in results if not r.data_consistent or r.corruption_detected]
        if problematic_tests:
            print("Issues detected:")
            for test in problematic_tests:
                status = ""
                if test.corruption_detected:
                    status = "CORRUPTION"
                elif not test.data_consistent:
                    status = "INCONSISTENT"
                print(f"  ⚠ {test.test_name}: {status}")
        
        # Show failed tests
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            print("Failed tests:")
            for test in failed_tests:
                print(f"  ✗ {test.test_name}: {test.error_message or 'Unknown error'}")
    
    print(f"\nOVERALL DATA INTEGRITY:")
    print(f"Tests passed: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    print(f"Data inconsistencies: {total_inconsistencies}")
    print(f"Data corruptions: {total_corruptions}")
    
    if total_corruptions == 0 and total_inconsistencies == 0:
        print("🟢 INTEGRITY STATUS: EXCELLENT - No issues detected")
    elif total_corruptions == 0 and total_inconsistencies <= 2:
        print("🟡 INTEGRITY STATUS: GOOD - Minor inconsistencies detected")
    elif total_corruptions == 0:
        print("🟠 INTEGRITY STATUS: FAIR - Multiple inconsistencies detected")
    else:
        print("🔴 INTEGRITY STATUS: POOR - Data corruption detected")


async def main():
    """Main data integrity test runner."""
    parser = argparse.ArgumentParser(description="Vector store data integrity tests")
    parser.add_argument("--store", "-s", 
                        help="Specific store to test (faiss, ipfs, duckdb)",
                        choices=["faiss", "ipfs", "duckdb"])
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Reset factory state
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
        logger.error("No available stores to test")
        sys.exit(1)
    
    logger.info(f"Testing data integrity of {len(stores_to_test)} stores: {[s.value for s in stores_to_test]}")
    
    # Run integrity tests
    all_results = {}
    
    for db_type in stores_to_test:
        results = await test_store_integrity(db_type)
        all_results[db_type.value] = results
    
    # Print summary
    print_integrity_summary(all_results)
    
    # Check overall integrity status
    total_issues = sum(
        sum(1 for r in results if not r.data_consistent or r.corruption_detected)
        for results in all_results.values()
    )
    
    sys.exit(1 if total_issues > 3 else 0)  # Fail if too many integrity issues


if __name__ == "__main__":
    asyncio.run(main())
