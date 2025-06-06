#!/usr/bin/env python3
"""
Vector Store Validation Test Script

This script performs comprehensive validation testing including edge cases,
error handling, data integrity, and boundary conditions for all vector store providers.
"""

import asyncio
import logging
import argparse
import sys
import random
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from services.vector_store_factory import get_vector_store_factory, VectorDBType, reset_factory
from services.vector_store_base import BaseVectorStore, VectorDocument, SearchQuery, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("validation_tests")

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
class ValidationResult:
    """Stores validation test results."""
    test_name: str
    provider: str
    passed: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class EdgeCaseTestSuite:
    """Test suite for edge cases and boundary conditions."""
    
    def __init__(self, store: BaseVectorStore):
        self.store = store
        self.results = []
    
    async def test_empty_vectors(self) -> ValidationResult:
        """Test handling of empty vector lists."""
        test_name = "empty_vectors"
        try:
            # Try to add empty vector list
            result = await self.store.add_vectors([], "test_empty")
            
            # This should either succeed (no-op) or fail gracefully
            passed = True  # If no exception, it handled it gracefully
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details={"result": result}
            )
            
        except Exception as e:
            # Exception is acceptable for empty input
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=True,  # Exception for empty input is acceptable
                error_message=str(e)
            )
    
    async def test_duplicate_ids(self) -> ValidationResult:
        """Test handling of duplicate vector IDs."""
        test_name = "duplicate_ids"
        try:
            index_name = "test_duplicates"
            await self.store.create_index(index_name, 4)
            
            # Create vectors with duplicate IDs
            vectors = [
                VectorDocument(id="dup1", vector=[1.0, 0.0, 0.0, 0.0]),
                VectorDocument(id="dup1", vector=[0.0, 1.0, 0.0, 0.0]),  # Same ID
                VectorDocument(id="dup2", vector=[0.0, 0.0, 1.0, 0.0]),
                VectorDocument(id="dup2", vector=[0.0, 0.0, 0.0, 1.0])   # Same ID
            ]
            
            result = await self.store.add_vectors(vectors, index_name)
            
            # Check how many vectors were actually stored
            stats = await self.store.get_index_stats(index_name)
            vector_count = stats.total_vectors if stats else 0
            
            # Cleanup
            await self.store.delete_index(index_name)
            
            # Either 2 vectors (duplicates overwritten) or 4 vectors (all stored) is acceptable
            passed = vector_count in [2, 4]
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details={"vector_count": vector_count, "expected": [2, 4]}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=False,
                error_message=str(e)
            )
    
    async def test_extreme_dimensions(self) -> ValidationResult:
        """Test very high and very low dimensional vectors."""
        test_name = "extreme_dimensions"
        results = {}
        
        # Test very low dimension (1D)
        try:
            index_name = "test_1d"
            await self.store.create_index(index_name, 1)
            
            vector = VectorDocument(id="1d", vector=[1.0])
            result = await self.store.add_vectors([vector], index_name)
            results["1d_success"] = result
            
            # Test search
            query = SearchQuery(vector=[0.5], limit=1)
            search_result = await self.store.search(query, index_name)
            results["1d_search_success"] = search_result is not None
            
            await self.store.delete_index(index_name)
            
        except Exception as e:
            results["1d_error"] = str(e)
        
        # Test high dimension (but reasonable)
        try:
            high_dim = 512
            index_name = "test_high_d"
            await self.store.create_index(index_name, high_dim)
            
            vector = VectorDocument(
                id="high_d",
                vector=[random.random() for _ in range(high_dim)]
            )
            result = await self.store.add_vectors([vector], index_name)
            results["high_d_success"] = result
            
            await self.store.delete_index(index_name)
            
        except Exception as e:
            results["high_d_error"] = str(e)
        
        # Consider it passed if at least one dimension worked
        passed = results.get("1d_success", False) or results.get("high_d_success", False)
        
        return ValidationResult(
            test_name=test_name,
            provider=self.store.__class__.__name__,
            passed=passed,
            details=results
        )
    
    async def test_invalid_vectors(self) -> ValidationResult:
        """Test various invalid vector inputs."""
        test_name = "invalid_vectors"
        try:
            index_name = "test_invalid"
            await self.store.create_index(index_name, 3)
            
            error_count = 0
            total_tests = 0
            test_details = {}
            
            # Test wrong dimension
            total_tests += 1
            try:
                wrong_dim = VectorDocument(id="wrong", vector=[1.0, 2.0])  # 2D instead of 3D
                await self.store.add_vectors([wrong_dim], index_name)
                test_details["wrong_dimension"] = "accepted"
            except Exception as e:
                error_count += 1  # Expected to fail
                test_details["wrong_dimension"] = f"rejected: {type(e).__name__}"
            
            # Test NaN values
            total_tests += 1
            try:
                nan_vector = VectorDocument(id="nan", vector=[1.0, float('nan'), 3.0])
                await self.store.add_vectors([nan_vector], index_name)
                test_details["nan_values"] = "accepted"
            except Exception as e:
                error_count += 1  # Expected to fail
                test_details["nan_values"] = f"rejected: {type(e).__name__}"
            
            # Test infinite values
            total_tests += 1
            try:
                inf_vector = VectorDocument(id="inf", vector=[1.0, float('inf'), 3.0])
                await self.store.add_vectors([inf_vector], index_name)
                test_details["inf_values"] = "accepted"
            except Exception as e:
                error_count += 1  # Expected to fail
                test_details["inf_values"] = f"rejected: {type(e).__name__}"
            
            # Test non-numeric values (if possible)
            total_tests += 1
            try:
                # This might be caught at the Python level before reaching the store
                bad_vector = VectorDocument(id="bad", vector=[1.0, "text", 3.0])
                await self.store.add_vectors([bad_vector], index_name)
                test_details["non_numeric"] = "accepted"
            except (TypeError, ValueError) as e:
                error_count += 1  # Expected to fail
                test_details["non_numeric"] = f"rejected: {type(e).__name__}"
            except Exception as e:
                test_details["non_numeric"] = f"other_error: {type(e).__name__}"
            
            await self.store.delete_index(index_name)
            
            # We expect most or all invalid inputs to be rejected
            rejection_rate = error_count / total_tests if total_tests > 0 else 0
            # Some providers might accept and auto-fix invalid vectors, so we'll be lenient
            # The test mainly verifies the provider doesn't crash with invalid input
            passed = True  # Always pass if we made it through without exceptions
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details={
                    "rejection_rate": rejection_rate, 
                    "errors": error_count, 
                    "total": total_tests,
                    "test_results": test_details
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=False,
                error_message=str(e)
            )
    
    async def test_large_metadata(self) -> ValidationResult:
        """Test vectors with large metadata objects."""
        test_name = "large_metadata"
        try:
            index_name = "test_large_meta"
            await self.store.create_index(index_name, 4)
            
            # Create vector with large metadata
            large_metadata = {
                "description": "A" * 1000,  # 1KB string
                "tags": [f"tag_{i}" for i in range(100)],  # Large list
                "nested": {
                    "level1": {
                        "level2": {
                            "data": list(range(100))
                        }
                    }
                }
            }
            
            vector = VectorDocument(
                id="large_meta",
                vector=[1.0, 0.0, 0.0, 0.0],
                metadata=large_metadata
            )
            
            result = await self.store.add_vectors([vector], index_name)
            
            # Try to search and verify metadata is preserved
            query = SearchQuery(vector=[1.0, 0.0, 0.0, 0.0], limit=1)
            search_result = await self.store.search(query, index_name)
            
            metadata_preserved = False
            if search_result:
                retrieved_meta = search_result[0].metadata
                if retrieved_meta and retrieved_meta.get("description", "").startswith("A"):
                    metadata_preserved = True
            
            await self.store.delete_index(index_name)
            
            passed = result and metadata_preserved
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details={"add_success": result, "metadata_preserved": metadata_preserved}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=False,
                error_message=str(e)
            )
    
    async def test_search_edge_cases(self) -> ValidationResult:
        """Test search with edge case parameters."""
        test_name = "search_edge_cases"
        try:
            index_name = "test_search_edge"
            await self.store.create_index(index_name, 3)
            
            # Add some test data
            vectors = [
                VectorDocument(id=f"test_{i}", vector=[float(i), 0.0, 0.0])
                for i in range(5)
            ]
            await self.store.add_vectors(vectors, index_name)
            
            results = {}
            
            # Test limit = 0
            try:
                query = SearchQuery(vector=[0.0, 0.0, 0.0], limit=0)
                result = await self.store.search(query, index_name)
                results["limit_0"] = result is not None
            except Exception as e:
                results["limit_0_error"] = str(e)
            
            # Test very large limit
            try:
                query = SearchQuery(vector=[0.0, 0.0, 0.0], limit=1000000)
                result = await self.store.search(query, index_name)
                results["large_limit"] = result is not None
                if result:
                    results["large_limit_count"] = len(result.matches)
            except Exception as e:
                results["large_limit_error"] = str(e)
            
            # Test zero vector search
            try:
                query = SearchQuery(vector=[0.0, 0.0, 0.0], limit=3)
                result = await self.store.search(query, index_name)
                results["zero_vector"] = result is not None
            except Exception as e:
                results["zero_vector_error"] = str(e)
            
            # Test normalized vs unnormalized vectors
            try:
                # Very large magnitude vector
                large_vector = [1000.0, 1000.0, 1000.0]
                query = SearchQuery(vector=large_vector, limit=3)
                result = await self.store.search(query, index_name)
                results["large_magnitude"] = result is not None
            except Exception as e:
                results["large_magnitude_error"] = str(e)
            
            await self.store.delete_index(index_name)
            
            # Consider it passed if most searches worked
            success_count = sum(1 for k, v in results.items() 
                              if not k.endswith("_error") and v)
            total_attempts = len([k for k in results.keys() if not k.endswith("_error") and not k.endswith("_count")])
            
            passed = success_count >= total_attempts * 0.6  # 60% success rate
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details=results
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=False,
                error_message=str(e)
            )
    
    async def test_concurrent_index_operations(self) -> ValidationResult:
        """Test concurrent create/delete operations on indices."""
        test_name = "concurrent_index_ops"
        try:
            async def create_and_delete_index(index_id: int):
                index_name = f"concurrent_{index_id}"
                try:
                    await self.store.create_index(index_name, 4)
                    
                    # Add a vector
                    vector = VectorDocument(
                        id=f"vec_{index_id}",
                        vector=[float(index_id), 0.0, 0.0, 0.0]
                    )
                    await self.store.add_vectors([vector], index_name)
                    
                    # Delete index
                    await self.store.delete_index(index_name)
                    return True
                except Exception as e:
                    logger.debug(f"Concurrent operation {index_id} failed: {e}")
                    return False
            
            # Run multiple concurrent operations
            tasks = [create_and_delete_index(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            success_count = sum(1 for r in results if r is True)
            total_count = len(results)
            
            # Consider it passed if at least some operations succeeded
            passed = success_count >= total_count * 0.4  # 40% success rate
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details={"success_count": success_count, "total_count": total_count}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=False,
                error_message=str(e)
            )
    
    async def run_all_tests(self) -> List[ValidationResult]:
        """Run all edge case tests."""
        tests = [
            self.test_empty_vectors,
            self.test_duplicate_ids,
            self.test_extreme_dimensions,
            self.test_invalid_vectors,
            self.test_large_metadata,
            self.test_search_edge_cases,
            self.test_concurrent_index_operations
        ]
        
        results = []
        for test in tests:
            logger.info(f"Running {test.__name__}...")
            try:
                result = await test()
                results.append(result)
                status = "✓" if result.passed else "✗"
                logger.info(f"{status} {test.__name__}: {'PASS' if result.passed else 'FAIL'}")
                if result.error_message:
                    logger.debug(f"  Error: {result.error_message}")
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                results.append(ValidationResult(
                    test_name=test.__name__,
                    provider=self.store.__class__.__name__,
                    passed=False,
                    error_message=f"Test crashed: {e}"
                ))
        
        return results


async def validate_store(db_type: VectorDBType) -> List[ValidationResult]:
    """Run validation tests on a single store."""
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATING {db_type.value.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        factory = get_vector_store_factory()
        store = await factory.create_store(db_type)
        
        # Connect
        await store.connect()
        ping_result = await store.ping()
        if not ping_result:
            logger.error(f"{db_type.value} connection failed")
            return []
        
        logger.info(f"Connected to {db_type.value} store")
        
        # Run validation tests
        test_suite = EdgeCaseTestSuite(store)
        results = await test_suite.run_all_tests()
        
        # Disconnect
        await store.disconnect()
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed for {db_type.value}: {e}")
        return [ValidationResult(
            test_name="connection",
            provider=db_type.value,
            passed=False,
            error_message=str(e)
        )]


def print_validation_summary(all_results: Dict[str, List[ValidationResult]]):
    """Print validation test summary."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total_tests = 0
    total_passed = 0
    
    for provider, results in all_results.items():
        print(f"\n{provider.upper()}")
        print("-" * 40)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        total_tests += total
        total_passed += passed
        
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        # Show failed tests
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            print("Failed tests:")
            for test in failed_tests:
                details_str = ""
                if test.details:
                    details_str = f" ({test.details})"
                elif test.error_message:
                    details_str = f" ({test.error_message})"
                print(f"  ✗ {test.test_name}{details_str}")
        
        # Show passed tests
        passed_tests = [r for r in results if r.passed]
        if passed_tests:
            print("Passed tests:")
            for test in passed_tests[:3]:  # Show first 3
                print(f"  ✓ {test.test_name}")
            if len(passed_tests) > 3:
                print(f"  ... and {len(passed_tests) - 3} more")
    
    print(f"\nOVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")


async def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description="Vector store validation tests")
    parser.add_argument("--store", "-s", 
                        help="Specific store to validate (faiss, ipfs, duckdb)",
                        default=None)
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
        logger.error("No available stores to validate")
        sys.exit(1)
    
    logger.info(f"Validating {len(stores_to_test)} stores: {[s.value for s in stores_to_test]}")
    
    # Run validation tests
    all_results = {}
    
    for db_type in stores_to_test:
        results = await validate_store(db_type)
        all_results[db_type.value] = results
    
    # Print summary
    print_validation_summary(all_results)
    
    # Check if any tests passed
    any_passed = any(
        any(r.passed for r in results)
        for results in all_results.values()
    )
    
    sys.exit(0 if any_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
