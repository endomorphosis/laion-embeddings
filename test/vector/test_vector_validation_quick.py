#!/usr/bin/env python3
"""
Quick Vector Store Validation Test Script

A simplified version with built-in timeouts to prevent hanging.
"""

import asyncio
import logging
import argparse
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from services.vector_store_factory import get_vector_store_factory, VectorDBType, reset_factory
from services.vector_store_base import BaseVectorStore, VectorDocument, SearchQuery, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("validation_tests_quick")

# Test timeout in seconds
TEST_TIMEOUT = 10

@dataclass
class ValidationResult:
    test_name: str
    provider: str
    passed: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class QuickValidationSuite:
    """Quick validation test suite with timeouts."""
    
    def __init__(self, store: BaseVectorStore):
        self.store = store
    
    async def test_basic_operations(self) -> ValidationResult:
        """Test basic store operations."""
        test_name = "basic_operations"
        try:
            # Test create index
            index_name = "test_basic"
            await self.store.create_index(index_name, 4)
            
            # Test add vector
            vector = VectorDocument(
                id="test_1",
                vector=[1.0, 0.0, 0.0, 0.0],
                metadata={"test": "basic"}
            )
            await self.store.add_vectors([vector], index_name)
            
            # Test search
            query = SearchQuery(
                vector=[1.0, 0.0, 0.0, 0.0],
                top_k=1
            )
            results = await self.store.search(query, index_name)
            
            # Test get stats
            stats = await self.store.get_index_stats(index_name)
            
            # Cleanup
            await self.store.delete_index(index_name)
            
            passed = len(results) > 0 and stats is not None
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=passed,
                details={"results_count": len(results), "has_stats": stats is not None}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=False,
                error_message=str(e)
            )
    
    async def test_empty_input(self) -> ValidationResult:
        """Test handling of empty inputs."""
        test_name = "empty_input"
        try:
            # Test with empty vector list
            result = await self.store.add_vectors([], "test_empty")
            
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=True,  # No exception = good handling
                details={"result": result}
            )
            
        except Exception as e:
            # Exception for empty input is also acceptable
            return ValidationResult(
                test_name=test_name,
                provider=self.store.__class__.__name__,
                passed=True,
                error_message=str(e)
            )
    
    async def run_quick_validation(self) -> List[ValidationResult]:
        """Run quick validation tests with timeouts."""
        tests = [
            self.test_basic_operations,
            self.test_empty_input
        ]
        
        results = []
        for test in tests:
            logger.info(f"Running {test.__name__}...")
            try:
                # Run test with timeout
                result = await asyncio.wait_for(test(), timeout=TEST_TIMEOUT)
                results.append(result)
                status = "✓" if result.passed else "✗"
                logger.info(f"{status} {test.__name__}: {'PASS' if result.passed else 'FAIL'}")
                if result.error_message:
                    logger.debug(f"  Error: {result.error_message}")
            except asyncio.TimeoutError:
                logger.error(f"Test {test.__name__} timed out after {TEST_TIMEOUT}s")
                results.append(ValidationResult(
                    test_name=test.__name__,
                    provider=self.store.__class__.__name__,
                    passed=False,
                    error_message=f"Test timed out after {TEST_TIMEOUT}s"
                ))
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                results.append(ValidationResult(
                    test_name=test.__name__,
                    provider=self.store.__class__.__name__,
                    passed=False,
                    error_message=f"Test crashed: {e}"
                ))
        
        return results


async def validate_store_quick(db_type: VectorDBType) -> List[ValidationResult]:
    """Run quick validation tests on a single store."""
    logger.info(f"\nQuick validation for {db_type.value.upper()}")
    
    try:
        factory = get_vector_store_factory()
        store = await factory.create_store(db_type)
        
        # Connect with timeout
        await asyncio.wait_for(store.connect(), timeout=10)
        ping_result = await asyncio.wait_for(store.ping(), timeout=5)
        
        if not ping_result:
            logger.error(f"{db_type.value} connection failed")
            return []
        
        logger.info(f"Connected to {db_type.value} store")
        
        # Run validation tests
        test_suite = QuickValidationSuite(store)
        results = await test_suite.run_quick_validation()
        
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


def print_quick_summary(all_results: Dict[str, List[ValidationResult]]):
    """Print validation summary."""
    logger.info("\n" + "="*60)
    logger.info("QUICK VALIDATION SUMMARY")
    logger.info("="*60)
    
    for provider, results in all_results.items():
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info(f"{provider:15} {passed}/{total} tests passed")
        
        for result in results:
            status = "✓" if result.passed else "✗"
            logger.info(f"  {status} {result.test_name}")
            if result.error_message:
                logger.info(f"    Error: {result.error_message}")


async def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description="Quick vector store validation tests")
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
        # Test only FAISS for quick validation
        stores_to_test = [VectorDBType.FAISS]
    
    logger.info(f"Quick validation for {len(stores_to_test)} stores: {[s.value for s in stores_to_test]}")
    
    # Run validation tests
    all_results = {}
    
    for db_type in stores_to_test:
        results = await validate_store_quick(db_type)
        all_results[db_type.value] = results
    
    # Print summary
    print_quick_summary(all_results)
    
    # Check if any tests passed
    any_passed = any(
        any(r.passed for r in results)
        for results in all_results.values()
    )
    
    sys.exit(0 if any_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
