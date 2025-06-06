#!/usr/bin/env python3
"""
Comprehensive Vector Store Test Suite

This script runs all vector store tests including basic functionality,
advanced features, and integration tests.
"""

import asyncio
import logging
import sys
import argparse
from typing import Dict, Any, List
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("comprehensive_test")


class TestSuite:
    """Comprehensive test suite for vector stores."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    async def run_basic_tests(self, specific_store=None, with_data=False):
        """Run basic functionality tests."""
        logger.info("Running basic vector store tests...")
        
        cmd = ["python", "test_vector_stores.py"]
        if specific_store:
            cmd.extend(["--store", specific_store])
        if with_data:
            cmd.append("--data")
        cmd.append("--factory")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            success = result.returncode == 0
            self.results["basic"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Basic tests passed")
            else:
                logger.error("✗ Basic tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Basic tests timed out")
            self.results["basic"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Basic tests failed to run: {e}")
            self.results["basic"] = {"success": False, "error": str(e)}
            return False
    
    async def run_advanced_tests(self, specific_store=None):
        """Run advanced feature tests."""
        logger.info("Running advanced vector store tests...")
        
        cmd = ["python", "test_vector_advanced.py"]
        if specific_store:
            cmd.extend(["--store", specific_store])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            success = result.returncode == 0
            self.results["advanced"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Advanced tests passed")
            else:
                logger.error("✗ Advanced tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Advanced tests timed out")
            self.results["advanced"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Advanced tests failed to run: {e}")
            self.results["advanced"] = {"success": False, "error": str(e)}
            return False
    
    async def run_integration_tests(self):
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        cmd = ["python", "test_vector_integration.py"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            self.results["integration"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Integration tests passed")
            else:
                logger.error("✗ Integration tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Integration tests timed out")
            self.results["integration"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Integration tests failed to run: {e}")
            self.results["integration"] = {"success": False, "error": str(e)}
            return False
    
    async def run_performance_tests(self):
        """Run performance tests."""
        logger.info("Running performance tests...")
        
        cmd = ["python", "test_vector_advanced.py", "--performance", "--dimension", "128", "--count", "500"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            success = result.returncode == 0
            self.results["performance"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Performance tests passed")
            else:
                logger.error("✗ Performance tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Performance tests timed out")
            self.results["performance"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Performance tests failed to run: {e}")
            self.results["performance"] = {"success": False, "error": str(e)}
            return False
    
    async def run_validation_tests(self):
        """Run validation tests."""
        logger.info("Running validation tests...")
        
        cmd = ["python", "test_vector_validation.py"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            self.results["validation"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Validation tests passed")
            else:
                logger.error("✗ Validation tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Validation tests timed out")
            self.results["validation"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Validation tests failed to run: {e}")
            self.results["validation"] = {"success": False, "error": str(e)}
            return False
    
    async def run_benchmark_tests(self):
        """Run benchmark tests."""
        logger.info("Running benchmark tests...")
        
        cmd = ["python", "test_vector_benchmarks.py"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            success = result.returncode == 0
            self.results["benchmarks"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Benchmark tests passed")
            else:
                logger.error("✗ Benchmark tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Benchmark tests timed out")
            self.results["benchmarks"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Benchmark tests failed to run: {e}")
            self.results["benchmarks"] = {"success": False, "error": str(e)}
            return False
    
    async def run_security_tests(self):
        """Run security tests."""
        logger.info("Running security tests...")
        
        cmd = ["python", "test_vector_security.py"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            self.results["security"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Security tests passed")
            else:
                logger.error("✗ Security tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Security tests timed out")
            self.results["security"] = {"success": False, "error": "timeout"}
            return False
        except Exception as e:
            logger.error(f"✗ Security tests failed to run: {e}")
            self.results["security"] = {"success": False, "error": str(e)}
            return False
    
    async def run_integrity_tests(self):
        """Run data integrity tests."""
        logger.info("Running data integrity tests...")
        
        cmd = ["python", "test_vector_integrity.py"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            self.results["integrity"] = {
                "success": success,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
            
            if success:
                logger.info("✓ Data integrity tests passed")
            else:
                logger.error("✗ Data integrity tests failed")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("✗ Data integrity tests timed out")
            self.results["integrity"] = {"success": False, "error": "timeout"}
            return False
    
    def print_summary(self):
        """Print test summary."""
        elapsed = time.time() - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("VECTOR STORE TEST SUITE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total execution time: {elapsed:.2f} seconds")
        logger.info("")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.results.items():
            status = "PASS" if result["success"] else "FAIL"
            logger.info(f"{test_name.upper():15} {status}")
            if not result["success"] and result.get("error"):
                logger.info(f"                Error: {result['error']}")
        
        logger.info("")
        logger.info(f"Overall: {passed_tests}/{total_tests} test suites passed")
        
        if failed_tests == 0:
            logger.info("🎉 All test suites completed successfully!")
        else:
            logger.error(f"❌ {failed_tests} test suite(s) failed")
        
        return failed_tests == 0


async def main():
    parser = argparse.ArgumentParser(description="Run comprehensive vector store tests")
    parser.add_argument("--store", "-s", 
                        help="Test specific store only (qdrant, elasticsearch, pgvector, faiss, ipfs, duckdb)")
    parser.add_argument("--basic", "-b", 
                        help="Run only basic tests", 
                        action="store_true")
    parser.add_argument("--advanced", "-a", 
                        help="Run only advanced tests", 
                        action="store_true")
    parser.add_argument("--integration", "-i", 
                        help="Run only integration tests", 
                        action="store_true")
    parser.add_argument("--performance", "-p", 
                        help="Run only performance tests", 
                        action="store_true")
    parser.add_argument("--validation", "-v", 
                        help="Run only validation tests", 
                        action="store_true")
    parser.add_argument("--benchmarks", 
                        help="Run only benchmark tests", 
                        action="store_true")
    parser.add_argument("--security", 
                        help="Run only security tests", 
                        action="store_true")
    parser.add_argument("--integrity", 
                        help="Run only integrity tests", 
                        action="store_true")
    parser.add_argument("--data", "-d", 
                        help="Include data operations in basic tests", 
                        action="store_true")
    parser.add_argument("--quick", "-q", 
                        help="Run quick test suite (basic + integration)", 
                        action="store_true")
    parser.add_argument("--full", "-f", 
                        help="Run full test suite (all tests)", 
                        action="store_true")
    args = parser.parse_args()
    
    suite = TestSuite()
    overall_success = True
    
    # Determine which tests to run
    any_specific = any([args.basic, args.advanced, args.integration, args.performance, 
                       args.validation, args.benchmarks, args.security, args.integrity])
    
    run_basic = args.basic or args.quick or args.full or not any_specific
    run_advanced = args.advanced or args.full or not any_specific
    run_integration = args.integration or args.quick or args.full or not any_specific
    run_performance = args.performance or args.full or not any_specific
    run_validation = args.validation or args.full
    run_benchmarks = args.benchmarks or args.full
    run_security = args.security or args.full
    run_integrity = args.integrity or args.full
    
    logger.info("Starting comprehensive vector store test suite...")
    test_plan = f"basic={run_basic}, advanced={run_advanced}, integration={run_integration}, performance={run_performance}"
    test_plan += f", validation={run_validation}, benchmarks={run_benchmarks}, security={run_security}, integrity={run_integrity}"
    logger.info(f"Test plan: {test_plan}")
    
    if args.store:
        logger.info(f"Testing specific store: {args.store}")
    
    # Run tests in logical order
    if run_basic:
        success = await suite.run_basic_tests(args.store, args.data)
        overall_success &= success
    
    if run_integration:
        success = await suite.run_integration_tests()
        overall_success &= success
    
    if run_advanced:
        success = await suite.run_advanced_tests(args.store)
        overall_success &= success
    
    if run_performance:
        success = await suite.run_performance_tests()
        overall_success &= success
    
    if run_validation:
        success = await suite.run_validation_tests()
        overall_success &= success
    
    if run_benchmarks:
        success = await suite.run_benchmark_tests()
        overall_success &= success
    
    if run_security:
        success = await suite.run_security_tests()
        overall_success &= success
    
    if run_integrity:
        success = await suite.run_integrity_tests()
        overall_success &= success
    
    # Print summary
    final_success = suite.print_summary()
    overall_success &= final_success
    
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    asyncio.run(main())
