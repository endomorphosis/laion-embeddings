#!/usr/bin/env python3
"""
Master Vector Store Test Suite

This script orchestrates all vector store tests including basic functionality,
advanced features, integration tests, performance benchmarks, and validation tests.
"""

import subprocess
import sys
import time
import logging
import argparse
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("master_test_suite")


class TestSuite:
    """Represents a test suite with its script and description."""
    def __init__(self, name: str, script: str, description: str, default_enabled: bool = True):
        self.name = name
        self.script = script
        self.description = description
        self.default_enabled = default_enabled


# Define all available test suites
TEST_SUITES = [
    TestSuite("BASIC", "test_vector_stores.py", "Basic vector store functionality", True),
    TestSuite("INTEGRATION", "test_vector_integration.py", "Integration and factory tests", True),
    TestSuite("ADVANCED", "test_vector_advanced.py", "Advanced features (quantization, sharding)", True),
    TestSuite("BENCHMARKS", "test_vector_benchmarks.py", "Comprehensive benchmarks", False),
    TestSuite("VALIDATION", "test_vector_validation.py", "Edge cases and validation", False),
]


def run_test_script(script_path: str, args: List[str] = None) -> Tuple[bool, str]:
    """Run a test script and return success status and output."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for benchmarks
        )
        
        return result.returncode == 0, result.stderr if result.stderr else result.stdout
        
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 10 minutes"
    except Exception as e:
        return False, f"Failed to run test: {e}"


def run_test_suites(test_plan: Dict[str, bool], specific_store: str = None) -> Dict[str, Tuple[bool, str]]:
    """Run all enabled tests according to the test plan."""
    results = {}
    
    for suite in TEST_SUITES:
        if not test_plan.get(suite.name.lower(), False):
            continue
            
        logger.info(f"Running {suite.description.lower()}...")
        
        # Prepare arguments
        args = []
        if specific_store:
            args.extend(["--store", specific_store])
        
        # Special handling for quick mode in benchmarks
        if suite.name == "BENCHMARKS" and test_plan.get("quick", False):
            args.append("--quick")
        
        success, output = run_test_script(suite.script, args)
        results[suite.name] = (success, output)
        
        if success:
            logger.info(f"✓ {suite.description} passed")
        else:
            logger.error(f"✗ {suite.description} failed")
            # Log first few lines of error for debugging
            error_lines = output.split('\n')[:3]
            for line in error_lines:
                if line.strip():
                    logger.error(f"                 Error: {line.strip()}")
    
    return results


def print_summary(results: Dict[str, Tuple[bool, str]], total_time: float):
    """Print a comprehensive test summary."""
    logger.info(f"\n{'='*80}")
    logger.info("MASTER VECTOR STORE TEST SUITE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("")
    
    passed_count = 0
    total_count = len(results)
    
    for suite_name, (success, output) in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"{suite_name:<15} {status}")
        
        if not success and output:
            # Show brief error info
            error_lines = output.split('\n')
            # Find the most relevant error line
            for line in error_lines[:10]:
                if any(keyword in line.lower() for keyword in ['error', 'traceback', 'failed', 'exception']):
                    logger.info(f"                 Error: {line.strip()}")
                    break
        
        if success:
            passed_count += 1
    
    logger.info("")
    logger.info(f"Overall: {passed_count}/{total_count} test suites passed")
    
    if passed_count == total_count:
        logger.info("🎉 All test suites completed successfully!")
        return True
    else:
        logger.error(f"❌ {total_count - passed_count} test suite(s) failed")
        return False


def main():
    """Main test orchestrator."""
    parser = argparse.ArgumentParser(description="Master vector store test suite")
    parser.add_argument("--store", "-s", 
                        help="Test specific store only (faiss, ipfs, duckdb, etc.)",
                        default=None)
    parser.add_argument("--basic", action="store_true",
                        help="Run basic tests only")
    parser.add_argument("--advanced", action="store_true", 
                        help="Run advanced tests only")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests only")
    parser.add_argument("--benchmarks", action="store_true",
                        help="Run benchmark tests")
    parser.add_argument("--validation", action="store_true",
                        help="Run validation tests")
    parser.add_argument("--all", action="store_true",
                        help="Run all available test suites")
    parser.add_argument("--extended", action="store_true",
                        help="Run extended test suite (includes benchmarks and validation)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test suite (basic + integration)")
    
    args = parser.parse_args()
    
    # Determine test plan
    test_plan = {}
    
    if args.all:
        # Run all test suites
        for suite in TEST_SUITES:
            test_plan[suite.name.lower()] = True
    elif args.extended:
        # Extended test suite
        test_plan = {"basic": True, "integration": True, "advanced": True, "benchmarks": True, "validation": True}
    elif args.quick:
        # Quick test suite
        test_plan = {"basic": True, "integration": True}
    elif any([args.basic, args.advanced, args.integration, args.benchmarks, args.validation]):
        # Specific test suites
        test_plan = {
            "basic": args.basic,
            "advanced": args.advanced,
            "integration": args.integration,
            "benchmarks": args.benchmarks,
            "validation": args.validation
        }
    else:
        # Default test plan (basic functionality)
        for suite in TEST_SUITES:
            test_plan[suite.name.lower()] = suite.default_enabled
    
    # Add quick flag to test plan for benchmark optimization
    if args.quick:
        test_plan["quick"] = True
    
    logger.info("Starting master vector store test suite...")
    enabled_tests = [name for name, enabled in test_plan.items() if enabled and name != "quick"]
    logger.info(f"Enabled test suites: {', '.join(enabled_tests)}")
    
    if args.store:
        logger.info(f"Testing specific store: {args.store}")
    
    # Run tests
    start_time = time.time()
    results = run_test_suites(test_plan, args.store)
    end_time = time.time()
    
    # Print summary
    success = print_summary(results, end_time - start_time)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
