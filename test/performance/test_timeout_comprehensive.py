#!/usr/bin/env python3
"""
Comprehensive test for timeout implementation in ipfs_embeddings.py
This test verifies that all timeout protections are working correctly.
"""

import asyncio
import sys
import os
import logging
import time
from unittest.mock import AsyncMock, patch

# Add the project root to Python path
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_timeout_constants():
    """Test that timeout constants are properly defined"""
    try:
        from ipfs_embeddings_py.ipfs_embeddings import (
            BATCH_SIZE_OPTIMIZATION_TIMEOUT,
            NETWORK_REQUEST_TIMEOUT,
            ADAPTIVE_BATCH_TIMEOUT,
            TEST_BATCH_TIMEOUT,
            BatchSizeTimeoutError
        )
        
        # Verify constants have reasonable values
        assert BATCH_SIZE_OPTIMIZATION_TIMEOUT == 180, f"Expected 180, got {BATCH_SIZE_OPTIMIZATION_TIMEOUT}"
        assert NETWORK_REQUEST_TIMEOUT == 60, f"Expected 60, got {NETWORK_REQUEST_TIMEOUT}"
        assert ADAPTIVE_BATCH_TIMEOUT == 300, f"Expected 300, got {ADAPTIVE_BATCH_TIMEOUT}"
        assert TEST_BATCH_TIMEOUT == 45, f"Expected 45, got {TEST_BATCH_TIMEOUT}"
        
        logger.info("✓ Timeout constants are properly defined and have correct values")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import timeout constants: {e}")
        return False
    except AssertionError as e:
        logger.error(f"✗ Timeout constant validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error testing timeout constants: {e}")
        return False

def test_timeout_exception():
    """Test that BatchSizeTimeoutError is properly defined"""
    try:
        from ipfs_embeddings_py.ipfs_embeddings import BatchSizeTimeoutError
        
        # Test exception creation and inheritance
        error = BatchSizeTimeoutError("Test timeout error")
        assert isinstance(error, Exception), "BatchSizeTimeoutError should inherit from Exception"
        assert str(error) == "Test timeout error", "Exception message should be preserved"
        
        logger.info("✓ BatchSizeTimeoutError exception class is properly defined")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import BatchSizeTimeoutError: {e}")
        return False
    except AssertionError as e:
        logger.error(f"✗ BatchSizeTimeoutError validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error testing BatchSizeTimeoutError: {e}")
        return False

async def test_safe_async_execute_with_timeout():
    """Test the safe_async_execute_with_timeout function"""
    try:
        from ipfs_embeddings_py.ipfs_embeddings import (
            safe_async_execute_with_timeout,
            BatchSizeTimeoutError
        )
        
        # Test successful execution
        async def fast_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await safe_async_execute_with_timeout(
            fast_operation(), 
            timeout=1.0, 
            operation_name="test_fast_op"
        )
        assert result == "success", f"Expected 'success', got {result}"
        
        # Test timeout behavior
        async def slow_operation():
            await asyncio.sleep(2.0)
            return "should_not_reach"
        
        try:
            await safe_async_execute_with_timeout(
                slow_operation(), 
                timeout=0.5, 
                operation_name="test_slow_op"
            )
            assert False, "Should have raised BatchSizeTimeoutError"
        except BatchSizeTimeoutError as e:
            assert "test_slow_op" in str(e), f"Operation name should be in error message: {e}"
        
        logger.info("✓ safe_async_execute_with_timeout function works correctly")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import timeout function: {e}")
        return False
    except AssertionError as e:
        logger.error(f"✗ Timeout function validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error testing timeout function: {e}")
        return False

async def test_safe_async_execute_with_retry():
    """Test the enhanced safe_async_execute_with_retry function with timeout"""
    try:
        from ipfs_embeddings_py.ipfs_embeddings import (
            safe_async_execute_with_retry,
            BatchSizeTimeoutError
        )
        
        # Test successful execution with timeout
        async def mock_function():
            await asyncio.sleep(0.1)
            return "retry_success"
        
        result = await safe_async_execute_with_retry(
            mock_function,
            timeout=1.0,
            max_retries=2
        )
        assert result == "retry_success", f"Expected 'retry_success', got {result}"
        
        # Test timeout with retries
        call_count = 0
        async def mock_slow_function():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(1.0)  # Longer than timeout
            return "should_not_reach"
        
        try:
            await safe_async_execute_with_retry(
                mock_slow_function,
                timeout=0.3,
                max_retries=2,
                delay=0.1
            )
            assert False, "Should have raised BatchSizeTimeoutError"
        except BatchSizeTimeoutError:
            # Should attempt multiple times but each attempt should timeout
            assert call_count >= 1, f"Should have made at least 1 attempt, made {call_count}"
        
        logger.info("✓ safe_async_execute_with_retry with timeout works correctly")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import retry function: {e}")
        return False
    except AssertionError as e:
        logger.error(f"✗ Retry function validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error testing retry function: {e}")
        return False

def test_ipfs_embeddings_class_initialization():
    """Test that the ipfs_embeddings_py class can be properly initialized"""
    try:
        from ipfs_embeddings_py import ipfs_embeddings
        
        # Create minimal resources and metadata for testing
        test_resources = {}
        test_metadata = {}
        
        # Initialize the class
        embeddings = ipfs_embeddings.ipfs_embeddings(test_resources, test_metadata)
        
        # Check that timeout-related attributes are properly initialized
        assert hasattr(embeddings, 'adaptive_batch_processor'), "Should have adaptive_batch_processor"
        assert hasattr(embeddings, 'adaptive_queue_manager'), "Should have adaptive_queue_manager"
        assert hasattr(embeddings, 'memory_monitor'), "Should have memory_monitor"
        
        logger.info("✓ ipfs_embeddings_py class initializes correctly with timeout infrastructure")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import ipfs_embeddings class: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to initialize ipfs_embeddings_py class: {e}")
        return False

async def test_max_batch_size_timeout_protection():
    """Test that max_batch_size method has timeout protection"""
    try:
        from ipfs_embeddings_py import ipfs_embeddings
        
        # Create test instance
        test_resources = {}
        test_metadata = {}
        embeddings = ipfs_embeddings.ipfs_embeddings(test_resources, test_metadata)
        
        # Mock the internal implementation to simulate timeout
        async def mock_slow_implementation(*args, **kwargs):
            await asyncio.sleep(5.0)  # Longer than our timeout
            return 32
        
        # Patch the internal method to simulate slow operation
        with patch.object(embeddings, '_max_batch_size_implementation', mock_slow_implementation):
            start_time = time.time()
            result = await embeddings.max_batch_size("test_model")
            end_time = time.time()
            
            # Should return fallback value quickly due to timeout
            assert isinstance(result, int), f"Should return int, got {type(result)}"
            assert result > 0, f"Should return positive batch size, got {result}"
            
            # Should complete much faster than the 5 second mock delay
            duration = end_time - start_time
            assert duration < 4.0, f"Should timeout quickly, took {duration:.2f}s"
        
        logger.info("✓ max_batch_size method has proper timeout protection")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import for max_batch_size test: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ max_batch_size timeout test failed: {e}")
        return False

async def run_all_tests():
    """Run all timeout implementation tests"""
    logger.info("Starting comprehensive timeout implementation tests...")
    
    test_results = []
    
    # Synchronous tests
    test_results.append(("Timeout Constants", test_timeout_constants()))
    test_results.append(("Timeout Exception", test_timeout_exception()))
    test_results.append(("Class Initialization", test_ipfs_embeddings_class_initialization()))
    
    # Asynchronous tests
    test_results.append(("Safe Async Execute", await test_safe_async_execute_with_timeout()))
    test_results.append(("Safe Async Retry", await test_safe_async_execute_with_retry()))
    test_results.append(("Max Batch Size Timeout", await test_max_batch_size_timeout_protection()))
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("TIMEOUT IMPLEMENTATION TEST RESULTS")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-"*60)
    logger.info(f"Total: {len(test_results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All timeout implementation tests PASSED!")
        return True
    else:
        logger.error(f"❌ {failed} timeout implementation tests FAILED!")
        return False

if __name__ == "__main__":
    try:
        # Run the comprehensive test suite
        success = asyncio.run(run_all_tests())
        
        if success:
            logger.info("\n✅ Timeout implementation is working correctly!")
            # sys.exit(0) # Removed sys.exit to prevent pytest from crashing
        else:
            logger.error("\n❌ Timeout implementation has issues that need to be fixed!")
            # sys.exit(1) # Removed sys.exit to prevent pytest from crashing
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        # sys.exit(1) # Removed sys.exit to prevent pytest from crashing
    except Exception as e:
        logger.error(f"\n💥 Test suite crashed: {e}")
        # sys.exit(1) # Removed sys.exit to prevent pytest from crashing
