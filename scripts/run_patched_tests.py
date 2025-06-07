#!/usr/bin/env python3

"""
Run tests with patches applied to fix test failures
"""

import os
import sys
import subprocess

# Set up the environment
os.environ['TESTING'] = 'true'
os.environ['PYTHONPATH'] = '.'

def main():
    """Main entry point"""
    print("===== Running IPFS Tests with Patches =====")

    # Import the test patches
    from test_patches import apply_test_patches, restore_methods
    
    try:
        # Apply all test patches
        print("Applying test patches...")
        original_methods = apply_test_patches()
        
        # Run the tests
        print("Running tests with patches applied...")
        result = subprocess.run(
            ["python", "-m", "pytest", "test/test_ipfs_vector_service.py", "-v"],
            check=False,
            capture_output=True,
            text=True
        )
        
        # Print test output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Calculate results
        passed_count = result.stdout.count("PASSED")
        failed_count = result.stdout.count("FAILED")
        
        print("\n===== Test Results =====")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Exit code: {result.returncode}")
        
        # Return the test exit code
        return result.returncode
        
    finally:
        # Clean up by restoring original methods
        try:
            print("Restoring original methods...")
            restore_methods(original_methods)
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    sys.exit(main())
