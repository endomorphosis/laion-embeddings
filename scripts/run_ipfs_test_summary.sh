#!/bin/bash

# Set the environment variables for testing
export TESTING=true
export PYTHONPATH=.

# Print environment information
echo "===== Environment Information ====="
python --version
echo "TESTING=$TESTING"
echo "PYTHONPATH=$PYTHONPATH"
echo ""

# Run the tests
echo "===== Running IPFS Tests ====="
python -m pytest test/test_ipfs_vector_service.py -v > ipfs_test_results.txt 2>&1
TEST_EXIT_CODE=$?

echo "Test run complete. Exit code: $TEST_EXIT_CODE"
echo ""

# Calculate statistics
PASSED_COUNT=$(grep "PASSED" ipfs_test_results.txt | wc -l)
FAILED_COUNT=$(grep "FAILED" ipfs_test_results.txt | wc -l)

echo "===== Test Results ====="
echo "Passed: $PASSED_COUNT"
echo "Failed: $FAILED_COUNT"
echo ""

# Show failing tests
if [ $FAILED_COUNT -gt 0 ]; then
    echo "===== Failing Tests ====="
    grep "FAILED" ipfs_test_results.txt
    echo ""
fi

# Show all test results
echo "===== All Test Results ====="
cat ipfs_test_results.txt

exit $TEST_EXIT_CODE
