#!/bin/bash

# Create a directory for test results
mkdir -p test_results

# Set test environment variables
export TESTING=true
export PYTHONPATH=.

# Run individual test classes
echo "Testing IPFSVectorStorage..."
pytest test/test_ipfs_vector_service.py::TestIPFSVectorStorage -v > test_results/storage_test.txt 2>&1
echo "IPFSVectorStorage exit code: $?" >> test_results/storage_test.txt

echo "Testing TestDistributedVectorIndex::test_add_vectors_distributed..."
pytest test/test_ipfs_vector_service.py::TestDistributedVectorIndex::test_add_vectors_distributed -v > test_results/add_distributed_test.txt 2>&1
echo "test_add_vectors_distributed exit code: $?" >> test_results/add_distributed_test.txt

echo "Testing TestDistributedVectorIndex::test_search_distributed..."
pytest test/test_ipfs_vector_service.py::TestDistributedVectorIndex::test_search_distributed -v > test_results/search_distributed_test.txt 2>&1
echo "test_search_distributed exit code: $?" >> test_results/search_distributed_test.txt

echo "Testing TestIPFSIntegration..."
pytest test/test_ipfs_vector_service.py::TestIPFSIntegration -v > test_results/integration_test.txt 2>&1
echo "TestIPFSIntegration exit code: $?" >> test_results/integration_test.txt

# Print out the failing test details
echo "=== FAILING TEST OUTPUT ==="
echo "test_add_vectors_distributed error:"
grep -A 20 "FAILED" test_results/add_distributed_test.txt

echo "test_search_distributed error:"
grep -A 20 "FAILED" test_results/search_distributed_test.txt
