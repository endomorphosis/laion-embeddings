#!/bin/bash

# Run tests with environment variables set and generate output
cd /home/barberb/laion-embeddings-1
TESTING=true pytest test/test_ipfs_vector_service.py::TestIPFSVectorStorage -v > ipfs_test_results.txt 2>&1
echo "Test exit code: $?" >> ipfs_test_results.txt
echo "" >> ipfs_test_results.txt

# Show the output
echo "===== TEST RESULTS ====="
cat ipfs_test_results.txt
