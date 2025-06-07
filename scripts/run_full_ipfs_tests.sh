#!/bin/bash

# Run tests with environment variables set and generate output
cd /home/barberb/laion-embeddings-1
echo "Running all IPFS vector service tests..."
TESTING=true pytest test/test_ipfs_vector_service.py -v > full_ipfs_results.txt 2>&1
RESULT=$?
echo "Test exit code: $RESULT" >> full_ipfs_results.txt
echo "" >> full_ipfs_results.txt

# Show the output
echo "===== TEST RESULTS SUMMARY ====="
grep -A 1 "FAILED " full_ipfs_results.txt
echo ""
echo "Passed tests:"
grep "PASSED" full_ipfs_results.txt | wc -l
echo "Failed tests:" 
grep "FAILED" full_ipfs_results.txt | wc -l
echo ""
echo "Test exit code: $RESULT"
