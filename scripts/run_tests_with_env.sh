#!/bin/bash

# Set environment variables for testing
export TESTING=true

# Run all tests
echo "Running tests with TESTING=true environment..."
python -m pytest test/test_*service*.py test/test_simple.py -v
