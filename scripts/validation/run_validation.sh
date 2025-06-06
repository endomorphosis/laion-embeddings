#!/bin/bash
cd /home/barberb/laion-embeddings-1
export PYTHONPATH="/home/barberb/laion-embeddings-1"
python3 validate_mcp_server.py > mcp_validation_results.txt 2>&1
echo "Validation completed. Results saved to mcp_validation_results.txt"
