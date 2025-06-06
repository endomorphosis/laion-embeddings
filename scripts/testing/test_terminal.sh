#!/bin/bash
echo "Testing terminal functionality..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Testing Python import..."
python -c "
import sys
print('Python import test successful')
print('Python path:', sys.executable)
try:
    import asyncio
    print('Asyncio available')
except:
    print('Asyncio not available')
"
echo "Test completed successfully"
