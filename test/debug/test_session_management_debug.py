#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/home/barberb/laion-embeddings-1')

try:
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/test_mcp_tools/test_session_management_tools.py', 
        '-x', '--tb=short'
    ], capture_output=True, text=True, timeout=30, env=dict(os.environ, **{
        'VIRTUAL_ENV': '/home/barberb/laion-embeddings-1/.venv',
        'PATH': '/home/barberb/laion-embeddings-1/.venv/bin:' + os.environ.get('PATH', '')
    }))
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")

except subprocess.TimeoutExpired:
    print("Test execution timed out")
except Exception as e:
    print(f"Error: {e}")
