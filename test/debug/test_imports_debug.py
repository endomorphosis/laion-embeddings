#!/usr/bin/env python3

import sys
sys.path.append('/home/barberb/laion-embeddings-1')

print("Testing imports...")

try:
    from src.mcp_server.tools.session_management_tools import create_session_tool
    print("✓ session_management_tools import OK")
except Exception as e:
    print(f"✗ session_management_tools import failed: {e}")

try:
    from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterManagementTool
    print("✓ ipfs_cluster_tools import OK")
except Exception as e:
    print(f"✗ ipfs_cluster_tools import failed: {e}")

try:
    from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
    print("✓ vector_store_tools import OK")
except Exception as e:
    print(f"✗ vector_store_tools import failed: {e}")

try:
    from src.mcp_server.tools.workflow_tools import execute_workflow_tool
    print("✓ workflow_tools import OK")
except Exception as e:
    print(f"✗ workflow_tools import failed: {e}")

print("Testing pytest collection...")
import subprocess
import os

os.chdir('/home/barberb/laion-embeddings-1')
env = dict(os.environ)
env['VIRTUAL_ENV'] = '/home/barberb/laion-embeddings-1/.venv'
env['PATH'] = '/home/barberb/laion-embeddings-1/.venv/bin:' + env.get('PATH', '')

try:
    result = subprocess.run([
        '/home/barberb/laion-embeddings-1/.venv/bin/python', '-m', 'pytest', 
        'tests/test_mcp_tools/test_session_management_tools.py', 
        '--collect-only', '-q'
    ], capture_output=True, text=True, timeout=30, env=env)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
except subprocess.TimeoutExpired:
    print("Pytest collection timed out")
except Exception as e:
    print(f"Error running pytest: {e}")
