#!/usr/bin/env python3
"""
Debug MCP Server - Check if it starts at all
"""

import subprocess
import sys
import os
import time

def debug_server():
    print("🔍 Debugging MCP Server startup...")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/barberb/laion-embeddings-1'
    
    print("Starting server process...")
    process = subprocess.Popen(
        [sys.executable, 'mcp_server_minimal.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd='/home/barberb/laion-embeddings-1'
    )
    
    print(f"Process PID: {process.pid}")
    
    # Wait a bit
    time.sleep(2)
    
    # Check if process is still running
    poll_result = process.poll()
    if poll_result is not None:
        print(f"❌ Process exited with code: {poll_result}")
        stderr_output = process.stderr.read()
        stdout_output = process.stdout.read()
        print(f"STDERR: {stderr_output}")
        print(f"STDOUT: {stdout_output}")
    else:
        print("✅ Process is still running")
        
        # Try sending a simple message
        try:
            print("Sending test message...")
            process.stdin.write('{"test": "message"}\n')
            process.stdin.flush()
            
            # Wait a bit more
            time.sleep(1)
            
            # Check stderr for any output
            import select
            if select.select([process.stderr], [], [], 1.0)[0]:
                stderr_output = process.stderr.read()
                print(f"STDERR: {stderr_output}")
                
        except Exception as e:
            print(f"Error sending message: {e}")
        
        # Terminate
        process.terminate()
        try:
            process.wait(timeout=5)
        except:
            process.kill()

if __name__ == "__main__":
    debug_server()
