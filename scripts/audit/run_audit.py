import subprocess
import sys

try:
    result = subprocess.run([sys.executable, "mcp_final_audit_report.py"], 
                          capture_output=True, text=True, cwd="/home/barberb/laion-embeddings-1")
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")
    
except Exception as e:
    print(f"Error: {e}")
