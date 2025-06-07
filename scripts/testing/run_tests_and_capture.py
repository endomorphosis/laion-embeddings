#!/usr/bin/env python3
"""
Script to run pytest and capture results to a file
"""
import subprocess
import sys
import os

def run_tests():
    """Run pytest and capture output"""
    try:
        # Change to project directory
        os.chdir('/home/barberb/laion-embeddings-1')
        
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', '-v', '--tb=short', '--no-header'
        ], capture_output=True, text=True, timeout=300)
        
        # Write results to file
        with open('test_results_capture.txt', 'w') as f:
            f.write("=== PYTEST STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== PYTEST STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")
        
        print(f"Tests completed with return code: {result.returncode}")
        print("Output saved to test_results_capture.txt")
        
        return result
        
    except subprocess.TimeoutExpired:
        print("Tests timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"Error running tests: {e}")
        return None

if __name__ == "__main__":
    run_tests()
