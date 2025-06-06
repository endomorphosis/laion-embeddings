#!/usr/bin/env python3
"""
Test Summary Report
"""
import subprocess
import sys
import time

def run_test_file(test_file):
    """Run a single test file and return results"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, 
            '--tb=no', '-q', '--disable-warnings'
        ], capture_output=True, text=True, timeout=20)
        
        output = result.stdout.strip()
        if result.returncode == 0:
            if 'passed' in output:
                return f"✅ {test_file}: {output}"
            else:
                return f"✅ {test_file}: PASSED (no output)"
        else:
            return f"❌ {test_file}: FAILED - {output}"
    except subprocess.TimeoutExpired:
        return f"⏰ {test_file}: TIMEOUT"
    except Exception as e:
        return f"💥 {test_file}: ERROR - {str(e)}"

def main():
    test_files = [
        'test/test_simple_debug.py',
        'test/test_simple.py', 
        'test/test_isolated_units.py',
        'test/test_vector_service.py',
        'test/test_ipfs_vector_service.py'
    ]
    
    print("🧪 PYTEST STATUS REPORT")
    print("=" * 50)
    
    total_files = len(test_files)
    passed_files = 0
    
    for test_file in test_files:
        result = run_test_file(test_file)
        print(result)
        if result.startswith("✅"):
            passed_files += 1
        time.sleep(0.5)  # Brief pause between tests
    
    print("=" * 50)
    print(f"📊 SUMMARY: {passed_files}/{total_files} test files passing")
    
    if passed_files == total_files:
        print("🎉 ALL TESTS PASSING!")
        return 0
    else:
        print(f"⚠️  {total_files - passed_files} test files need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
