#!/usr/bin/env python3
"""
Comprehensive test runner and status report for LAION embeddings project.

This script runs all available tests and provides a detailed status report.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os

# Set up test environment
os.environ['TESTING'] = 'true'
os.environ['DISABLE_TELEMETRY'] = 'true'

def run_command(cmd, timeout=60):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd='/home/barberb/laion-embeddings-1'
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': 124
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': 1
        }

def main():
    """Run comprehensive tests and generate report."""
    print("🚀 Starting Comprehensive Test Suite for LAION Embeddings Project")
    print("=" * 70)
    
    start_time = datetime.now()
    results = {}
    
    # Test categories
    test_suites = [
        {
            'name': 'Standalone Integration Tests',
            'command': 'python test_integration_standalone.py',
            'timeout': 120,
            'critical': True
        },
        {
            'name': 'Vector Service Unit Tests', 
            'command': 'python -m pytest test/test_vector_service.py -v --tb=short',
            'timeout': 60,
            'critical': True
        },
        {
            'name': 'IPFS Vector Service Unit Tests',
            'command': 'python -m pytest test/test_ipfs_vector_service.py -v --tb=short', 
            'timeout': 60,
            'critical': True
        },
        {
            'name': 'Clustering Service Unit Tests',
            'command': 'python -m pytest test/test_clustering_service.py -v --tb=short',
            'timeout': 60,
            'critical': True
        },
        {
            'name': 'Vector Service Integration Tests',
            'command': 'python -m pytest test/test_complete_integration.py::TestVectorServiceIntegration -v',
            'timeout': 90,
            'critical': True
        },
        {
            'name': 'Basic Import Tests',
            'command': 'python test_imports.py',
            'timeout': 30,
            'critical': False
        },
        {
            'name': 'Service Dependencies Check',
            'command': 'python -c "from services.vector_service import VectorService; from services.clustering_service import SmartShardingService; print(\\"All services import OK\\")"',
            'timeout': 30,
            'critical': True
        }
    ]
    
    # Run tests
    for i, test_suite in enumerate(test_suites, 1):
        print(f"\n[{i}/{len(test_suites)}] Running: {test_suite['name']}")
        print("-" * 50)
        
        result = run_command(test_suite['command'], test_suite['timeout'])
        results[test_suite['name']] = {
            'result': result,
            'critical': test_suite['critical'],
            'command': test_suite['command']
        }
        
        if result['success']:
            print(f"✅ PASSED")
            if result['stdout'] and 'passed' in result['stdout'].lower():
                # Extract test count if available
                if 'passed' in result['stdout']:
                    lines = result['stdout'].split('\\n')
                    for line in lines:
                        if 'passed' in line and ('warning' in line or 'error' in line):
                            print(f"   📊 {line.strip()}")
                            break
        else:
            status = "❌ FAILED" if test_suite['critical'] else "⚠️  FAILED (non-critical)"
            print(f"{status}")
            if result['stderr']:
                print(f"   🚫 Error: {result['stderr'][:200]}...")
            if result['returncode'] == 124:
                print(f"   ⏰ Timed out after {test_suite['timeout']} seconds")
    
    # Generate summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_suites)
    passed_tests = sum(1 for r in results.values() if r['result']['success'])
    critical_tests = len([t for t in test_suites if t['critical']])
    critical_passed = sum(1 for r in results.values() if r['critical'] and r['result']['success'])
    
    print(f"📈 Overall Results:")
    print(f"   Total Tests: {passed_tests}/{total_tests} passed")
    print(f"   Critical Tests: {critical_passed}/{critical_tests} passed")
    print(f"   Duration: {duration.total_seconds():.1f} seconds")
    
    print(f"\\n🔍 Detailed Results:")
    for name, data in results.items():
        status = "✅" if data['result']['success'] else ("❌" if data['critical'] else "⚠️")
        critical_text = " (Critical)" if data['critical'] else ""
        print(f"   {status} {name}{critical_text}")
    
    # Status determination
    if critical_passed == critical_tests:
        print(f"\\n🎉 SUCCESS: All critical tests passed!")
        print(f"   The LAION embeddings project is in good working condition.")
        overall_status = "SUCCESS"
    elif critical_passed >= critical_tests * 0.8:
        print(f"\\n⚠️  WARNING: Most critical tests passed ({critical_passed}/{critical_tests})")
        print(f"   The project is mostly functional but has some issues.")
        overall_status = "WARNING"
    else:
        print(f"\\n❌ FAILURE: Many critical tests failed ({critical_passed}/{critical_tests})")
        print(f"   The project needs significant fixes.")
        overall_status = "FAILURE"
    
    # Save detailed report
    report = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'overall_status': overall_status,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'critical_tests': critical_tests,
            'critical_passed': critical_passed
        },
        'test_results': results
    }
    
    report_file = Path('test_results') / f'comprehensive_test_report_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if overall_status == "SUCCESS" else (1 if overall_status == "WARNING" else 2)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
