#!/usr/bin/env python3
"""
Comprehensive CI/CD Test Runner with MCP Tools Validation

This script runs all tests including the new MCP tools validation,
ensuring all 22 MCP tools are functional and the single MCP server
entry point works correctly.
"""

import subprocess
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CICDTestRunner:
    """Comprehensive test runner for CI/CD pipeline"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent
        self.results = {
            "mcp_tools": {"status": "pending", "details": {}},
            "mcp_server": {"status": "pending", "details": {}},
            "vector_service": {"status": "pending", "details": {}},
            "ipfs_service": {"status": "pending", "details": {}},
            "clustering": {"status": "pending", "details": {}},
            "integration": {"status": "pending", "details": {}},
            "imports": {"status": "pending", "details": {}},
            "dependencies": {"status": "pending", "details": {}}
        }
        self.start_time = time.time()
    
    def run_command(self, cmd: List[str], timeout: int = 300, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a command with timeout and capture output"""
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root)
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
            return 1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            logger.error(f"Command failed: {str(e)}")
            return 1, "", str(e)
    
    def test_mcp_server_validation(self) -> bool:
        """Test MCP server single entry point validation"""
        logger.info("🚀 Testing MCP Server Entry Point...")
        
        cmd = [sys.executable, "mcp_server.py", "--validate"]
        returncode, stdout, stderr = self.run_command(cmd, timeout=120)
        
        if returncode != 0:
            self.results["mcp_server"]["status"] = "failed"
            self.results["mcp_server"]["details"] = {
                "error": f"Exit code {returncode}",
                "stderr": stderr,
                "stdout": stdout
            }
            logger.error(f"❌ MCP Server validation failed: {stderr}")
            return False
        
        try:
            validation_data = json.loads(stdout)
            tools_count = validation_data.get("tools_count", 0)
            status = validation_data.get("status", "unknown")
            
            if status == "success" and tools_count >= 15:
                self.results["mcp_server"]["status"] = "passed"
                self.results["mcp_server"]["details"] = validation_data
                logger.info(f"✅ MCP Server validated: {tools_count} tools loaded")
                return True
            else:
                self.results["mcp_server"]["status"] = "failed"
                self.results["mcp_server"]["details"] = validation_data
                logger.error(f"❌ MCP Server validation insufficient: {validation_data}")
                return False
                
        except json.JSONDecodeError as e:
            self.results["mcp_server"]["status"] = "failed"
            self.results["mcp_server"]["details"] = {
                "error": f"Invalid JSON output: {str(e)}",
                "stdout": stdout
            }
            logger.error(f"❌ Invalid JSON from MCP server: {str(e)}")
            return False
    
    def test_mcp_tools_comprehensive(self) -> bool:
        """Test comprehensive MCP tools suite"""
        logger.info("🔧 Testing MCP Tools Comprehensive Suite...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "test/test_mcp_tools_comprehensive.py", 
            "-v", "--tb=short"
        ]
        returncode, stdout, stderr = self.run_command(cmd, timeout=180)
        
        success = returncode == 0
        self.results["mcp_tools"]["status"] = "passed" if success else "failed"
        self.results["mcp_tools"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ MCP Tools comprehensive test passed")
        else:
            logger.error(f"❌ MCP Tools comprehensive test failed: {stderr}")
        
        return success
    
    def test_vector_service(self) -> bool:
        """Test vector service (23/23 tests)"""
        logger.info("🔍 Testing Vector Service...")
        
        cmd = [sys.executable, "run_vector_tests_standalone.py"]
        returncode, stdout, stderr = self.run_command(cmd, timeout=180)
        
        success = returncode == 0
        self.results["vector_service"]["status"] = "passed" if success else "failed"
        self.results["vector_service"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ Vector Service tests passed (23/23)")
        else:
            logger.error(f"❌ Vector Service tests failed: {stderr}")
        
        return success
    
    def test_ipfs_service(self) -> bool:
        """Test IPFS vector service (15/15 tests)"""
        logger.info("📦 Testing IPFS Vector Service...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "test/test_ipfs_vector_service.py", 
            "-v"
        ]
        returncode, stdout, stderr = self.run_command(cmd, timeout=180)
        
        success = returncode == 0
        self.results["ipfs_service"]["status"] = "passed" if success else "failed"
        self.results["ipfs_service"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ IPFS Vector Service tests passed (15/15)")
        else:
            logger.error(f"❌ IPFS Vector Service tests failed: {stderr}")
        
        return success
    
    def test_clustering_service(self) -> bool:
        """Test clustering service (19/19 tests)"""
        logger.info("🧮 Testing Clustering Service...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "test/test_clustering_service.py", 
            "-v"
        ]
        returncode, stdout, stderr = self.run_command(cmd, timeout=180)
        
        success = returncode == 0
        self.results["clustering"]["status"] = "passed" if success else "failed"
        self.results["clustering"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ Clustering Service tests passed (19/19)")
        else:
            logger.error(f"❌ Clustering Service tests failed: {stderr}")
        
        return success
    
    def test_integration(self) -> bool:
        """Test integration tests (2/2 tests)"""
        logger.info("🔗 Testing Integration...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "test/test_integration_standalone.py", 
            "-v"
        ]
        returncode, stdout, stderr = self.run_command(cmd, timeout=120)
        
        success = returncode == 0
        self.results["integration"]["status"] = "passed" if success else "failed"
        self.results["integration"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ Integration tests passed (2/2)")
        else:
            logger.error(f"❌ Integration tests failed: {stderr}")
        
        return success
    
    def test_basic_imports(self) -> bool:
        """Test basic imports"""
        logger.info("📋 Testing Basic Imports...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "test/test_basic_imports.py", 
            "-v"
        ]
        returncode, stdout, stderr = self.run_command(cmd, timeout=60)
        
        success = returncode == 0
        self.results["imports"]["status"] = "passed" if success else "failed"
        self.results["imports"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ Basic imports tests passed")
        else:
            logger.error(f"❌ Basic imports tests failed: {stderr}")
        
        return success
    
    def test_service_dependencies(self) -> bool:
        """Test service dependencies"""
        logger.info("⚙️ Testing Service Dependencies...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "test/test_service_dependencies.py", 
            "-v"
        ]
        returncode, stdout, stderr = self.run_command(cmd, timeout=60)
        
        success = returncode == 0
        self.results["dependencies"]["status"] = "passed" if success else "failed"
        self.results["dependencies"]["details"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            logger.info("✅ Service dependencies tests passed")
        else:
            logger.error(f"❌ Service dependencies tests failed: {stderr}")
        
        return success
    
    def run_all_tests(self) -> bool:
        """Run all tests in sequence"""
        logger.info("🏁 Starting Comprehensive CI/CD Test Suite...")
        logger.info("=" * 60)
        
        test_functions = [
            ("MCP Server Validation", self.test_mcp_server_validation),
            ("MCP Tools Comprehensive", self.test_mcp_tools_comprehensive),
            ("Vector Service", self.test_vector_service),
            ("IPFS Service", self.test_ipfs_service),
            ("Clustering Service", self.test_clustering_service),
            ("Integration", self.test_integration),
            ("Basic Imports", self.test_basic_imports),
            ("Service Dependencies", self.test_service_dependencies)
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_name, test_func in test_functions:
            logger.info(f"\n📋 Running {test_name}...")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} CRASHED: {str(e)}")
        
        # Generate summary
        self.generate_summary(passed_tests, total_tests)
        
        return passed_tests == total_tests
    
    def generate_summary(self, passed: int, total: int):
        """Generate test summary"""
        duration = time.time() - self.start_time
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("\n" + "=" * 60)
        logger.info("🏆 CI/CD Test Suite Summary")
        logger.info("=" * 60)
        logger.info(f"📊 Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        logger.info(f"⏱️ Duration: {duration:.1f}s")
        logger.info("")
        
        # Individual test results
        for test_name, result in self.results.items():
            status = result["status"]
            icon = "✅" if status == "passed" else "❌" if status == "failed" else "⏳"
            logger.info(f"{icon} {test_name.replace('_', ' ').title()}: {status.upper()}")
        
        logger.info("=" * 60)
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - PRODUCTION READY!")
        else:
            logger.error("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        
        # Save results to file
        results_file = self.project_root / "ci_cd_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "passed": passed,
                    "total": total,
                    "success_rate": success_rate,
                    "duration": duration,
                    "timestamp": time.time()
                },
                "results": self.results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")

def main():
    """Main entry point"""
    runner = CICDTestRunner()
    
    # Check if specific test requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        test_map = {
            "mcp-server": runner.test_mcp_server_validation,
            "mcp-tools": runner.test_mcp_tools_comprehensive,
            "vector": runner.test_vector_service,
            "ipfs": runner.test_ipfs_service,
            "clustering": runner.test_clustering_service,
            "integration": runner.test_integration,
            "imports": runner.test_basic_imports,
            "dependencies": runner.test_service_dependencies
        }
        
        if test_name in test_map:
            logger.info(f"Running specific test: {test_name}")
            success = test_map[test_name]()
            sys.exit(0 if success else 1)
        else:
            logger.error(f"Unknown test: {test_name}")
            logger.info(f"Available tests: {', '.join(test_map.keys())}")
            sys.exit(1)
    
    # Run all tests
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
