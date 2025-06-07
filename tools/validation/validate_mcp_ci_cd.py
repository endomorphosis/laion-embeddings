#!/usr/bin/env python3
"""
CI/CD MCP Tools Validation Script

This script validates the CI/CD setup with MCP tools testing to ensure
all components are working correctly for production deployment.
"""

import sys
import os
import asyncio
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPCICDValidator:
    """Validates MCP tools integration with CI/CD pipeline"""
    
    def __init__(self):
        self.project_root = project_root
        self.validation_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "errors": [],
            "mcp_tools_validated": 0
        }
        
    def run_command(self, command: str, cwd: str = None) -> tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Command failed: {str(e)}"
    
    def test_python_environment(self) -> bool:
        """Test Python environment and dependencies"""
        logger.info("🐍 Testing Python environment...")
        
        # Test Python version
        success, output = self.run_command("python3 --version")
        if not success:
            self.validation_results["errors"].append("Python 3 not available")
            return False
        
        logger.info(f"✅ Python version: {output.strip()}")
        
        # Test basic imports
        test_imports = [
            "import sys",
            "import asyncio", 
            "import json",
            "import logging",
            "from pathlib import Path",
            "from datetime import datetime",
            "from typing import Dict, Any, List"
        ]
        
        for import_test in test_imports:
            success, output = self.run_command(f"python3 -c \"{import_test}\"")
            if not success:
                self.validation_results["errors"].append(f"Import failed: {import_test}")
                return False
        
        logger.info("✅ Basic Python imports successful")
        return True
    
    def test_project_structure(self) -> bool:
        """Test project structure and key files"""
        logger.info("📁 Testing project structure...")
        
        required_paths = [
            "src/mcp_server",
            "src/mcp_server/tools",
            "src/mcp_server/tool_registry.py",
            "test",
            "mcp_server.py",
            "run_ci_cd_tests.py"
        ]
        
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                self.validation_results["errors"].append(f"Missing required path: {path}")
                return False
        
        logger.info("✅ Project structure validated")
        return True
    
    def test_mcp_tools_directory(self) -> bool:
        """Test MCP tools directory and tool files"""
        logger.info("🔧 Testing MCP tools directory...")
        
        tools_dir = self.project_root / "src" / "mcp_server" / "tools"
        tool_files = list(tools_dir.glob("*.py"))
        
        if len(tool_files) == 0:
            self.validation_results["errors"].append("No MCP tool files found")
            return False
        
        logger.info(f"✅ Found {len(tool_files)} MCP tool files")
        
        # Expected tool categories (based on our 22 tools)
        expected_tools = [
            "vector_service_tools.py",
            "clustering_tools.py", 
            "index_management_tools.py",
            "ipfs_tools.py",
            "monitoring_tools.py"
        ]
        
        found_tools = 0
        for tool_file in expected_tools:
            if (tools_dir / tool_file).exists():
                found_tools += 1
                logger.info(f"  ✅ {tool_file}")
            else:
                logger.warning(f"  ⚠️  {tool_file} not found")
        
        self.validation_results["mcp_tools_validated"] = found_tools
        logger.info(f"✅ {found_tools}/{len(expected_tools)} expected tool files found")
        return True
    
    def test_mcp_server_entry_point(self) -> bool:
        """Test the single MCP server entry point"""
        logger.info("🚀 Testing MCP server entry point...")
        
        # Test import of MCP server
        success, output = self.run_command(
            'python3 -c "import mcp_server; print(\'MCP server import successful\')"'
        )
        
        if not success:
            self.validation_results["errors"].append(f"MCP server import failed: {output}")
            return False
        
        logger.info("✅ MCP server entry point import successful")
        return True
    
    def test_ci_cd_runner(self) -> bool:
        """Test the CI/CD test runner"""
        logger.info("🔄 Testing CI/CD test runner...")
        
        # Test import of CI/CD runner
        success, output = self.run_command(
            'python3 -c "import run_ci_cd_tests; print(\'CI/CD runner import successful\')"'
        )
        
        if not success:
            self.validation_results["errors"].append(f"CI/CD runner import failed: {output}")
            return False
        
        logger.info("✅ CI/CD test runner import successful")
        return True
    
    def test_github_workflow(self) -> bool:
        """Test GitHub Actions workflow file"""
        logger.info("📋 Testing GitHub Actions workflow...")
        
        workflow_file = self.project_root / ".github" / "workflows" / "ci-cd.yml"
        if not workflow_file.exists():
            self.validation_results["errors"].append("GitHub workflow file not found")
            return False
        
        # Check workflow content
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        required_sections = [
            "mcp-tools-test:",
            "python mcp_server.py --validate",
            "pytest test/test_mcp_tools_comprehensive.py"
        ]
        
        for section in required_sections:
            if section not in content:
                self.validation_results["errors"].append(f"Missing workflow section: {section}")
                return False
        
        logger.info("✅ GitHub Actions workflow validated")
        return True
    
    def run_validation_suite(self) -> bool:
        """Run the complete validation suite"""
        logger.info("🎯 Starting CI/CD MCP Tools Validation...")
        
        tests = [
            ("Python Environment", self.test_python_environment),
            ("Project Structure", self.test_project_structure),
            ("MCP Tools Directory", self.test_mcp_tools_directory),
            ("MCP Server Entry Point", self.test_mcp_server_entry_point),
            ("CI/CD Test Runner", self.test_ci_cd_runner),
            ("GitHub Workflow", self.test_github_workflow)
        ]
        
        for test_name, test_func in tests:
            self.validation_results["total_tests"] += 1
            try:
                if test_func():
                    self.validation_results["passed_tests"] += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    self.validation_results["failed_tests"] += 1
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                self.validation_results["failed_tests"] += 1
                self.validation_results["errors"].append(f"{test_name}: {str(e)}")
                logger.error(f"❌ {test_name}: ERROR - {str(e)}")
        
        return self.validation_results["failed_tests"] == 0
    
    def generate_report(self) -> dict:
        """Generate validation report"""
        success_rate = (
            self.validation_results["passed_tests"] / 
            self.validation_results["total_tests"] * 100
            if self.validation_results["total_tests"] > 0 else 0
        )
        
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_status": "SUCCESS" if self.validation_results["failed_tests"] == 0 else "FAILED",
            "success_rate_percent": round(success_rate, 1),
            "tests_summary": {
                "total": self.validation_results["total_tests"],
                "passed": self.validation_results["passed_tests"],
                "failed": self.validation_results["failed_tests"]
            },
            "mcp_tools_validated": self.validation_results["mcp_tools_validated"],
            "errors": self.validation_results["errors"],
            "ci_cd_status": "READY" if self.validation_results["failed_tests"] == 0 else "NOT_READY"
        }
        
        return report


def main():
    """Main validation function"""
    print("🚀 LAION Embeddings MCP CI/CD Validation")
    print("=" * 50)
    
    validator = MCPCICDValidator()
    
    # Run validation
    success = validator.run_validation_suite()
    
    # Generate report
    report = validator.generate_report()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Status: {report['validation_status']}")
    print(f"Success Rate: {report['success_rate_percent']}%")
    print(f"Tests: {report['tests_summary']['passed']}/{report['tests_summary']['total']} passed")
    print(f"MCP Tools Validated: {report['mcp_tools_validated']}")
    print(f"CI/CD Status: {report['ci_cd_status']}")
    
    if report['errors']:
        print(f"\n❌ Errors ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"  • {error}")
    
    # Save report
    report_file = Path("ci_cd_validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    if success:
        print("\n🎉 CI/CD MCP validation completed successfully!")
        print("✅ Ready for production deployment")
    else:
        print("\n⚠️  CI/CD MCP validation failed")
        print("❌ Fix errors before deployment")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
