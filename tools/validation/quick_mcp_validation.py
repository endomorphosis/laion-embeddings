#!/usr/bin/env python3
"""
Quick MCP CI/CD Validation Tool

This tool provides quick validation for CI/CD deployment specifically
for the MCP tools without complex dependencies.
"""

import sys
import os
import json
import importlib.util
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_tool_imports():
    """Test that we can import MCP tools without dependency issues"""
    print("🔧 Testing MCP tool imports...")
    
    tools_dir = project_root / "src" / "mcp_server" / "tools"
    if not tools_dir.exists():
        print(f"❌ Tools directory not found: {tools_dir}")
        return False
    
    tool_files = list(tools_dir.glob("*.py"))
    if not tool_files:
        print("❌ No tool files found")
        return False
    
    print(f"📁 Found {len(tool_files)} tool files")
    
    # Test basic tool file syntax
    importable_tools = 0
    tool_classes = []
    
    for tool_file in tool_files:
        if tool_file.name.startswith("__"):
            continue
            
        try:
            # Test file syntax by reading and compiling
            with open(tool_file, 'r') as f:
                content = f.read()
            
            compile(content, str(tool_file), 'exec')
            print(f"  ✅ {tool_file.name} - valid syntax")
            importable_tools += 1
            
            # Count tool classes (basic heuristic)
            class_count = content.count("class ") - content.count("# class ")
            tool_classes.append((tool_file.name, class_count))
            
        except SyntaxError as e:
            print(f"  ❌ {tool_file.name} - syntax error: {e}")
        except Exception as e:
            print(f"  ⚠️  {tool_file.name} - error: {e}")
    
    print(f"✅ {importable_tools}/{len(tool_files)} tool files have valid syntax")
    
    total_classes = sum(count for _, count in tool_classes)
    print(f"📊 Estimated {total_classes} tool classes across all files")
    
    return importable_tools > 0

def test_core_imports():
    """Test core MCP imports"""
    print("🧠 Testing core imports...")
    
    try:
        # Test tool registry import
        from src.mcp_server.tool_registry import ClaudeMCPTool, ToolRegistry
        print("  ✅ ClaudeMCPTool base class")
        print("  ✅ ToolRegistry")
        
        # Test that we can create instances
        registry = ToolRegistry()
        print("  ✅ ToolRegistry instantiation")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_project_structure():
    """Test that all required CI/CD files exist"""
    print("📋 Testing CI/CD project structure...")
    
    required_files = {
        "mcp_server.py": "MCP server entry point",
        "run_ci_cd_tests.py": "CI/CD test runner",
        ".github/workflows/ci-cd.yml": "GitHub Actions workflow",
        "test/test_mcp_tools_comprehensive.py": "Comprehensive MCP test suite"
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {description} - {file_path}")
        else:
            print(f"  ❌ {description} - {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_tool_registry_basic():
    """Test basic tool registry functionality"""
    print("🔧 Testing tool registry basics...")
    
    try:
        from src.mcp_server.tool_registry import ToolRegistry, ClaudeMCPTool
        
        # Create a simple test tool
        class TestTool(ClaudeMCPTool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "A test tool"
                self.input_schema = {"type": "object", "properties": {}}
                self.category = "test"
                
            async def execute(self, parameters):
                return {"type": "test", "result": "success", "message": "Test tool executed"}
        
        # Test registry
        registry = ToolRegistry()
        test_tool = TestTool()
        registry.register_tool(test_tool)
        
        tools = registry.get_all_tools()
        print(f"  ✅ Registry created and tool registered")
        print(f"  ✅ {len(tools)} tool(s) in registry")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Quick MCP CI/CD Validation")
    print("=" * 40)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Tool Imports", test_tool_imports),
        ("Project Structure", test_project_structure),
        ("Tool Registry Basic", test_tool_registry_basic)
    ]
    
    passed = 0
    total = len(tests)
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        try:
            success = test_func()
            if success:
                passed += 1
                results[test_name] = "PASSED"
                print(f"✅ {test_name}: PASSED")
            else:
                results[test_name] = "FAILED"
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 VALIDATION SUMMARY")
    print("=" * 40)
    
    success_rate = (passed / total) * 100
    print(f"Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if passed == total:
        status = "✅ READY FOR CI/CD"
    elif passed >= total * 0.75:
        status = "⚠️  MOSTLY READY (minor issues)"
    else:
        status = "❌ NOT READY (major issues)"
    
    print(f"Status: {status}")
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "validation_type": "quick_mcp_ci_cd",
        "success_rate": success_rate,
        "tests_passed": passed,
        "tests_total": total,
        "status": status,
        "results": results
    }
    
    # Save report
    report_file = project_root / "quick_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved: {report_file}")
    
    # Final message
    if passed == total:
        print("\n🎉 All validations passed!")
        print("✅ CI/CD pipeline is ready for MCP tools testing")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed")
        print("🔧 Fix issues before running CI/CD pipeline")
        return 1

if __name__ == "__main__":
    sys.exit(main())
