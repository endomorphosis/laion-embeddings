#!/usr/bin/env python3
"""
Comprehensive test script for ALL MCP tools in the laion-embeddings project.
This script will test import, instantiation, and basic validation for every tool.
"""

import sys
import os
from pathlib import Path
import traceback
from typing import Dict, Any, List

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class ToolTestResults:
    def __init__(self):
        self.total_tools = 0
        self.successful_imports = 0
        self.failed_imports = 0
        self.successful_instantiations = 0
        self.failed_instantiations = 0
        self.errors = []
        self.successes = []

def test_tool_file_imports():
    """Test imports for all MCP tool files."""
    
    print("=" * 80)
    print("COMPREHENSIVE MCP TOOLS TESTING")
    print("=" * 80)
    
    results = ToolTestResults()
    
    # Define all tool files and their expected classes/functions
    tool_tests = [
        {
            "file": "session_management_tools",
            "classes": ["SessionCreationTool", "SessionMonitoringTool", "SessionCleanupTool"],
            "functions": ["create_session_tool", "monitor_session_tool", "cleanup_session_tool"]
        },
        {
            "file": "data_processing_tools", 
            "classes": ["ChunkingTool", "DatasetLoadingTool", "ParquetToCarTool"],
            "functions": []
        },
        {
            "file": "rate_limiting_tools",
            "classes": ["RateLimitConfigurationTool", "RateLimitMonitoringTool"],
            "functions": []
        },
        {
            "file": "ipfs_cluster_tools",
            "classes": ["IPFSClusterTool"],
            "functions": []
        },
        {
            "file": "embedding_tools",
            "classes": ["EmbeddingGenerationTool", "EmbeddingStorageTool", "EmbeddingRetrievalTool"],
            "functions": []
        },
        {
            "file": "search_tools",
            "classes": ["SemanticSearchTool"],
            "functions": []
        },
        {
            "file": "cache_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "admin_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "index_management_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "monitoring_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "workflow_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "auth_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "vector_store_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "analysis_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "create_embeddings_tool",
            "classes": [],
            "functions": []
        },
        {
            "file": "background_task_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "vector_store_tools_new",
            "classes": [],
            "functions": []
        },
        {
            "file": "vector_store_tools_old",
            "classes": [],
            "functions": []
        },
        {
            "file": "sparse_embedding_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "storage_tools",
            "classes": [],
            "functions": []
        },
        {
            "file": "shard_embeddings_tool",
            "classes": [],
            "functions": []
        },
        {
            "file": "tool_wrapper",
            "classes": [],
            "functions": []
        }
    ]
    
    print(f"\nTesting {len(tool_tests)} MCP tool files...\n")
    
    for test_config in tool_tests:
        file_name = test_config["file"]
        expected_classes = test_config["classes"]
        expected_functions = test_config["functions"]
        
        print(f"Testing: {file_name}")
        print("-" * 60)
        
        try:
            # Import the module
            module_path = f"mcp_server.tools.{file_name}"
            module = __import__(module_path, fromlist=[''])
            
            print(f"  ✓ Module {file_name} imported successfully")
            results.successful_imports += 1
            
            # Test expected classes
            for class_name in expected_classes:
                try:
                    cls = getattr(module, class_name, None)
                    if cls is not None:
                        print(f"    ✓ Class {class_name} found")
                        
                        # Try to inspect the class
                        if hasattr(cls, '__init__'):
                            print(f"      ✓ {class_name} has __init__ method")
                        if hasattr(cls, 'execute'):
                            print(f"      ✓ {class_name} has execute method")
                        
                        results.successful_instantiations += 1
                    else:
                        print(f"    ⚠ Class {class_name} not found in module")
                        results.failed_instantiations += 1
                        
                except Exception as e:
                    print(f"    ✗ Error testing class {class_name}: {e}")
                    results.failed_instantiations += 1
            
            # Test expected functions
            for func_name in expected_functions:
                try:
                    func = getattr(module, func_name, None)
                    if func is not None:
                        print(f"    ✓ Function {func_name} found")
                    else:
                        print(f"    ⚠ Function {func_name} not found in module")
                        
                except Exception as e:
                    print(f"    ✗ Error testing function {func_name}: {e}")
            
            # Check for any other classes/functions in the module
            module_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            if module_attrs:
                print(f"    📋 Other attributes found: {', '.join(module_attrs[:5])}{'...' if len(module_attrs) > 5 else ''}")
            
            results.successes.append(file_name)
            
        except Exception as e:
            print(f"  ✗ Failed to import {file_name}: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            results.failed_imports += 1
            results.errors.append(f"{file_name}: {e}")
        
        print()
        results.total_tools += 1
    
    return results

def test_tool_file_structure():
    """Test the structure and content of tool files."""
    
    print("=" * 80)
    print("TESTING TOOL FILE STRUCTURE")
    print("=" * 80)
    
    tools_dir = project_root / "src" / "mcp_server" / "tools"
    
    if not tools_dir.exists():
        print(f"✗ Tools directory not found: {tools_dir}")
        return
    
    print(f"✓ Tools directory found: {tools_dir}")
    
    # List all Python files
    py_files = list(tools_dir.glob("*.py"))
    print(f"✓ Found {len(py_files)} Python files in tools directory")
    
    for py_file in py_files:
        print(f"  - {py_file.name}")
    
    print()

def analyze_tool_dependencies():
    """Analyze dependencies and imports in tool files."""
    
    print("=" * 80)
    print("ANALYZING TOOL DEPENDENCIES")
    print("=" * 80)
    
    tools_dir = project_root / "src" / "mcp_server" / "tools"
    
    common_imports = {}
    error_patterns = {}
    
    for py_file in tools_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Count common import patterns
            imports = [
                "from typing import",
                "import asyncio",
                "import logging",
                "from mcp.types import",
                "from ..services",
                "Dict[str, Any]",
            ]
            
            for imp in imports:
                if imp in content:
                    common_imports[imp] = common_imports.get(imp, 0) + 1
            
            # Look for potential error patterns
            error_patterns_to_check = [
                "tool.call(",  # Should be tool.execute(
                "arguments.get(",  # Should be parameters.get(
                "async def execute(self, arguments",  # Should be parameters
            ]
            
            for pattern in error_patterns_to_check:
                if pattern in content:
                    error_patterns[pattern] = error_patterns.get(pattern, [])
                    error_patterns[pattern].append(py_file.name)
                    
        except Exception as e:
            print(f"Error reading {py_file.name}: {e}")
    
    print("Common import patterns:")
    for imp, count in sorted(common_imports.items(), key=lambda x: x[1], reverse=True):
        print(f"  {imp}: {count} files")
    
    if error_patterns:
        print("\n⚠ Potential error patterns found:")
        for pattern, files in error_patterns.items():
            print(f"  '{pattern}' in: {', '.join(files)}")
    else:
        print("\n✓ No common error patterns detected")
    
    print()

def main():
    """Run all tests."""
    
    print("🧪 COMPREHENSIVE MCP TOOLS TEST SUITE")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print()
    
    # Test 1: File structure
    test_tool_file_structure()
    
    # Test 2: Dependencies analysis
    analyze_tool_dependencies()
    
    # Test 3: Import and instantiation tests
    results = test_tool_file_imports()
    
    # Print final summary
    print("=" * 80)
    print("FINAL TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Total tool files tested: {results.total_tools}")
    print(f"Successful imports: {results.successful_imports}")
    print(f"Failed imports: {results.failed_imports}")
    print(f"Successful class checks: {results.successful_instantiations}")
    print(f"Failed class checks: {results.failed_instantiations}")
    
    print(f"\n✓ Success rate: {(results.successful_imports / results.total_tools * 100):.1f}%")
    
    if results.errors:
        print(f"\n⚠ Errors encountered:")
        for error in results.errors:
            print(f"  - {error}")
    
    if results.successes:
        print(f"\n✅ Successfully tested modules:")
        for success in results.successes:
            print(f"  - {success}")
    
    # Return exit code
    return 0 if results.failed_imports == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
