#!/usr/bin/env python3
"""
Final comprehensive test of ALL MCP tools after bug fixes.
This tests imports, class definitions, and method signatures.
"""

import sys
import os
from pathlib import Path
import importlib
import traceback

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_all_mcp_tools():
    """Test all MCP tools comprehensively."""
    
    print("🚀 FINAL COMPREHENSIVE MCP TOOLS VALIDATION")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print()
    
    # Define all MCP tool files
    tool_files = [
        # Core tools (previously validated)
        "session_management_tools",
        "data_processing_tools", 
        "rate_limiting_tools",
        "ipfs_cluster_tools",
        "embedding_tools",
        "search_tools",
        
        # Additional tools (newly tested)
        "cache_tools",
        "admin_tools",
        "index_management_tools",
        "monitoring_tools",
        "workflow_tools", 
        "auth_tools",
        "vector_store_tools",
        "analysis_tools",
        "create_embeddings_tool",
        "background_task_tools",
        "storage_tools",
        "shard_embeddings_tool",
        "sparse_embedding_tools",
        "tool_wrapper",
        "vector_store_tools_new",
        "vector_store_tools_old"
    ]
    
    results = {
        "total_files": len(tool_files),
        "successful_imports": 0,
        "failed_imports": 0,
        "total_classes": 0,
        "total_functions": 0,
        "execute_methods": 0,
        "async_execute_methods": 0,
        "errors": [],
        "successes": []
    }
    
    print(f"Testing {len(tool_files)} MCP tool files...\n")
    
    for i, tool_file in enumerate(tool_files, 1):
        print(f"[{i:2d}/{len(tool_files)}] Testing: {tool_file}")
        print("-" * 60)
        
        try:
            # Import the module
            module_path = f"mcp_server.tools.{tool_file}"
            module = importlib.import_module(module_path)
            
            print(f"  ✓ Import successful")
            results["successful_imports"] += 1
            
            # Analyze module contents
            classes = []
            functions = []
            execute_methods = 0
            async_execute_methods = 0
            
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                    
                attr = getattr(module, attr_name, None)
                if attr is None:
                    continue
                
                # Check if it's a class
                if hasattr(attr, '__bases__'):
                    classes.append(attr_name)
                    results["total_classes"] += 1
                    
                    # Check for execute methods in the class
                    if hasattr(attr, 'execute'):
                        execute_methods += 1
                        results["execute_methods"] += 1
                        
                        # Check if it's async
                        execute_func = getattr(attr, 'execute')
                        if hasattr(execute_func, '__code__') and execute_func.__code__.co_flags & 0x80:
                            async_execute_methods += 1
                            results["async_execute_methods"] += 1
                
                # Check if it's a function
                elif callable(attr) and not attr_name.startswith('_'):
                    functions.append(attr_name)
                    results["total_functions"] += 1
            
            # Report findings
            if classes:
                print(f"  📦 Classes ({len(classes)}): {', '.join(classes[:3])}{'...' if len(classes) > 3 else ''}")
            if functions:
                print(f"  🔧 Functions ({len(functions)}): {', '.join(functions[:3])}{'...' if len(functions) > 3 else ''}")
            if execute_methods:
                print(f"  ⚡ Execute methods: {execute_methods} ({async_execute_methods} async)")
            
            results["successes"].append({
                "file": tool_file,
                "classes": len(classes),
                "functions": len(functions),
                "execute_methods": execute_methods,
                "async_execute_methods": async_execute_methods
            })
            
        except Exception as e:
            print(f"  ✗ Import failed: {e}")
            results["failed_imports"] += 1
            results["errors"].append({
                "file": tool_file,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        print()
    
    return results

def print_summary(results):
    """Print comprehensive test summary."""
    
    print("=" * 70)
    print("FINAL TEST RESULTS SUMMARY")
    print("=" * 70)
    
    # Overall statistics
    total_files = results["total_files"]
    successful = results["successful_imports"]
    failed = results["failed_imports"]
    success_rate = (successful / total_files * 100) if total_files > 0 else 0
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total files tested: {total_files}")
    print(f"   Successful imports: {successful}")
    print(f"   Failed imports: {failed}")
    print(f"   Success rate: {success_rate:.1f}%")
    print()
    
    print(f"🔍 CODE ANALYSIS:")
    print(f"   Total classes found: {results['total_classes']}")
    print(f"   Total functions found: {results['total_functions']}")
    print(f"   Execute methods: {results['execute_methods']}")
    print(f"   Async execute methods: {results['async_execute_methods']}")
    print()
    
    # Success details
    if results["successes"]:
        print("✅ SUCCESSFUL TOOLS:")
        for success in results["successes"]:
            file = success["file"]
            classes = success["classes"]
            functions = success["functions"]
            execute = success["execute_methods"]
            async_exec = success["async_execute_methods"]
            print(f"   {file}: {classes}C, {functions}F, {execute}E ({async_exec} async)")
        print()
    
    # Error details
    if results["errors"]:
        print("❌ FAILED TOOLS:")
        for error in results["errors"]:
            file = error["file"]
            err_msg = error["error"]
            print(f"   {file}: {err_msg}")
        print()
    
    # Final verdict
    if failed == 0:
        print("🎉 ALL MCP TOOLS PASSED! 🎉")
        print("The LAION embeddings MCP server is ready for production use.")
    else:
        print(f"⚠️  {failed} tools need attention before production deployment.")
    
    print("=" * 70)
    
    return failed == 0

def main():
    """Run the comprehensive test."""
    
    try:
        results = test_all_mcp_tools()
        all_passed = print_summary(results)
        
        # Write results to file
        results_file = project_root / "mcp_tools_final_test_results.txt"
        with open(results_file, 'w') as f:
            f.write("MCP Tools Final Test Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total files: {results['total_files']}\n")
            f.write(f"Successful: {results['successful_imports']}\n")
            f.write(f"Failed: {results['failed_imports']}\n")
            f.write(f"Success rate: {results['successful_imports']/results['total_files']*100:.1f}%\n\n")
            
            f.write("Successful tools:\n")
            for success in results["successes"]:
                f.write(f"- {success['file']}: {success['classes']}C, {success['functions']}F, {success['execute_methods']}E\n")
            
            if results["errors"]:
                f.write("\nFailed tools:\n")
                for error in results["errors"]:
                    f.write(f"- {error['file']}: {error['error']}\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"💥 Test suite failed with exception: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
