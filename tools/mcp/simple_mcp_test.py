#!/usr/bin/env python3
"""Comprehensive MCP tools test script"""

import sys
import os
from pathlib import Path
import traceback

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("🧪 COMPREHENSIVE MCP TOOLS TESTING")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"Python version: {sys.version}")
print()

def test_core_tools():
    """Test the 6 core tools we already validated"""
    print("Testing CORE tools (previously validated):")
    print("-" * 50)
    
    core_tools = [
        ("session_management_tools", ["SessionCreationTool", "SessionMonitoringTool", "SessionCleanupTool"]),
        ("data_processing_tools", ["ChunkingTool", "DatasetLoadingTool", "ParquetToCarTool"]),
        ("rate_limiting_tools", ["RateLimitConfigurationTool", "RateLimitMonitoringTool"]),
        ("ipfs_cluster_tools", ["IPFSClusterTool"]),
        ("embedding_tools", ["EmbeddingGenerationTool", "BatchEmbeddingTool", "MultimodalEmbeddingTool"]),
        ("search_tools", ["SemanticSearchTool"]),
    ]
    
    core_success = 0
    
    for module_name, class_names in core_tools:
        try:
            module = __import__(f"mcp_server.tools.{module_name}", fromlist=class_names)
            
            found_classes = []
            for class_name in class_names:
                if hasattr(module, class_name):
                    found_classes.append(class_name)
            
            print(f"✓ {module_name}: {len(found_classes)}/{len(class_names)} classes found")
            if len(found_classes) == len(class_names):
                core_success += 1
            
        except Exception as e:
            print(f"✗ {module_name}: {e}")
    
    print(f"Core tools result: {core_success}/{len(core_tools)} modules working")
    return core_success

def test_additional_tools():
    """Test all additional MCP tools"""
    print("\nTesting ADDITIONAL tools:")
    print("-" * 50)
    
    additional_tools = [
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
    
    success_count = 0
    total_count = len(additional_tools)
    detailed_results = []
    
    for tool_name in additional_tools:
        try:
            module = __import__(f"mcp_server.tools.{tool_name}", fromlist=[''])
            
            # Get all non-private attributes
            attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            classes = []
            functions = []
            
            for attr_name in attrs:
                attr = getattr(module, attr_name, None)
                if attr and hasattr(attr, '__bases__'):  # It's a class
                    classes.append(attr_name)
                elif callable(attr) and not attr_name.startswith('_'):
                    functions.append(attr_name)
            
            result = f"✓ {tool_name}: {len(classes)} classes, {len(functions)} functions"
            print(result)
            detailed_results.append((tool_name, True, len(classes), len(functions), None))
            success_count += 1
            
        except Exception as e:
            result = f"✗ {tool_name}: {e}"
            print(result)
            detailed_results.append((tool_name, False, 0, 0, str(e)))
    
    print(f"\nAdditional tools result: {success_count}/{total_count} modules working")
    print(f"Success rate: {(success_count / total_count * 100):.1f}%")
    
    return success_count, total_count, detailed_results

def test_tool_structure():
    """Analyze the structure of all tools"""
    print("\nTool structure analysis:")
    print("-" * 50)
    
    tools_dir = project_root / "src" / "mcp_server" / "tools"
    
    if not tools_dir.exists():
        print(f"✗ Tools directory not found: {tools_dir}")
        return
    
    py_files = list(tools_dir.glob("*.py"))
    print(f"✓ Found {len(py_files)} Python files in tools directory")
    
    # Check for common patterns
    execute_method_count = 0
    async_execute_count = 0
    
    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "def execute(" in content:
                execute_method_count += 1
            if "async def execute(" in content:
                async_execute_count += 1
                
        except Exception as e:
            print(f"Error reading {py_file.name}: {e}")
    
    print(f"✓ Files with execute methods: {execute_method_count}")
    print(f"✓ Files with async execute methods: {async_execute_count}")

def main():
    """Run all tests"""
    
    # Test core tools
    core_success = test_core_tools()
    
    # Test additional tools  
    additional_success, additional_total, detailed = test_additional_tools()
    
    # Test structure
    test_tool_structure()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    total_core = 6
    total_additional = additional_total
    total_all = total_core + total_additional
    total_success = core_success + additional_success
    
    print(f"Core tools: {core_success}/{total_core} working")
    print(f"Additional tools: {additional_success}/{total_additional} working")
    print(f"Overall: {total_success}/{total_all} tools working")
    print(f"Overall success rate: {(total_success / total_all * 100):.1f}%")
    
    # Show problematic tools
    print("\nDetailed results for additional tools:")
    for tool_name, success, classes, functions, error in detailed:
        if success:
            print(f"  ✓ {tool_name}: {classes}C, {functions}F")
        else:
            print(f"  ✗ {tool_name}: {error}")
    
    return 0 if total_success == total_all else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"\nTest completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)