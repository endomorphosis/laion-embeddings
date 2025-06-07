#!/usr/bin/env python3
"""
Module import checker - logs results to a file
"""
import sys
import os
import traceback

# Add project paths
project_root = '/home/barberb/laion-embeddings-1'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Clear and initialize log file
log_file = 'import_check_log.txt'
with open(log_file, 'w') as f:
    f.write("=== Module Import Check Log ===\n\n")

def log_message(message):
    """Log message to both console and file."""
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)

def check_module_import(module_name, description=""):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        log_message(f"✓ {module_name} - {description}")
        return True
    except Exception as e:
        log_message(f"✗ {module_name} - {description}")
        log_message(f"  Error: {str(e)}")
        log_message(f"  Traceback: {traceback.format_exc()}")
        return False

# Test core framework imports
log_message("=== Core Framework Imports ===")
check_module_import("src.mcp_server.tool_registry", "Base tool classes")
check_module_import("src.mcp_server.validators", "Validation utilities")
check_module_import("src.mcp_server.error_handlers", "Error handling")

# Test class-based tools
log_message("\n=== Class-based Tool Imports ===")
check_module_import("src.mcp_server.tools.auth_tools", "Authentication tools")
check_module_import("src.mcp_server.tools.analysis_tools", "Analysis tools")
check_module_import("src.mcp_server.tools.embedding_tools", "Embedding tools")
check_module_import("src.mcp_server.tools.index_management_tools", "Index management")
check_module_import("src.mcp_server.tools.vector_store_tools", "Vector store tools")

# Test function-based tools
log_message("\n=== Function-based Tool Imports ===")
check_module_import("src.mcp_server.tools.create_embeddings_tool", "Create embeddings functions")
check_module_import("src.mcp_server.tools.shard_embeddings_tool", "Shard embeddings functions")
check_module_import("src.mcp_server.tools.search_tools", "Search functions") 
check_module_import("src.mcp_server.tools.storage_tools", "Storage functions")
check_module_import("src.mcp_server.tools.session_management_tools", "Session management functions")

# Test external dependencies
log_message("\n=== External Dependencies ===")
check_module_import("create_embeddings.create_embeddings", "Create embeddings processor")
check_module_import("pytest", "Testing framework")
check_module_import("numpy", "Numerical computing")

# Test conftest
log_message("\n=== Test Configuration ===")
check_module_import("tests.test_mcp_tools.conftest", "Test configuration and fixtures")

log_message(f"\n=== Import check complete. Results saved to {log_file} ===")
