#!/usr/bin/env python3
"""
Test MCP tools directly without subprocess
"""
import os
import sys
import traceback

# Add project root to path
project_root = '/home/barberb/laion-embeddings-1'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def log_result(message):
    """Log a result to file."""
    with open('direct_test_log.txt', 'a') as f:
        f.write(message + '\n')
    print(message)

def test_tool_imports():
    """Test importing all MCP tools."""
    
    log_result("=== Testing Tool Imports ===")
    
    # Test modules to import
    test_modules = [
        ('src.mcp_server.tools.auth_tools', ['AuthenticationTool', 'UserInfoTool']),
        ('src.mcp_server.tools.analysis_tools', ['ClusterAnalysisTool', 'QualityAssessmentTool', 'DimensionalityReductionTool']),
        ('src.mcp_server.tools.index_management_tools', ['IndexLoadingTool', 'ShardManagementTool', 'IndexStatusTool']),
        ('src.mcp_server.tools.embedding_tools', ['EmbeddingGenerationTool', 'BatchEmbeddingTool', 'MultimodalEmbeddingTool']),
        ('src.mcp_server.error_handlers', ['MCPError', 'ValidationError']),
        ('src.mcp_server.validators', ['validator']),
        ('src.mcp_server.tool_registry', ['ClaudeMCPTool']),
    ]
    
    success_count = 0
    total_count = len(test_modules)
    
    for module_name, classes in test_modules:
        try:
            module = __import__(module_name, fromlist=classes)
            
            # Check if all expected classes exist
            missing_classes = []
            for class_name in classes:
                if not hasattr(module, class_name):
                    missing_classes.append(class_name)
            
            if missing_classes:
                log_result(f"✗ {module_name}: Missing classes {missing_classes}")
            else:
                log_result(f"✓ {module_name}: All classes found")
                success_count += 1
                
        except Exception as e:
            log_result(f"✗ {module_name}: Import failed - {str(e)}")
            log_result(f"  Traceback: {traceback.format_exc()}")
    
    log_result(f"\nImport Summary: {success_count}/{total_count} modules imported successfully")
    return success_count == total_count

def test_conftest():
    """Test conftest imports."""
    
    log_result("\n=== Testing Conftest ===")
    
    try:
        from tests.test_mcp_tools.conftest import (
            MockCreateEmbeddingsProcessor, create_sample_file,
            TEST_MODEL_NAME, TEST_BATCH_SIZE, TEST_EMBEDDING_DIM
        )
        log_result("✓ Conftest imports successful")
        log_result(f"  TEST_MODEL_NAME: {TEST_MODEL_NAME}")
        log_result(f"  TEST_BATCH_SIZE: {TEST_BATCH_SIZE}")
        return True
    except Exception as e:
        log_result(f"✗ Conftest import failed: {str(e)}")
        log_result(f"  Traceback: {traceback.format_exc()}")
        return False

def test_tool_instantiation():
    """Test creating tool instances."""
    
    log_result("\n=== Testing Tool Instantiation ===")
    
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        tool = AuthenticationTool()
        log_result(f"✓ AuthenticationTool created: {tool.name}")
        return True
    except Exception as e:
        log_result(f"✗ AuthenticationTool instantiation failed: {str(e)}")
        log_result(f"  Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Clear log file
    with open('direct_test_log.txt', 'w') as f:
        f.write("=== Direct MCP Test Log ===\n")
    
    log_result("Starting direct MCP tool tests...")
    
    imports_ok = test_tool_imports()
    conftest_ok = test_conftest()
    instantiation_ok = test_tool_instantiation() if imports_ok else False
    
    overall_success = imports_ok and conftest_ok and instantiation_ok
    
    log_result(f"\n=== Final Result: {'SUCCESS' if overall_success else 'FAILURE'} ===")
    log_result("Check direct_test_log.txt for detailed results.")
