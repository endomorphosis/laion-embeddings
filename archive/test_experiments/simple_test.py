#!/usr/bin/env python3
"""
Simple test to check if our fixed tests work
"""
import os
import sys

# Add project root to path
project_root = '/home/barberb/laion-embeddings-1'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_basic_imports():
    """Test basic imports that we've fixed."""
    
    print("=== Testing Basic Imports ===")
    
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool, UserInfoTool
        print("✓ auth_tools imports successful")
    except Exception as e:
        print(f"✗ auth_tools import failed: {e}")
        return False
    
    try:
        from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool, QualityAssessmentTool, DimensionalityReductionTool
        print("✓ analysis_tools imports successful")
    except Exception as e:
        print(f"✗ analysis_tools import failed: {e}")
        return False
    
    try:
        from src.mcp_server.tools.index_management_tools import IndexLoadingTool, ShardManagementTool, IndexStatusTool
        print("✓ index_management_tools imports successful")
    except Exception as e:
        print(f"✗ index_management_tools import failed: {e}")
        return False
    
    try:
        from tests.test_mcp_tools.conftest import MockCreateEmbeddingsProcessor, create_sample_file, TEST_MODEL_NAME
        print("✓ conftest imports successful")
    except Exception as e:
        print(f"✗ conftest import failed: {e}")
        return False
    
    return True

def test_tool_creation():
    """Test creating tool instances."""
    
    print("\n=== Testing Tool Creation ===")
    
    try:
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        tool = AuthenticationTool()
        print(f"✓ AuthenticationTool created: {tool.name}")
    except Exception as e:
        print(f"✗ AuthenticationTool creation failed: {e}")
        return False
    
    try:
        from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool
        tool = ClusterAnalysisTool()
        print(f"✓ ClusterAnalysisTool created: {tool.name}")
    except Exception as e:
        print(f"✗ ClusterAnalysisTool creation failed: {e}")
        return False
    
    return True

def write_results(success):
    """Write test results to file."""
    result = "SUCCESS" if success else "FAILURE"
    with open('simple_test_results.txt', 'w') as f:
        f.write(f"Simple test result: {result}\n")
    print(f"\nResult written to simple_test_results.txt: {result}")

if __name__ == "__main__":
    print("Running simple tests...")
    
    import_success = test_basic_imports()
    creation_success = test_tool_creation() if import_success else False
    
    overall_success = import_success and creation_success
    write_results(overall_success)
    
    if overall_success:
        print("\n🎉 All simple tests passed!")
    else:
        print("\n❌ Some tests failed")
