#!/usr/bin/env python3
"""
Comprehensive MCP Tools Test Suite for CI/CD

This test validates core MCP tools functionality for CI/CD integration.
Focuses on proven working components after cleanup validation.
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class TestMCPToolsComprehensive:
    """Comprehensive test suite for core MCP tools"""
    
    def test_core_imports(self):
        """Test that all core MCP tools can be imported"""
        # These imports were validated to work after cleanup
        from src.mcp_server.tools.auth_tools import AuthenticationTool, TokenValidationTool
        from src.mcp_server.tools.session_management_tools import SessionManager
        from src.mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
        from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
        from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool
        
        assert True, "All core imports successful"
    
    def test_authentication_tool_instantiation(self):
        """Test AuthenticationTool can be instantiated"""
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        
        tool = AuthenticationTool()
        assert tool is not None
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'name')
        assert tool.name == "authenticate_user"
    
    def test_token_validation_tool_instantiation(self):
        """Test TokenValidationTool can be instantiated"""
        from src.mcp_server.tools.auth_tools import TokenValidationTool
        
        tool = TokenValidationTool()
        assert tool is not None
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'name')
        assert tool.name == "validate_token"
    
    def test_session_manager_instantiation(self):
        """Test SessionManager can be instantiated"""
        from src.mcp_server.tools.session_management_tools import SessionManager
        
        manager = SessionManager()
        assert manager is not None
        assert hasattr(manager, 'sessions')
        assert hasattr(manager, 'create_session')
    
    def test_rate_limit_tool_instantiation(self):
        """Test RateLimitConfigurationTool can be instantiated"""
        from src.mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
        
        tool = RateLimitConfigurationTool()
        assert tool is not None
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'name')
        assert tool.name == "configure_rate_limits"
    
    def test_vector_store_tool_with_parameters(self):
        """Test create_vector_store_tool works with parameters"""
        from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = os.path.join(temp_dir, 'test_store')
            
            tool = create_vector_store_tool(store_path=store_path, dimension=384)
            assert tool is not None
            assert hasattr(tool, 'execute')
            assert hasattr(tool, 'name')
    
    def test_mcp_server_enhanced_import(self):
        """Test that the enhanced MCP server can be imported and instantiated"""
        from mcp_server_enhanced import LAIONEmbeddingsMCPServer
        
        server = LAIONEmbeddingsMCPServer()
        assert server is not None
        # Don't test full initialization to avoid complex dependencies
    
    def test_project_structure_integrity(self):
        """Test that required project directories exist after cleanup"""
        required_dirs = [
            "src/mcp_server/tools",
            "test",
            "docs",
            "archive",
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Required directory missing: {dir_path}"
            assert full_path.is_dir(), f"Path is not a directory: {dir_path}"
    
    def test_critical_files_exist(self):
        """Test that critical files exist in the right locations"""
        critical_files = [
            "src/mcp_server/tools/auth_tools.py",
            "src/mcp_server/tools/session_management_tools.py", 
            "src/mcp_server/tools/rate_limiting_tools.py",
            "src/mcp_server/tools/vector_store_tools.py",
            "src/mcp_server/tools/ipfs_cluster_tools.py",
            "mcp_server_enhanced.py",
            "requirements.txt",
            "README.md"
        ]
        
        for file_path in critical_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Critical file missing: {file_path}"
            assert full_path.is_file(), f"Path is not a file: {file_path}"

    def test_tool_metadata_integrity(self):
        """Test that tools have proper metadata structure"""
        from src.mcp_server.tools.auth_tools import AuthenticationTool
        
        tool = AuthenticationTool()
        
        # Check required metadata attributes
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert len(tool.name) > 0
        assert len(tool.description) > 0
    
    def test_cleanup_validation_status(self):
        """Test that cleanup maintained core functionality"""
        # This test verifies the key finding: 5/5 tools working after cleanup
        
        working_tools = []
        
        # Test each core tool category
        try:
            from src.mcp_server.tools.auth_tools import AuthenticationTool
            AuthenticationTool()
            working_tools.append("auth")
        except:
            pass
            
        try:
            from src.mcp_server.tools.session_management_tools import SessionManager
            SessionManager()
            working_tools.append("session")
        except:
            pass
            
        try:
            from src.mcp_server.tools.rate_limiting_tools import RateLimitConfigurationTool
            RateLimitConfigurationTool()
            working_tools.append("rate_limit")
        except:
            pass
            
        try:
            from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
            # Just test import, not instantiation (requires params)
            working_tools.append("vector_store")
        except:
            pass
            
        try:
            from src.mcp_server.tools.ipfs_cluster_tools import IPFSClusterTool
            # Just test import, not instantiation (requires params)
            working_tools.append("ipfs_cluster")
        except:
            pass
        
        # Should have all 5 core tool categories working
        assert len(working_tools) == 5, f"Expected 5 working tool categories, got {len(working_tools)}: {working_tools}"
        
        expected_tools = {"auth", "session", "rate_limit", "vector_store", "ipfs_cluster"}
        assert set(working_tools) == expected_tools, f"Missing tools: {expected_tools - set(working_tools)}"

if __name__ == "__main__":
    # Allow running this test directly
    pytest.main([__file__, "-v"])
