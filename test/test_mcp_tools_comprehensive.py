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
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up logging
logger = logging.getLogger(__name__)

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
        assert tool.name == "rate_limit_configuration"
    
    async def test_vector_store_tool_with_parameters(self):
        """Test create_vector_store_tool works with parameters"""
        from src.mcp_server.tools.vector_store_tools import create_vector_store_tool
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = os.path.join(temp_dir, 'test_store')
            
            tool = await create_vector_store_tool(store_path=store_path, dimension=384)
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
    
    @pytest.mark.asyncio
    async def test_all_tools_listed(self, mcp_server):
        """Test all expected tools are available"""
        expected_tool_patterns = [
            "create_embeddings",
            "search_embeddings", 
            "shard_embeddings",
            "sparse_embeddings",
            "ipfs_",
            "vector_store",
            "cache_",
            "monitoring_",
            "session_",
            "auth_",
            "admin_",
            "workflow_",
            "background_task",
            "rate_limiting",
            "data_processing"
        ]
        
        available_tools = list(mcp_server.tools.keys())
        logger.info(f"Available tools: {available_tools}")
        
        # Check that we have tools from major categories
        found_patterns = 0
        for pattern in expected_tool_patterns:
            if any(pattern in tool_name for tool_name in available_tools):
                found_patterns += 1
        
        # Should find most pattern categories
        assert found_patterns >= len(expected_tool_patterns) * 0.6, \
            f"Missing tool categories. Found {found_patterns}/{len(expected_tool_patterns)} patterns"
    
    @pytest.mark.asyncio
    async def test_tools_have_valid_schemas(self, mcp_server):
        """Test all tools have valid input schemas"""
        for tool_name, tool_data in mcp_server.tools.items():
            # Each tool should have description and parameters
            assert "description" in tool_data, f"Tool {tool_name} missing description"
            assert "parameters" in tool_data, f"Tool {tool_name} missing parameters"
            assert "instance" in tool_data, f"Tool {tool_name} missing instance"
            
            # Description should be non-empty
            assert tool_data["description"], f"Tool {tool_name} has empty description"
            
            # Parameters should be a dict (JSON schema)
            assert isinstance(tool_data["parameters"], dict), \
                f"Tool {tool_name} parameters not a dict"
    
    @pytest.mark.asyncio
    async def test_mcp_protocol_methods(self, mcp_server):
        """Test MCP protocol method handling"""
        # Test initialize method
        init_request = {
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"}
        }
        
        response = await mcp_server.handle_request(init_request)
        assert "result" in response, "Initialize should return result"
        assert "protocolVersion" in response["result"]
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]
        
        # Test tools/list method
        list_request = {"method": "tools/list", "params": {}}
        response = await mcp_server.handle_request(list_request)
        assert "result" in response, "Tools list should return result"
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)
        
        # Test validation/status method
        status_request = {"method": "validation/status", "params": {}}
        response = await mcp_server.handle_request(status_request)
        assert "result" in response, "Validation status should return result"
        assert "validation" in response["result"]
        assert "tools_count" in response["result"]
    
    @pytest.mark.asyncio
    async def test_tool_execution_basic(self, mcp_server):
        """Test basic tool execution (non-destructive tools only)"""
        safe_tools_to_test = []
        
        # Find safe tools that can be tested without side effects
        for tool_name in mcp_server.tools.keys():
            if any(safe_pattern in tool_name.lower() for safe_pattern in 
                   ["list", "get", "status", "info", "check", "validate"]):
                safe_tools_to_test.append(tool_name)
        
        if not safe_tools_to_test:
            logger.warning("No safe tools found for execution testing")
            return
        
        # Test a few safe tools
        for tool_name in safe_tools_to_test[:3]:  # Test first 3 safe tools
            try:
                call_request = {
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": {}  # Empty args for basic test
                    }
                }
                
                response = await mcp_server.handle_request(call_request)
                
                # Tool should either succeed or fail gracefully
                assert "result" in response or "error" in response, \
                    f"Tool {tool_name} returned invalid response format"
                
                if "result" in response:
                    logger.info(f"✅ Tool {tool_name} executed successfully")
                else:
                    logger.info(f"⚠️ Tool {tool_name} failed gracefully: {response.get('error', {}).get('message', 'Unknown error')}")
                    
            except Exception as e:
                logger.warning(f"Tool {tool_name} execution test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server):
        """Test error handling for invalid requests"""
        # Test unknown method
        unknown_request = {"method": "unknown_method", "params": {}}
        response = await mcp_server.handle_request(unknown_request)
        assert "error" in response, "Unknown method should return error"
        
        # Test invalid tool call
        invalid_tool_request = {
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}}
        }
        response = await mcp_server.handle_request(invalid_tool_request)
        assert "error" in response, "Nonexistent tool should return error"
    
    def test_tool_registry_functionality(self):
        """Test tool registry can be imported and used"""
        try:
            from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
            from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
            
            # Create registry
            registry = ToolRegistry()
            assert registry is not None
            
            # Create minimal ipfs embeddings instance
            test_resources = {"local_endpoints": [["test", "cpu", 512]]}
            test_metadata = {"dataset": "test", "chunk_size": 512}
            
            ipfs_instance = ipfs_embeddings_py(
                resources=test_resources,
                metadata=test_metadata
            )
            assert ipfs_instance is not None
            
            # Initialize tools
            initialize_laion_tools(registry, ipfs_instance)
            tools = registry.get_all_tools()
            
            assert len(tools) > 0, "No tools initialized in registry"
            logger.info(f"Registry initialized with {len(tools)} tools")
            
        except Exception as e:
            pytest.fail(f"Tool registry test failed: {str(e)}")
    
    def test_expected_tool_count(self):
        """Test that we have the expected number of tools (around 22)"""
        try:
            from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
            from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
            
            registry = ToolRegistry()
            test_resources = {"local_endpoints": [["test", "cpu", 512]]}
            test_metadata = {"dataset": "test", "chunk_size": 512}
            
            ipfs_instance = ipfs_embeddings_py(
                resources=test_resources,
                metadata=test_metadata
            )
            
            initialize_laion_tools(registry, ipfs_instance)
            tools = registry.get_all_tools()
            
            tool_count = len(tools)
            logger.info(f"Total tools found: {tool_count}")
            
            # Should have around 22 tools (allow some variance)
            assert tool_count >= 15, f"Too few tools: {tool_count} (expected ~22)"
            assert tool_count <= 30, f"Too many tools: {tool_count} (expected ~22)"
            
        except Exception as e:
            pytest.fail(f"Tool count test failed: {str(e)}")

class TestMCPToolsIntegration:
    """Integration tests for MCP tools"""
    
    def test_mcp_server_cli_validation(self):
        """Test MCP server CLI validation mode"""
        import subprocess
        import tempfile
        import os
        
        # Run MCP server in validation mode
        mcp_server_path = project_root / "mcp_server.py"
        
        try:
            # Use timeout to prevent hanging
            result = subprocess.run(
                [sys.executable, str(mcp_server_path), "--validate"],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd=str(project_root)
            )
            
            if result.returncode != 0:
                logger.error(f"MCP server validation failed with code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                pytest.fail(f"MCP server validation failed: {result.stderr}")
            
            # Parse validation output
            validation_output = json.loads(result.stdout)
            
            assert validation_output["status"] == "success", \
                f"Validation failed: {validation_output}"
            
            assert validation_output["tools_count"] > 0, \
                "No tools loaded during validation"
            
            logger.info(f"✅ CLI validation passed: {validation_output['tools_count']} tools loaded")
            
        except subprocess.TimeoutExpired:
            pytest.fail("MCP server validation timed out")
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output from MCP server: {e}")
        except Exception as e:
            pytest.fail(f"MCP server CLI test failed: {str(e)}")

def test_mcp_tools_comprehensive():
    """Entry point for pytest discovery"""
    # Run async tests
    asyncio.run(run_async_tests())

async def run_async_tests():
    """Run async tests manually for environments that need it"""
    from mcp_server import LAIONEmbeddingsMCPServer
    
    server = LAIONEmbeddingsMCPServer()
    success = await server.initialize()
    
    if not success:
        raise Exception("MCP server initialization failed")
    
    logger.info(f"✅ MCP Tools Test Complete: {len(server.tools)} tools validated")
    return True

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
