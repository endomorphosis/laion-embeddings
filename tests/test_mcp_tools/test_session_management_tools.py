"""
Tests for session management MCP tools.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import sys
sys.path.append('/home/barberb/laion-embeddings-1/tests/test_mcp_tools')
sys.path.append('/home/barberb/laion-embeddings-1')
from tests.test_mcp_tools.conftest import create_sample_json_file


@pytest.mark.asyncio
class TestSessionManagementTools:
    """Test suite for session management MCP tools."""
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_create_session_tool_success(self, mock_manager_class):
        """Test successful session creation."""
        from src.mcp_server.tools.session_management_tools import create_session_tool
        
        # Setup mock manager
        mock_manager = Mock()
        session_id = str(uuid.uuid4())
        mock_manager.create_session = AsyncMock(return_value={
            "session_id": session_id,
            "user_id": "user123",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "status": "active"
        })
        mock_manager_class.return_value = mock_manager
        
        # Execute tool
        result = await create_session_tool(
            user_id="user123",
            session_type="embedding_session",
            timeout_minutes=1440,
            metadata={"project": "test_project"}
        )
        
        # Verify result
        assert result["success"] is True
        assert result["session_id"] == session_id
        assert result["user_id"] == "user123"
        assert result["session_type"] == "embedding_session"
        assert result["timeout_minutes"] == 1440
        assert "created_at" in result
        assert "expires_at" in result
        assert result["status"] == "active"
        
        # Verify manager was called correctly
        mock_manager.create_session.assert_called_once_with(
            user_id="user123",
            session_type="embedding_session",
            timeout_minutes=1440,
            metadata={"project": "test_project"}
        )
    
    async def test_create_session_tool_invalid_user(self):
        """Test session creation with invalid user."""
        from src.mcp_server.tools.session_management_tools import create_session_tool
        
        result = await create_session_tool(user_id="")
        
        # Verify error handling
        assert result["success"] is False
        assert "Invalid user_id" in result["error"]
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_get_session_tool_success(self, mock_manager_class):
        """Test successful session retrieval."""
        from src.mcp_server.tools.session_management_tools import get_session_tool
        
        # Setup mock manager
        mock_manager = Mock()
        session_id = str(uuid.uuid4())
        mock_manager.get_session = AsyncMock(return_value={
            "session_id": session_id,
            "user_id": "user123",
            "session_type": "embedding_session",
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "metadata": {"project": "test_project"},
            "context": {"current_step": 1, "total_steps": 3}
        })
        mock_manager_class.return_value = mock_manager
        
        result = await get_session_tool(
            session_id=session_id,
            include_context=True
        )
        
        # Verify result
        assert result["success"] is True
        assert result["session_id"] == session_id
        assert result["user_id"] == "user123"
        assert result["status"] == "active"
        assert "context" in result
        assert result["context"]["current_step"] == 1
    
    async def test_get_session_tool_nonexistent(self):
        """Test retrieval of nonexistent session."""
        from src.mcp_server.tools.session_management_tools import get_session_tool
        
        fake_session_id = str(uuid.uuid4())
        
        result = await get_session_tool(session_id=fake_session_id)
        
        assert result["success"] is False
        assert "Session not found" in result["error"]
        assert result["session_id"] == fake_session_id
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_update_session_tool_success(self, mock_manager_class):
        """Test successful session update."""
        from src.mcp_server.tools.session_management_tools import update_session_tool
        
        # Setup mock manager
        mock_manager = Mock()
        session_id = str(uuid.uuid4())
        mock_manager.update_session = AsyncMock(return_value={
            "session_id": session_id,
            "updated_fields": ["status", "metadata"],
            "status": "paused",
            "last_updated": datetime.now().isoformat()
        })
        mock_manager_class.return_value = mock_manager
        
        result = await update_session_tool(
            session_id=session_id,
            status="paused",
            metadata={"project": "updated_project", "step": 2},
            extend_timeout=30
        )
        
        # Verify result
        assert result["success"] is True
        assert result["session_id"] == session_id
        assert result["status"] == "paused"
        assert "updated_fields" in result
        assert "last_updated" in result
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_end_session_tool_success(self, mock_manager_class):
        """Test successful session termination."""
        from src.mcp_server.tools.session_management_tools import end_session_tool
        
        # Setup mock manager
        mock_manager = Mock()
        session_id = str(uuid.uuid4())
        mock_manager.end_session = AsyncMock(return_value={
            "session_id": session_id,
            "status": "ended",
            "ended_at": datetime.now().isoformat(),
            "duration_minutes": 45,
            "cleanup_performed": True
        })
        mock_manager_class.return_value = mock_manager
        
        result = await end_session_tool(
            session_id=session_id,
            reason="User requested",
            cleanup_resources=True
        )
        
        # Verify result
        assert result["success"] is True
        assert result["session_id"] == session_id
        assert result["status"] == "ended"
        assert result["reason"] == "User requested"
        assert result["cleanup_performed"] is True
        assert "duration_minutes" in result
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_list_sessions_tool_success(self, mock_manager_class):
        """Test successful session listing."""
        from src.mcp_server.tools.session_management_tools import list_sessions_tool
        
        # Setup mock manager
        mock_manager = Mock()
        sessions = [
            {
                "session_id": str(uuid.uuid4()),
                "user_id": "user123",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "session_id": str(uuid.uuid4()),
                "user_id": "user456",
                "status": "paused",
                "created_at": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]
        mock_manager.list_sessions = AsyncMock(return_value={
            "sessions": sessions,
            "total": 2,
            "filtered": 2
        })
        mock_manager_class.return_value = mock_manager
        
        result = await list_sessions_tool(
            user_id="user123",
            status_filter="active",
            limit=10,
            offset=0
        )
        
        # Verify result
        assert result["success"] is True
        assert result["total_sessions"] == 2
        assert result["filtered_sessions"] == 2
        assert len(result["sessions"]) == 2
        assert result["user_id"] == "user123"
        assert result["status_filter"] == "active"
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_cleanup_expired_sessions_tool_success(self, mock_manager_class):
        """Test successful cleanup of expired sessions."""
        from src.mcp_server.tools.session_management_tools import cleanup_expired_sessions_tool
        
        # Setup mock manager
        mock_manager = Mock()
        mock_manager.cleanup_expired_sessions = AsyncMock(return_value={
            "cleaned_sessions": 5,
            "freed_resources": True,
            "cleanup_time": 2.3
        })
        mock_manager_class.return_value = mock_manager
        
        result = await cleanup_expired_sessions_tool(
            max_age_hours=24,
            cleanup_resources=True,
            dry_run=False
        )
        
        # Verify result
        assert result["success"] is True
        assert result["cleaned_sessions"] == 5
        assert result["freed_resources"] is True
        assert result["cleanup_time"] == 2.3
        assert result["max_age_hours"] == 24
        assert result["dry_run"] is False
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_get_session_stats_tool_success(self, mock_manager_class):
        """Test successful session statistics retrieval."""
        from src.mcp_server.tools.session_management_tools import get_session_stats_tool
        
        # Setup mock manager
        mock_manager = Mock()
        mock_manager.get_session_stats = AsyncMock(return_value={
            "total_sessions": 100,
            "active_sessions": 25,
            "paused_sessions": 5,
            "ended_sessions": 70,
            "average_duration_minutes": 120,
            "sessions_by_type": {
                "embedding_session": 60,
                "search_session": 30,
                "workflow_session": 10
            },
            "peak_concurrent_sessions": 35
        })
        mock_manager_class.return_value = mock_manager
        
        result = await get_session_stats_tool(
            include_historical=True,
            time_range_hours=168  # 1 week
        )
        
        # Verify result
        assert result["success"] is True
        assert result["total_sessions"] == 100
        assert result["active_sessions"] == 25
        assert result["time_range_hours"] == 168
        assert "sessions_by_type" in result
        assert result["sessions_by_type"]["embedding_session"] == 60
        assert result["peak_concurrent_sessions"] == 35
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_session_manager_exception(self, mock_manager_class):
        """Test handling of session manager exceptions."""
        from src.mcp_server.tools.session_management_tools import create_session_tool
        
        # Setup mock to raise exception
        mock_manager = Mock()
        mock_manager.create_session = AsyncMock(side_effect=Exception("Session manager failed"))
        mock_manager_class.return_value = mock_manager
        
        result = await create_session_tool(user_id="user123")
        
        assert result["success"] is False
        assert "Session manager failed" in result["error"]
    
    def test_tool_metadata_structure(self):
        """Test that tool metadata is properly structured."""
        from src.mcp_server.tools.session_management_tools import TOOL_METADATA
        
        # Check create_session_tool metadata
        create_meta = TOOL_METADATA["create_session_tool"]
        assert create_meta["name"] == "create_session_tool"
        assert "description" in create_meta
        assert "parameters" in create_meta
        
        params = create_meta["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert "user_id" in params["required"]
        
        # Check default values
        properties = params["properties"]
        assert properties["session_type"]["default"] == "general"
        assert properties["timeout_minutes"]["default"] == 1440  # 24 hours
        
        # Check get_session_tool metadata
        get_meta = TOOL_METADATA["get_session_tool"]
        assert get_meta["name"] == "get_session_tool"
        assert "session_id" in get_meta["parameters"]["required"]
        
        get_props = get_meta["parameters"]["properties"]
        assert get_props["include_context"]["default"] is False
        
        # Check list_sessions_tool metadata
        list_meta = TOOL_METADATA["list_sessions_tool"]
        assert list_meta["name"] == "list_sessions_tool"
        
        list_props = list_meta["parameters"]["properties"]
        assert list_props["status_filter"]["default"] == "all"
        assert list_props["limit"]["default"] == 50
        assert list_props["offset"]["default"] == 0
        
        # Check cleanup_expired_sessions_tool metadata
        cleanup_meta = TOOL_METADATA["cleanup_expired_sessions_tool"]
        assert cleanup_meta["name"] == "cleanup_expired_sessions_tool"
        
        cleanup_props = cleanup_meta["parameters"]["properties"]
        assert cleanup_props["max_age_hours"]["default"] == 168  # 1 week
        assert cleanup_props["cleanup_resources"]["default"] is True
        assert cleanup_props["dry_run"]["default"] is False
    
    def test_session_validation_functions(self):
        """Test session validation helper functions."""
        from src.mcp_server.tools.session_management_tools import (
            validate_session_id, validate_user_id, validate_session_type
        )
        
        # Valid session ID (UUID format)
        valid_session_id = str(uuid.uuid4())
        assert validate_session_id(valid_session_id) is True
        
        # Invalid session ID
        assert validate_session_id("invalid-id") is False
        assert validate_session_id("") is False
        
        # Valid user ID
        assert validate_user_id("user123") is True
        assert validate_user_id("user_with_underscore") is True
        
        # Invalid user ID
        assert validate_user_id("") is False
        assert validate_user_id(None) is False
        
        # Valid session types
        assert validate_session_type("embedding_session") is True
        assert validate_session_type("search_session") is True
        assert validate_session_type("workflow_session") is True
        assert validate_session_type("general") is True
        
        # Invalid session type
        assert validate_session_type("invalid_type") is False
        assert validate_session_type("") is False
    
    @patch('src.mcp_server.tools.session_management_tools.SessionManager')
    async def test_create_session_tool_with_custom_metadata(self, mock_manager_class):
        """Test session creation with custom metadata."""
        from src.mcp_server.tools.session_management_tools import create_session_tool
        
        mock_manager = Mock()
        session_id = str(uuid.uuid4())
        mock_manager.create_session = AsyncMock(return_value={
            "session_id": session_id,
            "user_id": "user123",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=2)).isoformat(),
            "status": "active"
        })
        mock_manager_class.return_value = mock_manager
        
        custom_metadata = {
            "project_id": "proj_456",
            "experiment_name": "embedding_test_v2",
            "model_version": "1.2.3",
            "batch_size": 128
        }
        
        result = await create_session_tool(
            user_id="user123",
            session_type="embedding_session",
            timeout_minutes=120,
            metadata=custom_metadata
        )
        
        assert result["success"] is True
        assert result["timeout_minutes"] == 120
        
        # Verify custom metadata was passed to manager
        mock_manager.create_session.assert_called_once_with(
            user_id="user123",
            session_type="embedding_session", 
            timeout_minutes=120,
            metadata=custom_metadata
        )


if __name__ == "__main__":
    pytest.main([__file__])
