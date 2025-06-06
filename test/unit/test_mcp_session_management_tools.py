import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mcp_server.tools.session_management_tools import (
    SessionCreationTool,
    SessionMonitoringTool, 
    SessionCleanupTool
)


class TestSessionCreationTool:
    """Test session creation tool functionality"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        service = MagicMock()
        service.create_session = AsyncMock(return_value={
            "session_id": "test-session-123",
            "session_name": "test_session",
            "status": "active",
            "created_at": "2025-06-05T12:00:00Z",
            "expires_at": "2025-06-05T13:00:00Z",
            "config": {
                "models": ["gte-small"],
                "max_requests_per_minute": 100,
                "max_concurrent_requests": 10,
                "timeout_seconds": 3600,
                "auto_cleanup": True
            },
            "resources": {
                "memory_limit_mb": 2048,
                "cpu_cores": 1.0,
                "gpu_enabled": False,
                "priority": "normal"
            },
            "metadata": {},
            "endpoint": "ws://localhost:8080/sessions/test-session-123",
            "auth_token": "sess_token_test-ses"
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_embedding_service):
        """Create tool instance for testing"""
        return SessionCreationTool(mock_embedding_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "create_session"
        assert tool.category == "session_management"
        assert "session_name" in tool.input_schema["properties"]
        assert "session_config" in tool.input_schema["properties"]
        assert "resource_allocation" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_create_session_success(self, tool, mock_embedding_service):
        """Test successful session creation"""
        parameters = {
            "session_name": "test_embedding_session",
            "session_config": {
                "models": ["gte-small", "sentence-transformers/all-MiniLM-L6-v2"],
                "max_requests_per_minute": 200,
                "timeout_seconds": 7200
            },
            "resource_allocation": {
                "memory_limit_mb": 4096,
                "cpu_cores": 2.0,
                "gpu_enabled": True,
                "priority": "high"
            },
            "metadata": {
                "user": "test_user",
                "project": "embeddings_test"
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_creation"
        assert "result" in result
        session_result = result["result"]
        assert session_result["session_name"] == "test_embedding_session"
        assert session_result["status"] == "active"
        assert "session_id" in session_result
        assert "created_at" in session_result
        assert "expires_at" in session_result
        
        mock_embedding_service.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_session_minimal_params(self, tool, mock_embedding_service):
        """Test session creation with minimal parameters"""
        parameters = {
            "session_name": "minimal_session"
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_creation"
        assert result["result"]["session_name"] == "minimal_session"
        mock_embedding_service.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_session_mock_fallback(self):
        """Test session creation with mock fallback (no service)"""
        tool = SessionCreationTool(embedding_service=None)
        
        parameters = {
            "session_name": "mock_session",
            "session_config": {
                "models": ["test-model"],
                "timeout_seconds": 1800
            }
        }
        
        with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-9abc-123456789012')):
            result = await tool.execute(parameters)
        
        assert result["type"] == "session_creation"
        session_result = result["result"]
        assert session_result["session_id"] == "12345678-1234-5678-9abc-123456789012"
        assert session_result["session_name"] == "mock_session"
        assert session_result["config"]["models"] == ["test-model"]
        assert session_result["config"]["timeout_seconds"] == 1800
    
    @pytest.mark.asyncio
    async def test_create_session_validation_error(self, tool):
        """Test session creation with validation error"""
        parameters = {
            "session_name": "",  # Invalid: empty string
            "session_config": {
                "models": [],  # Invalid: empty models array
                "max_requests_per_minute": -1  # Invalid: negative value
            }
        }
        
        with pytest.raises(Exception):
            await tool.execute(parameters)
    
    @pytest.mark.asyncio
    async def test_create_session_service_error(self, tool, mock_embedding_service):
        """Test session creation service error handling"""
        mock_embedding_service.create_session.side_effect = Exception("Service unavailable")
        
        parameters = {
            "session_name": "error_session"
        }
        
        with pytest.raises(Exception) as exc_info:
            await tool.execute(parameters)
        
        assert "Service unavailable" in str(exc_info.value)


class TestSessionMonitoringTool:
    """Test session monitoring tool functionality"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        service = MagicMock()
        service.monitor_sessions = AsyncMock(return_value={
            "session_id": "test-session-123",
            "status": "active",
            "uptime_seconds": 1800,
            "metrics": {
                "cpu": {
                    "current_usage": 35.2,
                    "average_usage": 28.7,
                    "peak_usage": 78.3
                },
                "memory": {
                    "current_mb": 1456,
                    "allocated_mb": 2048,
                    "peak_mb": 1789
                },
                "requests": {
                    "total_requests": 1247,
                    "requests_per_minute": 42.3,
                    "success_rate": 98.5
                }
            },
            "last_activity": "2025-06-05T12:30:00Z"
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_embedding_service):
        """Create tool instance for testing"""
        return SessionMonitoringTool(mock_embedding_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "monitor_sessions"
        assert tool.category == "session_management"
        assert "session_id" in tool.input_schema["properties"]
        assert "monitoring_scope" in tool.input_schema["properties"]
        assert "metrics_requested" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_monitor_single_session(self, tool, mock_embedding_service):
        """Test monitoring a single session"""
        parameters = {
            "session_id": "test-session-123",
            "metrics_requested": ["cpu", "memory", "requests"],
            "time_window": {
                "duration_minutes": 30,
                "granularity": "5min"
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_monitoring"
        assert "result" in result
        session_result = result["result"]
        assert session_result["session_id"] == "test-session-123"
        assert session_result["status"] == "active"
        assert "metrics" in session_result
        assert "cpu" in session_result["metrics"]
        assert "memory" in session_result["metrics"]
        assert "requests" in session_result["metrics"]
        
        mock_embedding_service.monitor_sessions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitor_all_sessions(self, tool, mock_embedding_service):
        """Test monitoring all sessions"""
        mock_embedding_service.monitor_sessions.return_value = {
            "monitoring_scope": "all",
            "total_sessions": 3,
            "sessions": [
                {
                    "session_id": "session_1",
                    "name": "embedding_session_1",
                    "status": "active",
                    "cpu_usage": 25.0,
                    "memory_usage_mb": 1200
                },
                {
                    "session_id": "session_2",
                    "name": "embedding_session_2", 
                    "status": "active",
                    "cpu_usage": 30.0,
                    "memory_usage_mb": 1400
                }
            ],
            "summary": {
                "total_cpu_usage": 55.0,
                "total_memory_mb": 2600,
                "average_latency_ms": 145.3
            }
        }
        
        parameters = {
            "monitoring_scope": "all",
            "metrics_requested": ["cpu", "memory"]
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_monitoring"
        session_result = result["result"]
        assert session_result["monitoring_scope"] == "all"
        assert session_result["total_sessions"] == 3
        assert len(session_result["sessions"]) == 2
        assert "summary" in session_result
        
        mock_embedding_service.monitor_sessions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitor_high_usage_sessions(self, tool, mock_embedding_service):
        """Test monitoring high usage sessions"""
        parameters = {
            "monitoring_scope": "high_usage",
            "metrics_requested": ["cpu", "memory", "latency"],
            "time_window": {
                "duration_minutes": 60,
                "granularity": "15min"
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_monitoring"
        mock_embedding_service.monitor_sessions.assert_called_once_with(
            None, "high_usage", ["cpu", "memory", "latency"], {"duration_minutes": 60, "granularity": "15min"}
        )
    
    @pytest.mark.asyncio
    async def test_monitor_sessions_mock_fallback(self):
        """Test session monitoring with mock fallback (no service)"""
        tool = SessionMonitoringTool(embedding_service=None)
        
        parameters = {
            "session_id": "mock-session-123",
            "metrics_requested": ["cpu", "memory"]
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_monitoring"
        session_result = result["result"]
        assert session_result["session_id"] == "mock-session-123"
        assert session_result["status"] == "active"
        assert "metrics" in session_result
    
    @pytest.mark.asyncio
    async def test_monitor_sessions_validation_error(self, tool):
        """Test session monitoring with validation error"""
        parameters = {
            "session_id": "invalid-session-id-format",  # Invalid UUID format
            "time_window": {
                "duration_minutes": 2000  # Invalid: exceeds maximum
            }
        }
        
        with pytest.raises(Exception):
            await tool.execute(parameters)
    
    @pytest.mark.asyncio
    async def test_monitor_sessions_service_error(self, tool, mock_embedding_service):
        """Test session monitoring service error handling"""
        mock_embedding_service.monitor_sessions.side_effect = Exception("Monitoring service error")
        
        parameters = {
            "monitoring_scope": "active"
        }
        
        with pytest.raises(Exception) as exc_info:
            await tool.execute(parameters)
        
        assert "Monitoring service error" in str(exc_info.value)


class TestSessionCleanupTool:
    """Test session cleanup tool functionality"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        service = MagicMock()
        service.manage_session_cleanup = AsyncMock(return_value={
            "session_id": "test-session-123",
            "action": "terminate",
            "status": "terminated",
            "terminated_at": "2025-06-05T12:45:00Z",
            "resources_freed": {
                "memory_mb": 2048,
                "cpu_cores": 1.0,
                "gpu_memory_mb": 0
            }
        })
        return service
    
    @pytest.fixture
    def tool(self, mock_embedding_service):
        """Create tool instance for testing"""
        return SessionCleanupTool(mock_embedding_service)
    
    def test_tool_initialization(self, tool):
        """Test tool proper initialization"""
        assert tool.name == "manage_session_cleanup"
        assert tool.category == "session_management"
        assert "action" in tool.input_schema["properties"]
        assert "session_id" in tool.input_schema["properties"]
        assert "cleanup_criteria" in tool.input_schema["properties"]
    
    @pytest.mark.asyncio
    async def test_terminate_session(self, tool, mock_embedding_service):
        """Test session termination"""
        parameters = {
            "action": "terminate",
            "session_id": "test-session-123"
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_cleanup"
        assert "result" in result
        cleanup_result = result["result"]
        assert cleanup_result["session_id"] == "test-session-123"
        assert cleanup_result["action"] == "terminate"
        assert cleanup_result["status"] == "terminated"
        assert "resources_freed" in cleanup_result
        
        mock_embedding_service.manage_session_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extend_session(self, tool, mock_embedding_service):
        """Test session extension"""
        mock_embedding_service.manage_session_cleanup.return_value = {
            "session_id": "test-session-123",
            "action": "extend",
            "status": "extended",
            "extended_at": "2025-06-05T12:45:00Z",
            "new_expiry": "2025-06-05T14:45:00Z",
            "extended_by_seconds": 7200
        }
        
        parameters = {
            "action": "extend",
            "session_id": "test-session-123",
            "extension_config": {
                "extend_by_seconds": 7200
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_cleanup"
        cleanup_result = result["result"]
        assert cleanup_result["action"] == "extend"
        assert cleanup_result["status"] == "extended"
        assert cleanup_result["extended_by_seconds"] == 7200
        
        mock_embedding_service.manage_session_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, tool, mock_embedding_service):
        """Test cleanup of expired sessions"""
        mock_embedding_service.manage_session_cleanup.return_value = {
            "action": "cleanup_expired",
            "cleaned_sessions": [
                {
                    "session_id": "expired_session_1",
                    "name": "session_1",
                    "status": "cleaned",
                    "resources_freed": {"memory_mb": 1024, "cpu_cores": 0.5}
                }
            ],
            "total_cleaned": 1,
            "total_resources_freed": {
                "memory_mb": 1024,
                "cpu_cores": 0.5
            }
        }
        
        parameters = {
            "action": "cleanup_expired",
            "cleanup_criteria": {
                "max_idle_minutes": 30,
                "expired_only": True,
                "preserve_data": False
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_cleanup"
        cleanup_result = result["result"]
        assert cleanup_result["action"] == "cleanup_expired"
        assert cleanup_result["total_cleaned"] == 1
        assert len(cleanup_result["cleaned_sessions"]) == 1
        assert "total_resources_freed" in cleanup_result
        
        mock_embedding_service.manage_session_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_sessions(self, tool, mock_embedding_service):
        """Test cleanup of inactive sessions"""
        parameters = {
            "action": "cleanup_inactive",
            "cleanup_criteria": {
                "max_idle_minutes": 120,
                "force": False
            }
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_cleanup"
        mock_embedding_service.manage_session_cleanup.assert_called_once_with(
            "cleanup_inactive", None, {"max_idle_minutes": 120, "force": False}, {}
        )
    
    @pytest.mark.asyncio
    async def test_cleanup_sessions_mock_fallback(self):
        """Test session cleanup with mock fallback (no service)"""
        tool = SessionCleanupTool(embedding_service=None)
        
        parameters = {
            "action": "terminate",
            "session_id": "mock-session-123"
        }
        
        result = await tool.execute(parameters)
        
        assert result["type"] == "session_cleanup"
        cleanup_result = result["result"]
        assert cleanup_result["session_id"] == "mock-session-123"
        assert cleanup_result["action"] == "terminate"
        assert cleanup_result["status"] == "terminated"
    
    @pytest.mark.asyncio
    async def test_cleanup_validation_error(self, tool):
        """Test session cleanup with validation error"""
        parameters = {
            "action": "invalid_action",  # Invalid action
            "session_id": "test-session-123"
        }
        
        with pytest.raises(Exception):
            await tool.execute(parameters)
    
    @pytest.mark.asyncio
    async def test_cleanup_missing_session_id(self, tool):
        """Test session cleanup with missing session ID for terminate action"""
        # Create tool without service to use mock implementation
        tool = SessionCleanupTool(embedding_service=None)
        
        parameters = {
            "action": "terminate"
            # Missing required session_id for terminate action
        }
        
        with pytest.raises(ValueError, match="Session ID is required"):
            await tool.execute(parameters)
    
    @pytest.mark.asyncio
    async def test_cleanup_service_error(self, tool, mock_embedding_service):
        """Test session cleanup service error handling"""
        mock_embedding_service.manage_session_cleanup.side_effect = Exception("Cleanup service error")
        
        parameters = {
            "action": "cleanup_expired"
        }
        
        with pytest.raises(Exception) as exc_info:
            await tool.execute(parameters)
        
        assert "Cleanup service error" in str(exc_info.value)


class TestSessionManagementIntegration:
    """Integration tests for session management tools"""
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_workflow(self):
        """Test complete session lifecycle: create -> monitor -> cleanup"""
        # Mock services
        embedding_service = MagicMock()
        
        # Setup mock responses for each tool
        embedding_service.create_session = AsyncMock(return_value={
            "session_id": "workflow-session-123",
            "session_name": "lifecycle_test",
            "status": "active"
        })
        
        embedding_service.monitor_sessions = AsyncMock(return_value={
            "session_id": "workflow-session-123",
            "status": "active",
            "metrics": {"cpu": {"current_usage": 25.0}}
        })
        
        embedding_service.manage_session_cleanup = AsyncMock(return_value={
            "session_id": "workflow-session-123",
            "action": "terminate",
            "status": "terminated"
        })
        
        # Create tools
        creation_tool = SessionCreationTool(embedding_service)
        monitoring_tool = SessionMonitoringTool(embedding_service)
        cleanup_tool = SessionCleanupTool(embedding_service)
        
        # Step 1: Create session
        create_result = await creation_tool.execute({
            "session_name": "lifecycle_test_session"
        })
        assert create_result["result"]["session_name"] == "lifecycle_test_session"
        
        # Step 2: Monitor session
        monitor_result = await monitoring_tool.execute({
            "session_id": "workflow-session-123",
            "metrics_requested": ["cpu"]
        })
        assert monitor_result["result"]["session_id"] == "workflow-session-123"
        
        # Step 3: Cleanup session
        cleanup_result = await cleanup_tool.execute({
            "action": "terminate",
            "session_id": "workflow-session-123"
        })
        assert cleanup_result["result"]["status"] == "terminated"
        
        # Verify all services were called
        embedding_service.create_session.assert_called_once()
        embedding_service.monitor_sessions.assert_called_once()
        embedding_service.manage_session_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_session_operations(self):
        """Test bulk session monitoring and cleanup operations"""
        embedding_service = MagicMock()
        
        # Setup bulk monitoring response
        embedding_service.monitor_sessions = AsyncMock(return_value={
            "monitoring_scope": "all",
            "total_sessions": 5,
            "sessions": [
                {"session_id": f"session_{i}", "status": "active"} 
                for i in range(5)
            ]
        })
        
        # Setup bulk cleanup response
        embedding_service.manage_session_cleanup = AsyncMock(return_value={
            "action": "cleanup_inactive",
            "total_cleaned": 2,
            "cleaned_sessions": [
                {"session_id": "session_3", "status": "cleaned"},
                {"session_id": "session_4", "status": "cleaned"}
            ]
        })
        
        monitoring_tool = SessionMonitoringTool(embedding_service)
        cleanup_tool = SessionCleanupTool(embedding_service)
        
        # Monitor all sessions
        monitor_result = await monitoring_tool.execute({
            "monitoring_scope": "all"
        })
        assert monitor_result["result"]["total_sessions"] == 5
        
        # Cleanup inactive sessions
        cleanup_result = await cleanup_tool.execute({
            "action": "cleanup_inactive",
            "cleanup_criteria": {"max_idle_minutes": 60}
        })
        assert cleanup_result["result"]["total_cleaned"] == 2
        
        embedding_service.monitor_sessions.assert_called_once()
        embedding_service.manage_session_cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
