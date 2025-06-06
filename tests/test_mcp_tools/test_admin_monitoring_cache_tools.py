"""
Tests for admin, monitoring, and cache MCP tools.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock


@pytest.mark.asyncio
class TestAdminTools:
    """Test suite for admin MCP tools."""
    
    @patch('src.mcp_server.tools.admin_tools.AdminManager')
    async def test_get_system_status_tool_success(self, mock_manager_class):
        """Test successful system status retrieval."""
        from src.mcp_server.tools.admin_tools import get_system_status_tool
        
        # Setup mock manager
        mock_manager = Mock()
        mock_manager.get_system_status = AsyncMock(return_value={
            "status": "healthy",
            "uptime_seconds": 86400,
            "memory_usage": {
                "total": 16000000000,
                "used": 8000000000,
                "free": 8000000000,
                "percent": 50.0
            },
            "cpu_usage": {
                "percent": 25.5,
                "cores": 8
            },
            "disk_usage": {
                "total": 1000000000000,
                "used": 500000000000,
                "free": 500000000000,
                "percent": 50.0
            },
            "services": {
                "embedding_service": "running",
                "vector_store": "running",
                "search_service": "running"
            }
        })
        mock_manager_class.return_value = mock_manager
        
        result = await get_system_status_tool(include_detailed_metrics=True)
        
        # Verify result
        assert result["success"] is True
        assert result["status"] == "healthy"
        assert result["uptime_seconds"] == 86400
        assert result["memory_usage"]["percent"] == 50.0
        assert result["cpu_usage"]["percent"] == 25.5
        assert result["services"]["embedding_service"] == "running"
    
    @patch('src.mcp_server.tools.admin_tools.AdminManager')
    async def test_restart_service_tool_success(self, mock_manager_class):
        """Test successful service restart."""
        from src.mcp_server.tools.admin_tools import restart_service_tool
        
        mock_manager = Mock()
        mock_manager.restart_service = AsyncMock(return_value={
            "service_name": "embedding_service",
            "status": "restarted",
            "restart_time": 3.2,
            "previous_uptime": 86400,
            "new_pid": 12345
        })
        mock_manager_class.return_value = mock_manager
        
        result = await restart_service_tool(
            service_name="embedding_service",
            force_restart=False,
            wait_for_ready=True
        )
        
        assert result["success"] is True
        assert result["service_name"] == "embedding_service"
        assert result["status"] == "restarted"
        assert result["restart_time"] == 3.2
        assert result["force_restart"] is False
    
    @patch('src.mcp_server.tools.admin_tools.AdminManager')
    async def test_cleanup_resources_tool_success(self, mock_manager_class):
        """Test successful resource cleanup."""
        from src.mcp_server.tools.admin_tools import cleanup_resources_tool
        
        mock_manager = Mock()
        mock_manager.cleanup_resources = AsyncMock(return_value={
            "cleanup_type": "full",
            "freed_memory_bytes": 1000000000,
            "cleaned_temp_files": 150,
            "cleared_cache_entries": 5000,
            "cleanup_time": 8.5,
            "services_restarted": ["cache_service"]
        })
        mock_manager_class.return_value = mock_manager
        
        result = await cleanup_resources_tool(
            cleanup_type="full",
            restart_services=True,
            cleanup_temp_files=True
        )
        
        assert result["success"] is True
        assert result["cleanup_type"] == "full"
        assert result["freed_memory_bytes"] == 1000000000
        assert result["cleaned_temp_files"] == 150
        assert result["cleared_cache_entries"] == 5000
    
    @patch('src.mcp_server.tools.admin_tools.AdminManager')
    async def test_update_configuration_tool_success(self, mock_manager_class):
        """Test successful configuration update."""
        from src.mcp_server.tools.admin_tools import update_configuration_tool
        
        mock_manager = Mock()
        mock_manager.update_configuration = AsyncMock(return_value={
            "updated_keys": ["embedding.batch_size", "cache.max_size"],
            "restart_required": False,
            "backup_created": True,
            "config_version": "1.2.3"
        })
        mock_manager_class.return_value = mock_manager
        
        config_updates = {
            "embedding.batch_size": 64,
            "cache.max_size": 10000
        }
        
        result = await update_configuration_tool(
            config_updates=config_updates,
            create_backup=True,
            validate_config=True
        )
        
        assert result["success"] is True
        assert len(result["updated_keys"]) == 2
        assert result["restart_required"] is False
        assert result["backup_created"] is True


@pytest.mark.asyncio
class TestMonitoringTools:
    """Test suite for monitoring MCP tools."""
    
    @patch('src.mcp_server.tools.monitoring_tools.MonitoringService')
    async def test_get_metrics_tool_success(self, mock_service_class):
        """Test successful metrics retrieval."""
        from src.mcp_server.tools.monitoring_tools import get_metrics_tool
        
        mock_service = Mock()
        mock_service.get_metrics = AsyncMock(return_value={
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "embedding_requests_total": 10000,
                "embedding_requests_per_second": 15.5,
                "search_requests_total": 5000,
                "search_requests_per_second": 8.2,
                "cache_hit_rate": 0.85,
                "error_rate": 0.02,
                "response_time_p50": 0.15,
                "response_time_p95": 0.45,
                "response_time_p99": 0.95
            }
        })
        mock_service_class.return_value = mock_service
        
        result = await get_metrics_tool(
            metric_types=["requests", "performance", "errors"],
            time_range_minutes=60
        )
        
        assert result["success"] is True
        assert "metrics" in result
        assert result["metrics"]["embedding_requests_total"] == 10000
        assert result["metrics"]["cache_hit_rate"] == 0.85
        assert result["time_range_minutes"] == 60
    
    @patch('src.mcp_server.tools.monitoring_tools.MonitoringService')
    async def test_create_alert_tool_success(self, mock_service_class):
        """Test successful alert creation."""
        from src.mcp_server.tools.monitoring_tools import create_alert_tool
        
        mock_service = Mock()
        mock_service.create_alert = AsyncMock(return_value={
            "alert_id": "alert_123",
            "name": "High Error Rate",
            "status": "active",
            "created_at": datetime.now().isoformat()
        })
        mock_service_class.return_value = mock_service
        
        result = await create_alert_tool(
            name="High Error Rate",
            condition="error_rate > 0.05",
            threshold=0.05,
            notification_channels=["email", "slack"],
            severity="warning"
        )
        
        assert result["success"] is True
        assert result["alert_id"] == "alert_123"
        assert result["name"] == "High Error Rate"
        assert result["threshold"] == 0.05
        assert result["severity"] == "warning"
    
    @patch('src.mcp_server.tools.monitoring_tools.MonitoringService')
    async def test_list_alerts_tool_success(self, mock_service_class):
        """Test successful alert listing."""
        from src.mcp_server.tools.monitoring_tools import list_alerts_tool
        
        mock_service = Mock()
        mock_service.list_alerts = AsyncMock(return_value={
            "alerts": [
                {
                    "alert_id": "alert_123",
                    "name": "High Error Rate",
                    "status": "triggered",
                    "severity": "warning"
                },
                {
                    "alert_id": "alert_456",
                    "name": "Low Cache Hit Rate",
                    "status": "active",
                    "severity": "info"
                }
            ],
            "total": 2
        })
        mock_service_class.return_value = mock_service
        
        result = await list_alerts_tool(
            status_filter="all",
            severity_filter="all"
        )
        
        assert result["success"] is True
        assert result["total_alerts"] == 2
        assert len(result["alerts"]) == 2
        assert result["alerts"][0]["status"] == "triggered"
    
    @patch('src.mcp_server.tools.monitoring_tools.MonitoringService')
    async def test_export_metrics_tool_success(self, mock_service_class, temp_dir):
        """Test successful metrics export."""
        from src.mcp_server.tools.monitoring_tools import export_metrics_tool
        
        mock_service = Mock()
        mock_service.export_metrics = AsyncMock(return_value={
            "export_path": f"{temp_dir}/metrics_export.json",
            "records_exported": 1000,
            "export_size_bytes": 500000,
            "time_range": {
                "start": (datetime.now() - timedelta(hours=24)).isoformat(),
                "end": datetime.now().isoformat()
            }
        })
        mock_service_class.return_value = mock_service
        
        import os
        export_path = os.path.join(temp_dir, "metrics_export.json")
        
        result = await export_metrics_tool(
            export_path=export_path,
            time_range_hours=24,
            format="json",
            include_raw_data=True
        )
        
        assert result["success"] is True
        assert result["export_path"] == export_path
        assert result["records_exported"] == 1000
        assert result["format"] == "json"


@pytest.mark.asyncio
class TestCacheTools:
    """Test suite for cache MCP tools."""
    
    @patch('src.mcp_server.tools.cache_tools.CacheManager')
    async def test_get_cache_stats_tool_success(self, mock_manager_class):
        """Test successful cache statistics retrieval."""
        from src.mcp_server.tools.cache_tools import get_cache_stats_tool
        
        mock_manager = Mock()
        mock_manager.get_cache_stats = AsyncMock(return_value={
            "total_size_bytes": 1000000000,
            "used_size_bytes": 600000000,
            "free_size_bytes": 400000000,
            "hit_rate": 0.85,
            "miss_rate": 0.15,
            "entries_count": 50000,
            "evictions_count": 1000,
            "cache_types": {
                "embedding_cache": {
                    "size_bytes": 400000000,
                    "entries": 20000,
                    "hit_rate": 0.90
                },
                "search_cache": {
                    "size_bytes": 200000000,
                    "entries": 30000,
                    "hit_rate": 0.80
                }
            }
        })
        mock_manager_class.return_value = mock_manager
        
        result = await get_cache_stats_tool(
            include_detailed_breakdown=True,
            cache_type="all"
        )
        
        assert result["success"] is True
        assert result["total_size_bytes"] == 1000000000
        assert result["hit_rate"] == 0.85
        assert result["entries_count"] == 50000
        assert "cache_types" in result
        assert result["cache_types"]["embedding_cache"]["hit_rate"] == 0.90
    
    @patch('src.mcp_server.tools.cache_tools.CacheManager')
    async def test_clear_cache_tool_success(self, mock_manager_class):
        """Test successful cache clearing."""
        from src.mcp_server.tools.cache_tools import clear_cache_tool
        
        mock_manager = Mock()
        mock_manager.clear_cache = AsyncMock(return_value={
            "cache_type": "embedding_cache",
            "cleared_entries": 20000,
            "freed_bytes": 400000000,
            "clear_time": 2.1
        })
        mock_manager_class.return_value = mock_manager
        
        result = await clear_cache_tool(
            cache_type="embedding_cache",
            confirm_clear=True
        )
        
        assert result["success"] is True
        assert result["cache_type"] == "embedding_cache"
        assert result["cleared_entries"] == 20000
        assert result["freed_bytes"] == 400000000
        assert result["clear_time"] == 2.1
    
    @patch('src.mcp_server.tools.cache_tools.CacheManager')
    async def test_set_cache_config_tool_success(self, mock_manager_class):
        """Test successful cache configuration update."""
        from src.mcp_server.tools.cache_tools import set_cache_config_tool
        
        mock_manager = Mock()
        mock_manager.set_cache_config = AsyncMock(return_value={
            "cache_type": "embedding_cache",
            "updated_config": {
                "max_size_bytes": 2000000000,
                "ttl_seconds": 3600,
                "eviction_policy": "lru"
            },
            "restart_required": False
        })
        mock_manager_class.return_value = mock_manager
        
        new_config = {
            "max_size_bytes": 2000000000,
            "ttl_seconds": 3600,
            "eviction_policy": "lru"
        }
        
        result = await set_cache_config_tool(
            cache_type="embedding_cache",
            config=new_config,
            apply_immediately=True
        )
        
        assert result["success"] is True
        assert result["cache_type"] == "embedding_cache"
        assert result["updated_config"]["max_size_bytes"] == 2000000000
        assert result["restart_required"] is False
    
    @patch('src.mcp_server.tools.cache_tools.CacheManager')
    async def test_warm_cache_tool_success(self, mock_manager_class):
        """Test successful cache warming."""
        from src.mcp_server.tools.cache_tools import warm_cache_tool
        
        mock_manager = Mock()
        mock_manager.warm_cache = AsyncMock(return_value={
            "cache_type": "embedding_cache",
            "warmed_entries": 15000,
            "warm_time": 45.6,
            "cache_hit_improvement": 0.15
        })
        mock_manager_class.return_value = mock_manager
        
        result = await warm_cache_tool(
            cache_type="embedding_cache",
            warm_strategy="frequent_queries",
            max_entries=15000
        )
        
        assert result["success"] is True
        assert result["cache_type"] == "embedding_cache"
        assert result["warmed_entries"] == 15000
        assert result["warm_time"] == 45.6
        assert result["cache_hit_improvement"] == 0.15
    
    def test_tool_metadata_structure(self):
        """Test that tool metadata is properly structured."""
        # Admin tools
        from src.mcp_server.tools.admin_tools import TOOL_METADATA as admin_metadata
        
        system_status_meta = admin_metadata["get_system_status_tool"]
        assert system_status_meta["name"] == "get_system_status_tool"
        
        restart_meta = admin_metadata["restart_service_tool"]
        assert "service_name" in restart_meta["parameters"]["required"]
        
        # Monitoring tools
        from src.mcp_server.tools.monitoring_tools import TOOL_METADATA as monitoring_metadata
        
        metrics_meta = monitoring_metadata["get_metrics_tool"]
        assert metrics_meta["name"] == "get_metrics_tool"
        
        # Cache tools
        from src.mcp_server.tools.cache_tools import TOOL_METADATA as cache_metadata
        
        cache_stats_meta = cache_metadata["get_cache_stats_tool"]
        assert cache_stats_meta["name"] == "get_cache_stats_tool"
        
        clear_cache_meta = cache_metadata["clear_cache_tool"]
        assert "cache_type" in clear_cache_meta["parameters"]["required"]


if __name__ == "__main__":
    pytest.main([__file__])
