# tests/test_mcp_tools/test_rate_limiting_tools.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.mcp_server.tools.rate_limiting_tools import (
    RateLimitConfigurationTool,
    RateLimitMonitoringTool,
    RateLimitEnforcementTool,
    RateLimitBypassTool
)
from src.mcp_server.error_handlers import MCPError, ValidationError


class TestRateLimitConfigurationTool:
    """Test cases for RateLimitConfigurationTool."""

    @pytest.fixture
    def mock_rate_limit_service(self):
        """Mock rate limiting service."""
        service = Mock()
        service.get_rate_limit = AsyncMock(return_value={
            "endpoint": "/create_embeddings",
            "requests_per_minute": 100,
            "requests_per_hour": 5000,
            "burst_limit": 150,
            "window_size": 60,
            "enforcement": "strict"
        })
        service.set_rate_limit = AsyncMock(return_value={
            "endpoint": "/create_embeddings",
            "updated": True,
            "previous_config": {"requests_per_minute": 50},
            "new_config": {"requests_per_minute": 100}
        })
        service.list_rate_limits = AsyncMock(return_value={
            "rate_limits": [
                {
                    "endpoint": "/create_embeddings",
                    "requests_per_minute": 100,
                    "enforcement": "strict"
                },
                {
                    "endpoint": "/search",
                    "requests_per_minute": 500,
                    "enforcement": "warn"
                }
            ],
            "total_endpoints": 2
        })
        return service

    @pytest.fixture
    def config_tool(self, mock_rate_limit_service):
        """Create RateLimitConfigurationTool instance."""
        return RateLimitConfigurationTool(embedding_service=mock_rate_limit_service)

    @pytest.mark.asyncio
    async def test_get_rate_limit_config(self, config_tool):
        """Test getting rate limit configuration."""
        parameters = {
            "action": "get",
            "endpoint": "/create_embeddings"
        }

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "get"
        assert result["endpoint"] == "/create_embeddings"
        assert result["requests_per_minute"] == 100
        assert result["requests_per_hour"] == 5000
        assert result["burst_limit"] == 150

    @pytest.mark.asyncio
    async def test_set_rate_limit_config(self, config_tool):
        """Test setting rate limit configuration."""
        parameters = {
            "action": "set",
            "endpoint": "/create_embeddings",
            "rate_limit": {
                "requests_per_minute": 100,
                "requests_per_hour": 5000,
                "burst_limit": 150,
                "window_size": 60,
                "enforcement": "strict"
            }
        }

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "set"
        assert result["endpoint"] == "/create_embeddings"
        assert result["updated"] is True

    @pytest.mark.asyncio
    async def test_update_rate_limit_config(self, config_tool):
        """Test updating existing rate limit configuration."""
        parameters = {
            "action": "update",
            "endpoint": "/create_embeddings",
            "rate_limit": {
                "requests_per_minute": 200,
                "enforcement": "warn"
            }
        }

        config_tool.embedding_service.update_rate_limit = AsyncMock(return_value={
            "endpoint": "/create_embeddings",
            "updated": True,
            "changes": ["requests_per_minute", "enforcement"]
        })

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "update"
        assert result["updated"] is True

    @pytest.mark.asyncio
    async def test_delete_rate_limit_config(self, config_tool):
        """Test deleting rate limit configuration."""
        config_tool.embedding_service.delete_rate_limit = AsyncMock(return_value={
            "endpoint": "/create_embeddings",
            "deleted": True
        })

        parameters = {
            "action": "delete",
            "endpoint": "/create_embeddings"
        }

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "delete"
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_list_rate_limits(self, config_tool):
        """Test listing all rate limit configurations."""
        parameters = {
            "action": "list"
        }

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "list"
        assert len(result["rate_limits"]) == 2
        assert result["total_endpoints"] == 2

    @pytest.mark.asyncio
    async def test_reset_rate_limits(self, config_tool):
        """Test resetting rate limit counters."""
        config_tool.embedding_service.reset_rate_limits = AsyncMock(return_value={
            "endpoints_reset": ["/create_embeddings", "/search"],
            "reset_count": 2,
            "reset_time": datetime.now().isoformat()
        })

        parameters = {
            "action": "reset",
            "endpoint": "all"
        }

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "reset"
        assert result["reset_count"] == 2

    @pytest.mark.asyncio
    async def test_invalid_action(self, config_tool):
        """Test handling of invalid action."""
        parameters = {
            "action": "invalid_action",
            "endpoint": "/create_embeddings"
        }

        with pytest.raises(ValidationError):
            await config_tool.execute(parameters)

    @pytest.mark.asyncio
    async def test_invalid_rate_limit_values(self, config_tool):
        """Test handling of invalid rate limit values."""
        parameters = {
            "action": "set",
            "endpoint": "/create_embeddings",
            "rate_limit": {
                "requests_per_minute": -1,  # Invalid negative value
                "burst_limit": 50
            }
        }

        config_tool.embedding_service.set_rate_limit.side_effect = ValidationError("Invalid rate limit values")

        result = await config_tool.execute(parameters)

        assert result["status"] == "error"
        assert "Invalid rate limit values" in result["error"]

    @pytest.mark.parametrize("endpoint,rpm,rph", [
        ("/create_embeddings", 100, 5000),
        ("/search", 500, 25000),
        ("/load", 50, 2000),
        ("global", 1000, 50000)
    ])
    @pytest.mark.asyncio
    async def test_different_endpoints(self, config_tool, endpoint, rpm, rph):
        """Test configuration for different endpoints."""
        parameters = {
            "action": "set",
            "endpoint": endpoint,
            "rate_limit": {
                "requests_per_minute": rpm,
                "requests_per_hour": rph
            }
        }

        result = await config_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["endpoint"] == endpoint


class TestRateLimitMonitoringTool:
    """Test cases for RateLimitMonitoringTool."""

    @pytest.fixture
    def mock_monitoring_service(self):
        """Mock rate limit monitoring service."""
        service = Mock()
        service.get_usage_stats = AsyncMock(return_value={
            "endpoint": "/create_embeddings",
            "current_usage": {
                "requests_this_minute": 45,
                "requests_this_hour": 2850,
                "remaining_minute": 55,
                "remaining_hour": 2150
            },
            "historical_usage": {
                "peak_minute": 98,
                "peak_hour": 4800,
                "average_per_minute": 67,
                "average_per_hour": 3500
            },
            "violations": {
                "total_violations": 12,
                "violations_today": 2,
                "last_violation": "2024-01-15T14:30:00Z"
            }
        })
        return service

    @pytest.fixture
    def monitoring_tool(self, mock_monitoring_service):
        """Create RateLimitMonitoringTool instance."""
        return RateLimitMonitoringTool(monitoring_service=mock_monitoring_service)

    @pytest.mark.asyncio
    async def test_get_current_usage(self, monitoring_tool):
        """Test getting current usage statistics."""
        parameters = {
            "endpoint": "/create_embeddings",
            "metric_type": "current"
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["endpoint"] == "/create_embeddings"
        assert "current_usage" in result
        assert result["current_usage"]["requests_this_minute"] == 45
        assert result["current_usage"]["remaining_minute"] == 55

    @pytest.mark.asyncio
    async def test_get_historical_usage(self, monitoring_tool):
        """Test getting historical usage statistics."""
        parameters = {
            "endpoint": "/create_embeddings",
            "metric_type": "historical",
            "time_range": "24h"
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"
        assert "historical_usage" in result
        assert result["historical_usage"]["peak_minute"] == 98
        assert result["historical_usage"]["average_per_hour"] == 3500

    @pytest.mark.asyncio
    async def test_get_violation_stats(self, monitoring_tool):
        """Test getting rate limit violation statistics."""
        parameters = {
            "endpoint": "/create_embeddings",
            "metric_type": "violations"
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"
        assert "violations" in result
        assert result["violations"]["total_violations"] == 12
        assert result["violations"]["violations_today"] == 2

    @pytest.mark.asyncio
    async def test_get_all_endpoints_usage(self, monitoring_tool):
        """Test getting usage for all endpoints."""
        monitoring_tool.monitoring_service.get_all_usage = AsyncMock(return_value={
            "endpoints": [
                {
                    "endpoint": "/create_embeddings",
                    "requests_this_minute": 45,
                    "utilization": 0.45
                },
                {
                    "endpoint": "/search",
                    "requests_this_minute": 120,
                    "utilization": 0.24
                }
            ],
            "global_stats": {
                "total_requests": 165,
                "active_endpoints": 2,
                "highest_utilization": 0.45
            }
        })

        parameters = {
            "endpoint": "all",
            "metric_type": "summary"
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"
        assert len(result["endpoints"]) == 2
        assert result["global_stats"]["active_endpoints"] == 2

    @pytest.mark.asyncio
    async def test_get_usage_trends(self, monitoring_tool):
        """Test getting usage trends over time."""
        monitoring_tool.monitoring_service.get_usage_trends = AsyncMock(return_value={
            "endpoint": "/create_embeddings",
            "time_range": "7d",
            "trends": {
                "daily_averages": [3500, 3800, 3200, 4100, 3900, 3600, 3300],
                "peak_times": ["14:00", "15:30", "16:15"],
                "trend_direction": "stable",
                "growth_rate": 0.02
            },
            "predictions": {
                "next_peak": "2024-01-16T14:30:00Z",
                "expected_load": 4200
            }
        })

        parameters = {
            "endpoint": "/create_embeddings",
            "metric_type": "trends",
            "time_range": "7d"
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"
        assert "trends" in result
        assert "predictions" in result
        assert result["trends"]["trend_direction"] == "stable"

    @pytest.mark.asyncio
    async def test_get_alerts(self, monitoring_tool):
        """Test getting rate limit alerts."""
        monitoring_tool.monitoring_service.get_alerts = AsyncMock(return_value={
            "active_alerts": [
                {
                    "alert_id": "alert123",
                    "endpoint": "/create_embeddings",
                    "alert_type": "approaching_limit",
                    "threshold": 0.90,
                    "current_value": 0.95,
                    "created_at": "2024-01-15T14:45:00Z"
                }
            ],
            "alert_history": [
                {
                    "alert_id": "alert122",
                    "endpoint": "/search",
                    "alert_type": "limit_exceeded",
                    "resolved_at": "2024-01-15T13:30:00Z"
                }
            ]
        })

        parameters = {
            "metric_type": "alerts",
            "include_history": True
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"
        assert len(result["active_alerts"]) == 1
        assert len(result["alert_history"]) == 1

    @pytest.mark.parametrize("time_range", ["1h", "6h", "24h", "7d", "30d"])
    @pytest.mark.asyncio
    async def test_different_time_ranges(self, monitoring_tool, time_range):
        """Test monitoring for different time ranges."""
        parameters = {
            "endpoint": "/create_embeddings",
            "metric_type": "historical",
            "time_range": time_range
        }

        result = await monitoring_tool.execute(parameters)

        assert result["status"] == "success"


class TestRateLimitEnforcementTool:
    """Test cases for RateLimitEnforcementTool."""

    @pytest.fixture
    def mock_enforcement_service(self):
        """Mock rate limit enforcement service."""
        service = Mock()
        service.enforce_rate_limit = AsyncMock(return_value={
            "enforcement_id": "enf123",
            "client_id": "client456",
            "endpoint": "/create_embeddings",
            "action": "blocked",
            "reason": "Rate limit exceeded",
            "retry_after": 45
        })
        service.check_rate_limit = AsyncMock(return_value={
            "allowed": True,
            "remaining_requests": 55,
            "reset_time": datetime.now() + timedelta(seconds=45),
            "current_usage": 45
        })
        return service

    @pytest.fixture
    def enforcement_tool(self, mock_enforcement_service):
        """Create RateLimitEnforcementTool instance."""
        return RateLimitEnforcementTool(enforcement_service=mock_enforcement_service)

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, enforcement_tool):
        """Test checking rate limit when request is allowed."""
        parameters = {
            "action": "check",
            "client_id": "client456",
            "endpoint": "/create_embeddings",
            "request_weight": 1
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "check"
        assert result["allowed"] is True
        assert result["remaining_requests"] == 55

    @pytest.mark.asyncio
    async def test_check_rate_limit_blocked(self, enforcement_tool):
        """Test checking rate limit when request should be blocked."""
        enforcement_tool.enforcement_service.check_rate_limit.return_value = {
            "allowed": False,
            "remaining_requests": 0,
            "reset_time": datetime.now() + timedelta(seconds=45),
            "current_usage": 100,
            "limit_exceeded": True
        }

        parameters = {
            "action": "check",
            "client_id": "client456",
            "endpoint": "/create_embeddings"
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["allowed"] is False
        assert result["limit_exceeded"] is True

    @pytest.mark.asyncio
    async def test_enforce_rate_limit(self, enforcement_tool):
        """Test enforcing rate limit action."""
        parameters = {
            "action": "enforce",
            "client_id": "client456",
            "endpoint": "/create_embeddings",
            "enforcement_action": "block",
            "duration": 300
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "enforce"
        assert result["enforcement_action"] == "blocked"
        assert result["retry_after"] == 45

    @pytest.mark.asyncio
    async def test_throttle_requests(self, enforcement_tool):
        """Test throttling requests."""
        enforcement_tool.enforcement_service.throttle_requests = AsyncMock(return_value={
            "throttle_id": "throttle123",
            "client_id": "client456",
            "endpoint": "/create_embeddings",
            "throttle_rate": 0.5,
            "duration": 600,
            "status": "active"
        })

        parameters = {
            "action": "throttle",
            "client_id": "client456",
            "endpoint": "/create_embeddings",
            "throttle_rate": 0.5,
            "duration": 600
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "throttle"
        assert result["throttle_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_whitelist_client(self, enforcement_tool):
        """Test whitelisting a client."""
        enforcement_tool.enforcement_service.whitelist_client = AsyncMock(return_value={
            "client_id": "client456",
            "whitelisted": True,
            "exempted_endpoints": ["/create_embeddings", "/search"],
            "whitelist_duration": 3600
        })

        parameters = {
            "action": "whitelist",
            "client_id": "client456",
            "endpoints": ["/create_embeddings", "/search"],
            "duration": 3600
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "whitelist"
        assert result["whitelisted"] is True

    @pytest.mark.asyncio
    async def test_blacklist_client(self, enforcement_tool):
        """Test blacklisting a client."""
        enforcement_tool.enforcement_service.blacklist_client = AsyncMock(return_value={
            "client_id": "client456",
            "blacklisted": True,
            "blocked_endpoints": ["all"],
            "blacklist_duration": 7200,
            "reason": "Repeated violations"
        })

        parameters = {
            "action": "blacklist",
            "client_id": "client456",
            "duration": 7200,
            "reason": "Repeated violations"
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "blacklist"
        assert result["blacklisted"] is True

    @pytest.mark.asyncio
    async def test_get_client_status(self, enforcement_tool):
        """Test getting client enforcement status."""
        enforcement_tool.enforcement_service.get_client_status = AsyncMock(return_value={
            "client_id": "client456",
            "status": "normal",
            "whitelisted": False,
            "blacklisted": False,
            "throttled": False,
            "current_limits": {
                "/create_embeddings": {"remaining": 55, "reset_in": 45}
            }
        })

        parameters = {
            "action": "status",
            "client_id": "client456"
        }

        result = await enforcement_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "status"
        assert result["client_status"] == "normal"

    @pytest.mark.asyncio
    async def test_invalid_enforcement_action(self, enforcement_tool):
        """Test handling of invalid enforcement action."""
        parameters = {
            "action": "invalid_action",
            "client_id": "client456"
        }

        with pytest.raises(ValidationError):
            await enforcement_tool.execute(parameters)


class TestRateLimitBypassTool:
    """Test cases for RateLimitBypassTool."""

    @pytest.fixture
    def mock_bypass_service(self):
        """Mock rate limit bypass service."""
        service = Mock()
        service.create_bypass_token = AsyncMock(return_value={
            "bypass_token": "bypass_token_123",
            "token_id": "token456",
            "valid_until": datetime.now() + timedelta(hours=24),
            "permitted_endpoints": ["/create_embeddings"],
            "usage_limit": 1000
        })
        service.validate_bypass_token = AsyncMock(return_value={
            "valid": True,
            "token_id": "token456",
            "remaining_uses": 750,
            "expires_at": datetime.now() + timedelta(hours=20)
        })
        return service

    @pytest.fixture
    def bypass_tool(self, mock_bypass_service):
        """Create RateLimitBypassTool instance."""
        return RateLimitBypassTool(bypass_service=mock_bypass_service)

    @pytest.mark.asyncio
    async def test_create_bypass_token(self, bypass_tool):
        """Test creating a bypass token."""
        parameters = {
            "action": "create",
            "client_id": "client456",
            "endpoints": ["/create_embeddings"],
            "duration": 86400,
            "usage_limit": 1000,
            "reason": "Emergency maintenance"
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "create"
        assert "bypass_token" in result
        assert result["usage_limit"] == 1000

    @pytest.mark.asyncio
    async def test_validate_bypass_token(self, bypass_tool):
        """Test validating a bypass token."""
        parameters = {
            "action": "validate",
            "bypass_token": "bypass_token_123"
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "validate"
        assert result["valid"] is True
        assert result["remaining_uses"] == 750

    @pytest.mark.asyncio
    async def test_revoke_bypass_token(self, bypass_tool):
        """Test revoking a bypass token."""
        bypass_tool.bypass_service.revoke_bypass_token = AsyncMock(return_value={
            "token_id": "token456",
            "revoked": True,
            "revoked_at": datetime.now()
        })

        parameters = {
            "action": "revoke",
            "token_id": "token456",
            "reason": "Security concern"
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "revoke"
        assert result["revoked"] is True

    @pytest.mark.asyncio
    async def test_list_bypass_tokens(self, bypass_tool):
        """Test listing active bypass tokens."""
        bypass_tool.bypass_service.list_bypass_tokens = AsyncMock(return_value={
            "active_tokens": [
                {
                    "token_id": "token456",
                    "client_id": "client456",
                    "created_at": "2024-01-15T10:00:00Z",
                    "expires_at": "2024-01-16T10:00:00Z",
                    "remaining_uses": 750
                },
                {
                    "token_id": "token457",
                    "client_id": "client789",
                    "created_at": "2024-01-15T12:00:00Z",
                    "expires_at": "2024-01-16T12:00:00Z",
                    "remaining_uses": 500
                }
            ],
            "total_tokens": 2
        })

        parameters = {
            "action": "list",
            "client_id": "all"
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "list"
        assert len(result["active_tokens"]) == 2
        assert result["total_tokens"] == 2

    @pytest.mark.asyncio
    async def test_temporary_bypass(self, bypass_tool):
        """Test creating temporary bypass."""
        bypass_tool.bypass_service.create_temporary_bypass = AsyncMock(return_value={
            "bypass_id": "temp_bypass_123",
            "client_id": "client456",
            "endpoints": ["/create_embeddings"],
            "duration": 300,
            "auto_revoke": True,
            "active": True
        })

        parameters = {
            "action": "temporary",
            "client_id": "client456",
            "endpoints": ["/create_embeddings"],
            "duration": 300,
            "reason": "Emergency processing"
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["action"] == "temporary"
        assert result["duration"] == 300
        assert result["auto_revoke"] is True

    @pytest.mark.asyncio
    async def test_invalid_bypass_token(self, bypass_tool):
        """Test validation of invalid bypass token."""
        bypass_tool.bypass_service.validate_bypass_token.return_value = {
            "valid": False,
            "reason": "Token expired"
        }

        parameters = {
            "action": "validate",
            "bypass_token": "invalid_token"
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["valid"] is False
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_unauthorized_bypass_creation(self, bypass_tool):
        """Test unauthorized bypass token creation."""
        bypass_tool.bypass_service.create_bypass_token.side_effect = Exception("Insufficient privileges")

        parameters = {
            "action": "create",
            "client_id": "client456",
            "endpoints": ["/admin"],
            "duration": 86400
        }

        result = await bypass_tool.execute(parameters)

        assert result["status"] == "error"
        assert "Insufficient privileges" in result["error"]
