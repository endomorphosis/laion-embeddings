# src/mcp_server/tools/rate_limiting_tools.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..tool_registry import ClaudeMCPTool

logger = logging.getLogger(__name__)

class RateLimitConfigurationTool(ClaudeMCPTool):
    """
    Tool for configuring rate limiting policies and rules.
    """

    def __init__(self, embedding_service=None):
        super().__init__()
        self.name = "rate_limit_configuration"
        self.description = "Configures rate limiting policies for API endpoints including limits, windows, and rules."
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Rate limit configuration action.",
                    "enum": ["get", "set", "update", "delete", "list", "reset"]
                },
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint to configure rate limiting for.",
                    "examples": ["/create_embeddings", "/search", "/load", "global"]
                },
                "rate_limit": {
                    "type": "object",
                    "description": "Rate limit configuration.",
                    "properties": {
                        "requests_per_minute": {
                            "type": "integer",
                            "description": "Maximum requests per minute.",
                            "minimum": 1,
                            "maximum": 10000
                        },
                        "requests_per_hour": {
                            "type": "integer",
                            "description": "Maximum requests per hour.",
                            "minimum": 1,
                            "maximum": 100000
                        },
                        "burst_limit": {
                            "type": "integer",
                            "description": "Burst request limit.",
                            "minimum": 1,
                            "maximum": 1000
                        },
                        "window_size": {
                            "type": "string",
                            "description": "Rate limit window size.",
                            "enum": ["1m", "5m", "15m", "1h", "24h"],
                            "default": "1m"
                        },
                        "bypass_roles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "User roles that bypass rate limits."
                        }
                    }
                },
                "user_id": {
                    "type": "string",
                    "description": "Specific user ID for user-specific rate limits."
                },
                "ip_address": {
                    "type": "string",
                    "description": "IP address for IP-specific rate limits."
                }
            },
            "required": ["action"]
        }
        self.category = "rate_limiting"
        self.tags = ["configuration", "limits", "policy"]
        self.embedding_service = embedding_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rate limit configuration operations.
        """
        try:
            action = parameters["action"]
            endpoint = parameters.get("endpoint")
            rate_limit = parameters.get("rate_limit", {})
            user_id = parameters.get("user_id")
            ip_address = parameters.get("ip_address")
            
            logger.info(f"Executing rate limit configuration: {action}")
            
            # Mock rate limit configurations
            current_configs = {
                "/create_embeddings": {
                    "requests_per_minute": 10,
                    "requests_per_hour": 100,
                    "burst_limit": 20,
                    "window_size": "1m",
                    "bypass_roles": ["admin", "premium"]
                },
                "/search": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000,
                    "burst_limit": 100,
                    "window_size": "1m",
                    "bypass_roles": ["admin"]
                },
                "/load": {
                    "requests_per_minute": 5,
                    "requests_per_hour": 20,
                    "burst_limit": 10,
                    "window_size": "5m",
                    "bypass_roles": ["admin"]
                },
                "global": {
                    "requests_per_minute": 1000,
                    "requests_per_hour": 10000,
                    "burst_limit": 2000,
                    "window_size": "1m",
                    "bypass_roles": ["admin"]
                }
            }
            
            if action == "get":
                if not endpoint:
                    return {
                        "type": "error",
                        "result": None,
                        "message": "Endpoint is required for get action"
                    }
                
                config = current_configs.get(endpoint)
                if config:
                    result = {
                        "endpoint": endpoint,
                        "rate_limit": config,
                        "updated_at": datetime.now().isoformat()
                    }
                    message = f"Rate limit configuration retrieved for {endpoint}"
                else:
                    result = {"endpoint": endpoint, "rate_limit": None}
                    message = f"No rate limit configuration found for {endpoint}"
                
            elif action == "set":
                if not endpoint or not rate_limit:
                    return {
                        "type": "error",
                        "result": None,
                        "message": "Endpoint and rate_limit are required for set action"
                    }
                
                # In real implementation, would save to configuration store
                result = {
                    "endpoint": endpoint,
                    "rate_limit": rate_limit,
                    "action": "created",
                    "updated_at": datetime.now().isoformat()
                }
                message = f"Rate limit configuration created for {endpoint}"
                
            elif action == "update":
                if not endpoint or not rate_limit:
                    return {
                        "type": "error",
                        "result": None,
                        "message": "Endpoint and rate_limit are required for update action"
                    }
                
                # Merge with existing configuration
                existing = current_configs.get(endpoint, {})
                updated_config = {**existing, **rate_limit}
                
                result = {
                    "endpoint": endpoint,
                    "old_rate_limit": existing,
                    "new_rate_limit": updated_config,
                    "action": "updated",
                    "updated_at": datetime.now().isoformat()
                }
                message = f"Rate limit configuration updated for {endpoint}"
                
            elif action == "delete":
                if not endpoint:
                    return {
                        "type": "error",
                        "result": None,
                        "message": "Endpoint is required for delete action"
                    }
                
                result = {
                    "endpoint": endpoint,
                    "action": "deleted",
                    "deleted_at": datetime.now().isoformat()
                }
                message = f"Rate limit configuration deleted for {endpoint}"
                
            elif action == "list":
                result = {
                    "configurations": current_configs,
                    "total": len(current_configs),
                    "retrieved_at": datetime.now().isoformat()
                }
                message = "All rate limit configurations retrieved"
                
            elif action == "reset":
                if endpoint:
                    result = {
                        "endpoint": endpoint,
                        "action": "reset",
                        "reset_at": datetime.now().isoformat()
                    }
                    message = f"Rate limit counters reset for {endpoint}"
                else:
                    result = {
                        "action": "reset_all",
                        "endpoints_reset": list(current_configs.keys()),
                        "reset_at": datetime.now().isoformat()
                    }
                    message = "All rate limit counters reset"
                
            else:
                return {
                    "type": "error",
                    "result": None,
                    "message": f"Unknown action: {action}"
                }
            
            return {
                "type": "rate_limit_configuration",
                "result": result,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Rate limit configuration failed: {e}")
            return {
                "type": "error",
                "result": None,
                "message": f"Rate limit configuration failed: {str(e)}"
            }


class RateLimitMonitoringTool(ClaudeMCPTool):
    """
    Tool for monitoring rate limit usage and violations.
    """

    def __init__(self, embedding_service=None):
        super().__init__()
        self.name = "rate_limit_monitoring"
        self.description = "Monitors rate limit usage, violations, and provides analytics on API usage patterns."
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Rate limit monitoring action.",
                    "enum": ["current_usage", "violations", "analytics", "top_users", "blocked_ips"]
                },
                "endpoint": {
                    "type": "string",
                    "description": "Specific endpoint to monitor.",
                    "default": "all"
                },
                "time_range": {
                    "type": "object",
                    "description": "Time range for monitoring data.",
                    "properties": {
                        "start": {
                            "type": "string",
                            "description": "Start time in ISO format.",
                            "format": "date-time"
                        },
                        "end": {
                            "type": "string",
                            "description": "End time in ISO format.",
                            "format": "date-time"
                        },
                        "last_hours": {
                            "type": "integer",
                            "description": "Monitor last N hours.",
                            "minimum": 1,
                            "maximum": 168,
                            "default": 24
                        }
                    }
                },
                "user_id": {
                    "type": "string",
                    "description": "Monitor specific user."
                },
                "ip_address": {
                    "type": "string",
                    "description": "Monitor specific IP address."
                }
            },
            "required": ["action"]
        }
        self.category = "rate_limiting"
        self.tags = ["monitoring", "analytics", "violations"]
        self.embedding_service = embedding_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rate limit monitoring operations.
        """
        try:
            action = parameters["action"]
            endpoint = parameters.get("endpoint", "all")
            time_range = parameters.get("time_range", {})
            user_id = parameters.get("user_id")
            ip_address = parameters.get("ip_address")
            
            logger.info(f"Executing rate limit monitoring: {action}")
            
            if action == "current_usage":
                # Mock current usage data
                usage_data = {
                    "/create_embeddings": {
                        "current_minute": {"count": 7, "limit": 10, "remaining": 3},
                        "current_hour": {"count": 45, "limit": 100, "remaining": 55},
                        "burst_used": 2,
                        "last_request": datetime.now().isoformat()
                    },
                    "/search": {
                        "current_minute": {"count": 35, "limit": 60, "remaining": 25},
                        "current_hour": {"count": 420, "limit": 1000, "remaining": 580},
                        "burst_used": 5,
                        "last_request": datetime.now().isoformat()
                    },
                    "/load": {
                        "current_minute": {"count": 2, "limit": 5, "remaining": 3},
                        "current_hour": {"count": 8, "limit": 20, "remaining": 12},
                        "burst_used": 0,
                        "last_request": datetime.now().isoformat()
                    }
                }
                
                if endpoint == "all":
                    result = {
                        "endpoints": usage_data,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    result = {
                        "endpoint": endpoint,
                        "usage": usage_data.get(endpoint, {"error": "Endpoint not found"}),
                        "timestamp": datetime.now().isoformat()
                    }
                message = f"Current usage retrieved for {endpoint}"
                
            elif action == "violations":
                # Mock violation data
                violations = [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "endpoint": "/create_embeddings",
                        "user_id": "user_123",
                        "ip_address": "192.168.1.100",
                        "violation_type": "minute_limit_exceeded",
                        "limit": 10,
                        "actual": 15,
                        "action_taken": "request_blocked"
                    },
                    {
                        "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                        "endpoint": "/search",
                        "user_id": "user_456",
                        "ip_address": "192.168.1.200",
                        "violation_type": "burst_limit_exceeded",
                        "limit": 100,
                        "actual": 150,
                        "action_taken": "rate_limited"
                    }
                ]
                
                if endpoint != "all":
                    violations = [v for v in violations if v["endpoint"] == endpoint]
                if user_id:
                    violations = [v for v in violations if v["user_id"] == user_id]
                if ip_address:
                    violations = [v for v in violations if v["ip_address"] == ip_address]
                
                result = {
                    "violations": violations,
                    "total": len(violations),
                    "time_range": time_range,
                    "retrieved_at": datetime.now().isoformat()
                }
                message = f"Retrieved {len(violations)} violations"
                
            elif action == "analytics":
                # Mock analytics data
                result = {
                    "time_range": time_range,
                    "request_statistics": {
                        "total_requests": 12450,
                        "blocked_requests": 245,
                        "success_rate": 0.98,
                        "peak_hour": "14:00-15:00",
                        "peak_requests_per_minute": 85
                    },
                    "endpoint_analytics": {
                        "/search": {"requests": 8500, "blocks": 120, "avg_per_minute": 5.9},
                        "/create_embeddings": {"requests": 2800, "blocks": 80, "avg_per_minute": 1.9},
                        "/load": {"requests": 1150, "blocks": 45, "avg_per_minute": 0.8}
                    },
                    "user_analytics": {
                        "unique_users": 156,
                        "top_users": [
                            {"user_id": "user_123", "requests": 450, "blocks": 12},
                            {"user_id": "user_456", "requests": 380, "blocks": 8}
                        ]
                    },
                    "geographic_distribution": {
                        "US": 0.45,
                        "EU": 0.30,
                        "Asia": 0.20,
                        "Other": 0.05
                    },
                    "generated_at": datetime.now().isoformat()
                }
                message = "Rate limit analytics generated"
                
            elif action == "top_users":
                # Mock top users data
                result = {
                    "top_users": [
                        {
                            "user_id": "user_123",
                            "total_requests": 2450,
                            "violations": 15,
                            "last_violation": datetime.now().isoformat(),
                            "most_used_endpoint": "/search"
                        },
                        {
                            "user_id": "user_456",
                            "total_requests": 1890,
                            "violations": 8,
                            "last_violation": (datetime.now() - timedelta(hours=2)).isoformat(),
                            "most_used_endpoint": "/create_embeddings"
                        },
                        {
                            "user_id": "user_789",
                            "total_requests": 1650,
                            "violations": 3,
                            "last_violation": (datetime.now() - timedelta(days=1)).isoformat(),
                            "most_used_endpoint": "/search"
                        }
                    ],
                    "time_range": time_range,
                    "retrieved_at": datetime.now().isoformat()
                }
                message = "Top users by request volume retrieved"
                
            elif action == "blocked_ips":
                # Mock blocked IPs data
                result = {
                    "blocked_ips": [
                        {
                            "ip_address": "192.168.1.100",
                            "block_reason": "excessive_violations",
                            "blocked_at": datetime.now().isoformat(),
                            "violations_count": 25,
                            "block_duration": "1 hour",
                            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
                        },
                        {
                            "ip_address": "10.0.0.50",
                            "block_reason": "suspicious_pattern",
                            "blocked_at": (datetime.now() - timedelta(minutes=30)).isoformat(),
                            "violations_count": 50,
                            "block_duration": "24 hours",
                            "expires_at": (datetime.now() + timedelta(hours=23.5)).isoformat()
                        }
                    ],
                    "total_blocked": 2,
                    "retrieved_at": datetime.now().isoformat()
                }
                message = "Blocked IP addresses retrieved"
                
            else:
                return {
                    "type": "error",
                    "result": None,
                    "message": f"Unknown action: {action}"
                }
            
            return {
                "type": "rate_limit_monitoring",
                "result": result,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Rate limit monitoring failed: {e}")
            return {
                "type": "error",
                "result": None,
                "message": f"Rate limit monitoring failed: {str(e)}"
            }


class RateLimitManagementTool(ClaudeMCPTool):
    """
    Tool for managing rate limit enforcement and exceptions.
    """

    def __init__(self, embedding_service=None):
        super().__init__()
        self.name = "rate_limit_management"
        self.description = "Manages rate limit enforcement including whitelisting, blacklisting, and temporary exceptions."
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Rate limit management action.",
                    "enum": ["whitelist", "blacklist", "unblock", "temporary_bypass", "bulk_action"]
                },
                "target_type": {
                    "type": "string",
                    "description": "Type of target for the action.",
                    "enum": ["user", "ip", "api_key", "role"]
                },
                "target_value": {
                    "type": "string",
                    "description": "Value of the target (user ID, IP address, etc.)."
                },
                "duration": {
                    "type": "object",
                    "description": "Duration for temporary actions.",
                    "properties": {
                        "hours": {"type": "integer", "minimum": 1, "maximum": 168},
                        "days": {"type": "integer", "minimum": 1, "maximum": 30}
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the action.",
                    "maxLength": 500
                },
                "bulk_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple targets for bulk actions.",
                    "maxItems": 100
                }
            },
            "required": ["action", "target_type"]
        }
        self.category = "rate_limiting"
        self.tags = ["management", "whitelist", "blacklist", "enforcement"]
        self.embedding_service = embedding_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rate limit management operations.
        """
        try:
            action = parameters["action"]
            target_type = parameters["target_type"]
            target_value = parameters.get("target_value")
            duration = parameters.get("duration", {})
            reason = parameters.get("reason", "")
            bulk_targets = parameters.get("bulk_targets", [])
            
            logger.info(f"Executing rate limit management: {action} for {target_type}")
            
            if action in ["whitelist", "blacklist", "unblock", "temporary_bypass"] and not target_value:
                return {
                    "type": "error",
                    "result": None,
                    "message": f"Target value is required for {action} action"
                }
            
            if action == "whitelist":
                result = {
                    "action": "whitelist",
                    "target_type": target_type,
                    "target_value": target_value,
                    "status": "whitelisted",
                    "reason": reason,
                    "whitelisted_at": datetime.now().isoformat(),
                    "expires_at": None  # Permanent whitelist
                }
                message = f"{target_type} {target_value} added to whitelist"
                
            elif action == "blacklist":
                result = {
                    "action": "blacklist",
                    "target_type": target_type,
                    "target_value": target_value,
                    "status": "blacklisted",
                    "reason": reason,
                    "blacklisted_at": datetime.now().isoformat(),
                    "expires_at": None  # Permanent blacklist
                }
                message = f"{target_type} {target_value} added to blacklist"
                
            elif action == "unblock":
                result = {
                    "action": "unblock",
                    "target_type": target_type,
                    "target_value": target_value,
                    "status": "unblocked",
                    "reason": reason,
                    "unblocked_at": datetime.now().isoformat()
                }
                message = f"{target_type} {target_value} unblocked"
                
            elif action == "temporary_bypass":
                expires_at = datetime.now()
                if duration.get("hours"):
                    expires_at += timedelta(hours=duration["hours"])
                elif duration.get("days"):
                    expires_at += timedelta(days=duration["days"])
                else:
                    expires_at += timedelta(hours=1)  # Default 1 hour
                
                result = {
                    "action": "temporary_bypass",
                    "target_type": target_type,
                    "target_value": target_value,
                    "status": "bypass_active",
                    "reason": reason,
                    "bypass_granted_at": datetime.now().isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "duration": duration
                }
                message = f"Temporary bypass granted for {target_type} {target_value}"
                
            elif action == "bulk_action":
                if not bulk_targets:
                    return {
                        "type": "error",
                        "result": None,
                        "message": "Bulk targets are required for bulk action"
                    }
                
                # Process bulk actions
                processed_targets = []
                for target in bulk_targets[:100]:  # Limit to 100 targets
                    processed_targets.append({
                        "target": target,
                        "status": "processed",
                        "processed_at": datetime.now().isoformat()
                    })
                
                result = {
                    "action": "bulk_action",
                    "target_type": target_type,
                    "targets_processed": len(processed_targets),
                    "targets": processed_targets,
                    "reason": reason,
                    "bulk_processed_at": datetime.now().isoformat()
                }
                message = f"Bulk action processed for {len(processed_targets)} {target_type}s"
                
            else:
                return {
                    "type": "error",
                    "result": None,
                    "message": f"Unknown action: {action}"
                }
            
            return {
                "type": "rate_limit_management",
                "result": result,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Rate limit management failed: {e}")
            return {
                "type": "error",
                "result": None,
                "message": f"Rate limit management failed: {str(e)}"
            }


class RateLimitEnforcementTool(ClaudeMCPTool):
    """
    Tool for enforcing rate limits on API requests.
    """
    
    def __init__(self, enforcement_service=None):
        super().__init__()
        self.name = "rate_limit_enforcement"
        self.description = "Enforces rate limits for API requests and handles violations"
        self.enforcement_service = enforcement_service
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check", "enforce", "reset", "status"]
                },
                "endpoint": {"type": "string"},
                "user_id": {"type": "string"},
                "request_info": {"type": "object"}
            },
            "required": ["action"]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rate limit enforcement."""
        action = parameters.get("action")
        
        # Mock implementation for testing
        if action == "check":
            return {"allowed": True, "remaining": 100}
        elif action == "enforce":
            return {"action_taken": "none", "violation": False}
        elif action == "reset":
            return {"reset": True}
        elif action == "status":
            return {"status": "active", "violations": 0}
        else:
            return {"error": "Invalid action"}


class RateLimitBypassTool(ClaudeMCPTool):
    """
    Tool for bypassing rate limits for specific users or requests.
    """

    def __init__(self, bypass_service=None, embedding_service=None):
        super().__init__()
        self.bypass_service = bypass_service
        self.name = "rate_limit_bypass"
        self.description = "Manages rate limit bypass permissions for users or API keys."
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Bypass action to perform.",
                    "enum": ["grant", "revoke", "check", "list", "temporary"]
                },
                "user_id": {"type": "string"},
                "api_key": {"type": "string"},
                "endpoint": {"type": "string"},
                "duration": {"type": "integer", "description": "Duration in minutes for temporary bypass"},
                "reason": {"type": "string"}
            },
            "required": ["action"]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rate limit bypass operation."""
        action = parameters.get("action")
        
        # Mock implementation for testing
        if action == "grant":
            return {"bypass_granted": True, "user_id": parameters.get("user_id")}
        elif action == "revoke":
            return {"bypass_revoked": True, "user_id": parameters.get("user_id")}
        elif action == "check":
            return {"has_bypass": False, "user_id": parameters.get("user_id")}
        elif action == "list":
            return {"bypasses": []}
        elif action == "temporary":
            return {"temporary_bypass": True, "duration": parameters.get("duration", 60)}
        else:
            return {"error": "Invalid action"}
