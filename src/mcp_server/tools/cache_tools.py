# src/mcp_server/tools/cache_tools.py

import logging
from typing import Dict, Any, Optional
from ..tool_registry import ClaudeMCPTool

logger = logging.getLogger(__name__)

class CacheStatsTool(ClaudeMCPTool):
    """
    Tool for retrieving cache statistics and performance metrics.
    """
    
    def __init__(self, cache_service=None):
        super().__init__()
        self.name = "get_cache_stats"
        self.description = "Get cache statistics, performance metrics, and memory usage information."
        self.category = "cache"
        self.tags = ["cache", "statistics", "performance", "memory"]
        self.input_schema = {
            "type": "object",
            "properties": {
                "include_memory": {
                    "type": "boolean",
                    "description": "Include system memory information",
                    "default": True
                },
                "cache_type": {
                    "type": "string",
                    "description": "Type of cache to query",
                    "enum": ["embedding", "search", "all"],
                    "default": "all"
                }
            }
        }
        self.cache_service = cache_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache statistics retrieval."""
        try:
            include_memory = parameters.get("include_memory", True)
            cache_type = parameters.get("cache_type", "all")
            
            if self.cache_service:
                stats = await self.cache_service.get_stats(cache_type)
            else:
                # Mock cache stats
                stats = {
                    "cache_size": 150,
                    "hit_rate": 0.85,
                    "miss_rate": 0.15,
                    "total_requests": 1000,
                    "cache_hits": 850,
                    "cache_misses": 150,
                    "ttl_minutes": 30,
                    "max_entries": 1000,
                    "current_entries": 150,
                    "expired_entries_removed": 25
                }
            
            if include_memory:
                try:
                    import psutil
                    memory_info = {
                        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                        "memory_percent": psutil.Process().memory_percent(),
                        "system_memory_percent": psutil.virtual_memory().percent
                    }
                    stats.update(memory_info)
                except ImportError:
                    stats["memory_info"] = "psutil not available"
            
            return {
                "type": "cache_statistics",
                "result": stats,
                "message": "Cache statistics retrieved successfully"
            }
            
        except Exception as e:
            logger.error(f"Cache stats retrieval failed: {e}")
            return {
                "type": "cache_statistics",
                "result": {"error": str(e)},
                "message": f"Cache stats retrieval failed: {str(e)}"
            }


class CacheManagementTool(ClaudeMCPTool):
    """
    Tool for managing cache operations like clearing, configuring, and monitoring.
    """
    
    def __init__(self, cache_service=None):
        super().__init__()
        self.name = "manage_cache"
        self.description = "Clear expired cache entries, configure cache settings, and manage cache operations."
        self.category = "cache"
        self.tags = ["cache", "management", "clear", "configuration"]
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Cache management action",
                    "enum": ["clear_expired", "clear_all", "configure", "get_config", "warm_up"]
                },
                "cache_type": {
                    "type": "string",
                    "description": "Type of cache to manage",
                    "enum": ["embedding", "search", "all"],
                    "default": "all"
                },
                "config": {
                    "type": "object",
                    "description": "Cache configuration settings",
                    "properties": {
                        "ttl_minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                        "max_entries": {"type": "integer", "minimum": 10, "maximum": 10000}
                    }
                }
            },
            "required": ["action"]
        }
        self.cache_service = cache_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache management operation."""
        try:
            action = parameters.get("action")
            cache_type = parameters.get("cache_type", "all")
            config = parameters.get("config", {})
            
            if self.cache_service:
                result = await self.cache_service.manage_cache(action, cache_type, config)
            else:
                # Mock cache management
                if action == "clear_expired":
                    result = {
                        "expired_entries_removed": 25,
                        "current_entries": 125,
                        "message": "Expired cache entries cleared"
                    }
                elif action == "clear_all":
                    result = {
                        "entries_removed": 150,
                        "current_entries": 0,
                        "message": "All cache entries cleared"
                    }
                elif action == "configure":
                    result = {
                        "old_config": {"ttl_minutes": 30, "max_entries": 1000},
                        "new_config": config,
                        "message": "Cache configuration updated"
                    }
                elif action == "get_config":
                    result = {
                        "ttl_minutes": 30,
                        "max_entries": 1000,
                        "enabled": True,
                        "auto_cleanup": True
                    }
                elif action == "warm_up":
                    result = {
                        "preloaded_entries": 50,
                        "message": "Cache warmed up with frequently accessed data"
                    }
                else:
                    result = {"error": f"Unknown action: {action}"}
            
            return {
                "type": "cache_management",
                "result": result,
                "message": f"Cache {action} operation completed"
            }
            
        except Exception as e:
            logger.error(f"Cache management failed: {e}")
            return {
                "type": "cache_management",
                "result": {"error": str(e)},
                "message": f"Cache management failed: {str(e)}"
            }


class CacheMonitoringTool(ClaudeMCPTool):
    """
    Tool for monitoring cache performance and health.
    """
    
    def __init__(self, cache_service=None):
        super().__init__()
        self.name = "monitor_cache"
        self.description = "Monitor cache performance, health metrics, and usage patterns."
        self.category = "cache"
        self.tags = ["cache", "monitoring", "performance", "health"]
        self.input_schema = {
            "type": "object",
            "properties": {
                "time_window": {
                    "type": "string",
                    "description": "Time window for monitoring data",
                    "enum": ["1h", "6h", "24h", "7d"],
                    "default": "1h"
                },
                "metrics": {
                    "type": "array",
                    "description": "Specific metrics to monitor",
                    "items": {
                        "type": "string",
                        "enum": ["hit_rate", "miss_rate", "latency", "memory_usage", "eviction_rate"]
                    },
                    "default": ["hit_rate", "miss_rate", "latency"]
                }
            }
        }
        self.cache_service = cache_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache monitoring."""
        try:
            time_window = parameters.get("time_window", "1h")
            metrics = parameters.get("metrics", ["hit_rate", "miss_rate", "latency"])
            
            if self.cache_service:
                monitoring_data = await self.cache_service.get_monitoring_data(time_window, metrics)
            else:
                # Mock monitoring data
                monitoring_data = {
                    "time_window": time_window,
                    "hit_rate": {
                        "current": 0.85,
                        "average": 0.82,
                        "trend": "increasing"
                    },
                    "miss_rate": {
                        "current": 0.15,
                        "average": 0.18,
                        "trend": "decreasing"
                    },
                    "latency": {
                        "avg_ms": 2.5,
                        "p95_ms": 5.0,
                        "p99_ms": 8.0
                    },
                    "memory_usage": {
                        "current_mb": 45.2,
                        "max_mb": 100.0,
                        "utilization_percent": 45.2
                    },
                    "health_status": "healthy",
                    "alerts": []
                }
            
            return {
                "type": "cache_monitoring",
                "result": monitoring_data,
                "message": f"Cache monitoring data for {time_window} retrieved"
            }
            
        except Exception as e:
            logger.error(f"Cache monitoring failed: {e}")
            return {
                "type": "cache_monitoring",
                "result": {"error": str(e)},
                "message": f"Cache monitoring failed: {str(e)}"
            }
