# src/mcp_server/monitoring.py

import asyncio
import time
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .config import MCPConfig

logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """Container for metric data with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # 'healthy', 'warning', 'unhealthy'
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """
    Collects and manages metrics for the MCP server.
    Provides counters, gauges, histograms, and health checks.
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.enabled = config.metrics_enabled
        
        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_checks: Dict[str, HealthCheckResult] = {}
        
        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self.total_request_time = 0.0
        self.request_times: deque = deque(maxlen=100)
        
        # Tool metrics
        self.tool_call_counts: Dict[str, int] = defaultdict(int)
        self.tool_error_counts: Dict[str, int] = defaultdict(int)
        self.tool_execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Session metrics
        self.session_count = 0
        self.session_creation_times: deque = deque(maxlen=100)
        
        # System metrics
        self.system_metrics: Dict[str, float] = {}
        self.system_metrics_task: Optional[asyncio.Task] = None
        
        if self.enabled:
            self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start the system metrics collection task."""
        if self.system_metrics_task is None or self.system_metrics_task.done():
            self.system_metrics_task = asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self):
        """Periodically collect system metrics."""
        while True:
            try:
                # CPU and memory
                self.system_metrics['cpu_percent'] = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                self.system_metrics['memory_percent'] = memory.percent
                self.system_metrics['memory_used_mb'] = memory.used / 1024 / 1024
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.system_metrics['disk_percent'] = disk.percent
                self.system_metrics['disk_used_gb'] = disk.used / 1024 / 1024 / 1024
                
                # Network
                net_io = psutil.net_io_counters()
                self.system_metrics['network_bytes_sent'] = net_io.bytes_sent
                self.system_metrics['network_bytes_recv'] = net_io.bytes_recv
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if not self.enabled:
            return
        
        key = name
        if labels:
            key += ":" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        
        self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        if not self.enabled:
            return
        
        key = name
        if labels:
            key += ":" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        if not self.enabled:
            return
        
        key = name
        if labels:
            key += ":" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        
        self.histograms[key].append(MetricData(value))
    
    def record_request(self, duration: float, success: bool = True):
        """Record request metrics."""
        if not self.enabled:
            return
        
        self.request_count += 1
        self.total_request_time += duration
        self.request_times.append(duration)
        
        if not success:
            self.error_count += 1
        
        self.increment_counter('http_requests_total', labels={'status': 'success' if success else 'error'})
        self.record_histogram('http_request_duration_seconds', duration)
    
    def record_tool_call(self, tool_name: str, duration: float, success: bool = True):
        """Record tool execution metrics."""
        if not self.enabled:
            return
        
        self.tool_call_counts[tool_name] += 1
        self.tool_execution_times[tool_name].append(duration)
        
        if not success:
            self.tool_error_counts[tool_name] += 1
        
        self.increment_counter('tool_calls_total', labels={'tool': tool_name, 'status': 'success' if success else 'error'})
        self.record_histogram('tool_execution_duration_seconds', duration, labels={'tool': tool_name})
    
    def record_session_created(self):
        """Record session creation."""
        if not self.enabled:
            return
        
        self.session_count += 1
        self.session_creation_times.append(time.time())
        self.increment_counter('sessions_created_total')
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        if not self.enabled:
            return {"metrics_enabled": False}
        
        # Calculate request metrics
        avg_request_time = (self.total_request_time / self.request_count) if self.request_count > 0 else 0
        error_rate = (self.error_count / self.request_count) if self.request_count > 0 else 0
        
        # Calculate recent request rate (last 5 minutes)
        recent_time = time.time() - 300  # 5 minutes ago
        recent_requests = sum(1 for t in self.session_creation_times if t > recent_time)
        request_rate = recent_requests / 5.0  # requests per minute
        
        return {
            "metrics_enabled": True,
            "timestamp": datetime.utcnow().isoformat(),
            "requests": {
                "total_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "average_duration": avg_request_time,
                "recent_rate_per_minute": request_rate
            },
            "tools": {
                "call_counts": dict(self.tool_call_counts),
                "error_counts": dict(self.tool_error_counts),
                "average_durations": {
                    tool: sum(times) / len(times) if times else 0
                    for tool, times in self.tool_execution_times.items()
                }
            },
            "sessions": {
                "total_created": self.session_count,
                "recent_creation_rate": request_rate
            },
            "system": dict(self.system_metrics),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
    
    def add_health_check(self, component: str, check_func: Callable[[], HealthCheckResult]):
        """Add a health check function."""
        try:
            result = check_func()
            self.health_checks[component] = result
        except Exception as e:
            self.health_checks[component] = HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"Health check failed: {e}"
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        healthy_count = sum(1 for check in self.health_checks.values() if check.status == "healthy")
        warning_count = sum(1 for check in self.health_checks.values() if check.status == "warning")
        unhealthy_count = sum(1 for check in self.health_checks.values() if check.status == "unhealthy")
        
        overall_status = "healthy"
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif warning_count > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "healthy": healthy_count,
                "warning": warning_count,
                "unhealthy": unhealthy_count,
                "total": len(self.health_checks)
            },
            "checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details
                }
                for name, check in self.health_checks.items()
            }
        }
    
    def shutdown(self):
        """Shutdown the metrics collector."""
        if self.system_metrics_task and not self.system_metrics_task.done():
            self.system_metrics_task.cancel()

class PerformanceMonitor:
    """
    Context manager for monitoring performance of operations.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, operation_name: str, 
                 labels: Optional[Dict[str, str]] = None):
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.success = exc_type is None
            
            # Record based on operation type
            if self.operation_name.startswith('tool_'):
                tool_name = self.operation_name[5:]  # Remove 'tool_' prefix
                self.metrics_collector.record_tool_call(tool_name, duration, self.success)
            else:
                self.metrics_collector.record_histogram(
                    f"{self.operation_name}_duration_seconds", 
                    duration, 
                    self.labels
                )
                self.metrics_collector.increment_counter(
                    f"{self.operation_name}_total",
                    labels={**self.labels, 'status': 'success' if self.success else 'error'}
                )

def create_default_health_checks(config: MCPConfig) -> List[Callable[[], HealthCheckResult]]:
    """Create default health check functions."""
    
    def check_memory():
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return HealthCheckResult(
                component="memory",
                status="unhealthy",
                message=f"Memory usage critically high: {memory.percent}%",
                details={"memory_percent": memory.percent}
            )
        elif memory.percent > 75:
            return HealthCheckResult(
                component="memory",
                status="warning",
                message=f"Memory usage high: {memory.percent}%",
                details={"memory_percent": memory.percent}
            )
        else:
            return HealthCheckResult(
                component="memory",
                status="healthy",
                message=f"Memory usage normal: {memory.percent}%",
                details={"memory_percent": memory.percent}
            )
    
    def check_disk():
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return HealthCheckResult(
                component="disk",
                status="unhealthy",
                message=f"Disk usage critically high: {disk.percent}%",
                details={"disk_percent": disk.percent}
            )
        elif disk.percent > 85:
            return HealthCheckResult(
                component="disk",
                status="warning",
                message=f"Disk usage high: {disk.percent}%",
                details={"disk_percent": disk.percent}
            )
        else:
            return HealthCheckResult(
                component="disk",
                status="healthy",
                message=f"Disk usage normal: {disk.percent}%",
                details={"disk_percent": disk.percent}
            )
    
    def check_cpu():
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return HealthCheckResult(
                component="cpu",
                status="unhealthy",
                message=f"CPU usage critically high: {cpu_percent}%",
                details={"cpu_percent": cpu_percent}
            )
        elif cpu_percent > 80:
            return HealthCheckResult(
                component="cpu",
                status="warning",
                message=f"CPU usage high: {cpu_percent}%",
                details={"cpu_percent": cpu_percent}
            )
        else:
            return HealthCheckResult(
                component="cpu",
                status="healthy",
                message=f"CPU usage normal: {cpu_percent}%",
                details={"cpu_percent": cpu_percent}
            )
    
    return [check_memory, check_disk, check_cpu]
