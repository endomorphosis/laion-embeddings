"""
Monitoring and metrics collection for LAION Embeddings API.
Provides Prometheus-compatible metrics and system health monitoring.
"""

import time
import psutil
from typing import Dict, List
from collections import defaultdict, deque
from datetime import datetime, timedelta

class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_counts = defaultdict(int)
        self.request_durations = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.active_requests = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Keep only last 1000 duration measurements per endpoint
        self.max_duration_history = 1000
    
    def record_request_start(self, endpoint: str) -> float:
        """Record the start of a request and return start time."""
        self.active_requests += 1
        self.total_requests += 1
        self.request_counts[endpoint] += 1
        return time.time()
    
    def record_request_end(self, endpoint: str, start_time: float, status_code: int):
        """Record the end of a request."""
        duration = time.time() - start_time
        self.active_requests = max(0, self.active_requests - 1)
        
        # Store duration (keep only recent measurements)
        if len(self.request_durations[endpoint]) >= self.max_duration_history:
            self.request_durations[endpoint].pop(0)
        self.request_durations[endpoint].append(duration)
        
        # Count errors (4xx and 5xx status codes)
        if status_code >= 400:
            self.error_counts[endpoint] += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def get_system_metrics(self) -> Dict:
        """Get current system resource metrics."""
        try:
            process = psutil.Process()
            system = psutil
            
            # Memory metrics
            memory_info = process.memory_info()
            system_memory = system.virtual_memory()
            
            # CPU metrics
            cpu_percent = process.cpu_percent()
            system_cpu = system.cpu_percent(interval=None)
            
            # Disk metrics
            disk_usage = system.disk_usage('/')
            
            return {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "process_memory_percent": process.memory_percent(),
                "system_memory_percent": system_memory.percent,
                "system_memory_available_mb": system_memory.available / 1024 / 1024,
                "process_cpu_percent": cpu_percent,
                "system_cpu_percent": system_cpu,
                "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "disk_free_gb": disk_usage.free / 1024 / 1024 / 1024,
                "process_threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
        except Exception as e:
            return {"error": f"Failed to collect system metrics: {str(e)}"}
    
    def get_application_metrics(self) -> Dict:
        """Get application-specific metrics."""
        uptime = time.time() - self.start_time
        
        # Calculate request rate (requests per minute)
        request_rate = (self.total_requests / uptime) * 60 if uptime > 0 else 0
        
        # Calculate error rates per endpoint
        error_rates = {}
        for endpoint in self.request_counts:
            total = self.request_counts[endpoint]
            errors = self.error_counts[endpoint]
            error_rates[endpoint] = (errors / total) * 100 if total > 0 else 0
        
        # Calculate average response times per endpoint
        avg_response_times = {}
        for endpoint, durations in self.request_durations.items():
            if durations:
                avg_response_times[endpoint] = sum(durations) / len(durations)
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests) * 100 if total_cache_requests > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "request_rate_per_minute": request_rate,
            "request_counts_by_endpoint": dict(self.request_counts),
            "error_counts_by_endpoint": dict(self.error_counts),
            "error_rates_by_endpoint": error_rates,
            "avg_response_times_by_endpoint": avg_response_times,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate
        }
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-formatted metrics."""
        metrics = []
        
        # System metrics
        system_metrics = self.get_system_metrics()
        if "error" not in system_metrics:
            metrics.extend([
                f"# HELP process_memory_bytes Process memory usage in bytes",
                f"# TYPE process_memory_bytes gauge",
                f"process_memory_bytes {system_metrics['process_memory_mb'] * 1024 * 1024}",
                "",
                f"# HELP process_cpu_percent Process CPU usage percentage",
                f"# TYPE process_cpu_percent gauge", 
                f"process_cpu_percent {system_metrics['process_cpu_percent']}",
                "",
                f"# HELP system_memory_percent System memory usage percentage",
                f"# TYPE system_memory_percent gauge",
                f"system_memory_percent {system_metrics['system_memory_percent']}",
                ""
            ])
        
        # Application metrics
        app_metrics = self.get_application_metrics()
        
        metrics.extend([
            f"# HELP http_requests_total Total number of HTTP requests",
            f"# TYPE http_requests_total counter",
            f"http_requests_total {app_metrics['total_requests']}",
            "",
            f"# HELP http_requests_active Currently active HTTP requests",
            f"# TYPE http_requests_active gauge",
            f"http_requests_active {app_metrics['active_requests']}",
            "",
            f"# HELP application_uptime_seconds Application uptime in seconds",
            f"# TYPE application_uptime_seconds counter",
            f"application_uptime_seconds {app_metrics['uptime_seconds']}",
            ""
        ])
        
        # Per-endpoint metrics
        for endpoint, count in app_metrics['request_counts_by_endpoint'].items():
            endpoint_label = endpoint.replace('/', '_').replace('-', '_')
            metrics.extend([
                f"http_requests_total{{endpoint=\"{endpoint}\"}} {count}"
            ])
        
        metrics.append("")
        
        # Cache metrics
        metrics.extend([
            f"# HELP cache_hits_total Total number of cache hits",
            f"# TYPE cache_hits_total counter",
            f"cache_hits_total {app_metrics['cache_hits']}",
            "",
            f"# HELP cache_misses_total Total number of cache misses", 
            f"# TYPE cache_misses_total counter",
            f"cache_misses_total {app_metrics['cache_misses']}",
            ""
        ])
        
        return "\n".join(metrics)
    
    def get_health_status(self) -> Dict:
        """Get overall health status."""
        system_metrics = self.get_system_metrics()
        app_metrics = self.get_application_metrics()
        
        # Define health thresholds
        memory_threshold = 85  # percent
        cpu_threshold = 80     # percent
        error_rate_threshold = 5  # percent
        
        health_issues = []
        
        # Check system health
        if "error" not in system_metrics:
            if system_metrics["system_memory_percent"] > memory_threshold:
                health_issues.append(f"High memory usage: {system_metrics['system_memory_percent']:.1f}%")
            
            if system_metrics["process_cpu_percent"] > cpu_threshold:
                health_issues.append(f"High CPU usage: {system_metrics['process_cpu_percent']:.1f}%")
        
        # Check application health
        for endpoint, error_rate in app_metrics["error_rates_by_endpoint"].items():
            if error_rate > error_rate_threshold:
                health_issues.append(f"High error rate for {endpoint}: {error_rate:.1f}%")
        
        status = "healthy" if not health_issues else "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": app_metrics["uptime_seconds"],
            "issues": health_issues,
            "system_metrics": system_metrics,
            "application_metrics": app_metrics
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()
