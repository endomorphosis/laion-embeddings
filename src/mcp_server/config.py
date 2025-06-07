# src/mcp_server/config.py

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class MCPConfig:
    """
    Configuration settings for the MCP server.
    Loads settings from environment variables with sensible defaults.
    """
    # Server Configuration
    server_name: str = field(default_factory=lambda: os.getenv("MCP_SERVER_NAME", "laion-embeddings-mcp"))
    server_version: str = field(default_factory=lambda: os.getenv("MCP_SERVER_VERSION", "1.0.0"))
    server_description: str = field(default_factory=lambda: os.getenv("MCP_SERVER_DESCRIPTION", "LAION Embeddings MCP Server"))
    
    # Network Configuration
    host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("MCP_PORT", "8000")))
    allowed_origins: List[str] = field(default_factory=lambda: 
        os.getenv("MCP_ALLOWED_ORIGINS", "*").split(",") if os.getenv("MCP_ALLOWED_ORIGINS") else ["*"])
    
    # Security Configuration
    api_key: str = field(default_factory=lambda: os.getenv("MCP_API_KEY", ""))
    max_request_size: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_REQUEST_SIZE", "10485760")))  # 10MB
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("MCP_RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("MCP_RATE_LIMIT_WINDOW", "60")))  # seconds
    
    # Session Configuration
    session_timeout: int = field(default_factory=lambda: int(os.getenv("MCP_SESSION_TIMEOUT", "3600")))  # 1 hour
    max_sessions: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_SESSIONS", "1000")))
    session_cleanup_interval: int = field(default_factory=lambda: int(os.getenv("MCP_SESSION_CLEANUP_INTERVAL", "300")))  # 5 minutes
    
    # Tool Configuration
    enabled_tools: List[str] = field(default_factory=lambda: 
        os.getenv("MCP_ENABLED_TOOLS", "").split(",") if os.getenv("MCP_ENABLED_TOOLS") else [])
    tool_timeout: int = field(default_factory=lambda: int(os.getenv("MCP_TOOL_TIMEOUT", "300")))  # 5 minutes
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_BATCH_SIZE", "100")))
    
    # Database Configuration
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///mcp_server.db"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("MCP_CACHE_TTL", "3600")))
    
    # Model Configuration
    default_embedding_model: str = field(default_factory=lambda: 
        os.getenv("MCP_DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    model_cache_dir: str = field(default_factory=lambda: 
        os.getenv("MCP_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "mcp_models")))
    max_model_memory: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_MODEL_MEMORY", "2048")))  # MB
    
    # Storage Configuration
    storage_backend: str = field(default_factory=lambda: os.getenv("MCP_STORAGE_BACKEND", "local"))
    local_storage_path: str = field(default_factory=lambda: 
        os.getenv("MCP_LOCAL_STORAGE_PATH", str(Path.home() / "mcp_storage")))
    ipfs_node_url: str = field(default_factory=lambda: os.getenv("IPFS_NODE_URL", "http://localhost:5001"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("AWS_S3_BUCKET", ""))
    s3_region: str = field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    
    # Performance Configuration
    worker_threads: int = field(default_factory=lambda: int(os.getenv("MCP_WORKER_THREADS", "4")))
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_CONCURRENT_REQUESTS", "50")))
    request_queue_size: int = field(default_factory=lambda: int(os.getenv("MCP_REQUEST_QUEUE_SIZE", "1000")))
    
    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv("MCP_LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("MCP_LOG_FILE", ""))
    log_format: str = field(default_factory=lambda: 
        os.getenv("MCP_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Monitoring Configuration
    metrics_enabled: bool = field(default_factory=lambda: 
        os.getenv("MCP_METRICS_ENABLED", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("MCP_METRICS_PORT", "9090")))
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("MCP_HEALTH_CHECK_INTERVAL", "30")))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        
        if self.session_timeout <= 0:
            raise ValueError(f"Invalid session timeout: {self.session_timeout}")
        
        if self.tool_timeout <= 0:
            raise ValueError(f"Invalid tool timeout: {self.tool_timeout}")
        
        if self.max_batch_size <= 0:
            raise ValueError(f"Invalid max batch size: {self.max_batch_size}")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.local_storage_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> "MCPConfig":
        """Load configuration from environment file."""
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        return cls()
