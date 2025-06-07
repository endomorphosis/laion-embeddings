"""
Vector Database Configuration Management

This module provides unified configuration loading and management for all supported
vector databases (Qdrant, Elasticsearch, pgvector, FAISS).
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class VectorDBType(Enum):
    """Supported vector database types."""
    QDRANT = "qdrant"
    ELASTICSEARCH = "elasticsearch"
    PGVECTOR = "pgvector"
    FAISS = "faiss"
    IPFS = "ipfs"
    DUCKDB = "duckdb"


class DistanceMetric(Enum):
    """Supported distance metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorDBConfig:
    """Configuration for a vector database."""
    db_type: VectorDBType
    enabled: bool
    connection_params: Dict[str, Any]
    index_params: Dict[str, Any]
    search_params: Dict[str, Any]
    performance_params: Dict[str, Any]


@dataclass
class GlobalConfig:
    """Global configuration settings."""
    embedding: Dict[str, Any]
    search: Dict[str, Any]
    performance: Dict[str, Any]
    monitoring: Dict[str, Any]
    migration: Dict[str, Any]


class VectorDatabaseConfigManager:
    """Manages vector database configurations."""
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (development, production, testing)
        """
        self.environment = environment
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config" / "vector_databases.yaml"
        return str(config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply environment-specific overrides
            if 'environments' in config and self.environment in config['environments']:
                env_config = config['environments'][self.environment]
                config = self._merge_configs(config, env_config)
            
            # Expand environment variables
            config = self._expand_env_vars(config)
            
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _expand_env_vars(self, config: Any) -> Any:
        """Expand environment variables in configuration values."""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _validate_config(self):
        """Validate configuration structure and values."""
        required_sections = ['databases', 'global']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate database configurations
        for db_name, db_config in self.config['databases'].items():
            if not isinstance(db_config.get('enabled'), bool):
                raise ValueError(f"Database {db_name} must have 'enabled' boolean field")
    
    def get_database_config(self, db_type: VectorDBType) -> Optional[VectorDBConfig]:
        """
        Get configuration for a specific database type.
        
        Args:
            db_type: Type of vector database
            
        Returns:
            Database configuration or None if not enabled
        """
        db_name = db_type.value
        if db_name not in self.config['databases']:
            return None
        
        db_config = self.config['databases'][db_name]
        if not db_config.get('enabled', False):
            return None
        
        # Extract configuration sections
        connection_params = self._extract_connection_params(db_type, db_config)
        index_params = db_config.get('index', {})
        search_params = db_config.get('search', {})
        performance_params = self._extract_performance_params(db_config)
        
        return VectorDBConfig(
            db_type=db_type,
            enabled=db_config['enabled'],
            connection_params=connection_params,
            index_params=index_params,
            search_params=search_params,
            performance_params=performance_params
        )
    
    def _extract_connection_params(self, db_type: VectorDBType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract connection parameters for a database type."""
        if db_type == VectorDBType.QDRANT:
            return {
                'host': config.get('host', 'localhost'),
                'port': config.get('port', 6333),
                'collection_name': config.get('collection_name', 'embeddings'),
                'timeout': config.get('timeout', 30),
                'prefer_grpc': config.get('prefer_grpc', False),
                'https': config.get('https', False),
                'api_key': config.get('api_key'),
                'prefix': config.get('prefix'),
            }
        elif db_type == VectorDBType.ELASTICSEARCH:
            return {
                'host': config.get('host', 'localhost'),
                'port': config.get('port', 9200),
                'index_name': config.get('index_name', 'embeddings'),
                'timeout': config.get('timeout', 30),
                'max_retries': config.get('max_retries', 3),
            }
        elif db_type == VectorDBType.PGVECTOR:
            return {
                'connection_string': config.get('connection_string'),
                'table_name': config.get('table_name', 'embeddings'),
                'pool_size': config.get('pool_size', 10),
                'max_overflow': config.get('max_overflow', 20),
                'pool_timeout': config.get('pool_timeout', 30),
                'pool_recycle': config.get('pool_recycle', 3600),
            }
        elif db_type == VectorDBType.FAISS:
            return {
                'storage_path': config.get('storage_path', 'data/faiss_indexes'),
                'index_type': config.get('index_type', 'IndexHNSWFlat'),
                'metric_type': config.get('metric_type', 'METRIC_INNER_PRODUCT'),
            }
        else:
            return {}
    
    def _extract_performance_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance parameters."""
        global_perf = self.config.get('global', {}).get('performance', {})
        db_perf = config.get('performance', {})
        
        # Merge global and database-specific performance settings
        return {**global_perf, **db_perf}
    
    def get_global_config(self) -> GlobalConfig:
        """Get global configuration settings."""
        global_config = self.config.get('global', {})
        
        return GlobalConfig(
            embedding=global_config.get('embedding', {}),
            search=global_config.get('search', {}),
            performance=global_config.get('performance', {}),
            monitoring=global_config.get('monitoring', {}),
            migration=global_config.get('migration', {})
        )
    
    def get_enabled_databases(self) -> List[VectorDBType]:
        """Get list of enabled database types."""
        enabled = []
        for db_type in VectorDBType:
            config = self.get_database_config(db_type)
            if config and config.enabled:
                enabled.append(db_type)
        return enabled
    
    def get_default_database(self) -> VectorDBType:
        """Get the default database type."""
        default_name = self.config.get('default', 'qdrant')
        try:
            return VectorDBType(default_name)
        except ValueError:
            # Fallback to first enabled database
            enabled = self.get_enabled_databases()
            if enabled:
                return enabled[0]
            else:
                raise ValueError("No enabled vector databases found")
    
    def is_database_enabled(self, db_type: VectorDBType) -> bool:
        """Check if a database type is enabled."""
        config = self.get_database_config(db_type)
        return config is not None and config.enabled
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self.config.get('global', {}).get('embedding', {})
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self.config.get('global', {}).get('search', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.config.get('global', {}).get('monitoring', {})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config = self._merge_configs(self.config, updates)
        self._validate_config()
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


# Global configuration manager instance
_config_manager: Optional[VectorDatabaseConfigManager] = None


def get_config_manager(config_path: Optional[str] = None, 
                      environment: Optional[str] = None) -> VectorDatabaseConfigManager:
    """
    Get or create global configuration manager instance.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None:
        env = environment or os.getenv('VECTOR_DB_ENV', 'development')
        _config_manager = VectorDatabaseConfigManager(config_path, env)
    
    return _config_manager


def reset_config_manager() -> None:
    """Reset global configuration manager (for testing)."""
    global _config_manager
    _config_manager = None
