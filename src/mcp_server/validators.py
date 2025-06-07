# src/mcp_server/validators.py

import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Set
from urllib.parse import urlparse
from pathlib import Path

import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

from .error_handlers import ValidationError

logger = logging.getLogger(__name__)

class ParameterValidator:
    """
    Comprehensive parameter validation for MCP tools.
    Provides validation for various data types and formats.
    """
    
    # Model name patterns
    VALID_MODEL_PATTERNS = [
        r'^sentence-transformers/.*',
        r'^all-.*',
        r'^openai/.*',
        r'^cohere/.*',
        r'^huggingface/.*',
        r'^local/.*'
    ]
    
    # Collection name pattern (alphanumeric, hyphens, underscores)
    COLLECTION_NAME_PATTERN = r'^[a-zA-Z0-9_-]+$'
    
    # File extension patterns
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.xml', '.html'}
    
    def __init__(self):
        self.validation_cache: Dict[str, bool] = {}
    
    def validate_text_input(self, text: str, max_length: int = 10000, 
                           min_length: int = 1, allow_empty: bool = False) -> str:
        """Validate text input with length constraints."""
        if not isinstance(text, str):
            raise ValidationError("text", "Text input must be a string")
        
        if not allow_empty and len(text.strip()) < min_length:
            raise ValidationError("text", f"Text must be at least {min_length} characters long")
        
        if len(text) > max_length:
            raise ValidationError("text", f"Text must not exceed {max_length} characters")
        
        return text.strip()
    
    def validate_model_name(self, model_name: str) -> str:
        """Validate embedding model name."""
        if not isinstance(model_name, str):
            raise ValidationError("model_name", "Model name must be a string")
        
        if not model_name.strip():
            raise ValidationError("model_name", "Model name cannot be empty")
        
        # Check against known patterns
        for pattern in self.VALID_MODEL_PATTERNS:
            if re.match(pattern, model_name):
                return model_name
        
        # If no pattern matches, log warning but allow (for flexibility)
        logger.warning(f"Unknown model pattern: {model_name}")
        return model_name
    
    def validate_numeric_range(self, value: Union[int, float], param_name: str,
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> Union[int, float]:
        """Validate numeric value within specified range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(param_name, "Value must be a number")
        
        if min_val is not None and value < min_val:
            raise ValidationError(param_name, f"Value must be >= {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(param_name, f"Value must be <= {max_val}")
        
        return value
    
    def validate_collection_name(self, collection_name: str) -> str:
        """Validate collection name format."""
        if not isinstance(collection_name, str):
            raise ValidationError("collection_name", "Collection name must be a string")
        
        if not re.match(self.COLLECTION_NAME_PATTERN, collection_name):
            raise ValidationError(
                "collection_name", 
                "Collection name must contain only alphanumeric characters, hyphens, and underscores"
            )
        
        if len(collection_name) > 64:
            raise ValidationError("collection_name", "Collection name must not exceed 64 characters")
        
        return collection_name
    
    def validate_search_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search filter parameters."""
        if not isinstance(filters, dict):
            raise ValidationError("filters", "Filters must be a dictionary")
        
        validated_filters = {}
        
        for key, value in filters.items():
            # Validate filter key
            if not isinstance(key, str) or not key.strip():
                raise ValidationError("filters", f"Filter key '{key}' must be a non-empty string")
            
            # Validate filter value types
            if isinstance(value, (str, int, float, bool)):
                validated_filters[key] = value
            elif isinstance(value, list):
                # Validate list contents
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    validated_filters[key] = value
                else:
                    raise ValidationError("filters", f"Filter '{key}' contains invalid list items")
            elif isinstance(value, dict):
                # Handle range filters
                if set(value.keys()).issubset({'min', 'max', 'gte', 'lte', 'gt', 'lt'}):
                    validated_filters[key] = value
                else:
                    raise ValidationError("filters", f"Filter '{key}' contains invalid range operators")
            else:
                raise ValidationError("filters", f"Filter '{key}' has unsupported value type")
        
        return validated_filters
    
    def validate_file_path(self, file_path: str, check_exists: bool = False,
                          allowed_extensions: Optional[Set[str]] = None) -> str:
        """Validate file path format and optionally check existence."""
        if not isinstance(file_path, str):
            raise ValidationError("file_path", "File path must be a string")
        
        try:
            path = Path(file_path)
        except Exception as e:
            raise ValidationError("file_path", f"Invalid file path format: {e}")
        
        if allowed_extensions:
            if path.suffix.lower() not in allowed_extensions:
                raise ValidationError(
                    "file_path", 
                    f"File extension must be one of: {', '.join(allowed_extensions)}"
                )
        
        if check_exists and not path.exists():
            raise ValidationError("file_path", "File does not exist")
        
        return str(path)
    
    def validate_url(self, url: str) -> str:
        """Validate URL format."""
        if not isinstance(url, str):
            raise ValidationError("url", "URL must be a string")
        
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValidationError("url", "Invalid URL format")
        except Exception as e:
            raise ValidationError("url", f"Invalid URL: {e}")
        
        return url
    
    def validate_json_schema(self, data: Any, schema: Dict[str, Any], 
                           parameter_name: str = "data") -> Any:
        """Validate data against JSON schema."""
        try:
            validate(instance=data, schema=schema)
            return data
        except JsonSchemaValidationError as e:
            raise ValidationError(parameter_name, f"Schema validation failed: {e.message}")
    
    def validate_batch_size(self, batch_size: int, max_batch_size: int = 100) -> int:
        """Validate batch size parameter."""
        return int(self.validate_numeric_range(
            batch_size, "batch_size", min_val=1, max_val=max_batch_size
        ))
    
    def validate_algorithm_choice(self, algorithm: str, 
                                 allowed_algorithms: List[str]) -> str:
        """Validate algorithm choice from allowed options."""
        if not isinstance(algorithm, str):
            raise ValidationError("algorithm", "Algorithm must be a string")
        
        if algorithm not in allowed_algorithms:
            raise ValidationError(
                "algorithm", 
                f"Algorithm must be one of: {', '.join(allowed_algorithms)}"
            )
        
        return algorithm
    
    def validate_embedding_vector(self, embedding: List[float]) -> List[float]:
        """Validate embedding vector format."""
        if not isinstance(embedding, list):
            raise ValidationError("embedding", "Embedding must be a list")
        
        if not embedding:
            raise ValidationError("embedding", "Embedding cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValidationError("embedding", "Embedding must contain only numbers")
        
        return embedding
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata dictionary."""
        if not isinstance(metadata, dict):
            raise ValidationError("metadata", "Metadata must be a dictionary")
        
        # Check for reasonable size
        if len(json.dumps(metadata)) > 10000:  # 10KB limit
            raise ValidationError("metadata", "Metadata too large (max 10KB)")
        
        # Validate that all values are JSON serializable
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            raise ValidationError("metadata", f"Metadata must be JSON serializable: {e}")
        
        return metadata
    
    def validate_and_hash_args(self, args: Dict[str, Any]) -> str:
        """Validate arguments and return a hash for caching."""
        # Create a deterministic hash of the arguments
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(args_str.encode()).hexdigest()
    
    def create_tool_validator(self, schema: Dict[str, Any]):
        """Create a validator function for a specific tool schema."""
        def validator(args: Dict[str, Any]) -> Dict[str, Any]:
            return self.validate_json_schema(args, schema, "tool_arguments")
        
        return validator

# Predefined schemas for common tool parameters
COMMON_SCHEMAS = {
    "text_input": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 10000
            }
        },
        "required": ["text"]
    },
    
    "embedding_generation": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 10000
            },
            "model": {
                "type": "string",
                "minLength": 1
            },
            "normalize": {
                "type": "boolean",
                "default": True
            }
        },
        "required": ["text"]
    },
    
    "search_query": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1
            },
            "collection": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9_-]+$"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 10
            },
            "threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["query", "collection"]
    }
}

# Global validator instance
validator = ParameterValidator()
