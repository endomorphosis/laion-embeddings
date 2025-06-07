# src/mcp_server/error_handlers.py

import logging
import traceback
from typing import Optional, Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPError(Exception):
    """Base exception for MCP server errors."""
    def __init__(self, code: int, message: str, data: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(f"Error {code}: {message}")

class ToolExecutionError(MCPError):
    """Raised when tool execution fails."""
    def __init__(self, tool_name: str, message: str, original_error: Optional[Exception] = None):
        self.tool_name = tool_name
        self.original_error = original_error
        data = {
            "tool_name": tool_name,
            "original_error": str(original_error) if original_error else None,
            "traceback": traceback.format_exc() if original_error else None
        }
        super().__init__(-32000, f"Tool execution failed for '{tool_name}': {message}", data)

class ValidationError(MCPError):
    """Raised when parameter validation fails."""
    def __init__(self, parameter: str, message: str, value: Any = None):
        self.parameter = parameter
        self.value = value
        data = {
            "parameter": parameter,
            "value": str(value) if value is not None else None,
            "validation_message": message
        }
        super().__init__(-32602, f"Validation failed for parameter '{parameter}': {message}", data)

class SessionError(MCPError):
    """Raised when session management fails."""
    def __init__(self, session_id: str, message: str):
        self.session_id = session_id
        data = {"session_id": session_id}
        super().__init__(-32001, f"Session error for '{session_id}': {message}", data)

class ToolNotFoundError(MCPError):
    """Raised when a requested tool is not found."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        data = {"tool_name": tool_name}
        super().__init__(-32601, f"Tool '{tool_name}' not found", data)

class InvalidRequestError(MCPError):
    """Raised when the request format is invalid."""
    def __init__(self, message: str, request_data: Optional[Dict[str, Any]] = None):
        data = {"request_data": request_data}
        super().__init__(-32600, f"Invalid request: {message}", data)

class ParseError(MCPError):
    """Raised when JSON parsing fails."""
    def __init__(self, message: str):
        super().__init__(-32700, f"Parse error: {message}")

class InternalError(MCPError):
    """Raised for internal server errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(-32603, f"Internal error: {message}", details)

class ResourceNotFoundError(MCPError):
    """Raised when a requested resource is not found."""
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        data = {"resource_type": resource_type, "resource_id": resource_id}
        super().__init__(-32004, f"{resource_type} '{resource_id}' not found", data)

class RateLimitError(MCPError):
    """Raised when rate limit is exceeded."""
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        data = {"limit": limit, "window": window}
        super().__init__(-32005, f"Rate limit exceeded: {limit} requests per {window} seconds", data)

class AuthenticationError(MCPError):
    """Raised when authentication fails."""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(-32006, message)

class AuthorizationError(MCPError):
    """Raised when authorization fails."""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(-32007, message)

class TimeoutError(MCPError):
    """Raised when operation times out."""
    def __init__(self, operation: str, timeout: int):
        self.operation = operation
        self.timeout = timeout
        data = {"operation": operation, "timeout": timeout}
        super().__init__(-32008, f"Operation '{operation}' timed out after {timeout} seconds", data)

def format_error_response(error: Union[MCPError, Exception], request_id: Optional[str] = None) -> Dict[str, Any]:
    """Format an error as a JSON-RPC 2.0 error response."""
    if isinstance(error, MCPError):
        error_obj = {
            "code": error.code,
            "message": error.message,
            "data": {
                **error.data,
                "timestamp": error.timestamp
            }
        }
    else:
        # Handle unexpected exceptions
        logger.error(f"Unexpected error: {error}", exc_info=True)
        error_obj = {
            "code": -32603,
            "message": "Internal error",
            "data": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error_obj
    }
    
    return response

def handle_tool_error(tool_name: str, error: Exception) -> ToolExecutionError:
    """Convert various exceptions to ToolExecutionError."""
    if isinstance(error, ToolExecutionError):
        return error
    
    logger.error(f"Tool '{tool_name}' execution failed: {error}", exc_info=True)
    
    if isinstance(error, ValueError):
        return ToolExecutionError(tool_name, f"Validation error: {error}", error)
    elif isinstance(error, FileNotFoundError):
        return ToolExecutionError(tool_name, f"File not found: {error}", error)
    elif isinstance(error, PermissionError):
        return ToolExecutionError(tool_name, f"Permission denied: {error}", error)
    elif isinstance(error, TimeoutError):
        return ToolExecutionError(tool_name, f"Operation timed out: {error}", error)
    else:
        return ToolExecutionError(tool_name, str(error), error)

def log_error(error: Union[MCPError, Exception], context: Optional[Dict[str, Any]] = None):
    """Log error with appropriate level and context."""
    context = context or {}
    
    if isinstance(error, MCPError):
        log_data = {
            "error_code": error.code,
            "error_message": error.message,
            "error_data": error.data,
            **context
        }
        
        if error.code >= -32099 and error.code <= -32000:
            # Server errors
            logger.error(f"Server error: {error.message}", extra=log_data)
        elif error.code >= -32699 and error.code <= -32600:
            # Invalid request errors
            logger.warning(f"Invalid request: {error.message}", extra=log_data)
        else:
            # Application errors
            logger.info(f"Application error: {error.message}", extra=log_data)
    else:
        logger.error(f"Unexpected error: {error}", extra=context, exc_info=True)
