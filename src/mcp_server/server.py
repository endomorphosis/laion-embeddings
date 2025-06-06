import json
import sys
import asyncio
import logging
from typing import Optional, Dict, Any

from .config import MCPConfig
from .session_manager import SessionManager
from .monitoring import MetricsCollector, PerformanceMonitor
from .tool_registry import ToolRegistry
from .error_handlers import (
    format_error_response, handle_tool_error, log_error,
    ParseError, InvalidRequestError, ToolNotFoundError, InternalError
)

logger = logging.getLogger(__name__)

class MCPServer:
    """
    Enhanced MCP server with comprehensive session management, monitoring, and error handling.
    """
    
    def __init__(self, name: str, version: str, session_manager: SessionManager,
                 metrics_collector: MetricsCollector, tool_registry: ToolRegistry,
                 config: MCPConfig):
        self.name = name
        self.version = version
        self.session_manager = session_manager
        self.metrics_collector = metrics_collector
        self.tool_registry = tool_registry
        self.config = config
        
        # Server state
        self.is_running = False
        self.request_count = 0
        
        logger.info(f"MCP Server initialized: {name} v{version}")
    
    async def _handle_request(self, request_str: str) -> Optional[Dict[str, Any]]:
        """Handle an incoming JSON-RPC request."""
        start_time = asyncio.get_event_loop().time()
        request_id = None
        method = None # Initialize method to None
        success = False
        
        try:
            # Parse JSON request
            try:
                request = json.loads(request_str)
            except json.JSONDecodeError as e:
                error = ParseError(f"Invalid JSON: {e}")
                return format_error_response(error, None)
            
            # Extract request components
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if not method:
                error = InvalidRequestError("Missing 'method' field")
                return format_error_response(error, request_id)
            
            # Handle request based on method
            with PerformanceMonitor(self.metrics_collector, f"request_{method.replace('.', '_')}"):
                if method == "initialize":
                    result = await self._handle_initialize(params)
                elif method == "tools/list":
                    result = await self._handle_list_tools(params)
                elif method == "tools/call":
                    result = await self._handle_call_tool(params)
                elif method == "ping":
                    result = await self._handle_ping(params)
                elif method == "shutdown":
                    result = await self._handle_shutdown(params)
                    self.is_running = False
                else:
                    error = InvalidRequestError(f"Unknown method: {method}")
                    return format_error_response(error, request_id)
            
            success = True
            
            # Create success response
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
            return response
            
        except Exception as e:
            error = handle_tool_error("request_handler", e)
            log_error(error, {"request_id": request_id, "method": method or "unknown"})
            return format_error_response(error, request_id)
        
        finally:
            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            self.metrics_collector.record_request(duration, success)
            self.request_count += 1
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client initialization."""
        logger.info("Client initializing connection")
        
        # Extract client information
        client_info = {
            "client_name": params.get("clientInfo", {}).get("name", "unknown"),
            "client_version": params.get("clientInfo", {}).get("version", "unknown"),
            "capabilities": params.get("capabilities", {})
        }
        
        # Create session for client
        session_id = self.session_manager.create_session(client_info)
        self.metrics_collector.record_session_created()
        
        # Return server capabilities
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": self.version
            },
            "capabilities": {
                "tools": {
                    "listChanged": False
                },
                "resources": {},
                "prompts": {},
                "logging": {}
            },
            "sessionId": session_id
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools list request."""
        session_id = params.get("sessionId")
        
        # Validate session if provided
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                raise InvalidRequestError("Invalid session ID")
        
        # Get all available tools
        tools = self.tool_registry.get_all_tools() # Corrected method call
        
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema
                }
                for tool in tools
            ]
        }
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution request."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        session_id = params.get("sessionId")
        
        if not tool_name:
            raise InvalidRequestError("Missing tool 'name'")
        
        # Validate session if provided
        session = None
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                raise InvalidRequestError("Invalid session ID")
        
        # Check for cached result
        if session_id and session and self.config.cache_ttl > 0:
            args_hash = str(hash(json.dumps(tool_args, sort_keys=True)))
            cached_result = await self.session_manager.get_cached_tool_result(
                session_id, tool_name, args_hash
            )
            if cached_result is not None:
                logger.debug(f"Returning cached result for {tool_name}")
                return cached_result
        
        # Execute the tool
        try:
            with PerformanceMonitor(self.metrics_collector, f"tool_{tool_name}"):
                result = await self.tool_registry.execute_tool(tool_name, tool_args)
            
            # Cache the result if session exists
            if session_id and session and self.config.cache_ttl > 0:
                args_hash = str(hash(json.dumps(tool_args, sort_keys=True)))
                await self.session_manager.cache_tool_result(
                    session_id, tool_name, args_hash, result
                )
            
            return result
            
        except Exception as e:
            error = handle_tool_error(tool_name, e)
            log_error(error, {"tool_name": tool_name, "session_id": session_id})
            raise error
    
    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request for health checking."""
        return {
            "status": "ok",
            "timestamp": asyncio.get_event_loop().time(),
            "server": self.name,
            "version": self.version,
            "request_count": self.request_count
        }
    
    async def _handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Received shutdown request")
        return {"status": "shutting_down"}
    
    async def _send_response(self, response: Dict[str, Any]):
        """Send response to client via stdout."""
        try:
            response_str = json.dumps(response)
            sys.stdout.write(response_str + "\n")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            raise InternalError(f"Response sending failed: {e}")
    
    async def run(self):
        """Run the MCP server main loop."""
        self.is_running = True
        logger.info(f"MCP server {self.name} starting...")
        
        try:
            while self.is_running:
                # Read from stdin
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    
                    if not line:
                        # EOF reached
                        logger.info("EOF reached, shutting down")
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle the request
                    response = await self._handle_request(line)
                    
                    # Send response if we have one
                    if response:
                        await self._send_response(response)
                        
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    error_response = format_error_response(
                        InternalError(f"Server error: {e}"), None
                    )
                    await self._send_response(error_response)
                    
        except Exception as e:
            logger.error(f"Fatal server error: {e}")
            raise
        
        finally:
            logger.info("MCP server stopped")
    
    async def shutdown(self):
        """Shutdown the server gracefully."""
        logger.info("Shutting down MCP server...")
        self.is_running = False
        
        # Additional cleanup can be added here
        logger.info("MCP server shutdown complete")

if __name__ == "__main__":
    # This is kept for backward compatibility but main.py should be used instead
    import asyncio
    from .main import main
    asyncio.run(main())
