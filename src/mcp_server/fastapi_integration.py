import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.mcp_server.server import MCPServer
from src.mcp_server.tool_registry import ToolRegistry, initialize_laion_tools
from src.mcp_server.config import MCPConfig
from src.mcp_server.session_manager import SessionManager
from src.mcp_server.validators import validator
from src.mcp_server.error_handlers import MCPError, ToolNotFoundError, ValidationError
from src.mcp_server.monitoring import MetricsCollector, PerformanceMonitor
from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py

logger = logging.getLogger(__name__)

def create_fastapi_app() -> FastAPI:
    """
    Creates and configures the FastAPI application for the MCP server.
    """
    # Initialize configuration
    config = MCPConfig()
    
    app = FastAPI(
        title="LAION Embeddings MCP Server",
        description="Model Context Protocol server exposing LAION Embeddings functionalities.",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize core MCP components
    # Placeholder for resources and metadata, these might need to be loaded from config or passed in
    ipfs_embeddings_instance = ipfs_embeddings_py(resources={}, metadata={})
    
    # Initialize dependencies for MCPServer
    session_manager_instance = SessionManager(config)
    metrics_collector_instance = MetricsCollector(config=config)
    tool_registry_instance = ToolRegistry()
    
    # Initialize tools
    initialize_laion_tools(tool_registry_instance, ipfs_embeddings_instance)
    
    # Create MCP Server instance
    mcp_server_instance = MCPServer(
        name="laion-embeddings-mcp", 
        version="0.1.0", 
        session_manager=session_manager_instance,
        metrics_collector=metrics_collector_instance,
        tool_registry=tool_registry_instance,
        config=config
    )

    # Store instances in app state for access in endpoints
    app.state.mcp_server = mcp_server_instance
    app.state.tool_registry = tool_registry_instance
    app.state.session_manager = session_manager_instance
    app.state.ipfs_embeddings = ipfs_embeddings_instance
    app.state.metrics_collector = metrics_collector_instance # Store metrics collector in app state

    @app.on_event("startup")
    async def startup_event():
        logger.info("FastAPI application startup.")
        # You might want to initialize models or other heavy resources here
        # For example, pre-load a default embedding model
        # await app.state.ipfs_embeddings.init_endpoints(models=["sentence-transformers/all-MiniLM-L6-v2"])
        pass

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("FastAPI application shutdown.")
        # Clean up resources if necessary
        app.state.metrics_collector.shutdown() # Shutdown metrics collector
        pass

    @app.get("/mcp/list_tools")
    async def list_tools_endpoint():
        """
        Lists all available tools exposed by the MCP server.
        """
        try:
            # Use the new tool registry to get all tools
            tools_list = app.state.tool_registry.list_tools()
            return JSONResponse(content={"tools": tools_list})
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/mcp/call_tool")
    async def call_tool_endpoint(request: Request):
        """
        Calls a specific tool with provided arguments.
        """
        tool_name: Optional[str] = None
        try:
            body = await request.json()
            tool_name = body.get("tool_name")
            arguments = body.get("arguments", {})
            session_id = body.get("session_id")  # Optional session ID

            if not tool_name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tool name is required.")

            # Check if tool exists
            if not app.state.tool_registry.has_tool(tool_name):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found")

            # Validate arguments against tool schema
            if not app.state.tool_registry.validate_tool_parameters(tool_name, arguments):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tool arguments")

            # Execute the tool using the tool registry
            tool_result = await app.state.tool_registry.execute_tool(tool_name, arguments)
            
            return JSONResponse(content={"result": tool_result})

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return app

# This part is for running the FastAPI app directly (e.g., with `uvicorn`)
if __name__ == "__main__":
    import uvicorn
    app = create_fastapi_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
