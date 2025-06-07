import asyncio
import json
import sys

from src.mcp_server.server import MCPServer
from src.mcp_server.tool_registry import ToolRegistry
from src.mcp_server.tools.embedding_tools import EmbeddingGenerationTool
from src.mcp_server.tools.search_tools import SemanticSearchTool
from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
# from src.mcp_server.tools.storage_tools import StorageManagementTool, CollectionManagementTool, RetrievalTool
# from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool, QualityAssessmentTool, DimensionalityReductionTool

class ProtocolHandler:
    def __init__(self, mcp_server: MCPServer, ipfs_embeddings_instance: ipfs_embeddings_py):
        self.mcp_server = mcp_server
        self.ipfs_embeddings_instance = ipfs_embeddings_instance
        self.tool_registry = ToolRegistry(self.mcp_server, self.ipfs_embeddings_instance)
        self._register_all_tools()

    def _register_all_tools(self):
        # The ToolRegistry now handles registering all LAION-specific tools
        # It is initialized with the mcp_server and ipfs_embeddings_instance
        # so it will register the tools directly with the mcp_server.
        pass

        # Placeholder for other tool registrations
        # self.mcp_server.register_tool(
        #     "storage_management_tool",
        #     "Manages storage operations.",
        #     StorageManagementTool().input_schema,
        #     StorageManagementTool().run
        # )
        # self.mcp_server.register_tool(
        #     "cluster_analysis_tool",
        #     "Performs cluster analysis.",
        #     ClusterAnalysisTool().input_schema,
        #     ClusterAnalysisTool().run
        # )

    async def start(self):
        print("Starting LAION Embeddings MCP Server...")
        await self.mcp_server.run()

if __name__ == "__main__":
    # In a real scenario, mcp_server and ipfs_embeddings_instance would be passed from a higher level
    # For standalone execution, we'll instantiate them here.
    mcp_server_instance = MCPServer("laion-embeddings-mcp", "0.1.0")
    ipfs_embeddings_instance = ipfs_embeddings_py(resources={}, metadata={})
    handler = ProtocolHandler(mcp_server_instance, ipfs_embeddings_instance)
    asyncio.run(handler.start())
