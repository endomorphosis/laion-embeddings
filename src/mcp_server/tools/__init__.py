# src/mcp_server/tools/__init__.py

"""
MCP Tools Package

This package contains all the tool implementations for the LAION Embeddings MCP server.
Tools are organized into categories:

- embedding_tools: Text and multimodal embedding generation
- search_tools: Semantic and similarity search capabilities  
- storage_tools: File and collection management
- analysis_tools: Data analysis and clustering operations
"""

from .embedding_tools import (
    EmbeddingGenerationTool,
    BatchEmbeddingTool, 
    MultimodalEmbeddingTool
)

from .search_tools import (
    SemanticSearchTool,
    SimilaritySearchTool,
    FacetedSearchTool
)

from .storage_tools import (
    StorageManagementTool,
    CollectionManagementTool,
    RetrievalTool
)

from .analysis_tools import (
    ClusterAnalysisTool,
    QualityAssessmentTool,
    DimensionalityReductionTool
)

__all__ = [
    # Embedding tools
    "EmbeddingGenerationTool",
    "BatchEmbeddingTool",
    "MultimodalEmbeddingTool",
    
    # Search tools  
    "SemanticSearchTool",
    "SimilaritySearchTool", 
    "FacetedSearchTool",
    
    # Storage tools
    "StorageManagementTool",
    "CollectionManagementTool",
    "RetrievalTool",
    
    # Analysis tools
    "ClusterAnalysisTool",
    "QualityAssessmentTool", 
    "DimensionalityReductionTool",
]
