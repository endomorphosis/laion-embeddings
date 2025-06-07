# src/mcp_server/tools/vector_store_tools.py

import logging
from typing import Dict, Any, List, Optional, Union
from ..tool_registry import ClaudeMCPTool
from ..validators import validator

logger = logging.getLogger(__name__)

class VectorIndexTool(ClaudeMCPTool):
    """
    Tool for managing vector indexes.
    """
    
    def __init__(self, vector_service):
        super().__init__()
        if vector_service is None:
            raise ValueError("Vector service cannot be None")
            
        self.name = "manage_vector_index"
        self.description = "Create, update, or manage vector indexes for efficient search."
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "delete", "info"],
                    "description": "Action to perform on the vector index."
                },
                "index_name": {
                    "type": "string",
                    "description": "Name of the vector index.",
                    "minLength": 1,
                    "maxLength": 100
                },
                "config": {
                    "type": "object",
                    "description": "Configuration for index creation/update.",
                    "properties": {
                        "dimension": {"type": "integer", "minimum": 1},
                        "metric": {"type": "string", "enum": ["cosine", "euclidean", "dot"]},
                        "index_type": {"type": "string", "enum": ["faiss", "hnswlib", "annoy"]}
                    }
                }
            },
            "required": ["action", "index_name"]
        }
        self.vector_service = vector_service
    
    async def execute(self, action: str, index_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute vector index management operation."""
        try:
            # Validate inputs
            action = validator.validate_algorithm_choice(action, ["create", "update", "delete", "info"])
            index_name = validator.validate_text_input(index_name)
            
            # Call the vector service
            if action == "create":
                result = await self.vector_service.create_index(index_name, config or {})
            elif action == "update":
                result = await self.vector_service.update_index(index_name, config or {})
            elif action == "delete":
                result = await self.vector_service.delete_index(index_name)
            else:  # info
                result = await self.vector_service.get_index_info(index_name)
            
            return {
                "action": action,
                "index_name": index_name,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vector index operation failed: {e}")
            raise


class VectorRetrievalTool(ClaudeMCPTool):
    """
    Tool for retrieving vectors from storage.
    """
    
    def __init__(self, vector_service):
        super().__init__()
        if vector_service is None:
            raise ValueError("Vector service cannot be None")
            
        self.name = "retrieve_vectors"
        self.description = "Retrieve vectors from storage with optional filtering."
        self.input_schema = {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name to retrieve from.",
                    "default": "default"
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific vector IDs to retrieve.",
                    "minItems": 1,
                    "maxItems": 1000
                },
                "filters": {
                    "type": "object",
                    "description": "Metadata filters for retrieval."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of vectors to retrieve.",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 100
                }
            },
            "required": []
        }
        self.vector_service = vector_service
    
    async def execute(self, collection: str = "default", ids: Optional[List[str]] = None, 
                     filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> Dict[str, Any]:
        """Execute vector retrieval operation."""
        try:
            # Validate inputs
            collection = validator.validate_text_input(collection)
            
            if ids:
                for id_val in ids:
                    validator.validate_text_input(id_val)
            
            # Call the vector service
            vectors = await self.vector_service.retrieve_vectors(
                collection=collection,
                ids=ids,
                filters=filters or {},
                limit=limit
            )
            
            return {
                "collection": collection,
                "vectors": vectors,
                "count": len(vectors),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            raise


class VectorMetadataTool(ClaudeMCPTool):
    """
    Tool for managing vector metadata.
    """
    
    def __init__(self, vector_service):
        super().__init__()
        if vector_service is None:
            raise ValueError("Vector service cannot be None")
            
        self.name = "manage_vector_metadata"
        self.description = "Manage metadata associated with vectors."
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "update", "delete", "list"],
                    "description": "Action to perform on vector metadata."
                },
                "collection": {
                    "type": "string",
                    "description": "Collection name.",
                    "default": "default"
                },
                "vector_id": {
                    "type": "string",
                    "description": "ID of the vector (required for get, update, delete)."
                },
                "metadata": {
                    "type": "object",
                    "description": "Metadata to update (required for update action)."
                },
                "filters": {
                    "type": "object",
                    "description": "Filters for listing metadata."
                }
            },
            "required": ["action"]
        }
        self.vector_service = vector_service
    
    async def execute(self, action: str, collection: str = "default", 
                     vector_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute vector metadata management operation."""
        try:
            # Validate inputs
            action = validator.validate_algorithm_choice(action, ["get", "update", "delete", "list"])
            collection = validator.validate_text_input(collection)
            
            if vector_id:
                vector_id = validator.validate_text_input(vector_id)
            
            # Call the vector service
            if action == "get":
                if not vector_id:
                    raise ValueError("vector_id is required for get action")
                result = await self.vector_service.get_vector_metadata(collection, vector_id)
            elif action == "update":
                if not vector_id or not metadata:
                    raise ValueError("vector_id and metadata are required for update action")
                result = await self.vector_service.update_vector_metadata(collection, vector_id, metadata)
            elif action == "delete":
                if not vector_id:
                    raise ValueError("vector_id is required for delete action")
                result = await self.vector_service.delete_vector_metadata(collection, vector_id)
            else:  # list
                result = await self.vector_service.list_vector_metadata(collection, filters or {})
            
            return {
                "action": action,
                "collection": collection,
                "vector_id": vector_id,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vector metadata operation failed: {e}")
            raise
