"""
MCP tool wrapper for workflow orchestration and pipeline management.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
import json
import os
import uuid
from datetime import datetime

from create_embeddings.create_embeddings import CreateEmbeddingsProcessor
from shard_embeddings.shard_embeddings import ShardEmbeddingsProcessor
from ipfs_cluster_index.ipfs_cluster_index import IPFSClusterIndex
from services.vector_store_factory import VectorStoreFactory


class WorkflowExecutor:
    """
    Executor for complex multi-step workflows.
    """
    
    def __init__(self):
        self.workflows = {}
        self.execution_history = {}
    
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow with multiple steps.
        """
        workflow_id = str(uuid.uuid4())
        self.workflows[workflow_id] = workflow_definition
        
        try:
            steps = workflow_definition.get("steps", [])
            results = {}
            
            for i, step in enumerate(steps):
                step_id = f"step_{i}"
                step_type = step.get("type")
                step_params = step.get("parameters", {})
                
                # Execute step based on type
                if step_type == "create_embeddings":
                    result = await self._execute_create_embeddings(step_params)
                elif step_type == "shard_embeddings":
                    result = await self._execute_shard_embeddings(step_params)
                elif step_type == "index_cluster":
                    result = await self._execute_index_cluster(step_params)
                elif step_type == "vector_store_operation":
                    result = await self._execute_vector_store_operation(step_params)
                else:
                    result = {"success": False, "error": f"Unknown step type: {step_type}"}
                
                results[step_id] = result
                
                # Stop execution if step failed and workflow is not set to continue on error
                if not result.get("success") and not workflow_definition.get("continue_on_error", False):
                    break
            
            # Store execution history
            execution_record = {
                "workflow_id": workflow_id,
                "definition": workflow_definition,
                "results": results,
                "executed_at": datetime.utcnow().isoformat(),
                "success": all(r.get("success", False) for r in results.values())
            }
            
            self.execution_history[workflow_id] = execution_record
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": results,
                "execution_summary": execution_record
            }
            
        except Exception as e:
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }
    
    async def _execute_create_embeddings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute create embeddings step."""
        try:
            from .create_embeddings_tool import create_embeddings_tool
            return await create_embeddings_tool(**params)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_shard_embeddings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shard embeddings step."""
        try:
            from .shard_embeddings_tool import shard_embeddings_tool
            return await shard_embeddings_tool(**params)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_index_cluster(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute IPFS cluster indexing step."""
        try:
            cluster_index = IPFSClusterIndex(params.get("config", {}))
            result = await asyncio.to_thread(cluster_index.index_embeddings, **params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_vector_store_operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector store operation step."""
        try:
            operation = params.get("operation")
            if operation == "create":
                from .vector_store_tools import create_vector_store_tool
                return await create_vector_store_tool(**params.get("args", {}))
            elif operation == "add":
                from .vector_store_tools import add_embeddings_to_store_tool
                return await add_embeddings_to_store_tool(**params.get("args", {}))
            elif operation == "search":
                from .vector_store_tools import search_vector_store_tool
                return await search_vector_store_tool(**params.get("args", {}))
            else:
                return {"success": False, "error": f"Unknown vector store operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global workflow executor instance
_workflow_executor = WorkflowExecutor()


async def execute_workflow_tool(
    workflow_definition: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a complex multi-step workflow.
    
    Args:
        workflow_definition: Definition of the workflow with steps and parameters
    
    Returns:
        Dict containing workflow execution results
    """
    return await _workflow_executor.execute_workflow(workflow_definition)


async def create_embedding_pipeline_tool(
    input_path: str,
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    shard_embeddings: bool = False,
    shard_size: int = 1000000,
    index_to_cluster: bool = False,
    cluster_config: Optional[Dict[str, Any]] = None,
    store_in_vector_db: bool = False,
    vector_store_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a complete embedding pipeline from data to indexed storage.
    
    Args:
        input_path: Path to input data
        output_path: Base path for outputs
        model_name: Embedding model to use
        shard_embeddings: Whether to shard the embeddings
        shard_size: Size of each shard
        index_to_cluster: Whether to index to IPFS cluster
        cluster_config: IPFS cluster configuration
        store_in_vector_db: Whether to store in vector database
        vector_store_config: Vector store configuration
    
    Returns:
        Dict containing pipeline execution results
    """
    try:
        results = {}
        
        # Step 1: Create embeddings
        embedding_result = await _workflow_executor._execute_create_embeddings({
            "input_path": input_path,
            "output_path": f"{output_path}/embeddings",
            "model_name": model_name
        })
        results["create_embeddings"] = embedding_result
        
        if not embedding_result.get("success"):
            return {"success": False, "results": results, "error": "Failed to create embeddings"}
        
        # Step 2: Shard embeddings (optional)
        if shard_embeddings:
            shard_result = await _workflow_executor._execute_shard_embeddings({
                "input_path": f"{output_path}/embeddings",
                "output_dir": f"{output_path}/shards",
                "shard_size": shard_size
            })
            results["shard_embeddings"] = shard_result
            
            if not shard_result.get("success"):
                return {"success": False, "results": results, "error": "Failed to shard embeddings"}
        
        # Step 3: Index to IPFS cluster (optional)
        if index_to_cluster:
            index_result = await _workflow_executor._execute_index_cluster({
                "config": cluster_config or {},
                "embeddings_path": f"{output_path}/shards" if shard_embeddings else f"{output_path}/embeddings"
            })
            results["index_cluster"] = index_result
        
        # Step 4: Store in vector database (optional)
        if store_in_vector_db and vector_store_config:
            store_result = await _workflow_executor._execute_vector_store_operation({
                "operation": "add",
                "args": {
                    "provider": vector_store_config.get("provider", "faiss"),
                    "config": vector_store_config,
                    "embeddings": f"{output_path}/embeddings"
                }
            })
            results["vector_store"] = store_result
        
        return {
            "success": True,
            "pipeline": "embedding_pipeline",
            "input_path": input_path,
            "output_path": output_path,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "pipeline": "embedding_pipeline"
        }


async def get_workflow_status_tool(
    workflow_id: str
) -> Dict[str, Any]:
    """
    Get the status and results of a workflow execution.
    
    Args:
        workflow_id: ID of the workflow to check
    
    Returns:
        Dict containing workflow status and results
    """
    try:
        if workflow_id not in _workflow_executor.execution_history:
            return {
                "success": False,
                "error": f"Workflow {workflow_id} not found",
                "workflow_id": workflow_id
            }
        
        execution_record = _workflow_executor.execution_history[workflow_id]
        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": execution_record
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "workflow_id": workflow_id
        }


async def list_workflows_tool() -> Dict[str, Any]:
    """
    List all executed workflows.
    
    Returns:
        Dict containing list of workflows
    """
    try:
        workflows = []
        for workflow_id, record in _workflow_executor.execution_history.items():
            workflows.append({
                "workflow_id": workflow_id,
                "executed_at": record.get("executed_at"),
                "success": record.get("success"),
                "steps_count": len(record.get("results", {}))
            })
        
        return {
            "success": True,
            "workflows": workflows,
            "total_count": len(workflows)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Tool metadata for MCP registration
TOOL_METADATA = {
    "execute_workflow_tool": {
        "name": "execute_workflow_tool",
        "description": "Execute a complex multi-step workflow",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_definition": {
                    "type": "object",
                    "description": "Definition of the workflow with steps and parameters"
                }
            },
            "required": ["workflow_definition"]
        }
    },
    "create_embedding_pipeline_tool": {
        "name": "create_embedding_pipeline_tool",
        "description": "Execute a complete embedding pipeline from data to indexed storage",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to input data"
                },
                "output_path": {
                    "type": "string",
                    "description": "Base path for outputs"
                },
                "model_name": {
                    "type": "string",
                    "description": "Embedding model to use",
                    "default": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "shard_embeddings": {
                    "type": "boolean",
                    "description": "Whether to shard the embeddings",
                    "default": False
                },
                "shard_size": {
                    "type": "integer",
                    "description": "Size of each shard",
                    "default": 1000000
                },
                "index_to_cluster": {
                    "type": "boolean",
                    "description": "Whether to index to IPFS cluster",
                    "default": False
                },
                "cluster_config": {
                    "type": "object",
                    "description": "IPFS cluster configuration"
                },
                "store_in_vector_db": {
                    "type": "boolean",
                    "description": "Whether to store in vector database",
                    "default": False
                },
                "vector_store_config": {
                    "type": "object",
                    "description": "Vector store configuration"
                }
            },
            "required": ["input_path", "output_path"]
        }
    },
    "get_workflow_status_tool": {
        "name": "get_workflow_status_tool",
        "description": "Get the status and results of a workflow execution",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "ID of the workflow to check"
                }
            },
            "required": ["workflow_id"]
        }
    },
    "list_workflows_tool": {
        "name": "list_workflows_tool",
        "description": "List all executed workflows",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}
