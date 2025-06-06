"""
MCP tool wrapper for the create_embeddings pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional
import json
import os

from create_embeddings.create_embeddings import (
    CreateEmbeddingsProcessor,
    EmbeddingConfig,
    DataConfig,
    OutputConfig
)


async def create_embeddings_tool(
    input_path: str,
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    chunk_size: Optional[int] = None,
    max_length: Optional[int] = None,
    normalize: bool = True,
    use_gpu: bool = False,
    num_workers: int = 1,
    output_format: str = "parquet",
    compression: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create embeddings from input data using the create_embeddings pipeline.
    
    Args:
        input_path: Path to input data (file or directory)
        output_path: Path where embeddings will be saved
        model_name: Name of the embedding model to use
        batch_size: Batch size for processing
        chunk_size: Size of data chunks to process
        max_length: Maximum sequence length
        normalize: Whether to normalize embeddings
        use_gpu: Whether to use GPU acceleration
        num_workers: Number of worker processes
        output_format: Output format (parquet, hdf5, npz, etc.)
        compression: Compression method to use
        metadata: Additional metadata to include
    
    Returns:
        Dict containing operation results and metadata
    """
    try:
        # Validate input path
        if not os.path.exists(input_path):
            return {
                "success": False,
                "error": f"Input path does not exist: {input_path}",
                "input_path": input_path,
                "output_path": output_path
            }
        
        # Create configuration objects
        embedding_config = EmbeddingConfig(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            use_gpu=use_gpu
        )
        
        data_config = DataConfig(
            input_path=input_path,
            chunk_size=chunk_size,
            num_workers=num_workers
        )
        
        output_config = OutputConfig(
            output_path=output_path,
            format=output_format,
            compression=compression,
            metadata=metadata or {}
        )
        
        # Initialize processor
        processor = CreateEmbeddingsProcessor(
            embedding_config=embedding_config,
            data_config=data_config,
            output_config=output_config
        )
        
        # Run the embedding creation pipeline
        result = await asyncio.to_thread(processor.process)
        
        return {
            "success": True,
            "result": result,
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "batch_size": batch_size,
            "output_format": output_format,
            "embeddings_created": result.get("embeddings_count", 0),
            "processing_time": result.get("processing_time", 0),
            "output_size": result.get("output_size", 0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name
        }


async def batch_create_embeddings_tool(
    batch_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create embeddings for multiple datasets in batch.
    
    Args:
        batch_configs: List of configuration dictionaries for each embedding task
    
    Returns:
        Dict containing batch processing results
    """
    try:
        results = []
        successful = 0
        failed = 0
        
        for i, config in enumerate(batch_configs):
            try:
                result = await create_embeddings_tool(**config)
                results.append({
                    "batch_index": i,
                    "config": config,
                    "result": result
                })
                
                if result.get("success"):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "config": config,
                    "result": {
                        "success": False,
                        "error": str(e)
                    }
                })
                failed += 1
        
        return {
            "success": True,
            "total_batches": len(batch_configs),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_batches": len(batch_configs) if batch_configs else 0
        }


# Tool metadata for MCP registration
TOOL_METADATA = {
    "create_embeddings_tool": {
        "name": "create_embeddings_tool",
        "description": "Create embeddings from input data using the create_embeddings pipeline",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to input data (file or directory)"
                },
                "output_path": {
                    "type": "string", 
                    "description": "Path where embeddings will be saved"
                },
                "model_name": {
                    "type": "string",
                    "description": "Name of the embedding model to use",
                    "default": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size for processing",
                    "default": 32
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Size of data chunks to process"
                },
                "max_length": {
                    "type": "integer", 
                    "description": "Maximum sequence length"
                },
                "normalize": {
                    "type": "boolean",
                    "description": "Whether to normalize embeddings",
                    "default": True
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "Whether to use GPU acceleration",
                    "default": False
                },
                "num_workers": {
                    "type": "integer",
                    "description": "Number of worker processes",
                    "default": 1
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format (parquet, hdf5, npz, etc.)",
                    "default": "parquet"
                },
                "compression": {
                    "type": "string",
                    "description": "Compression method to use"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata to include"
                }
            },
            "required": ["input_path", "output_path"]
        }
    },
    "batch_create_embeddings_tool": {
        "name": "batch_create_embeddings_tool", 
        "description": "Create embeddings for multiple datasets in batch",
        "parameters": {
            "type": "object",
            "properties": {
                "batch_configs": {
                    "type": "array",
                    "description": "List of configuration dictionaries for each embedding task",
                    "items": {
                        "type": "object",
                        "description": "Configuration for a single embedding task"
                    }
                }
            },
            "required": ["batch_configs"]
        }
    }
}
