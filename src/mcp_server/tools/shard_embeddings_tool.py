"""
MCP tool wrapper for the shard_embeddings pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional
import json
import os

from shard_embeddings.shard_embeddings import shard_embeddings, ShardEmbeddingsProcessor


async def shard_embeddings_tool(
    input_path: str,
    output_dir: str,
    shard_size: int = 1000000,
    max_shards: Optional[int] = None,
    overlap_size: int = 0,
    shuffle: bool = False,
    seed: Optional[int] = None,
    compression: str = "gzip",
    output_format: str = "parquet",
    metadata: Optional[Dict[str, Any]] = None,
    validate_shards: bool = True
) -> Dict[str, Any]:
    """
    Shard embeddings into smaller chunks for distributed processing.
    
    Args:
        input_path: Path to input embeddings file or directory
        output_dir: Directory where shards will be saved
        shard_size: Maximum number of embeddings per shard
        max_shards: Maximum number of shards to create
        overlap_size: Number of overlapping embeddings between shards
        shuffle: Whether to shuffle embeddings before sharding
        seed: Random seed for shuffling
        compression: Compression method (gzip, lz4, snappy)
        output_format: Output format for shards
        metadata: Additional metadata to include with shards
        validate_shards: Whether to validate shard integrity
    
    Returns:
        Dict containing sharding results and metadata
    """
    try:
        # Validate input path
        if not os.path.exists(input_path):
            return {
                "success": False,
                "error": f"Input path does not exist: {input_path}",
                "input_path": input_path,
                "output_dir": output_dir
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create configuration objects
        shard_config = ShardConfig(
            shard_size=shard_size,
            max_shards=max_shards,
            overlap_size=overlap_size,
            shuffle=shuffle,
            seed=seed
        )
        
        input_config = InputConfig(
            input_path=input_path
        )
        
        output_config = OutputConfig(
            output_dir=output_dir,
            compression=compression,
            format=output_format,
            metadata=metadata or {}
        )
        
        # Initialize processor
        processor = ShardEmbeddingsProcessor(
            shard_config=shard_config,
            input_config=input_config,
            output_config=output_config
        )
        
        # Run the sharding pipeline
        result = await asyncio.to_thread(processor.process)
        
        # Validate shards if requested
        validation_result = None
        if validate_shards:
            validation_result = await asyncio.to_thread(processor.validate_shards)
        
        return {
            "success": True,
            "result": result,
            "validation": validation_result,
            "input_path": input_path,
            "output_dir": output_dir,
            "shard_size": shard_size,
            "total_shards": result.get("total_shards", 0),
            "total_embeddings": result.get("total_embeddings", 0),
            "compression": compression,
            "output_format": output_format,
            "shard_files": result.get("shard_files", [])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "input_path": input_path,
            "output_dir": output_dir
        }


async def merge_shards_tool(
    shard_dir: str,
    output_path: str,
    shard_pattern: str = "shard_*.parquet",
    validate_merge: bool = True,
    remove_shards: bool = False
) -> Dict[str, Any]:
    """
    Merge multiple embedding shards back into a single file.
    
    Args:
        shard_dir: Directory containing shard files
        output_path: Path for the merged output file
        shard_pattern: Pattern to match shard files
        validate_merge: Whether to validate the merged result
        remove_shards: Whether to remove original shards after merging
    
    Returns:
        Dict containing merge results
    """
    try:
        if not os.path.exists(shard_dir):
            return {
                "success": False,
                "error": f"Shard directory does not exist: {shard_dir}",
                "shard_dir": shard_dir,
                "output_path": output_path
            }
        
        # Create processor for merging
        processor = ShardEmbeddingsProcessor(
            shard_config=ShardConfig(),
            input_config=InputConfig(input_path=shard_dir),
            output_config=OutputConfig(output_dir=os.path.dirname(output_path))
        )
        
        # Run the merge operation
        result = await asyncio.to_thread(
            processor.merge_shards,
            shard_dir=shard_dir,
            output_path=output_path,
            pattern=shard_pattern
        )
        
        # Validate merge if requested
        validation_result = None
        if validate_merge:
            validation_result = await asyncio.to_thread(
                processor.validate_merged_file,
                output_path
            )
        
        # Remove shards if requested and merge was successful
        if remove_shards and result.get("success"):
            await asyncio.to_thread(
                processor.cleanup_shards,
                shard_dir=shard_dir,
                pattern=shard_pattern
            )
        
        return {
            "success": True,
            "result": result,
            "validation": validation_result,
            "shard_dir": shard_dir,
            "output_path": output_path,
            "shards_processed": result.get("shards_processed", 0),
            "total_embeddings": result.get("total_embeddings", 0),
            "output_size": result.get("output_size", 0),
            "shards_removed": remove_shards and result.get("success", False)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "shard_dir": shard_dir,
            "output_path": output_path
        }


async def shard_info_tool(
    shard_path: str
) -> Dict[str, Any]:
    """
    Get information about a shard file or directory of shards.
    
    Args:
        shard_path: Path to shard file or directory
    
    Returns:
        Dict containing shard information
    """
    try:
        if not os.path.exists(shard_path):
            return {
                "success": False,
                "error": f"Shard path does not exist: {shard_path}",
                "shard_path": shard_path
            }
        
        processor = ShardEmbeddingsProcessor(
            shard_config=ShardConfig(),
            input_config=InputConfig(input_path=shard_path),
            output_config=OutputConfig(output_dir=".")
        )
        
        # Get shard information
        info = await asyncio.to_thread(processor.get_shard_info, shard_path)
        
        return {
            "success": True,
            "shard_path": shard_path,
            "info": info
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "shard_path": shard_path
        }


# Tool metadata for MCP registration
TOOL_METADATA = {
    "shard_embeddings_tool": {
        "name": "shard_embeddings_tool",
        "description": "Shard embeddings into smaller chunks for distributed processing",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to input embeddings file or directory"
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where shards will be saved"
                },
                "shard_size": {
                    "type": "integer",
                    "description": "Maximum number of embeddings per shard",
                    "default": 1000000
                },
                "max_shards": {
                    "type": "integer",
                    "description": "Maximum number of shards to create"
                },
                "overlap_size": {
                    "type": "integer",
                    "description": "Number of overlapping embeddings between shards",
                    "default": 0
                },
                "shuffle": {
                    "type": "boolean",
                    "description": "Whether to shuffle embeddings before sharding",
                    "default": False
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for shuffling"
                },
                "compression": {
                    "type": "string", 
                    "description": "Compression method (gzip, lz4, snappy)",
                    "default": "gzip"
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format for shards",
                    "default": "parquet"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata to include with shards"
                },
                "validate_shards": {
                    "type": "boolean",
                    "description": "Whether to validate shard integrity",
                    "default": True
                }
            },
            "required": ["input_path", "output_dir"]
        }
    },
    "merge_shards_tool": {
        "name": "merge_shards_tool",
        "description": "Merge multiple embedding shards back into a single file",
        "parameters": {
            "type": "object",
            "properties": {
                "shard_dir": {
                    "type": "string",
                    "description": "Directory containing shard files"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path for the merged output file"
                },
                "shard_pattern": {
                    "type": "string",
                    "description": "Pattern to match shard files",
                    "default": "shard_*.parquet"
                },
                "validate_merge": {
                    "type": "boolean",
                    "description": "Whether to validate the merged result",
                    "default": True
                },
                "remove_shards": {
                    "type": "boolean",
                    "description": "Whether to remove original shards after merging",
                    "default": False
                }
            },
            "required": ["shard_dir", "output_path"]
        }
    },
    "shard_info_tool": {
        "name": "shard_info_tool",
        "description": "Get information about a shard file or directory of shards",
        "parameters": {
            "type": "object",
            "properties": {
                "shard_path": {
                    "type": "string",
                    "description": "Path to shard file or directory"
                }
            },
            "required": ["shard_path"]
        }
    }
}
