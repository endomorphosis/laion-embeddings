# src/mcp_server/tools/data_processing_tools.py

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

from ..tool_registry import ClaudeMCPTool # Reusing ClaudeMCPTool base class
from ..validators import validator
from ..error_handlers import MCPError, ValidationError

logger = logging.getLogger(__name__)

# Import necessary modules from ipfs_embeddings_py with fallbacks
IPFS_EMBEDDINGS_AVAILABLE = False
ipfs_embeddings_py = None
chunker = None
ipfs_parquet_to_car_py = None

try:
    # Try to import without triggering torchvision dependencies
    import importlib.util
    spec = importlib.util.find_spec("ipfs_embeddings_py")
    if spec is not None:
        # Only import if we know the dependencies are available
        try:
            from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
            from ipfs_embeddings_py.chunker import chunker
            from ipfs_embeddings_py.ipfs_parquet_to_car import ipfs_parquet_to_car_py
            IPFS_EMBEDDINGS_AVAILABLE = True
        except (ImportError, RuntimeError) as inner_e:
            logger.warning(f"ipfs_embeddings_py components not available: {inner_e}")
    else:
        logger.warning("ipfs_embeddings_py package not found")
except Exception as e:
    logger.warning(f"Error checking ipfs_embeddings_py availability: {e}")

try:
    from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict
    DATASETS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"datasets not available: {e}")
    Dataset = None
    IterableDataset = None
    DatasetDict = None
    IterableDatasetDict = None
    DATASETS_AVAILABLE = False

class ChunkingTool(ClaudeMCPTool):
    """Tool for chunking text data."""
    
    name = "text_chunking"
    description = "Chunks text data using various methods (semantic, token, sentence, sliding window)."
    
    def __init__(self, ipfs_embeddings_instance: Optional[Any]):
        self.ipfs_embeddings = ipfs_embeddings_instance
        # Initialize chunker instance (assuming it can be initialized without resources/metadata directly)
        if IPFS_EMBEDDINGS_AVAILABLE and chunker:
            self.chunker_instance = chunker() # Assuming chunker can be initialized without args or with mock
        else:
            self.chunker_instance = None
        self.input_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text content to chunk."},
                "method": {
                    "type": "string",
                    "description": "Chunking method to use",
                    "enum": ["semantic", "tokens", "sentences", "sliding_window"],
                    "default": "tokens"
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Maximum size of each chunk (for token/sliding window methods)",
                    "minimum": 1,
                    "default": 256
                },
                "n_sentences": {
                    "type": "integer",
                    "description": "Number of sentences per chunk (for sentence method)",
                    "minimum": 1,
                    "default": 3
                },
                "step_size": {
                    "type": "integer",
                    "description": "Step size for sliding window method",
                    "minimum": 1,
                    "default": 128
                },
                "model": {
                    "type": "string",
                    "description": "Model for semantic chunking (e.g., 'sentence-transformers/all-MiniLM-L6-v2')",
                    "required": False
                }
            },
            "required": ["text"]
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text chunking."""
        # Extract parameters
        text = parameters.get("text", "")
        method = parameters.get("method", "tokens")
        
        try:
            # Basic input validation  
            if not text:
                raise ValidationError("text", "Text cannot be empty")
            if method not in ["semantic", "tokens", "sentences", "sliding_window"]:
                raise ValidationError("method", f"Invalid method: {method}")

            chunks = []
            
            # Check if chunker instance is available
            if self.chunker_instance is None:
                # Fallback to simple text splitting
                return await self._fallback_chunking(text, method, parameters)
            
            if method == "semantic":
                model = parameters.get("model")
                if not model:
                    raise ValidationError("model", "Model is required for semantic chunking.")
                # Assuming chunker_instance.chunk_semantically can be called directly
                chunks = self.chunker_instance.chunk_semantically(text, model)
            elif method == "tokens":
                chunks = self.chunker_instance.chunk_by_tokens(text, parameters.get("chunk_size", 256))
            elif method == "sentences":
                chunks = self.chunker_instance.chunk_by_sentences(text, parameters.get("n_sentences", 3))
            elif method == "sliding_window":
                chunks = self.chunker_instance.chunk_by_sliding_window(
                    text, parameters.get("chunk_size", 256), parameters.get("step_size", 128)
                )
            else:
                raise ValidationError("method", f"Unknown chunking method: {method}")
            
            return {"success": True, "method": method, "n_chunks": len(chunks), "chunks": chunks}
        except ValidationError as e:
            logger.error(f"Chunking tool validation error: {str(e)}")
            raise MCPError(-32602, f"Chunking tool validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Chunking tool error: {str(e)}")
            raise MCPError(-32000, f"Chunking tool failed: {str(e)}")

    async def _fallback_chunking(self, text: str, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback chunking when chunker_instance is not available."""
        chunks = []
        
        if method == "semantic":
            # Simple sentence-based chunking as fallback for semantic
            sentences = text.split('. ')
            chunk_size = parameters.get("chunk_size", 256)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) <= chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append(current_chunk.strip())
                
        elif method == "tokens":
            # Simple word-based chunking as fallback
            words = text.split()
            chunk_size = parameters.get("chunk_size", 256)
            chunk = []
            for word in words:
                chunk.append(word)
                if len(' '.join(chunk)) >= chunk_size:
                    chunks.append(' '.join(chunk))
                    chunk = []
            if chunk:
                chunks.append(' '.join(chunk))
                
        elif method == "sentences":
            # Split by sentences
            sentences = text.split('. ')
            n_sentences = parameters.get("n_sentences", 3)
            for i in range(0, len(sentences), n_sentences):
                chunk_sentences = sentences[i:i+n_sentences]
                chunks.append('. '.join(chunk_sentences))
                
        elif method == "sliding_window":
            # Simple sliding window by words
            words = text.split()
            chunk_size = parameters.get("chunk_size", 256)
            step_size = parameters.get("step_size", 128)
            for i in range(0, len(words), step_size):
                chunk_words = words[i:i+chunk_size]
                if chunk_words:
                    chunks.append(' '.join(chunk_words))
        
        return {"success": True, "method": method, "n_chunks": len(chunks), "chunks": chunks, "fallback": True}


class DatasetLoadingTool(ClaudeMCPTool):
    """Tool for loading datasets."""
    
    name = "load_dataset"
    description = "Loads a dataset from a specified source (e.g., Hugging Face, local path)."
    
    def __init__(self, ipfs_embeddings_instance: Optional[Any]):
        self.ipfs_embeddings = ipfs_embeddings_instance
        self.input_schema = {
            "type": "object",
            "properties": {
                "dataset_name_or_path": {"type": "string", "description": "Name of the dataset (e.g., 'laion/laion-400m') or local path."},
                "split": {"type": "string", "description": "Dataset split to load (e.g., 'train', 'validation').", "required": False}
            },
            "required": ["dataset_name_or_path"]
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dataset loading."""
        # Extract parameters
        dataset_name_or_path = parameters.get("dataset_name_or_path", "")
        split = parameters.get("split", None)
        
        try:
            # Basic input validation
            if not dataset_name_or_path:
                raise ValidationError("dataset_name_or_path", "Dataset name or path cannot be empty")

            # Check if ipfs_embeddings instance is available
            if self.ipfs_embeddings is None:
                return {
                    "success": False,
                    "error": "IPFS embeddings instance not available",
                    "fallback": True
                }

            print(f"Loading dataset: '{dataset_name_or_path}' split: '{split}'")
            # Assuming ipfs_embeddings.load_dataset can handle this
            await self.ipfs_embeddings.load_dataset(dataset_name_or_path, split=split)
            
            # After loading, ipfs_embeddings.dataset should be populated
            # Return some metadata about the loaded dataset
            dataset_info = {
                "name_or_path": dataset_name_or_path,
                "split": split,
                "num_rows": "N/A", # Default for streaming datasets
                "features": [] # Default for streaming datasets
            }

            # Handle dataset info extraction safely
            try:
                if (self.ipfs_embeddings is not None and 
                    hasattr(self.ipfs_embeddings, 'dataset') and 
                    self.ipfs_embeddings.dataset):
                    # Use getattr with defaults to avoid attribute errors
                    try:
                        dataset_info["num_rows"] = getattr(self.ipfs_embeddings.dataset, 'num_rows', 'Unknown')
                    except:
                        pass
                    
                    # Try to get features/columns safely
                    try:
                        if hasattr(self.ipfs_embeddings.dataset, 'column_names'):
                            dataset_info["features"] = getattr(self.ipfs_embeddings.dataset, 'column_names', [])
                        elif hasattr(self.ipfs_embeddings.dataset, 'features'):
                            features_attr = getattr(self.ipfs_embeddings.dataset, 'features', None)
                            if features_attr and hasattr(features_attr, 'keys'):
                                dataset_info["features"] = list(features_attr.keys())
                    except:
                        dataset_info["features"] = "Unable to extract features"
            except Exception as e:
                logger.warning(f"Could not extract dataset info: {e}")
                dataset_info["features"] = "Unable to extract features"
            
            return {"success": True, "message": "Dataset loaded successfully.", "dataset_info": dataset_info}
        except ValidationError as e:
            logger.error(f"Dataset loading validation error: {str(e)}")
            raise MCPError(-32602, f"Dataset loading validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Dataset loading error: {str(e)}")
            raise MCPError(-32000, f"Dataset loading failed: {str(e)}")

class ParquetToCarTool(ClaudeMCPTool):
    """Tool for converting Parquet files to IPFS CAR files."""
    
    name = "parquet_to_car"
    description = "Converts a Parquet file or directory of Parquet files to an IPFS CAR (Content Addressable Archive) file."
    
    def __init__(self, ipfs_embeddings_instance: Optional[Any]):
        self.ipfs_embeddings = ipfs_embeddings_instance
        # Initialize ipfs_parquet_to_car_py instance
        if IPFS_EMBEDDINGS_AVAILABLE and ipfs_parquet_to_car_py:
            self.parquet_to_car_instance = ipfs_parquet_to_car_py(resources={}, metadata={}) # Assuming it can be initialized with mock args
        else:
            self.parquet_to_car_instance = None
        self.input_schema = {
            "type": "object",
            "properties": {
                "src_path": {"type": "string", "description": "Source path to the Parquet file or directory."},
                "dst_path": {"type": "string", "description": "Destination path for the CAR file."},
            },
            "required": ["src_path", "dst_path"]
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Parquet to CAR conversion."""
        # Extract parameters
        src_path = parameters.get("src_path", "")
        dst_path = parameters.get("dst_path", "")
        
        try:
            # Basic input validation
            if not src_path:
                raise ValidationError("src_path", "Source path cannot be empty")
            if not dst_path:
                raise ValidationError("dst_path", "Destination path cannot be empty")

            # Check if parquet_to_car instance is available
            if self.parquet_to_car_instance is None:
                return {
                    "success": False,
                    "error": "Parquet to CAR converter not available",
                    "fallback": True
                }

            print(f"Converting Parquet from '{src_path}' to CAR at '{dst_path}'")
            # Assuming ipfs_parquet_to_car_py.run handles the conversion
            await self.parquet_to_car_instance.run(src_path, dst_path)
            
            return {"success": True, "src_path": src_path, "dst_path": dst_path, "message": "Parquet to CAR conversion successful."}
        except ValidationError as e:
            logger.error(f"Parquet to CAR conversion validation error: {str(e)}")
            raise MCPError(-32602, f"Parquet to CAR conversion validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Parquet to CAR conversion error: {str(e)}")
            raise MCPError(-32000, f"Parquet to CAR conversion failed: {str(e)}")
