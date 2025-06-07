from typing import Union, List, Optional, Any
import uvicorn
import logging
import json
import asyncio
import aiohttp
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Response
from search_embeddings import search_embeddings
from create_embeddings import create_embeddings
from shard_embeddings import shard_embeddings
from sparse_embeddings.sparse_embeddings import sparse_embeddings
from ipfs_cluster_index import ipfs_cluster_index
# DEPRECATED: storacha_clusters is deprecated, use ipfs_kit_py instead
from storacha_clusters import storacha_clusters
import warnings
warnings.warn("storacha_clusters is deprecated. Use ipfs_kit_py.storacha_kit instead.", DeprecationWarning)
from pydantic import BaseModel, Field
import re
import time
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import hashlib
from datetime import datetime, timedelta
from aiohttp import ClientTimeout

# Authentication imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Depends, status
from auth import (
    UserLogin, 
    authenticate_user, 
    create_access_token,
    get_current_user,
    require_admin,
    require_write,
    require_read,
    TokenData,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Monitoring imports
from monitoring import metrics_collector

# ==============================================================================
# TIMEOUT CONFIGURATION AND UTILITIES
# ==============================================================================

# Timeout constants (in seconds)
NETWORK_TIMEOUT = 300  # 5 minutes for network operations
BACKGROUND_TASK_TIMEOUT = 1800  # 30 minutes for background tasks
DATABASE_TIMEOUT = 60  # 1 minute for database operations
SEARCH_TIMEOUT = 120  # 2 minutes for search operations
IMPORT_TIMEOUT = 180  # 3 minutes for module imports
DEFAULT_ASYNC_TIMEOUT = 30  # 30 seconds for general async operations

class TimeoutError(Exception):
    """Custom timeout exception for clarity"""
    pass

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('laion_embeddings.log'),
        logging.StreamHandler()
    ]
)

app_logger = logging.getLogger(__name__)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('laion_embeddings.log'),
        logging.StreamHandler()
    ]
)

app_logger = logging.getLogger(__name__)

class StructuredLogger:
    """Structured logger for consistent logging across the application"""
    
    @staticmethod
    def log_request(endpoint: str, request_data: dict, user_id: Optional[str] = None):
        """Log incoming requests"""
        log_entry = {
            "event": "request_received",
            "endpoint": endpoint,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "request_size": len(str(request_data))
        }
        app_logger.info(json.dumps(log_entry))
    
    @staticmethod
    def log_response(endpoint: str, status_code: int, duration_ms: float, response_size: Optional[int] = None):
        """Log API responses"""
        log_entry = {
            "event": "request_completed",
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "response_size": response_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        app_logger.info(json.dumps(log_entry))
    
    @staticmethod
    def log_error(endpoint: str, error: Exception, request_data: Optional[dict] = None):
        """Log errors with context"""
        log_entry = {
            "event": "error_occurred",
            "endpoint": endpoint,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_data": request_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        app_logger.error(json.dumps(log_entry))
    
    @staticmethod
    def log_background_task(task_name: str, status: str, details: Optional[dict] = None):
        """Log background task status"""
        log_entry = {
            "event": "background_task",
            "task_name": task_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        app_logger.info(json.dumps(log_entry))

# Input Validation Class
class InputValidator:
    @staticmethod
    def validate_text_input(text: str) -> str:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if len(text) > 10000:
            raise HTTPException(status_code=400, detail="Text too long (max 10k characters)")
        return text.strip()
    
    @staticmethod
    def validate_model_name(model: str) -> str:
        allowed_models = [
            "thenlper/gte-small",
            "Alibaba-NLP/gte-large-en-v1.5", 
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        ]
        if model not in allowed_models:
            raise HTTPException(status_code=400, detail=f"Model {model} not allowed")
        return model
    
    @staticmethod
    def validate_dataset_name(dataset: str) -> str:
        if not dataset or len(dataset.strip()) == 0:
            raise HTTPException(status_code=400, detail="Dataset name cannot be empty")
        # Basic validation for dataset names (alphanumeric, hyphens, underscores, slashes)
        if not re.match(r'^[a-zA-Z0-9_/-]+$', dataset):
            raise HTTPException(status_code=400, detail="Invalid dataset name format")
        return dataset.strip()

class LoadIndexRequest(BaseModel):
    dataset: str = Field(..., description="Dataset identifier", min_length=1, max_length=200)
    knn_index: str = Field(..., description="KNN index identifier", min_length=1, max_length=200)
    dataset_split: Optional[str] = Field(None, description="Dataset split")
    knn_index_split: Optional[str] = Field(None, description="KNN index split")
    columns: List[str] = Field(..., description="List of columns to process")

class SearchRequest(BaseModel):
    collection: str = Field(..., description="Collection to search", min_length=1, max_length=100)
    text: str = Field(..., description="Search text", min_length=1, max_length=10000)
    n: int = Field(default=10, description="Number of results", ge=1, le=100)

class CreateEmbeddingsRequest(BaseModel):
    dataset: str = Field(..., description="Dataset identifier", min_length=1, max_length=200)
    split: str = Field(..., description="Dataset split", min_length=1, max_length=50)
    column: str = Field(..., description="Text column name", min_length=1, max_length=100)
    dst_path: str = Field(..., description="Destination path", min_length=1, max_length=500)
    models: List[str] = Field(..., description="List of model names", min_length=1, max_length=10)
    
class ShardEmbeddingsRequest(BaseModel):
    dataset: str = Field(..., description="Dataset identifier", min_length=1, max_length=200)
    split: str = Field(..., description="Dataset split", min_length=1, max_length=50)
    column: str = Field(..., description="Text column name", min_length=1, max_length=100)
    dst_path: str = Field(..., description="Destination path", min_length=1, max_length=500)
    models: List[str] = Field(..., description="List of model names", min_length=1, max_length=10)
 
class CreateSparseEmbeddingsRequest(BaseModel):
    dataset: str = Field(..., description="Dataset identifier", min_length=1, max_length=200)
    split: str = Field(..., description="Dataset split", min_length=1, max_length=50)
    column: str = Field(..., description="Text column name", min_length=1, max_length=100)
    dst_path: str = Field(..., description="Destination path", min_length=1, max_length=500)
    models: List[str] = Field(..., description="List of model names", min_length=1, max_length=10)
    
class IndexClusterRequest(BaseModel):
    resources: dict = Field(..., description="Resource configuration")
    metadata: dict = Field(..., description="Metadata configuration")
    
class StorachaClustersRequest(BaseModel):
    resources: dict = Field(..., description="Resource configuration")
    metadata: dict = Field(..., description="Metadata configuration")
    
metadata = {
    "dataset": "TeraflopAI/Caselaw_Access_Project",
    "column": "text",
    "split": "train",
    "models": [
        "thenlper/gte-small",
        "Alibaba-NLP/gte-large-en-v1.5",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ],
    "chunk_settings": {
        "chunk_size": 512,
        "n_sentences": 8,
        "step_size": 256,
        "method": "fixed",
        "embed_model": "thenlper/gte-small",
        "tokenizer": None
    },
    "dst_path": "/storage/teraflopai/tmp",
}
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "cpu", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cpu", 32768],
        ["thenlper/gte-small", "cuda:0", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cuda:0", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:0", 32768],
        ["thenlper/gte-small", "cuda:1", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cuda:1", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:1", 32768],
        ["thenlper/gte-small", "openvino", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "openvino", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "openvino", 32768],
        ["thenlper/gte-small", "llama_cpp", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "llama_cpp", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "llama_cpp", 32768],
        ["thenlper/gte-small", "ipex", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "ipex", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "ipex", 32768],
    ],
    "openvino_endpoints": [
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx0-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx0/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx1-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx1/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx2-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx2/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx3-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx3/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx4-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx4/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx5-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx5/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx6-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx6/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx7-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx7/infer", 1024]
    ],
    "tei_endpoints": [
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8080/embed-medium", 32768],
        ["thenlper/gte-small", "http://62.146.169.111:8080/embed-tiny", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8081/embed-small", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8081/embed-medium", 32768],
        ["thenlper/gte-small", "http://62.146.169.111:8081/embed-tiny", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8082/embed-small", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8082/embed-medium", 32768],
        ["thenlper/gte-small", "http://62.146.169.111:8082/embed-tiny", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8083/embed-small", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8083/embed-medium", 32768],
        ["thenlper/gte-small", "http://62.146.169.111:8083/embed-tiny", 512]
    ]
}

def metadata_generator(src_metadata):
    dst_metadata = src_metadata 
    return dst_metadata

def resources_generator(src_resources):
    dst_resources = src_resources
    return dst_resources    

search = search_embeddings(resources, metadata)
create = create_embeddings(resources, metadata)
sparse = sparse_embeddings(resources, metadata)
shards = shard_embeddings(resources, metadata)
storacha = storacha_clusters(resources, metadata)
index_cluster = ipfs_cluster_index(resources, metadata)

app = FastAPI(
    title="LAION Embeddings API",
    description="API for creating, searching, and managing embeddings",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "laion-embeddings"}

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {"message": "LAION Embeddings API", "docs_url": "/docs"}

@app.post("/add_endpoint")
async def add_endpoint(
    request: dict, 
    http_request: Request,
    current_user: TokenData = Depends(require_admin)
):
    """Add a new endpoint configuration for embeddings processing (admin only)."""
    StructuredLogger.log_request("add_endpoint", request.__dict__ if hasattr(request, '__dict__') else request)
    
    try:
        # Validate required fields
        required_fields = ["model", "endpoint", "type", "ctx_length"]
        for field in required_fields:
            if field not in request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        model = request["model"]
        endpoint = request["endpoint"]
        endpoint_type = request["type"]
        ctx_length = request["ctx_length"]
        
        # Validate input values
        InputValidator.validate_model_name(model)
        
        # Validate endpoint type
        valid_types = ["libp2p", "https", "cuda", "local", "openvino"]
        if endpoint_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid endpoint type: {endpoint_type}. Must be one of: {valid_types}"
            )
        
        # Validate context length
        if not isinstance(ctx_length, int) or ctx_length <= 0:
            raise HTTPException(
                status_code=400,
                detail="Context length must be a positive integer"
            )
        
        # Add endpoint to all relevant objects
        objects = [search, sparse, shards, index_cluster, storacha, create]
        added_count = 0
        
        for obj in objects:
            method_name = f"add_{endpoint_type}_endpoint"
            if hasattr(obj, method_name):
                try:
                    method = getattr(obj, method_name)
                    method(model, endpoint, ctx_length)
                    added_count += 1
                except Exception as e:
                    # Log but don't fail the entire operation
                    logging.warning(f"Failed to add endpoint to {obj.__class__.__name__}: {str(e)}")
        
        response = {
            "message": "Endpoint added successfully",
            "model": model,
            "endpoint": endpoint,
            "type": endpoint_type,
            "ctx_length": ctx_length,
            "objects_updated": added_count
        }
        
        StructuredLogger.log_response("add_endpoint", 200, 0, len(str(response)))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to add endpoint: {str(e)}"
        StructuredLogger.log_error("add_endpoint", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

async def create_embeddings_task(request: CreateEmbeddingsRequest):
    """Create embeddings task with timeout protection"""
    try:
        dataset = request.dataset
        split = request.split
        column = request.column
        dst_path = request.dst_path
        models = request.models
        
        # Validate models
        for model in models:
            InputValidator.validate_model_name(model)
        
        # Execute with timeout protection
        create_embeddings_results = await safe_async_execute_with_timeout(
            create.create_embeddings(dataset, split, column, dst_path, models),
            timeout=BACKGROUND_TASK_TIMEOUT,
            operation_name="create_embeddings"
        )
        return create_embeddings_results
    except Exception as e:
        app_logger.error(f"Create embeddings task failed: {str(e)}")
        raise e

@app.post("/create_embeddings")
def create_embeddings_post(
    request: CreateEmbeddingsRequest, 
    background_tasks: BackgroundTasks,
    # current_user: TokenData = Depends(require_write) # Temporarily disable auth for testing
):
    """Create embeddings for a dataset (requires write permission)"""
    try:
        # Validate dataset name
        InputValidator.validate_dataset_name(request.dataset)
        
        # Validate models
        for model in request.models:
            InputValidator.validate_model_name(model)
            
        background_tasks.add_task(
            create_timeout_protected_background_task,
            create_embeddings_task,
            request,
            timeout=BACKGROUND_TASK_TIMEOUT,
            task_name="create_embeddings"
        )
        
        return {
            "message": "Embedding creation started in the background",
            "dataset": request.dataset,
            "split": request.split,
            "column": request.column,
            "models": request.models,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start embedding creation: {str(e)}")

async def load_index_task(dataset: str, knn_index: str, dataset_split: Optional[str], knn_index_split: Optional[str], columns: str):
    """Load index task with timeout protection"""
    try:
        vector_search = search_embeddings(resources, metadata)
        
        # Execute load operations with timeout protection
        await safe_async_execute_with_timeout(
            vector_search.load_qdrant_iter(dataset, knn_index, dataset_split, knn_index_split),
            timeout=DATABASE_TIMEOUT,
            operation_name="load_qdrant_iter"
        )
        
        await safe_async_execute_with_timeout(
            vector_search.ingest_qdrant_iter(columns),
            timeout=DATABASE_TIMEOUT,
            operation_name="ingest_qdrant_iter"
        )
        
        return None
    except Exception as e:
        app_logger.error(f"Load index task failed: {str(e)}")
        raise

@app.post("/load")
def load_index_post(request: LoadIndexRequest, background_tasks: BackgroundTasks):
    """Load index in background with proper error handling"""
    try:
        # Validate dataset name
        InputValidator.validate_dataset_name(request.dataset)
        
        # Start background task with timeout protection
        background_tasks.add_task(
            create_timeout_protected_background_task,
            load_index_task,
            request.dataset, 
            request.knn_index, 
            request.dataset_split, 
            request.knn_index_split, 
            ",".join(request.columns),  # Convert list to comma-separated string
            timeout=DATABASE_TIMEOUT,
            task_name="load_index"
        )
        
        StructuredLogger.log_background_task(
            task_name="load_index",
            status="started",
            details={
                "dataset": request.dataset,
                "knn_index": request.knn_index,
                "dataset_split": request.dataset_split,
                "knn_index_split": request.knn_index_split
            }
        )
        
        return {
            "message": "Index loading started in the background",
            "dataset": request.dataset,
            "knn_index": request.knn_index,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        StructuredLogger.log_error("load_index", e, request.dict())
        raise HTTPException(status_code=500, detail=f"Failed to start index loading: {str(e)}")

async def search_item_task(collection: str, text: str, n: int):
    """Search item task with timeout protection"""
    try:
        # Execute search with timeout protection
        result = await safe_async_execute_with_timeout(
            search.search(collection, text, n),
            timeout=SEARCH_TIMEOUT,
            operation_name="search_embeddings"
        )
        return result
    except Exception as e:
        app_logger.error(f"Search item task failed: {str(e)}")
        raise

@app.post("/search")
async def search_item_post(
    request: SearchRequest, 
    http_request: Request,
    # current_user: TokenData = Depends(require_read)
):
    """Search embeddings in a collection with caching support (requires read permission)."""
    StructuredLogger.log_request("search", {"collection": request.collection, "text": request.text[:100] + "...", "n": request.n})
    
    try:
        # Validate input using our validator
        validated_text = InputValidator.validate_text_input(request.text)
        
        # Check cache first
        cache_key_data = {
            "collection": request.collection,
            "n": request.n
        }
        cached_result = embedding_cache.get(validated_text, filters=cache_key_data)
        
        if cached_result:
            metrics_collector.record_cache_hit()
            StructuredLogger.log_response("search", 200, 0, len(str(cached_result)))
            return cached_result
        
        # Cache miss - record it
        metrics_collector.record_cache_miss()
        
        # Perform search
        search_results = await search.search(request.collection, validated_text, request.n)
        
        if not search_results:
            raise HTTPException(status_code=404, detail=f"No results found in collection: {request.collection}")
        
        response = {
            "collection": request.collection,
            "query": validated_text,
            "n_requested": request.n,
            "n_returned": len(search_results) if isinstance(search_results, list) else 1,
            "results": search_results,
            "cached": False
        }
        
        # Cache the results
        embedding_cache.set(validated_text, response, filters=cache_key_data)
        
        StructuredLogger.log_response("search", 200, 0, len(str(response)))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        StructuredLogger.log_error("search", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

async def shard_embeddings_task(request: ShardEmbeddingsRequest):
    """Shard embeddings task with timeout protection"""
    try:
        if request is None:
            # Create default request
            request = ShardEmbeddingsRequest(
                dataset="TeraflopAI/Caselaw_Access_Project",
                split="train",
                column="text",
                dst_path="/storage/teraflopai/tmp",
                models=[
                    "thenlper/gte-small",
                    "Alibaba-NLP/gte-large-en-v1.5",
                    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                ]
            )
        
        # Validate models
        for model in request.models:
            InputValidator.validate_model_name(model)
        
        # Call the shard embeddings function with timeout protection
        shard_embeddings_results = await safe_async_execute_with_timeout(
            shards.kmeans_cluster_split(
                request.dataset,
                request.split,
                request.column,
                request.dst_path,
                request.models
            ),
            timeout=BACKGROUND_TASK_TIMEOUT,
            operation_name="shard_embeddings"
        )
        return shard_embeddings_results
    except Exception as e:
        app_logger.error(f"Shard embeddings task failed: {str(e)}")
        raise

@app.post("/shard_embeddings")
async def shard_embeddings_post(
    request: ShardEmbeddingsRequest, 
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """Shard embeddings into smaller chunks for distributed processing."""
    StructuredLogger.log_request("shard_embeddings", request.dict())
    
    try:
        # Validate dataset name
        InputValidator.validate_dataset_name(request.dataset)
        
        # Validate models
        for model in request.models:
            InputValidator.validate_model_name(model)
        
        # Generate request ID for tracking
        request_id = f"shard_{datetime.now().isoformat()}"
        
        # Add background task with timeout protection
        background_tasks.add_task(
            create_timeout_protected_background_task,
            shard_embeddings_task,
            request,
            timeout=BACKGROUND_TASK_TIMEOUT,
            task_name="shard_embeddings"
        )
        
        response = {
            "status": "started",
            "request_id": request_id,
            "message": "Shard embeddings task initiated",
            "dataset": request.dataset,
            "models": request.models
        }
        
        StructuredLogger.log_response("shard_embeddings", 202, 0, len(str(response)))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start shard embeddings task: {str(e)}"
        StructuredLogger.log_error("shard_embeddings", e, request.dict())
        raise HTTPException(status_code=500, detail=error_msg)
 
 
async def index_sparse_embeddings_task(request: CreateSparseEmbeddingsRequest):
    """Index sparse embeddings task with timeout protection"""
    try:
        # Execute sparse indexing with timeout protection
        index_sparse_embeddings_results = await safe_async_execute_with_timeout(
            sparse.index_sparse_embeddings(request),
            timeout=BACKGROUND_TASK_TIMEOUT,
            operation_name="index_sparse_embeddings"
        )
        return index_sparse_embeddings_results
    except Exception as e:
        app_logger.error(f"Index sparse embeddings task failed: {str(e)}")
        raise

@app.post("/index_sparse_embeddings")
async def index_sparse_embeddings_post(
    request: CreateSparseEmbeddingsRequest, 
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """Index sparse embeddings for efficient retrieval."""
    StructuredLogger.log_request("index_sparse_embeddings", request.dict())
    
    try:
        # Validate dataset name
        InputValidator.validate_dataset_name(request.dataset)
        
        # Validate models
        for model in request.models:
            InputValidator.validate_model_name(model)
        
        # Generate request ID for tracking
        request_id = f"sparse_{datetime.now().isoformat()}"
        
        # Add background task with timeout protection
        background_tasks.add_task(
            create_timeout_protected_background_task,
            index_sparse_embeddings_task,
            request,
            timeout=BACKGROUND_TASK_TIMEOUT,
            task_name="index_sparse_embeddings"
        )
        
        response = {
            "status": "started",
            "request_id": request_id,
            "message": "Sparse embeddings indexing task initiated",
            "dataset": request.dataset,
            "models": request.models
        }
        
        StructuredLogger.log_response("index_sparse_embeddings", 202, 0, len(str(response)))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start sparse embeddings indexing: {str(e)}"
        StructuredLogger.log_error("index_sparse_embeddings", Exception(error_msg), request.dict())
        raise HTTPException(status_code=500, detail=error_msg)

async def index_cluster_task(request: IndexClusterRequest):
    """Index cluster task with timeout protection"""
    try:
        # Use timeout protection for the cluster test operation
        result = await safe_async_execute_with_timeout(
            index_cluster.test(),
            timeout=DATABASE_TIMEOUT,
            operation_name="index_cluster_test"
        )
        return result
    except Exception as e:
        app_logger.error(f"Index cluster task failed: {str(e)}")
        raise

@app.post("/index_cluster")
async def index_cluster_post(
    request: IndexClusterRequest, 
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """Index cluster embeddings for efficient search and retrieval."""
    StructuredLogger.log_request("index_cluster", request.dict())
    
    try:
        # Generate request ID for tracking
        request_id = f"cluster_{datetime.now().isoformat()}"
        
        # Add background task with timeout protection
        background_tasks.add_task(
            create_timeout_protected_background_task,
            index_cluster_task,
            request,
            timeout=DATABASE_TIMEOUT,
            task_name="index_cluster"
        )
        
        response = {
            "status": "started",
            "request_id": request_id,
            "message": "Cluster indexing task initiated"
        }
        
        StructuredLogger.log_response("index_cluster", 202, 0, len(str(response)))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start cluster indexing: {str(e)}"
        StructuredLogger.log_error("index_cluster", Exception(error_msg), request.dict())
        raise HTTPException(status_code=500, detail=error_msg)

async def storacha_clusters_task(request: StorachaClustersRequest):
    """Storacha clusters task with timeout protection"""
    try:
        # Create storacha instance with timeout protection
        storacha_instance = storacha_clusters(request.resources, request.metadata)
        
        # Execute test method with timeout protection
        result = await safe_async_execute_with_timeout(
            storacha_instance.test(),
            timeout=BACKGROUND_TASK_TIMEOUT,
            operation_name="storacha_test"
        )
        return result
    except Exception as e:
        app_logger.error(f"Storacha clusters task failed: {str(e)}")
        raise

@app.post("/storacha_clusters")
async def storacha_clusters_post(
    request: StorachaClustersRequest, 
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """Upload and manage clusters using Storacha distributed storage."""
    StructuredLogger.log_request("storacha_clusters", request.dict())
    
    try:
        # Generate request ID for tracking
        request_id = f"storacha_{datetime.now().isoformat()}"
        
        # Add background task with timeout protection
        background_tasks.add_task(
            create_timeout_protected_background_task,
            storacha_clusters_task,
            request,
            timeout=BACKGROUND_TASK_TIMEOUT,
            task_name="storacha_clusters"
        )
        
        response = {
            "status": "started",
            "request_id": request_id,
            "message": "Storacha clusters task initiated"
        }
        
        StructuredLogger.log_response("storacha_clusters", 202, 0, len(str(response)))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start Storacha clusters task: {str(e)}"
        StructuredLogger.log_error("storacha_clusters", Exception(error_msg), request.dict())
        raise HTTPException(status_code=500, detail=error_msg)

class EmbeddingCache:
    """Simple in-memory cache for embedding results with TTL."""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache: dict[str, tuple[Any, datetime]] = {}
        self.ttl_minutes = ttl_minutes
    
    def _generate_key(self, query: str, model: str = "", filters: Optional[dict] = None) -> str:
        """Generate a cache key from query parameters."""
        cache_data = {
            "query": query,
            "model": model,
            "filters": filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, model: str = "", filters: Optional[dict] = None) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(query, model, filters)
        
        if key in self.cache:
            data, timestamp = self.cache[key]
            # Check if cache entry is still valid
            if datetime.now() - timestamp < timedelta(minutes=self.ttl_minutes):
                return data
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, query: str, result: Any, model: str = "", filters: Optional[dict] = None) -> None:
        """Cache a result with current timestamp."""
        key = self._generate_key(query, model, filters)
        self.cache[key] = (result, datetime.now())
    
    def clear_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= timedelta(minutes=self.ttl_minutes)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        current_time = datetime.now()
        
        expired_count = sum(
            1 for _, timestamp in self.cache.values()
            if current_time - timestamp >= timedelta(minutes=self.ttl_minutes)
        )
        
        return {
            "total_entries": total_entries,
            "valid_entries": total_entries - expired_count,
            "expired_entries": expired_count,
            "ttl_minutes": self.ttl_minutes
        }

# Initialize cache
embedding_cache = EmbeddingCache(ttl_minutes=30)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(lambda: deque())

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.window_seconds = 60
    
    async def dispatch(self, request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Current time
        now = time.time()
        
        # Clean old entries (outside window)
        rate_limit_storage[client_ip] = deque([
            timestamp for timestamp in rate_limit_storage[client_ip]
            if now - timestamp < self.window_seconds
        ])
        
        # Check rate limit
        if len(rate_limit_storage[client_ip]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls_per_minute} requests per minute allowed",
                    "retry_after": 60
                }
            )
        
        # Add current request to tracking
        rate_limit_storage[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        return response

# # Add rate limiting middleware
# app.add_middleware(RateLimitMiddleware, calls_per_minute=999999)  # Temporarily high limit for testing

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""
    
    async def dispatch(self, request, call_next):
        # Record request start
        endpoint = request.url.path
        start_time = metrics_collector.record_request_start(endpoint)
        
        # Process request
        response = await call_next(request)
        
        # Record request end
        metrics_collector.record_request_end(endpoint, start_time, response.status_code)
        
        return response

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

@app.get("/cache/stats")
async def get_cache_stats(http_request: Request):
    """Get cache statistics and performance metrics."""
    StructuredLogger.log_request("cache_stats", {})
    
    try:
        stats = embedding_cache.get_stats()
        
        # Add system memory info if available
        try:
            import psutil
            memory_info = {
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "memory_percent": psutil.Process().memory_percent()
            }
            stats.update(memory_info)
        except ImportError:
            stats["memory_info"] = "psutil not available"
        
        StructuredLogger.log_response("cache_stats", 200, 0, len(str(stats)))
        return stats
        
    except Exception as e:
        error_msg = f"Failed to get cache stats: {str(e)}"
        StructuredLogger.log_error("cache_stats", e, {})
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/cache/clear")
async def clear_cache(http_request: Request):
    """Clear expired cache entries."""
    StructuredLogger.log_request("cache_clear", {})
    
    try:
        removed_count = embedding_cache.clear_expired()
        
        response = {
            "message": "Cache cleared successfully",
            "expired_entries_removed": removed_count
        }
        
        StructuredLogger.log_response("cache_clear", 200, 0, len(str(response)))
        return response
        
    except Exception as e:
        error_msg = f"Failed to clear cache: {str(e)}"
        StructuredLogger.log_error("cache_clear", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/auth/login")
async def login(user_credentials: UserLogin, http_request: Request):
    """Authenticate user and return JWT token."""
    StructuredLogger.log_request("login", {"username": user_credentials.username})
    
    try:
        user = authenticate_user(user_credentials.username, user_credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = create_access_token(
            data={"sub": user["username"], "role": user["role"]}
        )
        
        response = {
            "access_token": access_token,
            "token_type": "bearer",
            "role": user["role"],
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
        }
        
        StructuredLogger.log_response("login", 200, 0, len(str({"access_token": access_token})))
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Login failed: {str(e)}"
        StructuredLogger.log_error("login", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/auth/me")
async def get_current_user_info(
    http_request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    """Get current user information."""
    StructuredLogger.log_request("user_info", {})
    
    try:
        response = {
            "username": current_user.username,
            "role": current_user.role
        }
        
        StructuredLogger.log_response("user_info", 200, 0, len(str(response)))
        return response
        
    except Exception as e:
        error_msg = f"Failed to get user info: {str(e)}"
        StructuredLogger.log_error("user_info", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics."""
    metrics = metrics_collector.get_prometheus_metrics()
    return Response(content=metrics, media_type="text/plain")

@app.get("/metrics/json")
async def get_json_metrics(http_request: Request):
    """Get metrics in JSON format."""
    StructuredLogger.log_request("metrics_json", {})
    
    try:
        system_metrics = metrics_collector.get_system_metrics()
        app_metrics = metrics_collector.get_application_metrics()
        
        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics,
            "application": app_metrics
        }
        
        StructuredLogger.log_response("metrics_json", 200, 0, len(str(response)))
        return response
        
    except Exception as e:
        error_msg = f"Failed to collect metrics: {str(e)}"
        StructuredLogger.log_error("metrics_json", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health/detailed")
async def get_detailed_health(http_request: Request):
    """Get detailed health status with metrics."""
    StructuredLogger.log_request("health_detailed", {})
    
    try:
        health_status = metrics_collector.get_health_status()
        
        StructuredLogger.log_response("health_detailed", 200, 0, len(str(health_status)))
        return health_status
        
    except Exception as e:
        error_msg = f"Failed to get health status: {str(e)}"
        StructuredLogger.log_error("health_detailed", Exception(error_msg), {})
        raise HTTPException(status_code=500, detail=error_msg)

async def safe_async_execute_with_timeout(
    coroutine, 
    timeout: float = DEFAULT_ASYNC_TIMEOUT,
    operation_name: str = "async_operation"
) -> Any:
    """
    Execute an async operation with timeout protection.
    
    Args:
        coroutine: The async coroutine to execute
        timeout: Timeout in seconds
        operation_name: Name for logging purposes
        
    Returns:
        Result of the coroutine
        
    Raises:
        TimeoutError: If operation times out
        Exception: Any other exception from the coroutine
    """
    try:
        result = await asyncio.wait_for(coroutine, timeout=timeout)
        app_logger.info(f"✓ {operation_name} completed successfully within {timeout}s")
        return result
    except asyncio.TimeoutError:
        error_msg = f"⚠️ {operation_name} timed out after {timeout} seconds"
        app_logger.error(error_msg)
        raise TimeoutError(error_msg)
    except Exception as e:
        error_msg = f"✗ {operation_name} failed: {str(e)}"
        app_logger.error(error_msg)
        raise

async def safe_network_request_with_timeout(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    timeout: float = NETWORK_TIMEOUT,
    **kwargs
) -> Any:
    """
    Make network request with comprehensive timeout protection.
    
    Args:
        session: aiohttp session
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        timeout: Timeout in seconds
        **kwargs: Additional arguments for the request
        
    Returns:
        Response data
        
    Raises:
        TimeoutError: If request times out
        Exception: Other network-related exceptions
    """
    try:
        # Configure timeout
        client_timeout = ClientTimeout(total=timeout)
        
        async with session.request(
            method=method,
            url=url,
            timeout=client_timeout,
            **kwargs
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Network request failed: {response.status}"
                )
                
    except asyncio.TimeoutError:
        error_msg = f"Network request to {url} timed out after {timeout}s"
        app_logger.error(error_msg)
        raise TimeoutError(error_msg)
    except aiohttp.ClientError as e:
        error_msg = f"Network error for {url}: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)

def create_timeout_protected_background_task(
    func,
    *args,
    timeout: float = BACKGROUND_TASK_TIMEOUT,
    task_name: str = "background_task",
    **kwargs
):
    """
    Create a background task with timeout protection.
    
    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        timeout: Timeout in seconds
        task_name: Name for logging purposes
        **kwargs: Keyword arguments for the function
        
    Returns:
        Wrapped async function with timeout protection
    """
    async def timeout_protected_task():
        try:
            app_logger.info(f"🚀 Starting {task_name} with {timeout}s timeout")
            # Note: StructuredLogger will be defined below
            
            # Execute with timeout protection
            result = await safe_async_execute_with_timeout(
                func(*args, **kwargs),
                timeout=timeout,
                operation_name=task_name
            )
            
            app_logger.info(f"✅ {task_name} completed successfully")
            return result
            
        except TimeoutError as e:
            error_msg = f"⏰ {task_name} timed out: {str(e)}"
            app_logger.error(error_msg)
            raise HTTPException(status_code=408, detail=error_msg)
            
        except Exception as e:
            error_msg = f"💥 {task_name} failed: {str(e)}"
            app_logger.error(error_msg)
            raise
    
    return timeout_protected_task

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
