from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from search_embeddings import search_embeddings
from create_embeddings import create_embeddings
from shard_embeddings import shard_embeddings
from sparse_embeddings import sparse_embeddings
from ipfs_cluster_index import ipfs_cluster_index
from storacha_clusters import storacha_clusters
from pydantic import BaseModel

class LoadIndexRequest(BaseModel):
    dataset: str
    knn_index: str
    dataset_split: Union[str, None]
    knn_index_split: Union[str, None]
    columns: list

class SearchRequest(BaseModel):
    collection: str
    text: str
    n: int

class CreateEmbeddingsRequest(BaseModel):
    dataset = str,
    split = str,
    column = str,
    dst_path = str,
    models = list
    
class ShardEmbeddingsRequest(BaseModel):
    dataset = str,
    split = str,
    column = str,
    dst_path = str,
    models = list
 
class CreateSparseEmbeddingsRequest(BaseModel):
    dataset = str,
    split = str,
    column = str,
    dst_path = str,
    models = list
    
class IndexClusterRequest(BaseModel):
    resources: dict
    metadata: dict
    
class StorachaClustersRequest(BaseModel):
    resources: dict
    metadata: dict
    
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

search = search_embeddings.search_embeddings(resources, metadata)
create = create_embeddings.create_embeddings(resources, metadata)
sparse = sparse_embeddings.sparse_embeddings(resources, metadata)
shards = shard_embeddings.shard_embeddings(resources, metadata)
storacha = storacha_clusters.storacha_clusters(resources, metadata)
index_cluster = ipfs_cluster_index.ipfs_cluster_index(resources, metadata)

app = FastAPI(port=9999)

@app.post("/add_endpoint")
def add_endpoint(request: dict):
    model = request["model"]
    endpoint = request["endpoint"]
    type = request["type"]
    ctx_length = request["ctx_length"]
    objects = [ search, sparse, shards, index_cluster, storacha, create]
    for obj in objects:
        if type == "libp2p":
            if "add_libp2p_endpoint" not in dir(obj):
                pass
            else:
                obj.add_libp2p_endpoint(model, endpoint, ctx_length)
        elif type == "https":
            if "add_https_endpoint" not in dir(obj):
                pass
            else:
                obj.add_https_endpoint(model, endpoint, ctx_length)
        elif type == "cuda":
            if "add_cuda_endpoint" not in dir(obj):
                pass
            else:
                obj.add_cuda_endpoint(model, endpoint, ctx_length)
        elif type == "local":
            if "add_local_endpoint" not in dir(obj):
                pass
            else:
                obj.add_local_endpoint(model, endpoint, ctx_length)
        elif type == "openvino":
            if "add_openvino_endpoint" not in dir(obj):
                pass
            else:
                obj.add_openvino_endpoint(model, endpoint, ctx_length)
        
    return {"message": "Endpoint added"}

async def create_embeddings_task(request: CreateEmbeddingsRequest):
    dataset = request.dataset
    split = request.split
    column = request.column
    dst_path = request.dst_path
    models = request.models
    create_embeddings_results = await create.create_embeddings(dataset, split, column, dst_path, models)
    return create_embeddings_results

@app.post("/create_embeddings")
def create_embeddings_post(request: CreateEmbeddingsRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(create_embeddings_task, request)    
    return {"message": "Index creation started in the background"}

async def load_index_task(dataset: str, knn_index: str, dataset_split: Union[str, None], knn_index_split: Union[str, None], columns: str):
    vector_search = search_embeddings.search_embeddings(resources, metadata)
    await vector_search.load_qdrant_iter(dataset, knn_index, dataset_split, knn_index_split)
    await vector_search.ingest_qdrant_iter(columns)
    return None

@app.post("/load")
def load_index_post(request: LoadIndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(load_index_task, request.dataset, request.knn_index, request.dataset_split, request.knn_index_split, request.columns)
    return {"message": "Index loading started in the background"}

async def search_item_task(collection: str, text: str, n: int):
    return await search.search(collection, text, n)

@app.post("/search")
async def search_item_post(request: SearchRequest, background_tasks: BackgroundTasks):
    search_results = await search.search(request.collection, request.text, request.n)
    return search_results

async def shard_embeddings_task(request: ShardEmbeddingsRequest):
    if request is None:
        request = {
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
    else:
        if "dataset" not in request.keys():
            request["dataset"] = "TeraflopAI/Caselaw_Access_Project"
        if "column" not in request.keys():
            request["column"] = "text"
        if "split" not in request.keys():
            request["split"] = "train"
        if "models" not in request.keys():
            request["models"] = [
                "thenlper/gte-small",
                "Alibaba-NLP/gte-large-en-v1.5",
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            ]
        if "chunk_settings" not in request.keys():
            request["chunk_settings"] = {
                "chunk_size": 512,
                "n_sentences": 8,
                "step_size": 256,
                "method": "fixed",
                "embed_model": "thenlper/gte-small",
                "tokenizer": None
            }
        if "dst_path" not in request.keys():
            request["dst_path"] = "/storage/teraflopai/tmp"
    shard_embeddings_results = await shards.kmeans_cluster_split(request)
    return shard_embeddings_results

@app.post("/shard_embeddings")
async def shard_embeddings_post(request: dict, background_tasks: BackgroundTasks):
    shard_embeddings_task_results = background_tasks.add_task(shard_embeddings_task, request)
    return shard_embeddings_task_results
 
 
async def index_sparse_embeddings_task(request: CreateSparseEmbeddingsRequest):
    index_sparse_embeddings_results = await sparse.index_sparse_embeddings(request)
    return index_sparse_embeddings_results

@app.post("/index_sparse_embeddings")
async def index_sparse_embeddings_post(request: dict, background_tasks: BackgroundTasks):
    index_sparse_embeddings_results = background_tasks.add_task(index_sparse_embeddings_task, request)
    return index_sparse_embeddings_results

async def index_cluster_task(request: IndexClusterRequest):
    index_cluster_results = await index_cluster.test()
    return index_cluster_results

@app.post("/index_cluster")
async def index_cluster_post(request: dict , background_tasks: BackgroundTasks):
    index_cluster_task_results = background_tasks.add_task(index_cluster_task, request)
    return index_cluster_task_results

async def storacha_clusters_task(request: StorachaClustersRequest):
    storacha_clusters_results = await storacha_clusters.storacha_clusters(request)
    storacha_clusters_task_results = storacha_clusters_results.upload()
    return storacha_clusters_task_results

@app.post("/storacha_clusters")
async def storacha_clusters_post(request: dict, background_tasks: BackgroundTasks):
    storacha_clustsers_post_results = background_tasks.add_task(storacha_clusters_task, request)
    return storacha_clustsers_post_results



uvicorn.run(app, host="0.0.0.0", port=9999)