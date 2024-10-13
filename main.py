from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from search_embeddings import search_embeddings
from create_embeddings import create_embeddings
from ipfs_embeddings_py import ipfs_embeddings_py
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

class CreateIndexRequest(BaseModel):
    resources: dict
    metadata: dict  

metadata = {
    "dataset": "laion/Wikipedia-X-Concat",
    "faiss_index": "laion/Wikipedia-M3",
    "model": "BAAI/bge-m3"
}
resources = {
    # "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",1]],
    # "https_endpoints": [["BAAI/bge-m3", "http://127.0.0.1:8080/embed",1]],
    "https_endpoints": [],
    "libp2p_endpoints": []
}

vector_search = search_embeddings.search_embeddings(resources, metadata)
index_dataset = create_embeddings.create_embeddings(resources, metadata)

app = FastAPI(port=9999)

@app.post("/add_endpoint")
def add_endpoint(request: dict):
    model = request["model"]
    endpoint = request["endpoint"]
    ctx_length = request["ctx_length"]
    create_embeddings.add_https_endpoint(model, endpoint, ctx_length)
    search_embeddings.add_https_endpoint(model, endpoint, ctx_length)
    return {"message": "Endpoint added"}

async def create_index_task(request: LoadIndexRequest, background_tasks: BackgroundTasks):
    dataset = request.dataset
    split = request.split
    column = request.column
    dst_path = request.dst_path
    models = request.models
    index_dataset = await create_embeddings.index_dataset(dataset, split, column, dst_path, models)
    return None

@app.post("/create")
def create_index_post(request: CreateIndexRequest):
    resources = request.resources
    metadata = request.metadata
    create_index_task(resources, metadata)
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
    return await vector_search.search(collection, text, n)

@app.post("/search")
async def search_item_post(request: SearchRequest, background_tasks: BackgroundTasks):
    search_results = await vector_search.search(request.collection, request.text, request.n)
    return search_results

uvicorn.run(app, host="0.0.0.0", port=9999)