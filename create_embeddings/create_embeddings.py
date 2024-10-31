import asyncio
from aiohttp import ClientSession
from datasets import load_dataset, Dataset
import os
import sys
import datasets
sys.path.append('../ipfs_embeddings_py')
from ipfs_embeddings_py import ipfs_embeddings_py
import os
import sys
import subprocess
from transformers import AutoTokenizer
import random
from multiprocessing import Pool

class create_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.index =  {}
        self.cid_list = []
        if len(list(metadata.keys())) > 0:
            for key in metadata.keys():
                setattr(self, key, metadata[key])
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        if "https_endpoints" in resources.keys():
            for endpoint in resources["https_endpoints"]:
                self.ipfs_embeddings_py.add_https_endpoint(endpoint[0], endpoint[1], endpoint[2])
        self.join_column = None
        self.tokenizer = {}
        self.index_dataset = self.index_dataset

    def add_https_endpoint(self, model, endpoint, ctx_length):
        return self.ipfs_embeddings_py.add_https_endpoint(model, endpoint, ctx_length)

    async def create_embeddings(self, dataset, split, column, dst_path, models):
        await self.ipfs_embeddings_py.index_dataset(dataset, split, column, dst_path, models)
        return None
           
    async def __call__(self, dataset, split, column, dst_path, models):
        await self.ipfs_embeddings_py.index_dataset(dataset, split, column, dst_path, models)
        return None

    async def test(self, dataset, split, column, dst_path, models):
        https_endpoints = [
            # ["Alibaba-NLP/gte-large-en-v1.5", "http://127.0.0.1:8080/embed", 8192],
            # ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://127.0.0.1:8082/embed", 32768],
            # # ["Alibaba-NLP/gte-Qwen2-7B-instruct", "http://62.146.169.111:8080/embed-large", 32000],
            # ["Alibaba-NLP/gte-large-en-v1.5", "http://127.0.0.1:8081/embed", 8192],
            # ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://127.0.0.1:8083/embed", 32768],
            # # ["Alibaba-NLP/gte-Qwen2-7B-instruct", "http://62.146.169.111:8081/embed-large", 32000],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8080/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8080/embed-medium", 32000],
            # ["Alibaba-NLP/gte-Qwen2-7B-instruct", "http://62.146.169.111:8080/embed-large", 32000],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8081/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8081/embed-medium", 32000],
            # ["Alibaba-NLP/gte-Qwen2-7B-instruct", "http://62.146.169.111:8081/embed-large", 32000],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8082/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8082/embed-medium", 32000],
            # ["Alibaba-NLP/gte-Qwen2-7B-instruct", "http://62.146.169.111:8082/embed-large", 32000],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8083/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8083/embed-medium", 32000],
            # ["Alibaba-NLP/gte-Qwen2-7B-instruct", "http://62.146.169.111:8083/embed-large", 32000],
        ]
        for endpoint in https_endpoints:
            self.add_https_endpoint(endpoint[0], endpoint[1], endpoint[2])
        await self.index_dataset(dataset, split, column, dst_path, models)
        return None
    
if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "split": "train",
        "column": "text",
        "models": [
            "Alibaba-NLP/gte-large-en-v1.5",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            # "dunzhang/stella_en_1.5B_v5",
        ],
        "dst_path": "/storage/teraflopai/tmp"
    }
    resources = {
    }
    create_embeddings_batch = create_embeddings(resources, metadata)
    asyncio.run(create_embeddings_batch.test(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))    
