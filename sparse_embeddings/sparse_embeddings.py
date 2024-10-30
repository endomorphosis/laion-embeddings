import os
import sys
import json
import random
import datasets
import asyncio
import subprocess
import aiohttp
import requests
from aiohttp import ClientSession, ClientTimeout
from multiprocessing import Pool
from transformers import AutoTokenizer
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
from ..ipfs_embeddings_py import ipfs_embeddings_py
from ..ipfs_embeddings_py import *
from ..ipfs_embeddings_py import ipfs_multiformats_py
from ipfs_multiformats import *
from ..ipfs_embeddings_py import chunker
from chunker import Chunker
import time

class sparse_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        self.ipfs_multiformats_py = ipfs_multiformats_py(resources, metadata)
        self.index_dataset = self.index_dataset
        self.index_sparse_embeddings = self.index_sparse_embeddings        
        return None

    def index_dataset(self, dataset):
        return self.ipfs_embeddings_py.index_dataset(dataset)
    
    def index_sparse_embeddings(self, embeddings):
        return self.ipfs_embeddings_py.index_sparse_embeddings(embeddings)
    
    def test():
        results = {}
        test_ipfs_kit_init = None
        test_ipfs_kit = None
        test_ipfs_parquet_to_car = None
        test_storacha_clusters = None
        try:
            test_ipfs_kit_init = self.ipfs_kit.init()
        except Exception as e:
            test_ipfs_kit_init = e
            print(e)
            raise e

        try:
            test_ipfs_kit = self.ipfs_kit.test()
        except Exception as e:
            test_ipfs_kit = e
            print(e)
            raise e
        try:
            test_ipfs_parquet_to_car = self.ipfs_parquet_to_car.test()
        except Exception as e:
            test_ipfs_parquet_to_car = e
            print(e)
            raise e
        try:

            test_storacha_clusters = self.ipfs_kit_py.storacha_kit_py.test()
        except Exception as e:
            test_storacha_clusters = e
            print(e)
            raise e

if __name__ == "main":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "split": "train",
        "models": [
            "thenlper/gte-small",
            # "Alibaba-NLP/gte-large-en-v1.5",
            # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        ],
        "chunk_settings": {
            "chunk_size": 512,
            "n_sentences": 8,
            "step_size": 256,
            "method": "fixed",
            "embed_model": "thenlper/gte-small",
            "tokenizer": None
        },
        "dst_path": "/storage/teraflopai/tmp2",
    }
    resources = {
        "https_endpoints": [
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8080/embed-small", 8192],
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
    test_sparse_embeddings = sparse_embeddings(resources, metadata)
    results = test_sparse_embeddings.test()
    print(results)   