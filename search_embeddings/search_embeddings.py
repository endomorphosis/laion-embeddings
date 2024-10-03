
import datasets
import sys
sys.path.append('./ipfs_embeddings_py')
import ipfs_embeddings_py
from ipfs_embeddings_py import ipfs_embeddings_py, qdrant_kit_py
import numpy as np
import os
import json
import pandas as pd
import subprocess
import asyncio
import hashlib
import random
from multiprocessing import Pool

class search_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.dataset = []
        if len(list(metadata.keys())) > 0:
            for key in metadata.keys():
                setattr(self, key, metadata[key])
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        self.qdrant_kit_py = qdrant_kit_py(resources, metadata)
        if "https_endpoints" in resources.keys():
            for endpoint in resources["https_endpoints"]:
                self.ipfs_embeddings_py.add_https_endpoint(endpoint[0], endpoint[1], endpoint[2])
        else:
            self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed",1)
        self.join_column = None
        self.qdrant_found = False
        qdrant_port_cmd = "nc -zv localhost 6333"
        qdrant_port_cmd_results = os.system(qdrant_port_cmd)
        if qdrant_port_cmd_results != 0:
            self.qdrant_kit_py.start_qdrant()
            qdrant_port_cmd_results = os.system(qdrant_port_cmd)
            if qdrant_port_cmd_results == 0:
                self.qdrant_found = True
            else:
                print("Qdrant failed to start, fallback to faiss")
        else:
            self.qdrant_found = True
        self.add_https_endpoint = self.add_https_endpoint

    def add_https_endpoint(self, model, endpoint, ctx_length):
        return self.ipfs_embeddings_py.add_https_endpoint(model, endpoint, ctx_length)
    
    
    def rm_cache(self):
        homedir = os.path.expanduser("~")
        cache_dir = homedir + "/.cache/huggingface/datasets/"
        cache_dir = os.path.expanduser(cache_dir)
        os.system("rm -rf " + cache_dir)
        return None

    async def generate_embeddings(self, query, model=None):
        if model is not None:
            model = self.metadata["model"]
        if isinstance(query, str):
            query = [query]
        elif not isinstance(query, list):
            raise ValueError("Query must be a string or a list of strings")
        self.ipfs_embeddings_py.index_knn(query, "")
        selected_endpoint = self.ipfs_embeddings_py.choose_endpoint(self.model)
        embeddings = await self.ipfs_embeddings_py.index_knn(selected_endpoint, self.model)
        return embeddings
    
    def search_embeddings(self, embeddings):
        scores, samples = self.qdrant_kit_py.knn_index.get_nearest_examples(
           "embeddings", embeddings, k=5
        )
        return scores, samples 
        
    async def search(self, collection, query, n=5):
        if self.qdrant_found == True:
            query_embeddings = await self.ipfs_embeddings_py.index_knn(query, self.metadata["model"])
            vector_search = await self.qdrant_kit_py.search_qdrant(collection, query_embeddings, n)
        else:
            print("Qdrant failed to start")
            ## Fallback to faiss
            return None
        return vector_search

    async def test_low_memory(self):
        start = self.qdrant_kit_py.start_qdrant()
        # load_qdrant = await self.load_qdrant_iter("laion/Wikipedia-X-Concat", "laion/Wikipedia-M3", "enwiki_concat", "enwiki_embed")
        # ingest_qdrant = await self.ingest_qdrant_iter("Concat Abstract")
        load_qdrant = await self.qdrant_kit_py.load_qdrant_iter("laion/English-ConcatX-Abstract", "laion/English-ConcatX-M3")
        ingest_qdrant = await self.qdrant_kit_py.ingest_qdrant_iter("Concat Abstract")
        load_qdrant = await self.qdrant_kit_py.load_qdrant_iter("laion/German-ConcatX-Abstract", "laion/German-ConcatX-M3")
        ingest_qdrant = await self.qdrant_kit_py.ingest_qdrant_iter("Concat Abstract")
        results = await search_embeddings.search("laion/German-ConcatX-Abstract", "Machine Learning")
        results = await search_embeddings.search("German-ConcatX-Abstract", "Machine Learning")
        return None
    
    async def test_high_memory(self):
        start = self.qdrant_kit_py.start_qdrant()
        load_qdrant = await self.qdrant_kit_py.load_qdrant("laion/Wikipedia-X-Concat", "laion/Wikipedia-M3", "enwiki_concat", "enwiki_embed")
        results = await search_embeddings.search("Wikipedia-X-Concat", "Machine Learning")
        return results

    async def test_query(self):
        query = "Machine Learning"
        collection = "English-ConcatX-Abstract"
        search_results = await self.search(collection, query)
        print(search_results)
        return None
        
if __name__ == '__main__':
    metadata = {
        "dataset": "laion/Wikipedia-X-Concat",
        "faiss_index": "laion/Wikipedia-M3",
        "model": "BAAI/bge-m3"
    }
    resources = {
        "https_endpoints": [["BAAI/bge-m3" , "http://62.146.169.111:8083/embed-small" , 8192]]
    }
    search_embeddings = search_embeddings(resources, metadata)
    # asyncio.run(search_embeddings.test_high_memory())
    # asyncio.run(search_embeddings.test_low_memory())
    asyncio.run(search_embeddings.test_query())

    print()