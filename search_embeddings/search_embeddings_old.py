
import datasets
import sys
import ipfs_embeddings_py
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
        self.ipfs_embeddings_py = ipfs_embeddings_py.ipfs_embeddings_py(resources, metadata)
        self.qdrant_kit_py = ipfs_embeddings_py.qdrant_kit_py(resources, metadata)
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
    
    # def search_embeddings(self, embeddings):
    #     scores, samples = self.qdrant_kit_py.knn_index.get_nearest_examples(
    #        "embeddings", embeddings, k=5
    #     )
    #     return scores, samples 
        
    async def search(self, collection, query, n=5):
        query_embeddings = await self.generate_embeddings(query)
        if self.qdrant_found == True:
            vector_search = await self.qdrant_kit_py.search_qdrant(collection, query_embeddings, n)
        else:
            vector_search = self.search_faiss(collection, query_embeddings, n)
        return vector_search

    async def test_low_memory(self, collections=[], datasets=[], column=None, query=None):
        if query is None:
            query = "the quick brown fox jumps over the lazy dog"
        if column is None:
            column = "Concat Abstract"
        if len(datasets) == 0:
            datasets = ["laion/German-ConcatX-Abstract", "laion/German-ConcatX-M3"]
        if len(collections) == 0:
            collections = [ x for x in datasets if "/" in x]
            collections = [ x.split("/")[1] for x in collections]
        start_qdrant = self.qdrant_kit_py.start_qdrant()
        if start_qdrant == True:
            print("Qdrant started")
            datasets_pairs = ["",""]
            search_results = {
                collections: [],
                results: []
            }
            for i in range(len(datasets)):
                if i % 2 == 0:
                    datasets_pairs.append(datasets[i-1], datasets[i])
                await self.qdrant_kit_py.load_qdrant(datasets_pairs[0], datasets_pairs[1])
                await self.qdrant_kit_py.ingest_qdrant(column)
            for collection in collections:
                results = await self.search(collection, query)
                search_results[collection] = results

            return search_results
        else:
            start_faiss = self.ipfs_embeddings_py.start_faiss(collection, query)
            if start_faiss == True:
                print("Faiss started")
                datasets_pairs = ["",""]
                search_results = {
                    collections: [],
                    results: []
                }
                for i in range(len(datasets)):
                    if i % 2 == 0:
                        datasets_pairs.append(datasets[i-1], datasets[i])
                    await self.ipfs_embeddings_py.load_faiss(datasets_pairs[0], datasets_pairs[1])
                    await self.ipfs_embeddings_py.ingest_faiss(column)
                for collection in collections:
                    results = await self.search(collection, query)
                    search_results[collection] = results

                return search_results
            else:
                print("Faiss failed to start")
                return None
    
    async def load_qdrant_iter(self, dataset, knn_index, dataset_split=None, knn_index_split=None):
        await self.qdrant_kit_py.load_qdrant_iter(dataset, knn_index, dataset_split, knn_index_split)
        return None

    async def ingest_qdrant_iter(self, columns):
        await self.qdrant_kit_py.ingest_qdrant_iter(columns)
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
        
    async def start_faiss(self, collection, query):
        return self.ipfs_embeddings_py.start_faiss(collection, query)
    
    async def load_faiss(self, dataset, knn_index):
        return self.ipfs_embeddings_py.load_faiss(dataset, knn_index)
    
    async def ingest_faiss(self, column):
        return self.ipfs_embeddings_py.ingest_faiss(column)
    
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