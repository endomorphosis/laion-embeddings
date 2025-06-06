import os
import sys
import json
import random
import datasets
import asyncio
import subprocess
import aiohttp
import requests
import torch
import faiss
import math
import gc
import timeit
import time
import numpy as np
from aiohttp import ClientSession, ClientTimeout
import multiprocessing
from multiprocessing import Pool
import transformers
from transformers import AutoTokenizer, AutoModel
import datasets
import ipfs_accelerate_py
import chunker
import qdrant_kit
import elasticsearch_kit

from datasets import Dataset, concatenate_datasets, load_dataset
try:
    from .ipfs_multiformats import ipfs_multiformats_py
    from .ipfs_multiformats import *
except Exception as e:
    try:
        from ipfs_multiformats import ipfs_multiformats_py
        from ipfs_multiformats import *
    except Exception as e:
        try:
            import ipfs_multiformats
        except Exception as e:
            pass
    pass

try:
    from .chunker import chunker
    from .chunker import *
except Exception as e:
    try:
        from chunker import chunker
        from chunker import *
    except Exception as e:
        try:
            import chunker
        except Exception as e:
            pass
    pass

try:
    from .elasticsearch_kit import elasticsearch_kit
    from .elasticsearch_kit import *
except Exception as e:
    try:
        from elasticsearch_kit import elasticsearch_kit
        from elasticsearch_kit import *
    except Exception as e:
        pass
    pass

try:
    from .qdrant_kit import qdrant_kit_py
    from .qdrant_kit import *
except Exception as e:
    try:
        from qdrant_kit import qdrant_kit_py
        from qdrant_kit import *
    except Exception as e:
        pass
    pass

try:
    from .faiss_kit import faiss_kit_py
    from .faiss_kit import *
except Exception as e:
    try:
        from faiss_kit import faiss_kit_py
        from faiss_kit import *
    except Exception as e:
        pass
    pass


from multiprocessing import Process
import concurrent.futures
import concurrent
import json
from ipfs_kit_py.ipfs_kit import ipfs_kit
from ipfs_embeddings_py.ipfs_datasets import ipfs_datasets_py
from ipfs_kit_py.ipfs_kit import ipfs_kit
import ipfs_accelerate_py


class ipfs_embeddings_py:
    def __init__(self, resources, metadata):
        self.multiformats = ipfs_multiformats_py(resources, metadata)
        self.multiformats_py = ipfs_multiformats_py(resources, metadata)
        self.datasets = datasets.Dataset
        self.ipfs_datasets = ipfs_datasets_py(resources, metadata)
        self.chunker = chunker(resources, metadata)
        self.qdrant_kit_py = qdrant_kit_py(resources, metadata)
        self.elasticsearch_kit = elasticsearch_kit(resources, metadata)
        self.faiss_kit = faiss_kit_py(resources, metadata)
        self.ipfs_accelerate_py = ipfs_accelerate_py(resources, metadata)
        self.process_new_dataset_shard = self.ipfs_datasets.process_new_dataset_shard
        self.process_index_shard = self.ipfs_datasets.process_index_shard
        self.ipfs_parquet_to_car = self.ipfs_datasets.ipfs_parquet_to_car_py
        self.ipfs_parquet_to_car_test = self.ipfs_datasets.ipfs_parquet_to_car_py.test
        self.ipfs_parquet_to_car_install = self.ipfs_datasets.ipfs_parquet_to_car_py.install
        if "ipfs_embeddings" not in dir(self) and "ipfs_embeddings" in self.__dict__.keys():
            self.ipfs_embeddings = self
        self.parquet_to_car = self.ipfs_parquet_to_car
        # self.elasticsearch = elasticsearch_kit(resources, metadata)
        self.consumer_task_done = {}
        self.producer_task_done = False
        self.save_to_disk_task_done = False
        self.tei_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.local_endpoints = {}
        self.index =  {}
        self.schemas = {}
        self.queues = {}
        self.caches = {}
        self.chunk_cache = {}
        self.chunk_embeddings = {}
        self.cid_chunk_list = []
        self.cid_chunk_set = set()
        self.batch_sizes = {}
        self.cid_list = set()
        self.cid_set = set()
        self.new_dataset = None
        self.all_cid_list = {}
        self.all_cid_set = {}
        self.cid_chunk_queue = None
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.endpoint_handler = {}
        self.new_dataset = {}
        self.new_dataset_children = {}
        self.saved = False
        self.resources = resources
        self.metadata = metadata
        self.index_dataset = self.index_dataset
        self.max_batch_size = self.ipfs_kit.max_batch_size
        self.producer = self.producer
        self.save_checkpoints_to_disk = self.save_checkpoints_to_disk
        self.save_chunks_to_disk = self.save_chunks_to_disk
        self.index_cid = self.index_cid
        self.load_index = self.load_index
        self.async_generator = self.async_generator
        self.kmeans_cluster_split = self.kmeans_cluster_split
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.ipfs_kit.add_endpoint
        self.rm_endpoint = self.ipfs_kit.rm_endpoint
        self.init_endpoints = self.init_endpoints       
        return None

    async def init(self, models , endpoints ):
        resources = await self.init_endpoints(models, endpoints)
        resources_keys = ["queues", "batch_sizes", "endpoints", "models", "worker"]
        for resource in resources_keys:
            if resource in list(self.ipfs_kit.resources.keys()) and resources is not None:
                this_resource = resources[resource]
                if type(this_resource) is dict:
                    for key in list(this_resource.keys()):
                        self.resources[resource][key] = this_resource[key]
                elif type(this_resource) is object:
                    self.resources[resource] = this_resource
        test_endpoints = None
        test_endpoints = await self.ipfs_kit.test_endpoints(models)
        return test_endpoints
        
    async def async_generator(self, iterable):
        for item in iterable:
            yield item
            
    async def max_batch_size(self, model, endpoint, endpoit_handler):
        print("max batch size")
        return await self.ipfs_kit.max_batch_size(model, endpoint, endpoit_handler)
        
    async def chunk_item(self, item, column=None, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None):
        # Assuming `item` is a dictionary with required data
        cuda_test = False
        openvino_test = False
        tokenizer_types = list(self.ipfs_kit.resources["tokenizer"][embed_model].keys())
        cuda_tokenizer_types = [x for x in tokenizer_types if "cuda" in x]
        openvino_tokenizer_types = [x for x in tokenizer_types if "openvino" in x]
        if self.ipfs_kit.resources["hwtest"]["cuda"] == True:
            cuda_test = True
        if self.ipfs_kit.resources["hwtest"]["openvino"] == True:
            openvino_test = True
        if column is None:
            content = json.dumps(item)
        elif column not in list(item.keys()):
            content = json.dumps(item)
        else:
            content = item[column]
        if embed_model is None:
            if len(self.metadata["models"]) == 0:
                embed_model = "thenlper/gte-small"
            else:
                embed_model = self.metadata["models"][0]
        if chunk_size is None:
            chunk_size = 512
        if n_sentences is None:
            n_sentences = 8
        if step_size is None:
            step_size = 256
        if tokenizer is None:
            if embed_model not in list(self.ipfs_kit.resources["tokenizer"].keys()):
                self.tokenizer[embed_model] = {}
            if cuda_test == True:
                tokenizer_types = list(self.ipfs_kit.resources["tokenizer"][embed_model].keys())
                cuda_tokenizer_types = [x for x in tokenizer_types if "cuda" in x]
                random_cuda_tokenizer = random.choice(cuda_tokenizer_types)
                device = random_cuda_tokenizer
                tokenizer = self.ipfs_kit.resources["tokenizer"][embed_model][random_cuda_tokenizer]
                batch_size = self.ipfs_kit.resources["batch_sizes"][embed_model][random_cuda_tokenizer]
                if batch_size == 0 or batch_size is None:
                    batch_size = 32
            elif openvino_test == True:
                openvino_tokenizer_types = [x for x in tokenizer_types if "openvino" in x]
                random_openvino_tokenizer = random.choice(openvino_tokenizer_types)
                device = random_openvino_tokenizer
                tokenizer = self.ipfs_kit.resources["tokenizer"][embed_model][random_openvino_tokenizer]  
                batch_size = self.ipfs_kit.resources["batch_sizes"][embed_model][random_openvino_tokenizer]
                if batch_size == 0 or batch_size is None:
                    batch_size = 1
            elif "cpu" not in tokenizer_types:
                tokenizer = self.ipfs_kit.resources["tokenizer"][embed_model]["cpu"]                
                batch_size = self.ipfs_kit.resources["batch_sizes"][embed_model]["cpu"]
                device = "cpu"
                if batch_size == 0 or batch_size is None:
                    batch_size = 1
            else:
                device = "cpu"
                tokenizer =  AutoTokenizer.from_pretrained(embed_model, device='cpu', use_fast=True)
                batch_size = 1
        if method is None:
            fixed_chunk_list = self.chunker.chunk(content, tokenizer, "fixed", 512, 8, 256, self.metadata["models"][0], device, batch_size)
            semantic_chunk_list = self.chunker.chunk(content, tokenizer, "semantic", 512, 8, 256, self.metadata["models"][0], device, batch_size)
            sentences_chunk_list = self.chunker.chunk(content, tokenizer, "sentences", 512, 8, 256, self.metadata["models"][0], device, batch_size) 
            sliding_window_chunk_list = self.chunker.chunk(content, tokenizer, "sliding_window", 512, 8, 256, self.metadata["models"][0], device, batch_size)
            content_chunks = fixed_chunk_list + semantic_chunk_list + sentences_chunk_list + sliding_window_chunk_list
        else:
            content_chunks = self.chunker.chunk(content, tokenizer, method, chunk_size, n_sentences, step_size, embed_model)
        parent_cid = item["cid"]
        content_tokens = tokenizer.encode(content)
        ## sort content_chunks by the firt element of each tuple then the second element
        content_chunks = sorted(content_chunks, key=lambda x: (x[0], x[1]))
        ## filter out chunks that are larger than the chunk_size
        content_chunks = [chunk for chunk in content_chunks if chunk[1] - chunk[0] <= chunk_size]
        ## filter content_chunks to remove duplicates
        seen_chunks = set()
        unique_content_chunks = []
        for chunk in content_chunks:
            if chunk not in seen_chunks:
                unique_content_chunks.append(chunk)
                seen_chunks.add(chunk)
        content_chunks = unique_content_chunks
        if parent_cid in list(self.caches.keys()):
            pass
        else:
            cid_chunks = {"parent_cid": parent_cid, "items" : [], "children": []}
            for chunk in content_chunks:
                chunk_index = chunk
                chunk_content = content_tokens[chunk[0]:chunk[1]]
                chunk_text = tokenizer.decode(chunk_content)
                child_cid = self.multiformats.get_cid(chunk_text)
                child_content = {"cid": child_cid, "index": chunk_index, "content": chunk_text, "parent_cid": parent_cid}
                cid_chunks["children"].append(child_cid)
                cid_chunks["items"].append(child_content)
        
            if parent_cid not in self.cid_chunk_set:
                while self.cid_chunk_queue.full():
                    await asyncio.sleep(0.1)
                if not self.cid_chunk_queue.full():
                    self.cid_chunk_queue.put_nowait(cid_chunks)
                    print("Added chunked item to queue for CID " + cid_chunks["parent_cid"])
            else:
                print("CID " + cid_chunks["parent_cid"] + " already in chunk set")
        return cid_chunks

    async def save_chunks_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(60)
            if self.saved == False:
                if len(self.chunk_cache) > 0: 
                    for this_cid in list(self.chunk_cache.keys()):
                        this_chunk = self.chunk_cache[this_cid]
                        if len(this_chunk["children"]) == len(this_chunk["items"]):
                            this_cid_dataset = datasets.Dataset.from_dict({"items":this_chunk["items"]})
                            this_cid_path = os.path.join(dst_path, "checkpoints", "sparse_chunks", this_cid + ".parquet")
                            this_cid_dataset.to_parquet(this_cid_path)
                            print("Saved " + str(len(this_cid_dataset)) + " chunks to disk for CID " + this_cid + " at " + this_cid_path)
                            self.cid_chunk_set.add(this_cid)
                            self.cid_chunk_list.append(this_cid)
                            del self.chunk_cache[this_cid]
                            del this_cid_dataset
                    self.saved = True
                    await asyncio.sleep(0.01)            
                    chunk_dir_path = os.path.join(dst_path, "checkpoints", "sparse_chunks")
                    chunk_files = os.listdir(chunk_dir_path)
                    await asyncio.sleep(0.01)
                    chunk_files = [x for x in chunk_files if ".parquet" in x]
                    saved_chunk_cids = [x.split(".")[0] for x in chunk_files]
                    self.cid_chunk_list += saved_chunk_cids
                    self.cid_chunk_set = set(saved_chunk_cids)     
                    await asyncio.sleep(0.01)
        return None
    
    async def queue_size(self, model):
        print("Checking queue size")
        queue_size = 0
        for endpoint in list(self.ipfs_kit.resources["batch_sizes"][model].keys()):
            queue_size += self.ipfs_kit.resources["batch_sizes"][model][endpoint]
        if queue_size == 0:
            queue_size = 1
        return queue_size

    async def save_checkpoints_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(60)
            if self.saved == False:
                if not os.path.exists(os.path.join(dst_path, "checkpoints")):
                    os.makedirs(os.path.join(dst_path, "checkpoints"))
                if not os.path.exists(os.path.join(dst_path, "checkpoints", "sparse_chunks")):
                    os.makedirs(os.path.join(dst_path, "checkpoints", "sparse_chunks"))
                if not os.path.exists(os.path.join(dst_path, "checkpoints", "sparse_embeddings")):
                    os.makedirs(os.path.join(dst_path, "checkpoints", "sparse_embeddings"))
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                if self.caches["new_dataset"] and len(self.caches["new_dataset"]["items"]) > 0:
                    tmp_dataset = datasets.Dataset.from_dict(self.caches["new_dataset"])
                    tmp_dataset_cids = tmp_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                    self.all_cid_list["new_dataset"] += tmp_dataset_cids
                    self.all_cid_set["new_dataset"] = set(self.all_cid_set["new_dataset"].union(set(tmp_dataset_cids)))
                    tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": tmp_dataset_cids})
                    new_dataset_shards = [x for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x]
                    next_filename_shard = f"ipfs_{dataset.replace('/', '___')}_shard_{len(new_dataset_shards)}"
                    tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                    tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                    del tmp_dataset
                    del tmp_dataset_cids
                    del tmp_dataset_cids_dataset
                    del self.caches["new_dataset"]
                    self.caches["new_dataset"] = {"items" : []}
                for model in models:
                    if model in self.caches.keys():
                        if self.caches[model] and len(self.caches[model]["items"]) > 0:
                            tmp_dataset = datasets.Dataset.from_dict(self.caches[model])
                            tmp_dataset_cids = tmp_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                            self.all_cid_list[model] += tmp_dataset_cids
                            self.all_cid_set[model] = set(self.all_cid_set[model].union(set(tmp_dataset_cids)))
                            tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": list(tmp_dataset_cids)})
                            self.caches[model] = {"items" : []}
                            this_model_shards = [x for x in ls_checkpoints if model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                            next_filename_shard = f"{dataset.replace('/', '___')}_{model.replace('/', '___')}_shard_{len(this_model_shards)}"
                            tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                            tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                            print("Saved "+ str(len(tmp_dataset)) + " items to disk for model " + model + " at " + dst_path)
                            del tmp_dataset
                            del tmp_dataset_cids
                            del tmp_dataset_cids_dataset
                            self.caches[model] = {"items" : []}
                for this_cid in list(self.chunk_cache.keys()):
                    this_chunk = self.chunk_cache[this_cid]
                    this_cid_dataset = datasets.Dataset.from_dict({"items":this_chunk["items"]})
                    this_cid_dataset.to_parquet(os.path.join(dst_path, "checkpoints", "sparse_chunks", this_cid + ".parquet"))
                    print("Saved " + str(len(this_cid_dataset)) + " chunks to disk for CID " + this_cid + " at " + dst_path)
                    self.cid_chunk_set.add(this_cid)
                    self.cid_chunk_list.append(this_cid)
                    del self.chunk_cache[this_cid]
                    del this_cid_dataset
                self.saved = True
            # if self.producer_task_done and all(self.consumer_task_done.values()):
            #     self.save_to_disk_task_done = True
            #     break
        return None 

         
    async def save_checkpoints_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(60)
            if self.saved == False:
                if not os.path.exists(os.path.join(dst_path, "checkpoints")):
                    os.makedirs(os.path.join(dst_path, "checkpoints"))
                if not os.path.exists(os.path.join(dst_path, "checkpoints", "sparse_chunks")):
                    os.makedirs(os.path.join(dst_path, "checkpoints", "sparse_chunks"))
                if not os.path.exists(os.path.join(dst_path, "checkpoints", "sparse_embeddings")):
                    os.makedirs(os.path.join(dst_path, "checkpoints", "sparse_embeddings"))
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                if self.caches["new_dataset"] and len(self.caches["new_dataset"]["items"]) > 0:
                    tmp_dataset = datasets.Dataset.from_dict(self.caches["new_dataset"])
                    tmp_dataset_cids = tmp_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                    self.all_cid_list["new_dataset"] += tmp_dataset_cids
                    self.all_cid_set["new_dataset"] = set(self.all_cid_set["new_dataset"].union(set(tmp_dataset_cids)))
                    tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": tmp_dataset_cids})
                    new_dataset_shards = [x for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x]
                    next_filename_shard = f"ipfs_{dataset.replace('/', '___')}_shard_{len(new_dataset_shards)}"
                    tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                    tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                    del tmp_dataset
                    del tmp_dataset_cids
                    del tmp_dataset_cids_dataset
                    del self.caches["new_dataset"]
                    self.caches["new_dataset"] = {"items" : []}
                for model in models:
                    if model in self.caches.keys():
                        if self.caches[model] and len(self.caches[model]["items"]) > 0:
                            tmp_dataset = datasets.Dataset.from_dict(self.caches[model])
                            tmp_dataset_cids = tmp_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                            self.all_cid_list[model] += tmp_dataset_cids
                            self.all_cid_set[model] = set(self.all_cid_set[model].union(set(tmp_dataset_cids)))
                            tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": list(tmp_dataset_cids)})
                            self.caches[model] = {"items" : []}
                            this_model_shards = [x for x in ls_checkpoints if model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                            next_filename_shard = f"{dataset.replace('/', '___')}_{model.replace('/', '___')}_shard_{len(this_model_shards)}"
                            tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                            tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                            print("Saved "+ str(len(tmp_dataset)) + " items to disk for model " + model + " at " + dst_path)
                            del tmp_dataset
                            del tmp_dataset_cids
                            del tmp_dataset_cids_dataset
                            self.caches[model] = {"items" : []}
                for this_cid in list(self.chunk_cache.keys()):
                    this_chunk = self.chunk_cache[this_cid]
                    this_cid_dataset = datasets.Dataset.from_dict({"items":this_chunk["items"]})
                    this_cid_dataset.to_parquet(os.path.join(dst_path, "checkpoints", "sparse_chunks", this_cid + ".parquet"))
                    print("Saved " + str(len(this_cid_dataset)) + " chunks to disk for CID " + this_cid + " at " + dst_path)
                    self.cid_chunk_set.add(this_cid)
                    self.cid_chunk_list.append(this_cid)
                    del self.chunk_cache[this_cid]
                    del this_cid_dataset
                self.saved = True
            # if self.producer_task_done and all(self.consumer_task_done.values()):
            #     self.save_to_disk_task_done = True
            #     break
        return None 

    async def index_sparse_chunks(self, dataset, split, column, dst_path, models = None):
        self.queues = {}
        self.cid_set = set()
        self.all_cid_list = {}
        self.cid_chunk_queue = asyncio.Queue()
        # Initialize resources and endpoints
        resource_keys = list(self.resources.keys())
        endpoints = {resource: self.resources[resource] 
                    for resource in resource_keys if "endpoints" in resource}
        endpoint_types = list(endpoints.keys())
    
        # Load required data
        self.resources = await self.init(models, endpoints)

        await self.ipfs_datasets.load_clusters(dataset, split, dst_path)
        
        # Load dataset
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        
        columns = self.dataset.column_names
        columns.append("cid")
        await self.ipfs_datasets.load_checkpoints(dataset, split, dst_path, models)
        queue_size = await self.queue_size(models[0])
        self.cid_chunk_queue = asyncio.Queue(queue_size)
        chunk_dir_path = os.path.join(dst_path, "checkpoints", "sparse_chunks")
        chunk_files = os.listdir(chunk_dir_path)
        chunk_files = [x for x in chunk_files if ".parquet" in x]
        saved_chunk_cids = [x.split(".")[0] for x in chunk_files]
        self.cid_chunk_list = saved_chunk_cids
        self.cid_chunk_set = set(saved_chunk_cids)
        # Setup parallel processing
        num_workers = min(multiprocessing.cpu_count(), 1)  # Use up to 1 CPU cores
        # num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 CPU cores
        # num_workers = multiprocessing.cpu_count()
        consumer_tasks = []
        producer_tasks = []
        all_tasks = []
        
        # Create producer and consumer tasks
        for endpoint in list(self.ipfs_kit.resources["endpoint_handler"][models[0]].keys()):
            if self.ipfs_kit.resources["hwtest"]["cuda"] == True and "openvino:" in endpoint:
                continue
            this_batch_size = self.ipfs_kit.resources["batch_sizes"][models[0]][endpoint]
            this_endpoint_handler = self.ipfs_kit.resources["endpoint_handler"][models[0]][endpoint]
            chunk_consumer = asyncio.create_task(self.chunk_consumer(this_batch_size, models[0], endpoint, this_endpoint_handler))
            consumer_tasks.append(chunk_consumer)
            all_tasks.append(chunk_consumer)
            endpoint_consumer = asyncio.create_task(self.endpoint_consumer(this_batch_size, models[0], endpoint, this_endpoint_handler))
            consumer_tasks.append(endpoint_consumer)
            all_tasks.append(endpoint_consumer)
        
        save_task = asyncio.create_task(self.save_chunks_to_disk(dataset, dst_path, models))
        all_tasks.append(save_task)
        for _ in range(num_workers):
            producer_task = asyncio.create_task(self.chunk_producer(self.dataset, column, None, None, None, None, None, models[0], dst_path))
            producer_tasks.append(producer_task)
            all_tasks.append(producer_task)
                
        # Wait for all tasks to complete
        
        # await asyncio.gather(*consumer_tasks, *producer_tasks, save_task)
        # await asyncio.gather(*all_tasks, save_task)
        await asyncio.gather(*all_tasks)
        return None

    async def chunk_consumer(self, batch_size, model_name, endpoint, endpoint_handler):
        try:
            batch_size = await self.max_batch_size(model_name, endpoint, endpoint_handler)
            queue_size = await self.queue_size(model_name)
            self.ipfs_kit.resources["queues"][model_name][endpoint] = asyncio.Queue(batch_size)
            self.ipfs_kit.resources["queue"][model_name] = asyncio.Queue(queue_size) 
            self.cid_chunk_queue = asyncio.Queue(queue_size) 
        except Exception as e:
            batch_size = 0
            print(e)
            # return None
        while True:
            while batch_size == 0:
                await asyncio.sleep(0.1)
                try:
                    batch_size = await self.max_batch_size(model_name, endpoint, endpoint_handler)
                    queue_size = await self.queue_size(model_name)
                    self.cid_chunk_queue = asyncio.Queue(queue_size)
                    self.ipfs_kit.resources["queue"][model_name] = asyncio.Queue(queue_size)
                    self.ipfs_kit.resources["queues"][model_name][endpoint] = asyncio.Queue(batch_size)
                except Exception as e:
                    batch_size = 0
            
            cid_chunk_queue = True if "cid_chunk_queue" in dir(self) else False
            empty = True if "cid_chunk_queue" in dir(self) and "empty" in dir(self.cid_chunk_queue) else False
            queue_not_empty = not self.cid_chunk_queue.empty()

            test_ready = all([
                cid_chunk_queue,
                empty,
                queue_not_empty
            ])
            while not test_ready:
                await asyncio.sleep(0.1)
                cid_chunk_queue = True if "cid_chunk_queue" in dir(self) else False
                empty = True if "cid_chunk_queue" in dir(self) and "empty" in dir(self.cid_chunk_queue) else False
                queue_empty = self.cid_chunk_queue.empty()
                queue_not_empty = not self.cid_chunk_queue.empty()
                test_ready = all([
                    cid_chunk_queue,
                    empty,
                    queue_not_empty
                ])
                pass     
            chunked_item = await self.cid_chunk_queue.get()
            batch_results = []
            batch = []
            chunk_data = []
            if chunked_item["parent_cid"] not in list(self.chunk_cache.keys()):
               self.chunk_cache[chunked_item["parent_cid"]] = {}
            self.chunk_cache[chunked_item["parent_cid"]]["children"] = chunked_item["children"]
            self.chunk_cache[chunked_item["parent_cid"]]["parent_cid"] = chunked_item["parent_cid"]
            self.chunk_cache[chunked_item["parent_cid"]]["items"] = []
            if chunked_item is not None:
                for i in range(len(chunked_item["items"])):
                    item = chunked_item["items"][i]
                    while self.ipfs_kit.resources["queues"][model_name][endpoint].full():
                        await asyncio.sleep(0.01)
                    self.ipfs_kit.resources["queues"][model_name][endpoint].put_nowait(item)
                await asyncio.sleep(0.01)
            else:
                pass
            self.cid_chunk_queue.task_done()
            await asyncio.sleep(0.01)
        return None
    
    async def endpoint_consumer(self, batch_size, model_name, endpoint, endpoint_handler):                
        endpoint_queue = True if endpoint in list(self.ipfs_kit.resources["queues"][model_name].keys()) else False
        empty = True if endpoint in list(self.ipfs_kit.resources["queues"][model_name].keys()) and "empty" in dir(self.ipfs_kit.resources["queues"][model_name][endpoint]) else False
        queue_not_empty = not self.ipfs_kit.resources["queues"][model_name][endpoint].empty()
        batch_size = self.ipfs_kit.resources["batch_sizes"][model_name][endpoint]
        test_ready = all([
            endpoint_queue,
            empty,
            queue_not_empty
        ])
        while True:
            batch_results = []
            batch = []
            chunk_data = []
            while not test_ready or batch_size == 0:
                await asyncio.sleep(0.1)
                batch_size = self.ipfs_kit.resources["batch_sizes"][model_name][endpoint]
                endpoint_queue = True if endpoint in list(self.ipfs_kit.resources["queues"][model_name].keys()) else False
                empty = True if endpoint in list(self.ipfs_kit.resources["queues"][model_name].keys()) and "empty" in dir(self.ipfs_kit.resources["queues"][model_name][endpoint]) else False
                queue_not_empty = not self.ipfs_kit.resources["queues"][model_name][endpoint].empty()
                queue_not_full = not self.ipfs_kit.resources["queues"][model_name][endpoint].full()
                queue_full = self.ipfs_kit.resources["queues"][model_name][endpoint].full()
                test_ready = all([
                    endpoint_queue,
                    empty,
                    queue_full,
                ])
                pass
            batch_results = []
            batch = []
            chunk_data = []
            if endpoint in list(self.chunker.chunkers[model_name].keys()):
                del self.chunker.chunkers[model_name][endpoint]
            if "cuda" in endpoint:
                with torch.no_grad():
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'ipc_collect'):
                        torch.cuda.ipc_collect()
                gc.collect()
            
            while self.ipfs_kit.resources["queues"][model_name][endpoint].empty():
                asyncio.sleep(0.1)
            
            while not self.ipfs_kit.resources["queues"][model_name][endpoint].empty():
                item = await self.ipfs_kit.resources["queues"][model_name][endpoint].get()
                batch.append(item["content"])
                chunk_data.append(item)
                self.ipfs_kit.resources["queues"][model_name][endpoint].task_done()

            if len(batch) >= batch_size:
                results = endpoint_handler(batch)
                if "hidden_states" in results.keys():
                    hidden_states = results["hidden_states"]
                    embeddings = []
                    if isinstance(hidden_states, torch.Tensor):
                        embeddings = torch.mean(hidden_states, dim=1).detach().cpu().numpy()
                    else:
                        for state in hidden_states:
                            if isinstance(state, torch.Tensor):
                                embedding = torch.mean(state, dim=0).detach().cpu().numpy()
                                embeddings.append(embedding)
                            else:
                                embedding = np.mean(state, axis=0)
                                embedding = embedding.tolist()
                                embeddings.append(embedding)
                else:
                    print("this might be the wrong endpoint handler")

                for i in range(len(embeddings)):
                    this_embeddings = embeddings[i]
                    this_cid = chunk_data[i]["cid"]
                    this_index = chunk_data[i]["index"]
                    this_content = chunk_data[i]["content"]
                    this_parent_cid = chunk_data[i]["parent_cid"]
                    batch_results.append({"cid": this_cid, "index": this_index, "content": this_content , "embedding": this_embeddings, "parent_cid": this_parent_cid})

            if len(batch_results) > 0:
                batch_parent_cids = list(set([x["parent_cid"] for x in batch_results]))
                for this_parent_cids in batch_parent_cids:
                    if this_parent_cids not in list(self.chunk_cache.keys()):
                        self.chunk_cache[this_parent_cids] = {}
                    self.chunk_cache[this_parent_cids]["items"] = []
                    self.chunk_cache[this_parent_cids]["children"] = []
                    self.chunk_cache[this_parent_cids]["parent_cid"] = this_parent_cids                    
                for result in batch_results:
                    this_parent_cid = result["parent_cid"]
                    this_cid = result["cid"]
                    self.chunk_cache[this_parent_cid]["items"].append(result)
                    if this_cid not in set(self.chunk_cache[this_parent_cid]["children"]):  
                        self.chunk_cache[this_parent_cid]["children"].append(this_cid)
                    self.cid_chunk_set.add(result["cid"])
                    self.cid_chunk_list.append(result["cid"])
                self.saved = False
                batch_results = []
                batch = []
                chunk_data = []
                if "cuda" in endpoint:
                    with torch.no_grad():
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        if hasattr(torch.cuda, 'ipc_collect'):
                            torch.cuda.ipc_collect()
                gc.collect()
                queue_not_empty = not self.ipfs_kit.resources["queues"][model_name][endpoint].empty()
                queue_not_full = not self.ipfs_kit.resources["queues"][model_name][endpoint].full()
                queue_full = self.ipfs_kit.resources["queues"][model_name][endpoint].full()
                test_ready = all([
                    endpoint_queue,
                    empty,
                    queue_full,
                ])
                await asyncio.sleep(0.01)                

    
    async def chunk_producer(self, dataset_stream, column, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None, dst_path=None):
        async for item in self.async_generator(dataset_stream):
            while self.cid_chunk_queue.full():
                await asyncio.sleep(1)
            if column is not None:
                cid = self.multiformats.get_cid(item[column])
            else:
                json_item = json.dumps(item)
                cid = self.multiformats.get_cid(json_item)
            if cid not in list(item.keys()):
                item["cid"] = cid
            if cid not in self.cid_chunk_set:
                chunked_item = await self.chunk_item(item, column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model)
            else:
                pass
            await asyncio.sleep(0.01)
           
        return None

    async def index_dataset(self, dataset, split, column, dst_path, models = None):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.queues = {}
        self.cid_set = set()
        self.all_cid_list = {}
        consumer_tasks = {}
        batch_sizes = {}
        if models is None:
            models = list(self.tei_endpoints.keys())
        for model in models:
            if model not in self.queues:
                self.queues[model] = {}
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        await self.load_checkpoints( dataset, split, dst_path, models)
        consumer_tasks = {}
        try:
            gpus = torch.cuda.device_count()            
        except:
            gpus = 0
        try:
            cpus = torch.get_num_threads()
        except:
            cpus = 0
        for model in models:
            endpoints = self.get_endpoints(model)
            local = self.get_endpoints(model, "local")
            openvino = self.get_endpoints(model, "openvino")
            libp2p = self.get_endpoints(model, "libp2p")
            tei = self.get_endpoints(model, "tei")
            cuda = self.get_endpoints(model, "cuda")
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if model not in self.tokenizer.keys():
                self.tokenizer[model] = {}                    
            if len(cuda) > 0 and len(gpus) > 0:
                self.local_endpoints[model] = {"cuda:" + str(gpu) : None for gpu in range(gpus) } if gpus > 0 else {"cpu": None}
                for gpu in range(gpus):
                    self.tokenizer[model]["cuda:" + str(gpu)] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                    self.local_endpoints[model]["cuda:" + str(gpu)] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                    torch.cuda.empty_cache()  # Free up unused memory
                    self.queues[model]["cuda:" + str(gpu)] = asyncio.Queue(4)
                    batch_size = await self.max_batch_size(model, "cuda:" + str(gpu))
                    self.batch_sizes[model]["cuda:" + str(gpu)] = batch_size
                    consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.consumer(self.queues[model]["cuda:" + str(gpu)], column, batch_size, model, "cuda:" + str(gpu)))
            elif len(local) > 0 and len(cpus) > 0:
                #detect openvino locally
                openvino_test = None
                llama_cpp_test = None
                ipex_test = None
                try:
                    openvino_test = self.test_local_openvino()
                except Exception as e:
                    print(e)
                    pass
                try:
                    llama_cpp_test = self.test_llama_cpp()
                except Exception as e:
                    print(e)
                    pass
                try:
                    ipex_test = self.test_ipex()
                except Exception as e:
                    print(e)
                    pass
                
                print("local_endpoint_test")
                results = {
                    "openvino": openvino_test,
                    "llama_cpp": llama_cpp_test,
                    "ipex": ipex_test
                }
                print(results)
                if not openvino_test and not llama_cpp_test and not ipex_test:                
                    self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                    self.queues[model]["cpu"] = asyncio.Queue()
                    consumer_tasks[(model, "cpu")] = asyncio.create_task(self.consumer(self.queues[model]["cpu"], column, 1, model, "cpu"))
                elif openvino_test:
                    ov_count = 0
                    for endpoint in local:
                        if "openvino" in endpoint:
                            endpoint_name = "openvino:"+str(ov_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint_name] = batch_size
                            if self.batch_sizes[model][endpoint_name] > 0:
                                self.queues[model][endpoint_name] = asyncio.Queue()
                                consumer_tasks[(model, endpoint_name )] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            openvino_count = openvino_count + 1
                elif llama_cpp_test:
                    llama_count = 0
                    for endpoint in local:
                        if "llama_cpp" in endpoint:
                            endpoint_name = "llama:"+str(ov_count)
                            batch_size = 0                            
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue()
                                consumer_tasks[(model, endpoint)] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            llama_count = llama_count + 1
                elif ipex_test:
                    ipex_count = 0
                    for endpoint in local:
                        if "ipex" in endpoint:
                            endpoint_name = "ipex:"+str(ov_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue()
                                consumer_tasks[(model, endpoint)] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            ipex_count = ipex_count + 1
            if len(openvino) > 0:
                for endpoint in openvino:
                    batch_size = 0
                    if model not in self.batch_sizes:
                        self.batch_sizes[model] = {}
                    if model not in self.queues:
                        self.queues[model] = {}
                    if endpoint not in list(self.batch_sizes[model].keys()):
                        batch_size = await self.max_batch_size(model, endpoint)
                        self.batch_sizes[model][endpoint] = batch_size
                    if self.batch_sizes[model][endpoint] > 0:
                        self.queues[model][endpoint] = asyncio.Queue()  # Unbounded queue
                        consumer_tasks[(model, endpoint)] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
            if not endpoints:
                raise ValueError("No endpoints available for model " + model)
                  
            if len(tei) > 0:
                for endpoint in tei:
                    batch_size = 0
                    if model not in self.batch_sizes:
                        self.batch_sizes[model] = {}
                    if model not in self.queues:
                        self.queues[model] = {}
                    if endpoint not in list(self.batch_sizes[model].keys()):
                        batch_size = await self.max_batch_size(model, endpoint)
                        self.batch_sizes[model][endpoint] = batch_size
                    if self.batch_sizes[model][endpoint] > 0:
                        self.queues[model][endpoint] = asyncio.Queue()  # Unbounded queue
                        consumer_tasks[(model, endpoint)] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
            if not endpoints:
                raise ValueError("No endpoints available for model " + model)
        
        # Compute commonn
        self.cid_set = set.intersection(*self.all_cid_set.values())
        producer_task = asyncio.create_task(self.producer(self.dataset, column, self.queues))        
        save_task = asyncio.create_task(self.save_checkpoints_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, *consumer_tasks.values(), save_task)
        self.save_checkpoints_to_disk(dataset, dst_path, models)
        return None 
    

    async def producer(self, dataset_stream, column, queues):
        tasks = []
        self.producer_task_done = False
        async for item in self.async_generator(dataset_stream):
            task = self.process_item(item, column, queues)
            tasks.append(task)
            if len(tasks) >= 1:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
        self.producer_task_done = True
        return None
    
    
    async def sparse_producer(self, dataset_stream, column, queues):
        tasks = []
        self.producer_task_done = False
        async for item in self.async_generator(dataset_stream):
            task = self.process_item(item, column, queues)
            tasks.append(task)
            if len(tasks) >= 1:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
        self.producer_task_done = True
        return None
        
    async def process_new_dataset_shard(self, dataset, split=None):
        results = await self.ipfs_datasets.process_new_dataset_shard(dataset, split)
        return results
    
    async def init_endpoints(self, models, endpoint_list=None):
        resources = await self.ipfs_kit.init_endpoints(models, endpoint_list)
        resources_keys = list(resources.keys())
        for resource in resources_keys:
            this_resource = resources[resource]
            if type(this_resource) is dict:
                if resource not in list(self.resources.keys()):
                    self.resources[resource] = {} 
                for key in list(this_resource.keys()):
                    self.resources[resource][key] = this_resource[key]
            elif type(this_resource) is object:
                self.resources[resource] = this_resource
        resources_list = ["queues", "batch_sizes", "endpoints", "models", "worker"]
        new_resources = {}
        for resource in resources_list:
            if resource in list(self.ipfs_kit.resources.keys()):
                new_resources[resource] = self.ipfs_kit.resources[resource]
        return new_resources

    def load_index(self, index):
        self.index = index
        return None 
    
    async def load_dataset(self, dataset, split=None):
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        return None

    def index_cid(self, samples):
        results = []
        if samples is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
        if isinstance(samples, list):
            for this_sample in samples:
                this_sample_cid = self.multiformats.get_cid(this_sample)
                self.cid_index[this_sample_cid] = this_sample
                results.append(this_sample_cid)
        else:
            raise ValueError("samples must be a list or string")
        return results
            
        
    async def load_dataset(self, dataset, split=None):
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        return None


    async def kmeans_cluster_split(self, dataset, split, columns, dst_path, models, max_splits=None):
        # await self.load_clusters(dataset, split, dst_path)
        # await self.load_checkpoints(dataset, split, dst_path, models)
        await self.ipfs_datasets.load_clusters(dataset, split, dst_path)
        await self.ipfs_datasets.load_clusters(dataset, split, dst_path)
        await self.load_dataset(dataset, split)

        centroids = []
        embeddings_np = []                    
        ipfs_cids = []
        kmeans = None
        if os.path.exists(os.path.join(dst_path, dataset.replace("/", "___") + "_centroids.parquet")):
            centroids_dataset = load_dataset('parquet', data_files=os.path.join(dst_path, dataset.replace("/", "___") + "_centroids.parquet"))["train"]
            centroids = centroids_dataset["centroids"]
            centroids = np.array(centroids)
            max_splits = len(centroids)
        else:
            new_dataset_download_size = self.new_dataset.dataset_size            
            embeddings_size = {}
            for model in self.metadata["models"]:
                embeddings_size[model] = self.index[model].dataset_size
            largest_embeddings_dataset = max(embeddings_size, key=embeddings_size.get)
            largest_embeddings_size = embeddings_size[max(embeddings_size, key=embeddings_size.get)]
            embeddings_size["new_dataset"] = new_dataset_download_size
            largest_embedding_dataset_rows = len(self.index[largest_embeddings_dataset])                
            largest_dataset_size = embeddings_size[max(embeddings_size, key=embeddings_size.get)]
            max_size = 50 * 1024 * 1024 # 50 MB
            max_rows_in_powers_of_64 = math.ceil(math.log(largest_embedding_dataset_rows, 64))                        
            max_splits_size = round(largest_dataset_size / max_size)
            max_splits_rows = 64 ** (max_rows_in_powers_of_64 - 2)
            if max_splits == None:                
                if max_splits_rows > max_splits_size:
                    max_splits = max_splits_rows
                else:
                    max_splits = max_splits_size
            num_items = len(self.index[largest_embeddings_dataset])
            embedding_dim = len(self.index[largest_embeddings_dataset][0]["items"]["embedding"])
            embeddings_np = np.zeros((num_items, embedding_dim))
            
            for i, item in enumerate(self.index[largest_embeddings_dataset]):
                    embeddings_np[i] = item["items"]["embedding"]
                    ipfs_cids.append(item["items"]["cid"])

            # Perform KMeans clustering using faiss
            kmeans = faiss.Kmeans(d=embeddings_np.shape[1], k=max_splits, niter=100, verbose=True)
            kmeans.train(embeddings_np)               
            # Get centroids
            centroids = kmeans.centroids
            # Save centroids to disk
            centroids_dataset = datasets.Dataset.from_dict({"centroids": centroids.tolist()})
            centroids_dataset.to_parquet(os.path.join(dst_path, dataset.replace("/", "___") + "_centroids.parquet"))

        if os.path.exists(os.path.join(dst_path, dataset.replace("/", "___") + "_cluster_cids.parquet")):
            cluster_cids_dataset = load_dataset('parquet', data_files=os.path.join(dst_path, dataset.replace("/", "___") + "_cluster_cids.parquet"))["train"]
            ipfs_cid_clusters_list = cluster_cids_dataset["cluster_cids"]
            ipfs_cid_clusters_set = [set(x) for x in ipfs_cid_clusters_list]
            ipfs_cid_set = set([cid for sublist in ipfs_cid_clusters_list for cid in sublist])
        else:
            if kmeans is None:
                new_dataset_download_size = self.new_dataset.dataset_size            
                embeddings_size = {}
                for model in self.metadata["models"]:
                    embeddings_size[model] = self.index[model].dataset_size
                largest_embeddings_dataset = max(embeddings_size, key=embeddings_size.get)
                largest_embeddings_size = embeddings_size[max(embeddings_size, key=embeddings_size.get)]
                embeddings_size["new_dataset"] = new_dataset_download_size
                largest_embedding_dataset_rows = len(self.index[largest_embeddings_dataset])                
                largest_dataset_size = embeddings_size[max(embeddings_size, key=embeddings_size.get)]
                max_size = 50 * 1024 * 1024 # 50 MB
                max_rows_in_powers_of_64 = math.ceil(math.log(largest_embedding_dataset_rows, 64))                        
                max_splits_size = round(largest_dataset_size / max_size)
                max_splits_rows = 64 ** (max_rows_in_powers_of_64 - 2)
                if max_splits_rows > max_splits_size:
                    max_splits = max_splits_rows
                else:
                    max_splits = max_splits_size
                num_items = len(self.index[largest_embeddings_dataset])
                embedding_dim = len(self.index[largest_embeddings_dataset][0]["items"]["embedding"])
                embeddings_np = np.zeros((num_items, embedding_dim))
                for i, item in enumerate(self.index[largest_embeddings_dataset]):
                    embeddings_np[i] = item["items"]["embedding"]
                    ipfs_cids.append(item["items"]["cid"])
                kmeans = faiss.Kmeans(d=embeddings_np.shape[1], k=max_splits, niter=100, verbose=True)
                kmeans.centroids = centroids
                pass
            
            if len(ipfs_cids) == 0:
                new_dataset_download_size = self.new_dataset.dataset_size            
                embeddings_size = {}
                for model in self.metadata["models"]:
                    embeddings_size[model] = self.index[model].dataset_size
                largest_embeddings_dataset = max(embeddings_size, key=embeddings_size.get)
                num_items = len(self.index[largest_embeddings_dataset])
                embedding_dim = len(self.index[largest_embeddings_dataset][0]["items"]["embedding"])
                embeddings_np = np.zeros((num_items, embedding_dim))
                for i, item in enumerate(self.index[largest_embeddings_dataset]):
                    embeddings_np[i] = item["items"]["embedding"]
                    ipfs_cids.append(item["items"]["cid"])
        
            max_splits = len(centroids)
            index = faiss.IndexFlatL2(centroids.shape[1])
            index.add(centroids)
            _, cluster_assignments = index.search(embeddings_np, 1)
            cluster_assignments = cluster_assignments.flatten()  # Flatten the cluster_assignments array
            ipfs_cid_clusters_list = [[] for _ in range(max_splits)]
            ipfs_cid_clusters_set = [set() for _ in range(max_splits)]
            for cid, cluster_id in zip(ipfs_cids, cluster_assignments):
                ipfs_cid_clusters_list[cluster_id].append(cid)
                ipfs_cid_clusters_set[cluster_id].add(cid) 
            ipfs_cid_set = set([cid for sublist in ipfs_cid_clusters_list for cid in sublist])
            cluster_cids_dataset = datasets.Dataset.from_dict({"cluster_cids": ipfs_cid_clusters_list})
            cluster_cids_dataset.to_parquet(os.path.join(dst_path, dataset.replace("/", "___") + "_cluster_cids.parquet"))

        max_splits = len(centroids)
        for model in list(self.index.keys()):
            kmeans_embeddings_splits = {}
            if not os.path.exists(os.path.join(dst_path, dataset.replace("/", "___") + model.replace("/", "___") + "_clusters")):
                os.makedirs(os.path.join(dst_path, dataset.replace("/", "___") + model.replace("/", "___") + "_clusters"))
            model_splits = os.listdir(os.path.join(dst_path, dataset.replace("/", "___") + model.replace("/", "___") + "_clusters"))
            if len(model_splits) == max_splits:
                pass 
            else:
                kmeans_embeddings_splits_set = set()
                cluster_id_list = []
                cluster_id_set = set()
                for cluster_id in range(max_splits):
                    if cluster_id not in kmeans_embeddings_splits_set:
                        kmeans_embeddings_splits[cluster_id] = {}
                first_item = self.index[model][0]
                embedding_dim = len(first_item["items"]["embedding"])
                kmeans_embeddings_splits = [
                    {
                        key: (np.zeros((len(ipfs_cid_clusters_list[cluster_id]), embedding_dim)) if key == "embedding" else ["" for _ in range(len(ipfs_cid_clusters_list[cluster_id]))])
                        for key in first_item["items"].keys()
                    }
                    for cluster_id in range(max_splits)
                ]

                def process_item(item):
                    for cluster_id in range(max_splits):
                        if item["items"]["cid"] in ipfs_cid_clusters_set[cluster_id]:
                            for key in item["items"].keys():
                                kmeans_embeddings_splits[cluster_id][key][
                                    ipfs_cid_clusters_list[cluster_id].index(item["items"]["cid"])
                                ] = np.array(item["items"][key]) if key == "embedding" else item["items"][key]
                            break

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(process_item, [item for item in self.index[model] if item["items"]["cid"] in ipfs_cid_set])
                
                for cluster_id in range(max_splits):
                    if cluster_id not in list(kmeans_embeddings_splits.keys()):
                        continue
                    cluster_dataset = datasets.Dataset.from_dict(kmeans_embeddings_splits[cluster_id])
                    cluster_dataset.to_parquet(os.path.join(dst_path, dataset.replace("/", "___") + model.replace("/", "___") + "_clusters", f"cluster_{cluster_id}.parquet"))
        
        kmeans_embeddings_splits = {}
        cluster_folder = os.path.join(dst_path, dataset.replace("/", "___") + "_clusters")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        model_splits = os.listdir(cluster_folder)
        if len(model_splits) == max_splits:
            pass 
        else:
            cluster_id_list = []
            for cluster_id in range(max_splits):
                if cluster_id not in cluster_id_list:
                    cluster_id_list.append(cluster_id)
                    kmeans_embeddings_splits[cluster_id] = {}
            first_item = self.new_dataset[0]
            if "items" in list(first_item.keys()):
                keys_list = list(first_item["items"].keys())
            else:
                keys_list = list(first_item.keys())
            keys_set = set(keys_list)          
            cluster_id_set = set(cluster_id_list)
            kmeans_embeddings_splits_list = []
            kmeans_embeddings_splits_set = set()       
            kmeans_embeddings_splits_list = [
                cluster_id
                for cluster_id in range(max_splits)
                if cluster_id not in kmeans_embeddings_splits_list
                for key in keys_list
                if not kmeans_embeddings_splits[cluster_id].__setitem__(key, [""] * len(ipfs_cid_clusters_list[cluster_id]))
            ]
            kmeans_embeddings_splits_set = set(kmeans_embeddings_splits_list)
            [
                kmeans_embeddings_splits[cluster_id][key].__setitem__(
                    ipfs_cid_clusters_list[cluster_id].index(this_cid),
                    item["items"][key] if "items" in list(item.keys()) else item[key]
                )
                for item in self.new_dataset
                for this_cid in [item["items"]["cid"] if "items" in list(item.keys()) else item["cid"]]
                if this_cid in ipfs_cid_set
                for cluster_id in range(max_splits)
                if this_cid in ipfs_cid_clusters_set[cluster_id]
                for key in keys_list
            ]
            for cluster_id in range(max_splits):
                cluster_filename = os.path.join(cluster_folder, dataset.replace("/", "___") + "_cluster_" + str(cluster_id) + ".parquet")
                if cluster_id not in list(kmeans_embeddings_splits.keys()):
                    continue
                cluster_dataset = datasets.Dataset.from_dict(kmeans_embeddings_splits[cluster_id])
                cluster_dataset.to_parquet(cluster_filename)
        return True
    
    
if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "namespace": "TeraflopAI/Caselaw_Access_Project",
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
            ["thenlper/gte-small", "openvino:0", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "openvino:0", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "openvino:0", 32768],
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
    create_embeddings_batch = ipfs_embeddings_py(resources, metadata)
    # asyncio.run(create_embeddings_batch.index_dataset(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))    
    # asyncio.run(create_embeddings_batch.combine_checkpoints(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))
    # asyncio.run(create_embeddings_batch.kmeans_cluster_split(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"], 10))
    asyncio.run(create_embeddings_batch.index_sparse_chunks(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))
