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

from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process
import concurrent.futures
import concurrent
import json
from ipfs_datasets import ipfs_datasets_py
from ipfs_accelerate_py import ipfs_accelerate_py
import multiformats
from queue import Queue

manager = Manager()
caches = manager.dict()
chunk_cache = manager.dict()
cid_cache = manager.dict()
cid_chunk_queue = manager.Queue()
cid_queue = manager.Queue()
cid_set = manager.list()
cid_chunk_set = manager.list()
all_cid_set = manager.dict()
batch_sizes = manager.dict()
metadata = manager.dict()
caches["hashed_dataset"] = manager.dict()

ipfs_multiformats = ipfs_multiformats_py()
this_chunker = chunker()
def index_cid(samples):
    results = []
    if samples is None:
        raise ValueError("samples must be a list")
    if isinstance(samples, str):
        samples = [samples]
    if isinstance(samples, list):
        for this_sample in samples:
            this_sample_cid = ipfs_multiformats.get_cid(this_sample)
            results.append(this_sample_cid)
    else:
        raise ValueError("samples must be a list or string")
    return results

def tokenize_batch(batch, tokenizer, column):
    if type(batch) is Dataset:
        len_columns = len(batch.column_names)
        batch = batch.to_dict()
    else:
        len_columns = len(list(batch.keys()))
    if len_columns > 1 and column is None:
        batch = [json.dumps(item) for item in batch]
        pass
    elif len_columns > 1 and column not in list(batch.keys()):
        batch = [json.dumps(item) for item in batch]
    elif len_columns > 1 and column in list(batch.keys()):
        batch = batch[column]

    new_token_data = {}
    try:
        results = tokenizer(batch, padding=True, return_tensors="pt")
        new_token_data["tokens_list"] = [ results["input_ids"][i].tolist() for i in range(len(results["input_ids"])) ]
        new_token_data["tokens_text"] = [ tokenizer.decode(tokens) for tokens in new_token_data["tokens_list"]]
        new_token_data["text_list"] = [ tokenizer.decode(tokens) for tokens in new_token_data["tokens_list"]]
        new_token_data["token_lengths"] = [new_token_data["tokens_list"][i].count(0) for i in range(len(new_token_data["tokens_list"])) ]
        return new_token_data
    except Exception as e:
        print("Error tokenizing batch: ", e)
        return e

def process_dataset(dataset_stream, column=None, caches=None, this_cid_list=None):
    if "hashed_dataset" not in list(all_cid_set.keys()):
        all_cid_set["hashed_dataset"] = []
    if this_cid_list is not None:
        for cid in this_cid_list:
            if cid not in cid_set:
                cid_set.append(cid)
                all_cid_set["hashed_dataset"].append(cid)
    dataset = {}
    for item in generator(dataset_stream):
        column_names = list(item.keys())
        if column is None:
            payload = str(json.dumps(item))
        elif column not in column_names:
            payload = str(json.dumps(item))
        else:
            payload = str(item[column])
        try:
            ## convert payload to a bytes like object 
            payload = payload.encode()
            payload = payload.decode('utf-8')
            this_cid = ipfs_multiformats.get_cid(payload)
        except Exception as e:
            # print("Error getting CID: ", e)
            try:
                this_cid = ipfs_multiformats.get_cid(payload)
            except Exception as e:
                print("Error getting CID: ", e)
                try:
                    this_cid = ipfs_multiformats.get_cid(payload)
                except Exception as e:
                    print("Error getting CID: ", e)
                    print(payload)
                    this_cid = ipfs_multiformats.get_cid(json.dumps(item))            
            
        if "cid" not in column_names:
            item["cid"] = this_cid
        elif item["cid"] is None:
            item["cid"] = this_cid
        # Check if cid is in index
        if this_cid in cid_set:
            print(f"CID {this_cid} already in index, skipping item.")
        else:
            while cid_queue.full():
                 time.sleep(0.1)
            if this_cid in this_cid_list["hashed_dataset"] or this_cid in cid_set:
                pass    
            # if this_cid not in this_cid_list["hashed_dataset"] and this_cid not in cid_set:
            else:
                cid_set.append(this_cid)
                all_cid_set["hashed_dataset"].append(this_cid)
                # caches["hashed_dataset"][this_cid] = item
                dataset[item["cid"]] = item
                cid_queue.put_nowait
                print("Added item to queue for CID " + str(this_cid))
    return dataset              

def chunk_producer(dataset_stream, column, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None, dst_path=None, chunk_item=None, process_item=None, cid_queue=None, cid_chunk_set=None, chunker=None, metadata=None, caches=None, all_cid_set=None):
    batch_size = 2048  # Adjust batch size based on your needs
    current_batch = []
    current_processed_items = []
    current_items = []
    current_results = []
    num_shards = 32
    shard_id = 0
    dataset = {}
    shards = []
    if "hashed_dataset" in list(dataset_stream.keys()) and dataset_stream["hashed_dataset"] is not None:
        for i in range(num_shards): 
            shards.append(dataset_stream["hashed_dataset"].shard(num_shards=num_shards, index=i))
    
    with Pool(processes=32) as pool:
        len_hashed_dataset_cids = len(all_cid_set["hashed_dataset"])
        len_hashed_dataset = dataset_stream["hashed_dataset"].num_rows
        len_dataset_stream = dataset_stream["dataset"].num_rows
        len_model_dataset_cids = len(all_cid_set[embed_model])
        if column == None and len_hashed_dataset_cids <= len_dataset_stream or len_model_dataset_cids <= len_hashed_dataset:        
            args = [(shards[i], column, caches, all_cid_set) for i in range(len(shards))]
            processed_dataset = pool.starmap(process_dataset, args)
            for shard in processed_dataset:
                dataset.update(shard)
            if len(list(dataset.keys())) > 0:
                hashed_dataset = datasets.Dataset.from_dict(dataset)
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                this_model_shards = [os.path.join(dst_path, "checkpoints", x)  for x in ls_checkpoints if embed_model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                next_model_shard = os.path.join(dst_path, "checkpoints", embed_model.replace("/", "___") + "_shard_" + str(len(this_model_shards)))
                hashed_dataset.to_parquet(next_model_shard)
                this_model_shards.append(next_model_shard)
                dataset_stream["hashed_dataset"] = load_dataset(shards=this_model_shards)
                hashed_dataset = dataset_stream["hashed_dataset"]
                shards = []
                for i in range(num_shards) and "hashed_dataset" and hashed_dataset is not None:
                    shards.append(hashed_dataset.shard(num_shards=num_shards, index=i))
        elif column != None and len_hashed_dataset_cids <= len_dataset_stream or len_model_dataset_cids <= len_hashed_dataset:
            dataset_column_row = dataset_stream["dataset"][column]
            unique_dataset_column_row = set(dataset_column_row)
            len_unique_dataset_columns_rows = len(unique_dataset_column_row)
            if len_unique_dataset_columns_rows >  len_hashed_dataset_cids or len_unique_dataset_columns_rows > len_model_dataset_cids:
                args = [(shards[i], column, caches, all_cid_set) for i in range(len(shards))]
                processed_dataset = pool.starmap(process_dataset, args)
                for shard in processed_dataset:
                    dataset.update(shard)
                if len(list(dataset.keys())) > 0:
                    hashed_dataset = datasets.Dataset.from_dict(dataset)
                    ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                    this_model_shards = [os.path.join(dst_path, "checkpoints", x)  for x in ls_checkpoints if embed_model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                    next_model_shard = os.path.join(dst_path, "checkpoints", embed_model.replace("/", "___") + "_shard_" + str(len(this_model_shards)))
                    hashed_dataset.to_parquet(next_model_shard)
                    dataset_stream["hashed_dataset"] = load_dataset(shards=this_model_shards)
                    hashed_dataset = dataset_stream["hashed_dataset"]
                    shards = []
                    for i in range(num_shards) and "hashed_dataset" and hashed_dataset is not None:
                        shards.append(hashed_dataset.shard(num_shards=num_shards, index=i))
            else:
                hashed_dataset = dataset_stream["hashed_dataset"]
        else:
            hashed_dataset = dataset_stream["hashed_dataset"]
            pass
        
        args = [(shards[i], column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model, chunker, metadata) for i in range(len(shards))]
        tokenized_texts = pool.starmap(tokenize_batch, [(shards[i], tokenizer, column) for i in range(len(shards))])    
        if "processed_items"  not in list(tokenized_texts[0].keys()):
            tokenized_texts[0]["processed_items"] = []
        for i in range(len(tokenized_texts[0]["text_list"])):
            if i >= len(tokenized_texts[0]["processed_items"]):
                tokenized_texts[0]["processed_items"].append(current_processed_items[i])
            
        tokenized_texts_datasets = datasets.Dataset.from_dict(tokenized_texts[0])
        tokenized_texts_shards = []
        for i in range(num_shards):
            tokenized_texts_shards.append(tokenized_texts_datasets.shard(num_shards=num_shards, index=i))
        args = [(tokenized_texts_shards[i], column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model, chunker, metadata) for i in range(len(tokenized_texts_shards))]    
        tokenized_chunks = pool.starmap(chunk_items, args)
        # if "parent_cid" not in list(chunk_cache.keys()):
        #     chunk_cache[cid] = manager.dict()
        # for chunk in tokenized_chunks:
        #     if chunk is not None:
        #         chunk_cache[chunk["parent_cid"]] = chunk
        #         current_results.append(chunk)
        #         cid_chunk_set.append(chunk["parent_cid"])
        # current_batch = []
        # current_processed_items = []
        return tokenized_chunks
        pass
             
        
        # Process remaining items
        if current_batch:
            tokenized_texts = pool.starmap(tokenize_batch, [(batch, tokenizer) for batch in chunks(current_batch, len(pool._pool))])
            if "processed_items"  not in list(tokenized_texts[0].keys()):
                tokenized_texts[0]["processed_items"] = []
            for i in range (len(tokenized_texts[0]["text_list"])):
                tokenized_texts[0]["processed_items"].append(current_processed_items[i])
                
            tokenized_texts_datasets = datasets.Dataset.from_dict(tokenized_texts[0])
            args = [(tokenized_texts_datasets[i], column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model, chunker, metadata) for i in range(len(tokenized_texts_datasets))]    
            tokenized_chunks = pool.starmap(chunk_items, args)
            if "parent_cid" not in list(chunk_cache.keys()):
                chunk_cache[cid] = manager.dict()
            for chunk in tokenized_chunks:
                chunk_cache[chunk["parent_cid"]] = chunk
                current_results.append(chunk)
                cid_chunk_set.append(chunk["parent_cid"])
                
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_items(item_data, column=None, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None, chunker=None, metadata=None):
    item = item_data["processed_items"]
    batch_size = 1
    device = 'cpu'
    cid_chunks = None
    if type(item) == dict and column not in list(item.keys()):
        content = json.dumps(item)
    if type(item) == dict and column in list(item.keys()):
        content = item[column]
    else:
        content = json.dumps(item)
    if embed_model is None:
        if len(metadata["models"]) == 0:
            embed_model = "thenlper/gte-small"
        else:
            embed_model = metadata["models"][0]
    if chunk_size is None:
        chunk_size = 512
    if n_sentences is None:
        n_sentences = 8
    if step_size is None:
        step_size = 256    
    if method is None:
        fixed_chunk_list = this_chunker.chunk(content, tokenizer, "fixed", 512, 8, 256, metadata["models"][0], device, batch_size)
        # semantic_chunk_list = self.chunker.chunk(content, tokenizer, "semantic", 512, 8, 256, self.metadata["models"][0], device, batch_size)
        sentences_chunk_list = this_chunker.chunk(content, tokenizer, "sentences", 512, 8, 256, metadata["models"][0], device, batch_size) 
        sliding_window_chunk_list = this_chunker.chunk(content, tokenizer, "sliding_window", 512, 8, 256, metadata["models"][0], device, batch_size)
        # content_chunks = fixed_chunk_list + semantic_chunk_list + sentences_chunk_list + sliding_window_chunk_list
        content_chunks = fixed_chunk_list + sentences_chunk_list + sliding_window_chunk_list
    else:
        content_chunks = this_chunker.chunk(content, tokenizer, method, chunk_size, n_sentences, step_size, embed_model)
        
    if item is not None and type(item) is dict and "cid" in list(item.keys()):
        parent_cid = item["cid"]
    else:
        if item is not None and "colmuns" in list(item.keys()):
            parent_cid = ipfs_multiformats.get_cid(item[column])
        elif item is not None and "columns" not in list(item.keys()):
            parent_cid = ipfs_multiformats.get_cid(json.dumps(item))
        else:
            parent_cid = ipfs_multiformats.get_cid(json.dumps(item))

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
    if parent_cid in cid_chunk_set:
        pass
    else:
        cid_chunks = {"parent_cid": parent_cid, "items" : [], "children": []}
        for chunk in content_chunks:
            chunk_index = chunk
            chunk_content = content_tokens[chunk[0]:chunk[1]]
            chunk_text = tokenizer.decode(chunk_content)
            child_cid = ipfs_multiformats.get_cid(chunk_text)
            child_content = {"cid": child_cid, "index": chunk_index, "content": chunk_text, "parent_cid": parent_cid}
            cid_chunks["children"].append(child_cid)
            cid_chunks["items"].append(child_content)

        if type(cid_chunks["parent_cid"]) is not str:
            print("Parent CID is not a string")
    
        if parent_cid not in cid_chunk_set and type(cid_chunks["parent_cid"]) is str:
            while cid_chunk_queue.full():
                time.sleep(0.1)
            if not cid_chunk_queue.full():
                cid_chunk_set.append(parent_cid)
                cid_chunk_queue.put_nowait(cid_chunks)
                print("Added chunked item to queue for CID " + cid_chunks["parent_cid"])
        else:
            print("CID " + cid_chunks["parent_cid"] + " already in chunk set")
    return cid_chunks

def chunk_item(item_data, column=None, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None, chunker=None, metadata=None):
    item = item_data[0]
    batch_size = 1
    device = 'cpu'
    if column is None:
        content = json.dumps(item)
    elif type(item) is dict and column not in list(item.keys()):
        content = json.dumps(item)
    elif type(item) is dict and column in list(item.keys()):
        content = item[column]
    else:
        content = json.dumps(item)    
    if embed_model is None:
        if len(metadata["models"]) == 0:
            embed_model = "thenlper/gte-small"
        else:
            embed_model = metadata["models"][0]
    if chunk_size is None:
        chunk_size = 512
    if n_sentences is None:
        n_sentences = 8
    if step_size is None:
        step_size = 256    
    if method is None:
        fixed_chunk_list = this_chunker.chunk(content, tokenizer, "fixed", 512, 8, 256, metadata["models"][0], device, batch_size)
        # semantic_chunk_list = self.chunker.chunk(content, tokenizer, "semantic", 512, 8, 256, self.metadata["models"][0], device, batch_size)
        sentences_chunk_list = this_chunker.chunk(content, tokenizer, "sentences", 512, 8, 256, metadata["models"][0], device, batch_size) 
        sliding_window_chunk_list = this_chunker.chunk(content, tokenizer, "sliding_window", 512, 8, 256, metadata["models"][0], device, batch_size)
        # content_chunks = fixed_chunk_list + semantic_chunk_list + sentences_chunk_list + sliding_window_chunk_list
        content_chunks = fixed_chunk_list + sentences_chunk_list + sliding_window_chunk_list
    else:
        content_chunks = this_chunker.chunk(content, tokenizer, method, chunk_size, n_sentences, step_size, embed_model)
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
    if parent_cid in cid_chunk_set:
        pass
    else:
        cid_chunks = {"parent_cid": parent_cid, "items" : [], "children": []}
        for chunk in content_chunks:
            chunk_index = chunk
            chunk_content = content_tokens[chunk[0]:chunk[1]]
            chunk_text = tokenizer.decode(chunk_content)
            child_cid = ipfs_multiformats.get_cid(chunk_text)
            child_content = {"cid": child_cid, "index": chunk_index, "content": chunk_text, "parent_cid": parent_cid}
            cid_chunks["children"].append(child_cid)
            cid_chunks["items"].append(child_content)

        if type(cid_chunks["parent_cid"]) is not str:
            print("Parent CID is not a string")
    
        if parent_cid not in cid_chunk_set and type(cid_chunks["parent_cid"]) is str:
            while cid_chunk_queue.full():
                time.sleep(0.1)
            if not cid_chunk_queue.full():
                cid_chunk_set.append(parent_cid)
                cid_chunk_queue.put_nowait(cid_chunks)
                print("Added chunked item to queue for CID " + cid_chunks["parent_cid"])
        else:
            print("CID " + cid_chunks["parent_cid"] + " already in chunk set")
    return cid_chunks

        
def process_item(item, column=None, queues=None, caches=None, all_cid_set=None):
    # print(f"Processing item with CID {index_cid(item[column])[0]}")
    # if queues is None:
    #     queues = ipfs_accelerate_py.resources["queues"]
    column_names = list(item.keys())
    if column is None:
        this_cid = index_cid(json.dumps(item))[0]
    elif column not in column_names:
        this_cid =  index_cid(json.dumps(item))[0]
    else:
        this_cid = ipfs_multiformats.get_cid(item[column])
        index_cid(item[column])[0]
    if "cid" not in column_names:
        item["cid"] = this_cid
    elif item["cid"] is None:
        item["cid"] = this_cid
    # Check if cid is in index
    if this_cid in cid_set:
        # print(f"CID {this_cid} already in index, skipping item.")
        return None
    else:
        while cid_queue.full():
            time.sleep(0.1)    
        if this_cid not in cid_set:
            cid_set.append(this_cid)
            caches["hashed_dataset"][this_cid] = item
            cid_queue.put_nowait
            print("Added item to queue for CID " + str(this_cid))
            # self.saved = False
    return item
                        
async def async_generator(iterable):
    for item in iterable:
        yield item

def generator(iterable):
    for item in iterable:
        yield item

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
        self.process_hashed_dataset_shard = self.ipfs_datasets.process_hashed_dataset_shard
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
        self.item_cache = {}
        self.chunk_embeddings = {}
        self.cid_chunk_list = []
        self.cid_chunk_set = set()
        self.batch_sizes = {}
        self.cid_list = []
        self.cid_set = set()
        self.hashed_dataset = None
        self.all_cid_list = {}
        self.all_cid_set = {}
        self.cid_chunk_queue = None
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.endpoint_handler = {}
        self.hashed_dataset = {}
        self.chunk_item = chunk_item
        self.process_item = process_item
        self.hashed_dataset_children = {}
        self.saved = False
        self.resources = resources
        self.metadata = metadata
        self.index_dataset = self.index_dataset
        self.max_batch_size = self.ipfs_accelerate_py.max_batch_size
        # self.item_producer = self.item_producer
        self.save_checkpoints_to_disk = self.save_checkpoints_to_disk
        # self.save_chunks_to_disk = self.save_chunks_to_disk
        self.index_cid = self.index_cid
        self.load_index = self.load_index
        self.async_generator = self.async_generator
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.ipfs_accelerate_py.add_endpoint
        self.rm_endpoint = self.ipfs_accelerate_py.rm_endpoint
        self.init_endpoints = self.init_endpoints       
        return None

    async def init_endpoints(self, models , endpoints ):
        resources = await self.init_endpoints(models, endpoints)
        resources_keys = ["queues", "batch_sizes", "endpoints", "models", "worker"]
        for resource in resources_keys:
            if resource in list(self.ipfs_accelerate_py.resources.keys()) and resources is not None:
                this_resource = resources[resource]
                if type(this_resource) is dict:
                    for key in list(this_resource.keys()):
                        self.resources[resource][key] = this_resource[key]
                elif type(this_resource) is object:
                    self.resources[resource] = this_resource
        test_endpoints = None
        test_endpoints = await self.ipfs_accelerate_py.test_endpoints(models)
        
        return test_endpoints
        
    async def async_generator(self, iterable):
        for item in iterable:
            yield item
            
    async def max_batch_size(self, model, endpoint, endpoit_handler):
        print("max batch size")
        return await self.ipfs_accelerate_py.max_batch_size(model, endpoint, endpoit_handler)
        
    # async def chunk_item(self, item, column=None, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None):
    #     # Assuming `item` is a dictionary with required data
    #     cuda_test = False
    #     openvino_test = False
    #     tokenizer_types = list(self.ipfs_accelerate_py.resources["tokenizer"][embed_model].keys())
    #     cuda_tokenizer_types = [x for x in tokenizer_types if "cuda" in x]
    #     openvino_tokenizer_types = [x for x in tokenizer_types if "openvino" in x]
    #     if self.ipfs_accelerate_py.resources["hwtest"]["cuda"] == True:
    #         cuda_test = True
    #     if self.ipfs_accelerate_py.resources["hwtest"]["openvino"] == True:
    #         openvino_test = True
    #     if column is None:
    #         content = json.dumps(item)
    #     elif column not in list(item.keys()):
    #         content = json.dumps(item)
    #     else:
    #         content = item[column]
    #     if embed_model is None:
    #         if len(self.metadata["models"]) == 0:
    #             embed_model = "thenlper/gte-small"
    #         else:
    #             embed_model = self.metadata["models"][0]
    #     if chunk_size is None:
    #         chunk_size = 512
    #     if n_sentences is None:
    #         n_sentences = 8
    #     if step_size is None:
    #         step_size = 256
    #     if tokenizer is None:
    #         if embed_model not in list(self.ipfs_accelerate_py.resources["tokenizer"].keys()):
    #             self.tokenizer[embed_model] = {}
    #         if cuda_test == True:
    #             tokenizer_types = list(self.ipfs_accelerate_py.resources["tokenizer"][embed_model].keys())
    #             cuda_tokenizer_types = [x for x in tokenizer_types if "cuda" in x]
    #             random_cuda_tokenizer = random.choice(cuda_tokenizer_types)
    #             device = random_cuda_tokenizer
    #             tokenizer = self.ipfs_accelerate_py.resources["tokenizer"][embed_model][random_cuda_tokenizer]
    #             batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][embed_model][random_cuda_tokenizer]
    #             if batch_size == 0 or batch_size is None:
    #                 batch_size = 32
    #         elif openvino_test == True:
    #             openvino_tokenizer_types = [x for x in tokenizer_types if "openvino" in x]
    #             random_openvino_tokenizer = random.choice(openvino_tokenizer_types)
    #             device = random_openvino_tokenizer
    #             tokenizer = self.ipfs_accelerate_py.resources["tokenizer"][embed_model][random_openvino_tokenizer]  
    #             batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][embed_model][random_openvino_tokenizer]
    #             if batch_size == 0 or batch_size is None:
    #                 batch_size = 1
    #         elif "cpu" not in tokenizer_types:
    #             tokenizer = self.ipfs_accelerate_py.resources["tokenizer"][embed_model]["cpu"]                
    #             batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][embed_model]["cpu"]
    #             device = "cpu"
    #             if batch_size == 0 or batch_size is None:
    #                 batch_size = 1
    #         else:
    #             device = "cpu"
    #             tokenizer =  AutoTokenizer.from_pretrained(embed_model, device='cpu', use_fast=True)
    #             batch_size = 1
    #     if method is None:
    #         fixed_chunk_list = self.chunker.chunk(content, tokenizer, "fixed", 512, 8, 256, self.metadata["models"][0], device, batch_size)
    #         # semantic_chunk_list = self.chunker.chunk(content, tokenizer, "semantic", 512, 8, 256, self.metadata["models"][0], device, batch_size)
    #         sentences_chunk_list = self.chunker.chunk(content, tokenizer, "sentences", 512, 8, 256, self.metadata["models"][0], device, batch_size) 
    #         sliding_window_chunk_list = self.chunker.chunk(content, tokenizer, "sliding_window", 512, 8, 256, self.metadata["models"][0], device, batch_size)
    #         # content_chunks = fixed_chunk_list + semantic_chunk_list + sentences_chunk_list + sliding_window_chunk_list
    #         content_chunks = fixed_chunk_list + sentences_chunk_list + sliding_window_chunk_list
    #     else:
    #         content_chunks = self.chunker.chunk(content, tokenizer, method, chunk_size, n_sentences, step_size, embed_model)
    #     parent_cid = item["cid"]
    #     content_tokens = tokenizer.encode(content)
    #     ## sort content_chunks by the firt element of each tuple then the second element
    #     content_chunks = sorted(content_chunks, key=lambda x: (x[0], x[1]))
    #     ## filter out chunks that are larger than the chunk_size
    #     content_chunks = [chunk for chunk in content_chunks if chunk[1] - chunk[0] <= chunk_size]
    #     ## filter content_chunks to remove duplicates
    #     seen_chunks = set()
    #     unique_content_chunks = []
    #     for chunk in content_chunks:
    #         if chunk not in seen_chunks:
    #             unique_content_chunks.append(chunk)
    #             seen_chunks.add(chunk)
    #     content_chunks = unique_content_chunks
    #     if parent_cid in list(self.caches.keys()):
    #         pass
    #     else:
    #         cid_chunks = {"parent_cid": parent_cid, "items" : [], "children": []}
    #         for chunk in content_chunks:
    #             chunk_index = chunk
    #             chunk_content = content_tokens[chunk[0]:chunk[1]]
    #             chunk_text = tokenizer.decode(chunk_content)
    #             child_cid = self.multiformats.get_cid(chunk_text)
    #             child_content = {"cid": child_cid, "index": chunk_index, "content": chunk_text, "parent_cid": parent_cid}
    #             cid_chunks["children"].append(child_cid)
    #             cid_chunks["items"].append(child_content)
    
    #         if type(cid_chunks["parent_cid"]) is not str:
    #             print("Parent CID is not a string")
        
    #         if parent_cid not in self.cid_chunk_set and type(cid_chunks["parent_cid"]) is str:
    #             while self.cid_chunk_queue.full():
    #                 await asyncio.sleep(0.1)
    #             if not self.cid_chunk_queue.full():
    #                 self.cid_chunk_set.add(parent_cid)
    #                 self.cid_chunk_queue.put_nowait(cid_chunks)
    #                 print("Added chunked item to queue for CID " + cid_chunks["parent_cid"])
    #         else:
    #             print("CID " + cid_chunks["parent_cid"] + " already in chunk set")
    #     return cid_chunks
    
    

    async def queue_size(self, model):
        print("Checking queue size")
        queue_size = 0
        for endpoint in list(self.ipfs_accelerate_py.resources["batch_sizes"][model].keys()):
            queue_size += self.ipfs_accelerate_py.resources["batch_sizes"][model][endpoint]
        if queue_size == 0:
            queue_size = 1
        return queue_size

    async def save_checkpoints_to_disk(self, dataset, dst_path, models):
        # self.saved = False
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
                if "hashed_dataset" not in list(self.all_cid_list.keys()):
                    self.all_cid_list["hashed_dataset"] = []
                if "hashed_dataset" not in list(self.all_cid_set.keys()):
                    self.all_cid_set["hashed_dataset"] = set()
                if "hashed_dataset" in list(self.caches.keys()):
                    for cid in list(self.caches["hashed_dataset"].keys()):
                        if "embedding" in list(self.caches["hashed_dataset"][cid].keys()):
                            self.caches[model][cid] = self.caches["hashed_dataset"][cid]["embedding"]
                            self.all_cid_list[model].append(cid)
                            self.all_cid_set[model].add(cid)
                            del self.caches["hashed_dataset"][cid]["embedding"]
                    if self.caches["hashed_dataset"] and len(self.caches["hashed_dataset"]) > 0:
                        tmp_dataset = datasets.Dataset.from_dict(self.caches["hashed_dataset"])
                        tmp_dataset_cids = list(self.caches["hashed_dataset"].keys())
                        self.all_cid_list["hashed_dataset"] += tmp_dataset_cids
                        self.all_cid_set["hashed_dataset"] = set(self.all_cid_set["hashed_dataset"].union(set(tmp_dataset_cids)))
                        tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": tmp_dataset_cids})
                        hashed_dataset_shards = [x for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x]
                        next_filename_shard = f"ipfs_{dataset.replace('/', '___')}_shard_{len(hashed_dataset_shards)}"
                        tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                        tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                        del tmp_dataset
                        del tmp_dataset_cids
                        del tmp_dataset_cids_dataset
                        del self.caches["hashed_dataset"]
                        self.caches["hashed_dataset"] = {}
                        pass
                for model in models:
                    if model in list(self.item_cache.keys()):
                        if "items" in list(self.item_cache[model].keys()):
                            del self.item_cache[model]["items"]
                        for result in list(self.item_cache[model].keys()):
                            if len(list(self.item_cache[model][result].keys())) == 0:
                                del self.item_cache[model][result]
                            this_result = self.item_cache[model][result]
                            this_cid = None if "cid" not in list(this_result.keys()) else this_result["cid"]
                            this_embedding = None if "embedding" not in list(this_result.keys()) else this_result["embedding"]
                            this_data = None if "data" not in list(this_result.keys()) else this_result["data"]                            
                            parent_cid = None if "parent_cid" not in list(this_result.keys()) else this_result["parent_cid"]
                            children = None if "children" not in list(this_result.keys()) else this_result["children"]
                            self.all_cid_list[model].append(this_cid)
                            self.all_cid_set[model].add(this_cid)
                            if "embedding" in list(this_result.keys()):
                                self.caches[model][this_cid] = np.array(this_embedding)
                                del self.item_cache[model][result]["embedding"]
                            if "children" not in list(this_result.keys()) and "parent_cid" not in list(this_result.keys()):
                                if this_embedding is None:
                                    if this_cid not in list(self.caches["hashed_dataset"].keys()):
                                        # self.caches["hashed_dataset"][this_cid] = this_embedding
                                        pass
                                    pass
                                else:
                                    if this_cid not in list(self.caches[model].keys()):
                                        self.caches[model][this_cid] = np.array(this_embedding)
                                        del self.item_cache[model][result]
                            pass
                        pass
                    else:
                        pass
                for model in models:
                    if model in list(self.chunk_cache.keys()) and model != "hashed_dataset":
                        if "items" in list(self.chunk_cache[model].keys()):
                            del self.chunk_cache[model]["items"]
                        if len(list(self.chunk_cache[model].keys())) > 0:
                            for this_cid in list(self.chunk_cache[model].keys()):
                                this_chunk = self.chunk_cache[model][this_cid]
                                if "children" in list(this_chunk.keys()) and "items" in list(this_chunk.keys()):
                                    parent_cid = this_chunk["parent_cid"]
                                    for this_child in this_chunk["children"]:
                                        if this_child in list(self.item_cache[model].keys()):
                                            this_child_index = this_chunk["children"].index(this_child)
                                            this_child_item = self.item_cache[model][this_child]
                                            if  len(list(this_child_item.keys())) > 0 and "embedding" in list(this_chunk["items"][this_child_index].keys()):
                                                self.caches[model][this_child] = this_chunk["items"][this_child_index]["embedding"]
                                                pass
                                            if type(this_child_item) == list and len(this_child_item) > 0:
                                                self.chunk_cache[model][parent_cid]["items"].append(this_child_item)
                                                del self.item_cache[model][this_child]
                                            if type(this_child_item) == dict and len(list(this_child_item.keys())) > 0:
                                                self.chunk_cache[model][parent_cid]["items"].append(this_child_item)
                                                del self.item_cache[model][this_child]
                                        pass
                                    pass
                                else:
                                    print("Chunk is not ready for CID " + this_cid)
                                    pass
                for model in models:
                    if len( list(self.chunk_cache[model].keys())) > 0 and model != "hashed_dataset":           
                        chunk_cids = list(self.chunk_cache[model].keys())
                        for cid in chunk_cids:
                            len_children = len(self.chunk_cache[model][cid]["children"])
                            len_items = len(self.chunk_cache[model][cid]["items"])
                            children_cids = self.chunk_cache[model][cid]["children"]
                            if len_items == len_children:
                                this_chunk = self.chunk_cache[model][cid]
                                this_chunk_items = this_chunk["items"]
                                this_children_cid_dataset = datasets.Dataset.from_dict({"cids": children_cids})
                                this_children_cid_path = os.path.join(dst_path, "checkpoints", "sparse_chunks", cid + "_cids.parquet")
                                this_children_cid_dataset.to_parquet(this_children_cid_path)
                                this_cid_dataset = datasets.Dataset.from_dict(this_chunk["items"])
                                this_cid_path = os.path.join(dst_path, "checkpoints", "sparse_chunks", cid + ".parquet")
                                this_cid_dataset.to_parquet(this_cid_path)
                                print("Saved " + str(len(this_cid_dataset)) + " chunks to disk for CID " + cid + " at " + this_cid_path)
                                self.cid_chunk_set.add(cid)
                                self.cid_chunk_list.append(cid)
                                del self.chunk_cache[model][cid]
                                del this_cid_dataset
                for model in models:
                    if model in list(self.caches.keys()) and len(self.caches[model]) > 0:
                        if "items" in list(self.caches[model].keys()):
                            del self.caches[model]["items"]
                        if len(list(self.caches[model].keys())) > 0:
                            cache_dataset = datasets.Dataset.from_dict(self.caches[model])
                            cache_cid_list = list(self.caches[model].keys())
                            cache_cids_dataset = datasets.Dataset.from_dict({"cids": cache_cid_list})
                            ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints", model.replace('/', '___')))
                            hashed_dataset_shards = [x for x in ls_checkpoints if f"ipfs_{dataset.replace('/', '___')}_{model.replace('/', '___')}_shard" in x and "_cids" not in x]
                            next_filename_shard = f"ipfs_{dataset.replace('/', '___')}_{model.replace('/', '___')}_shard_{len(hashed_dataset_shards)}"
                            cache_dataset.to_parquet(os.path.join(dst_path, "checkpoints", model.replace('/', '___'), next_filename_shard + ".parquet"))
                            cache_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", model.replace('/', '___'), next_filename_shard + "_cids.parquet"))
                            print("Saved " + str(len(cache_cid_list)) + " embeddings to disk at " + dst_path)
                            self.caches[model] = {}
                self.saved = True
            #     if self.producer_task_done and all(self.consumer_task_done.values()):
            #     self.save_to_disk_task_done = True
            #     break
        return None 
    
    async def init_datasets(self, models, dataset, split, column, dst_path):
        columns = []
                
        try:
            # Load dataset
            if split is None:
                self.dataset = load_dataset(dataset).shuffle(random.randint(0,65536))
            else:
                self.dataset = load_dataset(dataset, split=split).shuffle(random.randint(0,65536))
            columns = self.dataset.column_names
            columns.append("cid")
        except Exception as e:
            print(e)
            self.dataset = None

        try:
            init_load_clusters = await self.ipfs_datasets.load_clusters(dataset, split, dst_path)
        except Exception as e:
            print(e)
            init_load_clusters = e

        try:
            init_load_checkpoints = await self.ipfs_datasets.load_checkpoints(dataset, split, dst_path, models)        
        except Exception as e:
            print(e)
            init_load_checkpoints = e
    
        len_datasets_list = self.dataset.num_rows
        len_cid_list = len(self.ipfs_datasets.cid_list)
        len_cid_set = len(self.ipfs_datasets.cid_set)
        if len_cid_list == len_datasets_list:
            self.cid_list = self.ipfs_datasets.cid_list
            self.cid_set = set(self.cid_list)
            self.all_cid_list["hashed_dataset"] = self.cid_list
            self.all_cid_set["hashed_dataset"] = self.cid_set
            pass
        elif len_cid_list < len_datasets_list:
            self.cid_list = self.ipfs_datasets.cid_list
            self.cid_set = set(self.cid_list)
            self.all_cid_list["hashed_dataset"] = self.cid_list
            self.all_cid_set["hashed_dataset"] = self.cid_set
            pass
        elif len_cid_list > len_datasets_list:
            self.cid_list = self.ipfs_datasets.hashed_dataset["cid"][:len_datasets_list]
            self.cid_set = set(self.cid_list)
            self.all_cid_list["hashed_dataset"] = self.cid_list
            self.all_cid_set["hashed_dataset"] = self.cid_set
            pass
        
                                
        results = { "load_clusters": init_load_clusters, "load_checkpoints": init_load_checkpoints, "columns": columns }
        return results

    async def init_queues(self, models, endpoints, dst_path):
        self.queues = {}
        self.cid_set = set()
        self.all_cid_list = {}
        self.cid_chunk_queue = asyncio.Queue()
        queue_size = await self.queue_size(models[0])
        self.cid_chunk_queue = asyncio.Queue(queue_size)
        chunk_dir_path = os.path.join(dst_path, "checkpoints", "sparse_chunks")
        chunk_files = os.listdir(chunk_dir_path)
        chunk_files = [x for x in chunk_files if ".parquet" in x]
        saved_chunk_cids = [x.split(".")[0] for x in chunk_files]
        self.cid_chunk_list = saved_chunk_cids
        self.cid_chunk_set = set(saved_chunk_cids)
        for model_name in models:
            if model_name not in list(self.item_cache.keys()):
                self.item_cache[model_name] = {}
            if model_name not in list(self.chunk_cache.keys()):
                self.chunk_cache[model_name] = {}
        return None
    
    async def chunk_consumer(self, model_name, endpoint, endpoint_handler):
        try:
            batch_size = await self.max_batch_size(model_name, endpoint, endpoint_handler)
            queue_size = await self.queue_size(model_name)
            self.ipfs_accelerate_py.resources["queues"][model_name][endpoint] = asyncio.Queue(batch_size)
            self.ipfs_accelerate_py.resources["queue"][model_name] = asyncio.Queue(queue_size) 
            self.cid_chunk_queue = asyncio.Queue(queue_size)
            self.cid_queue = asyncio.Queue(queue_size) 
        except Exception as e:
            batch_size = 0
            print(e)
            # return None
        while True:
            while batch_size == 0:
                try:
                    batch_size = await self.max_batch_size(model_name, endpoint, endpoint_handler)
                    queue_size = await self.queue_size(model_name)
                    self.cid_chunk_queue = asyncio.Queue(queue_size)
                    self.cid_queue = asyncio.Queue(queue_size) 
                    self.ipfs_accelerate_py.resources["queue"][model_name] = asyncio.Queue(queue_size)
                    self.ipfs_accelerate_py.resources["queues"][model_name][endpoint] = asyncio.Queue(batch_size)
                    if batch_size == 0:
                        await asyncio.sleep(300)
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    batch_size = 0
                    await asyncio.sleep(300)
            
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
            while not self.cid_chunk_queue.empty() and not self.ipfs_accelerate_py.resources["queue"][model_name].full():
                while self.cid_queue.empty():
                    await asyncio.sleep(0.1)
                while not self.cid_queue.empty():
                    processed_item = await self.cid_queue.get()
                    if processed_item is not None and (type(processed_item) == dict and len(list(processed_item.keys())) > 0):
                        if "parent_cid" not in list(processed_item.keys()):
                            if "cid" in list(processed_item.keys()):
                                this_cid = processed_item["cid"]
                                if this_cid not in list(self.item_cache[model_name].keys()):
                                    self.item_cache[model_name][this_cid] = {}
                                self.item_cache[model_name][this_cid] = processed_item
                                while self.ipfs_accelerate_py.resources["queue"][model_name].full():
                                    await asyncio.sleep(1)
                                self.ipfs_accelerate_py.resources["queue"][model_name].put_nowait(processed_item)
                                self.cid_queue.task_done()
                                await asyncio.sleep(0.001)

                while self.cid_chunk_queue.empty():
                    await asyncio.sleep(0.1)
                while not self.cid_chunk_queue.empty():                    
                    chunked_item = await self.cid_chunk_queue.get()
                    batch_results = []
                    batch = []
                    chunk_data = []
                    if chunked_item["parent_cid"] not in list(self.chunk_cache[model_name].keys()):
                        self.chunk_cache[model_name][chunked_item["parent_cid"]] = {}
                        self.chunk_cache[model_name][chunked_item["parent_cid"]]["children"] = chunked_item["children"]
                        self.chunk_cache[model_name][chunked_item["parent_cid"]]["parent_cid"] = chunked_item["parent_cid"]
                        self.chunk_cache[model_name][chunked_item["parent_cid"]]["items"] = {}
                    if chunked_item is not None:
                        for i in range(len(chunked_item["items"])):
                            item = chunked_item["items"][i]
                            while self.ipfs_accelerate_py.resources["queue"][model_name].full():
                                await asyncio.sleep(1)
                            self.ipfs_accelerate_py.resources["queue"][model_name].put_nowait(item)
                        # await asyncio.sleep(0.001)
                    else:
                        pass
                    self.cid_chunk_queue.task_done()
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.001)
        return None

    async def config_queues(self, models, column, endpoint, endpoint_handler):
        consumer_tasks = []
        all_tasks = []
        self.cid_queue = asyncio.Queue() 
        for endpoint in list(self.ipfs_accelerate_py.resources["endpoint_handler"][models[0]].keys()):
            if self.ipfs_accelerate_py.resources["hwtest"]["cuda"] == True and "openvino:" in endpoint:
                continue
            this_endpoint_handler = self.ipfs_accelerate_py.resources["endpoint_handler"][models[0]][endpoint]
            chunk_consumer = asyncio.create_task(self.chunk_consumer(models[0], endpoint, this_endpoint_handler))
            consumer_tasks.append(chunk_consumer)
            # all_tasks.append(item_consumer)
            model_consumer = asyncio.create_task(self.model_consumer(models[0], endpoint, this_endpoint_handler))
            consumer_tasks.append(model_consumer)
            # all_tasks.append(model_consumer)               
            endpoint_consumer = asyncio.create_task(self.endpoint_consumer( models[0], endpoint, this_endpoint_handler, column))
            consumer_tasks.append(endpoint_consumer)
            # all_tasks.append(endpoint_consumer)
            
        return consumer_tasks 
        
    async def index_dataset(self, dataset, split, column, dst_path, models = None):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        # Initialize resources and endpoints
        resource_keys = list(self.resources.keys())
        endpoints = {resource: self.resources[resource] 
                    for resource in resource_keys if "endpoints" in resource}
        self.cid_queue = asyncio.Queue() 
        # Load required data
        self.resources = await self.init_endpoints(models, endpoints)
        await self.init_datasets(models, dataset, split, column, dst_path)
        await self.init_queues(models, endpoints, dst_path)
        self.index["hashed_dataset"] = self.ipfs_datasets.hashed_dataset
        self.index["dataset"] = self.dataset
        for key in list(self.ipfs_datasets.index.keys()):
            self.index[key] = self.ipfs_datasets.index[key]
        self.cid_chunk_set = self.ipfs_datasets.cid_chunk_set
        self.cid_chunk_list = self.ipfs_datasets.cid_chunk_list
        self.cid_list = self.ipfs_datasets.cid_list
        self.cid_set = self.ipfs_datasets.cid_set
        cid_list = self.ipfs_datasets.cid_list
        cid_set = self.ipfs_datasets.cid_set
        all_cid_list = self.ipfs_datasets.all_cid_list
        all_cid_set = self.ipfs_datasets.all_cid_set
            
        # num_workers = min(multiprocessing.cpu_count(), 1)  # Use up to 1 CPU cores
        # num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 CPU cores
        num_workers = round(multiprocessing.cpu_count() / 2)
        # num_workers = round(multiprocessing.cpu_count())
        # num_workers = multiprocessing.cpu_count()
        consumer_tasks = []
        producer_tasks = []
        all_tasks = []
        metadata = self.metadata
        
        # def chunk_producer(dataset_stream, column, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None, dst_path=None, chunk_item=None, process_item=None, cid_queue=None, cid_chunk_set=None, chunker=None, metadata=None, caches=None, all_cid_set=None):

        # with multiprocessing.Pool() as pool:
        #     args = [
        #         (self.dataset.shard(num_shards= num_workers, index=worker_id), column, None, self.ipfs_accelerate_py.resources["tokenizer"][models[0]]["cuda:"+str(worker_id % 2)] , None, None, None, models[0], dst_path, chunk_item, process_item, cid_queue, cid_chunk_set, chunker, metadata, caches, all_cid_set)
        #         for worker_id in range(num_workers)
        #     ]
        #     pool.starmap(chunk_producer, args)
            
        args = [self.index, column, None, self.ipfs_accelerate_py.resources["tokenizer"][models[0]]["cuda:0"], None, None, None, models[0], dst_path, chunk_item, process_item, cid_queue, cid_chunk_set, chunker, metadata, caches, all_cid_set]        
        chunk_producer_results = chunk_producer(*args)
        # Create producer tasks directly as asyncio tasks
        # for worker_id in range(num_workers):
        #     producer_task = asyncio.create_task(
        #         chunk_producer(
        #             self.dataset.shard(num_shards=num_workers, index=worker_id),
        #             column,
        #             None,  # method
        #             None,  # tokenizer
        #             None,  # chunk_size
        #             None,  # n_sentences
        #             None,  # step_size
        #             models[0],  # embed_model
        #             dst_path,
        #             chunk_item,  # chunk_item
        #             process_item,  # process_item
        #             self.cid_queue,
        #             self.cid_chunk_set
        #         )
        #     )
        #     producer_tasks.append(producer_task)
        #     all_tasks.append(producer_task)
            
        consumer_tasks = await self.config_queues(models, column, endpoints, dst_path)
        all_tasks.extend(consumer_tasks)
        # for _ in range(num_workers):
        #     producer_task = asyncio.create_task(self.chunk_producer(self.dataset, column, self.queues, None, None, None, None, None, models[0], dst_path))
        #     producer_tasks.append(producer_task)
        #     all_tasks.append(producer_task)
        
        save_task = asyncio.create_task(self.save_checkpoints_to_disk(dataset, dst_path, models))
        all_tasks.append(save_task)

        # Wait for all tasks to complete
        # await asyncio.gather(*consumer_tasks, *producer_tasks, save_task)
        # await asyncio.gather(*all_tasks, save_task)
        await asyncio.gather(*all_tasks)
        return None        
    
    async def model_consumer(self, model_name, endpoint, endpoint_handler):
        print("model consumer started for model " + model_name + " at endpoint " + endpoint)
        while True:
            batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][model_name][endpoint]           
            model_queue = True if model_name in list(self.ipfs_accelerate_py.resources["queues"].keys()) else False
            endpoint_queue = True if endpoint in list(self.ipfs_accelerate_py.resources["queues"][model_name].keys()) else False
            empty = True if model_name in list(self.ipfs_accelerate_py.resources["queues"].keys()) and "empty" in dir(self.ipfs_accelerate_py.resources["queues"][model_name][endpoint]) else False
            queue_not_empty = not self.ipfs_accelerate_py.resources["queue"][model_name].empty()
            endpoint_queue_not_full = not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full()
            queue_size = self.ipfs_accelerate_py.resources["queue"][model_name].qsize()
            batch_ready = True if batch_size > 0 else False
            test_ready = all([
                model_queue,
                endpoint_queue,
                empty,
                queue_not_empty,
                endpoint_queue_not_full,
                batch_ready
            ])
            while not test_ready:
                batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][model_name][endpoint]           
                if batch_size == 0:
                    await asyncio.sleep(300)
                    batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][model_name][endpoint]
                while self.ipfs_accelerate_py.resources["queue"][model_name].empty():
                    await asyncio.sleep(1)
                while self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full():
                    await asyncio.sleep(1)     
                model_queue = True if model_name in list(self.ipfs_accelerate_py.resources["queue"].keys()) else False
                endpoint_queue = True if endpoint in list(self.ipfs_accelerate_py.resources["queues"][model_name].keys()) else False
                empty = True if model_name in list(self.ipfs_accelerate_py.resources["queue"].keys()) and "empty" in dir(self.ipfs_accelerate_py.resources["queue"][model_name]) else False
                queue_not_empty = not self.ipfs_accelerate_py.resources["queue"][model_name].empty()
                queue_not_full = not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full()
                endpoint_queue_not_full = not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full()
                queue_size = self.ipfs_accelerate_py.resources["queue"][model_name].qsize()
                batch_ready = True if batch_size > 0 else False
                test_ready = all([
                    model_queue,
                    endpoint_queue,
                    empty,
                    queue_not_empty,
                    batch_ready,
                    queue_not_full,
                    endpoint_queue_not_full
                ])
                await asyncio.sleep(1)
                pass
            while not self.ipfs_accelerate_py.resources["queue"][model_name].empty() and not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full():        
                processed_item = await self.ipfs_accelerate_py.resources["queue"][model_name].get()
                batch_results = []
                batch = []
                chunk_data = []
                if model_name not in list(self.item_cache.keys()):
                    self.item_cache[model_name] = {}
                if model_name not in list(self.all_cid_list.keys()):
                    self.all_cid_list[model_name] = []
                if model_name not in list(self.all_cid_set.keys()):
                    self.all_cid_set[model_name] = set()
                if model_name not in list(self.caches.keys()):
                    self.caches[model_name] = {}
                if "parent_cid" in list(processed_item.keys()):
                    if model_name not in list(self.chunk_cache.keys()):
                        self.chunk_cache[model_name] = {}
                    if processed_item["parent_cid"] not in list(self.chunk_cache[model_name].keys()):
                        self.chunk_cache[model_name][processed_item["parent_cid"]] = {}
                else:
                    if model_name not in list(self.item_cache.keys()):
                        self.item_cache[model_name] = {}
                    if processed_item["cid"] not in list(self.item_cache[model_name].keys()):
                        self.item_cache[model_name][processed_item["cid"]] = processed_item
                queue_sizes = {}
                max_sizes = {}
                queue_remaining = {}
                if processed_item is not None:
                    self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].put_nowait(processed_item)
                    self.ipfs_accelerate_py.resources["queue"][model_name].task_done()
                    await asyncio.sleep(0.001)
                else:
                    pass
                await asyncio.sleep(0.001)
        return None

    async def endpoint_consumer(self, model_name, endpoint, endpoint_handler, column = None):                
        endpoint_queue = True if endpoint in list(self.ipfs_accelerate_py.resources["queues"][model_name].keys()) else False
        empty = True if endpoint in list(self.ipfs_accelerate_py.resources["queues"][model_name].keys()) and "empty" in dir(self.ipfs_accelerate_py.resources["queues"][model_name][endpoint]) else False
        queue_not_empty = not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].empty()
        batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][model_name][endpoint]
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
                while batch_size == 0:
                    await asyncio.sleep(300)
                    batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][model_name][endpoint]
                while not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full():
                    queue_size = self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].qsize()
                    await asyncio.sleep(1)
                batch_size = self.ipfs_accelerate_py.resources["batch_sizes"][model_name][endpoint]
                endpoint_queue = True if endpoint in list(self.ipfs_accelerate_py.resources["queues"][model_name].keys()) else False
                empty = True if endpoint in list(self.ipfs_accelerate_py.resources["queues"][model_name].keys()) and "empty" in dir(self.ipfs_accelerate_py.resources["queues"][model_name][endpoint]) else False
                queue_not_empty = not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].empty()
                queue_not_full = not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full()
                queue_full = self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full()
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
            
            # while self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].empty():
            #     await asyncio.sleep(0.1)
            
            while not self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].empty():
                item = await self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].get()
                if "parent_cid" in list(item.keys()):
                    batch.append(item["content"])
                else:
                    if column is not None:
                        batch.append(item[column])
                    else:
                        batch.append(item["text"])
                chunk_data.append(item)
                self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].task_done()

            if len(batch) >= batch_size:
                if "cuda" in endpoint:
                    with torch.no_grad():
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        if hasattr(torch.cuda, 'ipc_collect'):
                            torch.cuda.ipc_collect()
                gc.collect()
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
                    if "parent_cid" in list(chunk_data[i].keys()):
                        this_cid = chunk_data[i]["cid"]
                        this_index = chunk_data[i]["index"]
                        this_content = chunk_data[i]["content"]
                        this_parent_cid = chunk_data[i]["parent_cid"]
                        batch_results.append({"cid": this_cid, "index": this_index, "content": this_content , "embedding": this_embeddings, "parent_cid": this_parent_cid})
                    else:
                        this_cid = chunk_data[i]["cid"]
                        data = chunk_data[i]
                        data["embedding"] = this_embeddings
                        batch_results.append(data)
            
            batch_parent_cids = []
            for i in range(len(batch_results)):
                batch_result = batch_results[i]
                if "parent_cid" in list(batch_result.keys()):
                    if batch_result["parent_cid"] not in set(batch_parent_cids):
                        batch_parent_cids.append(batch_result["parent_cid"])
                    if batch_result["parent_cid"] not in list(self.chunk_cache[model_name].keys()):
                        self.chunk_cache[model_name][batch_result["parent_cid"]] = {}
                        pass
                    if "items" not in list(self.chunk_cache[model_name][batch_result["parent_cid"]].keys()):
                        self.chunk_cache[model_name][batch_result["parent_cid"]]["items"] = {}
                    if "children" not in list(self.chunk_cache[model_name][batch_result["parent_cid"]].keys()):
                        self.chunk_cache[model_name][batch_result["parent_cid"]]["children"] = []
                    if "parent_cid" not in list(self.chunk_cache[model_name][batch_result["parent_cid"]].keys()):
                        self.chunk_cache[model_name][batch_result["parent_cid"]]["parent_cid"] = batch_result["parent_cid"]
  
                    if batch_result["cid"] not in self.chunk_cache[model_name][batch_result["parent_cid"]]["children"]:
                        self.chunk_cache[model_name][batch_result["parent_cid"]]["children"].append(batch_result["cid"])
                    child_cid_list = list(self.chunk_cache[model_name][batch_result["parent_cid"]]["items"].keys())
                    if batch_result["cid"] not in child_cid_list:
                        self.chunk_cache[model_name][batch_result["parent_cid"]]["items"][batch_result["cid"]] = batch_result
                    if "embedding" in list(batch_result.keys()):
                        self.caches[model_name][this_cid] = batch_result["embedding"]                  
                    if batch_result["parent_cid"] not in self.cid_chunk_list:
                        self.cid_chunk_list.append(batch_result["parent_cid"])
                        self.cid_chunk_set.add(batch_result["parent_cid"])
                else:
                    this_cid = batch_result["cid"]
                    if this_cid not in list(self.item_cache[model_name].keys()):
                        self.item_cache[model_name][this_cid] = {}
                    if "embedding" in list(batch_result.keys()):
                        self.caches[model_name][this_cid] = batch_result["embedding"]
                    if this_cid not in list(self.caches["hashed_dataset"].keys()):
                        if "embedding" in list(batch_result.keys()):
                            self.caches[model_name][this_cid] = batch_result["embedding"]
                            del batch_result["embedding"]
                    if this_cid not in self.all_cid_set[model_name]:
                        self.all_cid_list[model_name].append(this_cid)
                        self.all_cid_set[model_name].add(this_cid)
                    if batch_result["cid"] not in self.cid_list:
                        self.cid_list.append(batch_result["cid"])
                        self.cid_set.add(batch_result["cid"])
                    
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
            queue_full = self.ipfs_accelerate_py.resources["queues"][model_name][endpoint].full()
            test_ready = all([
                endpoint_queue,
                empty,
                queue_full,
            ])
            await asyncio.sleep(0.001)                

    # async def process_item(self, item, column=None, queues=None, dst_path=None, embed_model=None, ):
    #     # Assuming `item` is a dictionary with required data
    #     if "hashed_dataset" not in list(self.caches.keys()):
    #         self.caches["hashed_dataset"] = {}
    #     if "hashed_dataset" not in list(self.all_cid_set.keys()):
    #         self.all_cid_set["hashed_dataset"] = set()
    #     # print(f"Processing item with CID {index_cid(item[column])[0]}")
    #     if queues is None:
    #         queues = self.ipfs_accelerate_py.resources["queues"]
    #     column_names = list(item.keys())
    #     if column is None:
    #         this_cid = self.index_cid(json.dumps(item))[0]
    #     elif column not in column_names:
    #         this_cid = self.index_cid(json.dumps(item))[0]
    #     else:
    #         this_cid = self.index_cid(item[column])[0]
    #     if "cid" not in column_names:
    #         item["cid"] = this_cid
    #     elif item["cid"] is None:
    #         item["cid"] = this_cid
    #     # Check if cid is in index
    #     if this_cid in self.cid_set:
    #         # print(f"CID {this_cid} already in index, skipping item.")
    #         return None
    #     else:
    #         while self.cid_queue.full():
    #             await asyncio.sleep(0.1)    
    #         self.cid_set.add(this_cid)
    #         if this_cid not in self.all_cid_set["hashed_dataset"]:
    #             self.caches["hashed_dataset"][this_cid] = item
    #             self.cid_queue.put_nowait(item)
    #             print("Added item to queue for CID " + str(this_cid))
    #             # self.saved = False
    #     return item
                
    # async def chunk_producer(self, dataset_stream, column, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None, dst_path=None):
    #     tasks = []
    #     self.producer_task_done = False
        
    #     async for item in self.async_generator(dataset_stream):
    #         while self.cid_queue.full():
    #             await asyncio.sleep(0.1)
    #         if column is not None:
    #             cid = self.multiformats.get_cid(item[column])
    #         else:
    #             json_item = json.dumps(item)
    #             cid = self.multiformats.get_cid(json_item)
    #         if cid not in list(item.keys()):
    #             item["cid"] = cid
    #         if cid not in self.cid_chunk_set:
    #             processed_item = await self.process_item(item, column, None, embed_model, dst_path)
    #             chunked_item = await self.chunk_item(item, column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model)
    #         else:
    #             pass
    #         await asyncio.sleep(0.001)
           
    #     return None
        
    async def process_hashed_dataset_shard(self, dataset, split=None):
        results = await self.ipfs_datasets.process_hashed_dataset_shard(dataset, split)
        return results
    
    async def init_endpoints(self, models, endpoint_list=None):
        resources = await self.ipfs_accelerate_py.init_endpoints(models, endpoint_list)
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
            if resource in list(self.ipfs_accelerate_py.resources.keys()):
                new_resources[resource] = self.ipfs_accelerate_py.resources[resource]
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

    
if __name__ == "__main__":
    metadata = {
        "dataset": "laion/gpt4v-dataset",
        "namespace": "laion/gpt4v-dataset",
        "column": "link",
        "split": "train",
        "models": [
            "lmms-lab/llava-onevision-qwen2-0.5b-si"
            "lmms-lab/llava-onevision-qwen2-0.5b-ov"
            "OpenGVLab/InternVL2_5-1B"
            # "OpenGVLab/InternVL2_5-8B"
            # "PVC-InternVL2-8B"
            # "lmms-lab/llava-onevision-qwen2-7b-si-chat"
            # "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
            # "lmms-lab/LLaVA-Video-7B-Qwen2",
        ],
        "chunk_settings": {

        },
        "dst_path": "/storage/gpt4v-dataset/data",
    }
    resources = {
        "local_endpoints": [
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "cpu", 512],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "cpu", 8192],
            ["OpenGVLab/InternVL2_5-1B", "cpu", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "cuda:0", 512],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "cuda:0", 8192],
            ["OpenGVLab/InternVL2_5-1B", "cuda:0", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "cuda:1", 512],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "cuda:1", 8192],
            ["OpenGVLab/InternVL2_5-1B", "cuda:1", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "openvino", 512],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "openvino", 8192],
            ["OpenGVLab/InternVL2_5-1B", "openvino", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "llama_cpp", 512],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "llama_cpp", 8192],
            ["OpenGVLab/InternVL2_5-1B", "llama_cpp", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "ipex", 512],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "ipex", 8192],
            ["OpenGVLab/InternVL2_5-1B", "ipex", 32768],
        ],
        "openvino_endpoints": [
        ],
        "tei_endpoints": [
        ],
    }
    create_embeddings_batch = ipfs_embeddings_py(resources, metadata)
    asyncio.run(create_embeddings_batch.index_dataset(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))   
    # asyncio.run(create_embeddings_batch.summarize_dataset(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))    
    # asyncio.run(create_embeddings_batch.combine_checkpoints(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))
    # asyncio.run(create_embeddings_batch.kmeans_cluster_split(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"], 10))
    # asyncio.run(create_embeddings_batch.index_sparse_chunks(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))