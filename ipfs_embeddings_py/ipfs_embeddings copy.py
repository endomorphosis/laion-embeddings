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
from ipfs_embeddings_py.ipfs_datasets import ipfs_datasets_py
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
        self.ipfs_accelerate = ipfs_accelerate_py.ipfs_accelerate_py(resources, metadata)
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
        self.get_https_endpoint = self.get_https_endpoint
        self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.request_tei_endpoint = self.request_tei_endpoint
        self.request_libp2p_endpoint = self.request_libp2p_endpoint
        self.request_openvino_endpoint = self.request_openvino_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        self.test_tei_https_endpoint = self.test_tei_https_endpoint
        self.test_libp2p_endpoint = self.test_libp2p_endpoint
        self.test_openvino_endpoint = self.test_openvino_endpoint
        self.test_local_endpoint = self.test_local_endpoint
        self.index_knn = self.index_knn
        self.index_knn_openvino = self.index_knn_openvino
        self.make_post_request = self.make_post_request
        self.choose_endpoint = self.choose_endpoint
        self.get_endpoints = self.get_endpoints
        self.max_batch_size = self.max_batch_size
        self.consumer = self.consumer
        self.producer = self.producer
        self.process_item = self.process_item
        self.save_checkpoints_to_disk = self.save_checkpoints_to_disk
        self.save_chunks_to_disk = self.save_chunks_to_disk
        self.status = self.status
        self.setStatus = self.setStatus
        self.index_cid = self.index_cid
        self.load_index = self.load_index
        self.async_generator = self.async_generator
        self.send_batch_to_endpoint = self.send_batch_to_endpoint
        self.kmeans_cluster_split = self.kmeans_cluster_split
        # Initialize endpoints
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.init_endpoints = self.init_endpoints
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.init_endpoints = self.init_endpoints       
        return None
    

    async def add_endpoint(self, model, endpoint, context_length, endpoint_type):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if endpoint_type not in list(dir(self)):
                    self.__dict__[endpoint_type]= {}
                if model not in list(self.__dict__[endpoint_type].keys()):
                    self.__dict__[endpoint_type][model] = {}
                if endpoint not in list(self.__dict__[endpoint_type][model].keys()):
                    self.__dict__[endpoint_type][model][endpoint] = context_length
                self.endpoint_status[endpoint] = context_length
                success = True
            except Exception as e:
                print(e)
                pass
            return success        
        return None
    
    async def rm_endpoint(self, model, endpoint, endpoint_type):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if model in self.__dict__[endpoint_type] and endpoint in self.__dict__[endpoint_type][model]:
                    del self.__dict__[endpoint_type][model][endpoint]
                if endpoint in self.endpoint_status:
                    del self.endpoint_status[endpoint]
                success = True
            except Exception as e:
                print(e)
                pass
            return success
        return None
    
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

    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_endpoints and endpoint in self.tei_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False
    
    def test_openvino_endpoint(self, model, endpoint):
        if model in self.openvino_endpoints and endpoint in self.openvino_endpoints[model]:
            return True
        return False
    
    def test_local_endpoint(self, model, endpoint):
        if model in self.local_endpoints and endpoint in self.local_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.tei_endpoints:
            return self.tei_endpoints[model]
        return None

    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_tei_endpoint(self, model, endpoint, endpoint_type, batch):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    async def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model, batch_size):
        if model in self.llama_cpp_endpoints:
            for endpoint in self.llama_cpp_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
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
    
    async def parse_knn_errors(self, request, model, endpoint, endpoint_type=None):
        fatal = False
        return fatal
    
    
    async def parse_knn(self, request, model, endpoint, endpoint_type=None):
        token_length_size = 0
        incoming_batch_size = len(request)
        endpoint_batch_size = self.batch_sizes[model]
        embeddings = request
        embeddings_request = embeddings
        endpoint_context_size = 0
        if endpoint_type is None:
            raise ValueError("Endpoint type must be defined")
        if endpoint_type == "local_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                if "cuda" in endpoint or "cpu" in endpoint:
                    response = self.request_local_endpoint(model, endpoint, endpoint_type)
                elif "openvino:" in endpoint:
                    response_= self.request_openvino_endpoint(model, endpoint, endpoint_type)
                elif "llama_cpp" in endpoint:
                    response = self.request_llama_cpp_endpoint(model, endpoint, endpoint_type)
                else:
                    response = ValueError("Endpoint not found")
        if endpoint_type == "tei_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                response = self.request_tei_endpoint(model, endpoint, endpoint_type)
        if endpoint_type == "openvino_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                response = self.request_openvino_endpoint(model, endpoint, endpoint_type)
        if endpoint_type == "libp2p_endpoints":
            if incoming_batch_size > endpoint_batch_size:
                raise ValueError("Batch size too large")
            else:
                response = self.request_libp2p_endpoint(model, endpoint, endpoint_type)
        
        errors = await self.parse_knn_errors(response, model, endpoint, endpoint_type)
        
        return not errors
    
        # try:
        #     if not isinstance(embeddings, list):
        #         if isinstance(embeddings, ValueError):
        #             fail_reason = embeddings.args[0]
        #             if "413" in str(fail_reason):
        #                 error = fail_reason
        #                 if error.status == 413:
        #                     if error.reason == "Payload Too Large":
        #                         error_content = error.content._buffer[0].decode("utf-8")
        #                         error_content = json.loads(error_content)
        #                         if "error" in error_content.keys() and "error_type" in error_content.keys():
        #                             if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
        #                                 expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
        #                                 given = int(error_content["error"].split("Given: ")[1])
        #                                 difference = given - expected
        #                                 self.tei_endpoints[model][endpoint] = token_length_size - difference
        #                                 return await self.max_batch_size(model, endpoint)
        #             if "502" in str(fail_reason):
        #                 self.endpoint_status[endpoint] = 0
        #                 return 0
        #             if "504" in str(fail_reason):
        #                 self.endpoint_status[endpoint] = 0
        #                 return 0
        #             if "400" in str(fail_reason):
        #                 return await self.max_batch_size(model, endpoint)
        #         raise Exception(embeddings)
        #     exponent += 1
        #     batch_size = 2**exponent
        # except Exception as e:
        #     fail_reason = e.args[0]
        #     embed_fail = True
        #     if isinstance(e, ValueError) or isinstance(e, Exception):
        #         if "CUDA out of memory" in str(fail_reason):
        #             if exponent == 0:
        #                 self.endpoint_status[endpoint] = 0
        #                 return 1
        #             else:
        #                 self.endpoint_status[endpoint] = 2**(exponent-1)
        #                 return 2 ** (exponent-1)
        #         if "413" in str(fail_reason):
        #             error = fail_reason.args[0]
        #             if error.status == 413:
        #                 if error.reason == "Payload Too Large":
        #                     error_content = error.content._buffer[0].decode("utf-8")
        #                     error_content = json.loads(error_content)
        #                     if "error" in error_content.keys() and "error_type" in error_content.keys():
        #                         if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
        #                             expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
        #                             given = int(error_content["error"].split("Given: ")[1])
        #                             difference = given - expected
        #                             self.tei_endpoints[model][endpoint] = self.tei_endpoints[model][endpoint] - difference
        #                             results = await self.max_batch_size(model, endpoint)
        #                             return results
        #             pass
        #         if "504" in str(fail_reason):
        #             self.endpoint_status[endpoint] = 0
        #             return 0
        #         if "502" in str(fail_reason):
        #             self.endpoint_status[endpoint] = 0
        #             return 0
        #     pass
        
        # if isinstance(results, list):
        #     return results
        # elif isinstance(results, str):
        #     if "cid" in results:
        #         return results
        #     else:
        #         return results
        # else:
        #     return results
        
    async def request_knn(self, request_batch, model, endpoint, endpoint_type):
        request = None
        if endpoint_type is None:
            request = None
            pass
        elif endpoint_type == "tei_endpoints":
            request = await self.request_tei_endpoint(model, len(request_batch))
            pass
        elif endpoint_type == "openvino_endpoints":
            request = await self.request_openvino_endpoint(model, len(request_batch))
            pass
        elif endpoint_type == "libp2p_endpoints":
            request = await self.request_libp2p_endpoint(model, len(request_batch))
            pass
        elif endpoint_type == "local_endpoints":
            request = await self.request_local_endpoint(model, len(request_batch))
            pass
        else:
            request = None
            pass
        if request is not None:
            return request
        else:   
            return None

    async def max_batch_size(self, model, endpoint=None, endpoint_type=None ):
        embed_fail = False
        exponent = 0
        batch = []
        token_length_size = 0
        batch_size = 2**exponent
        if endpoint_type is None:
            this_model = None
            this_endpoint = None
            this_context_length = None
            if "/embed" in endpoint:
                endpoint_type = "tei_endpoints"
            elif "/infer" in endpoint:
                endpoint_type = "openvino_endpoints"
            elif "http" in endpoint:
                endpoint_type = "tei_endpoints"
            elif "cuda" in endpoint or "cpu" in endpoint or "local" in endpoint:
                endpoint_type = "local_endpoints"
            elif "libp2p" in endpoint:
                endpoint_type = "libp2p_endpoints"
            if endpoint_type is None:
                print('Endpoint not found')
                return 0
            else:
                pass
                  
        for this_endpoint in self.endpoints[endpoint_type]:
            if "cuda" in this_endpoint[1] or "cpu" in this_endpoint[1] or "local" in this_endpoint[1]:
                this_endpoint_index = self.endpoints[endpoint_type].index(this_endpoint)
                token_length_size = round(self.endpoints["local_endpoints"][this_endpoint_index][2] * 0.99)
            elif model is this_endpoint[0]:
                this_endpoint_index = self.endpoints[endpoint_type].index(this_endpoint)
                token_length_size = round(self.endpoints[endpoint_type][this_endpoint_index][2] * 0.99) 
        
        test_tokens = []
        if model not in self.tokenizer.keys():
            self.tokenizer[model] = {}
        if "cpu" not in self.tokenizer[model].keys():
            self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu')
        find_token_str = str("z")
        find_token_int = self.tokenizer[model]["cpu"].encode(find_token_str)
        if len(find_token_int) == 3:
            find_token_int = find_token_int[1]
        elif len(find_token_int) == 2:
            find_token_int = find_token_int[1]
        elif len(find_token_int) == 1:
            find_token_int = find_token_int[0]
        for i in range(token_length_size):
             test_tokens.append(find_token_int)
        test_text = self.tokenizer[model]["cpu"].decode(test_tokens)
        if endpoint is None:
            endpoint = self.choose_endpoint(model)
        while not embed_fail:
            test_batch = []
            for i in range(batch_size):
                test_batch.append(test_text)
            parsed_knn_embeddings = None
            embeddings = None
            request_knn_results = None
            try:
                request_knn_results = await self.request_knn(test_batch, model, endpoint, endpoint_type)
            except Exception as e:
                try:
                    embeddings = await self.index_knn(test_batch, model, endpoint)
                except Exception as e:
                        pass
            if request_knn_results != None and parsed_knn_embeddings == None:
                parsed_knn_embeddings = await self.parse_knn(request_knn_results, model, endpoint, endpoint_type)
            if parsed_knn_embeddings is not None:
               embeddings = parsed_knn_embeddings
            
        self.endpoint_status[endpoint] = 2**(exponent-1)
        if exponent == 0:
            return 1
        else:
            return 2**(exponent-1)
    
    async def save_chunks_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(60)
            if self.saved == False:
                if len(self.chunk_cache) > 0: 
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
        return None
        
    async def infer_local_cuda(self, model, samples):
        
        return None

    async def infer_local_cpu(self, model, samples):
        
        return None

    async def infer_local_llama(self, model, samples):
        
        return None
        
    async def infer_local_openvino(self, model, samples):
        
        return None

    async def index_knn(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is list or type(samples) is iter:
            this_query = {"inputs": samples}
            all_endpoints = { "tei_endpoints":  [ x[2] for x in self.tei_endpoints ], "openvino_endpoints":  [ x[2] for x in self.openvino_endpoints ], "libp2p_endpoints":  [ x[2] for x in self.libp2p_endpoints ], "local_endpoints": [ x[2] for x in self.local_endpoints ] }
            if len(all_endpoints["local_endpoints"]) > 0 and "cuda" in chosen_endpoint or "cpu" in chosen_endpoint:
                if chosen_endpoint is None:
                    if "chosen_endpoint" not in list(dir(self)) or self.chosen_local_endpoint is None or self.chosen_local_endpoint_model != model:
                        self.chosen_local_endpoint_model = model
                        self.chosen_local_endpoint = AutoModel.from_pretrained(model)
                        if model not in self.tokenizer.keys():
                            self.tokenizer[model] = {}
                        if "cpu" not in self.tokenizer[model].keys():
                            self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True)
                    chosen_endpoint = self.chosen_local_endpoint
                    chosen_endpoint.eval()
                    inputs = self.tokenizer[model]["cpu"](samples, return_tensors="pt")
                    with torch.no_grad():
                        output = chosen_endpoint(**inputs).last_hidden_state.mean(dim=1).tolist()
                        query_response = output[0]
            if len(all_endpoints["tei_endpoints"]) > 0 and "/embed" in chosen_endpoint and "cuda" not in chosen_endpoint and "cpu" not in chosen_endpoint:
                try:
                    query_response = await self.make_post_request(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            if len(all_endpoints["openvino_endpoints"]) > 0 and "/infer" in chosen_endpoint and "cuda" not in chosen_endpoint and "cpu" not in chosen_endpoint:
                try:
                    query_response = await self.make_post_request_openvino(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            if len(all_endpoints["libp2p_endpoints"]) > 0 and "/infer" not in chosen_endpoint and "/embed" not in chosen_endpoint and "cuda" not in chosen_endpoint and "cpu" not in chosen_endpoint:
                try:
                    query_response = await self.make_post_request_libp2p(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        return knn_stack
    
    async def index_knn_openvino(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is list or type(samples) is iter:
            this_query = {"inputs": samples}
            if "cuda" in chosen_endpoint or "cpu" in chosen_endpoint:
                if model not in self.local_endpoints.keys():
                    self.local_endpoints[model] = {}
                if model not in self.tokenizer.keys():
                    self.tokenizer[model] = {}
                if chosen_endpoint not in self.local_endpoints[model].keys():
                    self.local_endpoints[model][chosen_endpoint] = AutoModel.from_pretrained(model, device=chosen_endpoint)
                    self.tokenizer[model][chosen_endpoint] = AutoTokenizer.from_pretrained(model, device=chosen_endpoint, use_fast=True)
                query_response = await self.make_local_request(model, chosen_endpoint, samples)
                knn_stack = query_response
                return knn_stack
            else:
                try:
                    if model not in self.tokenizer.keys():
                        self.tokenizer[model] = {}
                    if "cpu" not in self.tokenizer[model].keys():
                        self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True)
                        pass
                    if len(samples) > 1:
                        raise ValueError("samples must be a list of one item")
                    inputs = []
                    for sample in samples:
                        max_length = 0
                        for resource in self.resources["tei_endpoints"]:
                            if model in resource and chosen_endpoint in resource:                            
                                max_length = resource[2]
                        input = self.tokenizer[model]["cpu"](sample, max_length=max_length, truncation=True, return_tensors='pt')
                        for item in list(input.keys()):
                            data = input[item].tolist()
                            data_len = len(data[0])
                            this_input = {
                                "name": item,
                                "shape": [1, data_len],
                                "datatype": "INT64",
                                "data": data
                            }
                            inputs.append(this_input)
                    data = {"inputs": inputs}
                    query_response = await self.make_post_request_openvino(chosen_endpoint, data)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                query_response_outputs = query_response["outputs"]
                data = query_response_outputs[0]
                vectors = data["data"]
                knn_stack = [vectors]
            pass
        return knn_stack
    
    async def make_local_request(self, model, endpoint, endpoint_type, data):
        device = torch.device(endpoint)
        inputs = self.tokenizer[model][endpoint](data, return_tensors="pt", padding=True, truncation=True).to(device)
        self.local_endpoints[model][endpoint].to(device).eval()
        with torch.no_grad():
            outputs = self.local_endpoints[model][endpoint](**inputs)
            query_response = outputs.last_hidden_state.mean(dim=1).tolist()  # Use mean of token embeddings
            results = query_response  # Return the entire batch of results
            del inputs, outputs  # Unallocate inputs and outputs
            torch.cuda.synchronize()  # Ensure all operations are complete
            torch.cuda.empty_cache()  # Free up GPU memory
        # self.local_endpoints[model][endpoint].to('cpu')  # Move model back to CPU
        torch.cuda.empty_cache()  # Free up GPU memory again
        return results

    async def make_local_request(self, model, endpoint, data):
        device = torch.device(endpoint)
        inputs = self.tokenizer[model][endpoint](data, return_tensors="pt", padding=True, truncation=True).to(device)
        self.local_endpoints[model][endpoint].to(device).eval()
        with torch.no_grad():
            outputs = self.local_endpoints[model][endpoint](**inputs)
            query_response = outputs.last_hidden_state.mean(dim=1).tolist()  # Use mean of token embeddings
            results = query_response  # Return the entire batch of results
            del inputs, outputs  # Unallocate inputs and outputs
            torch.cuda.synchronize()  # Ensure all operations are complete
            torch.cuda.empty_cache()  # Free up GPU memory
        # self.local_endpoints[model][endpoint].to('cpu')  # Move model back to CPU
        torch.cuda.empty_cache()  # Free up GPU memory again
        return results

    async def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")

    async def make_post_request_libp2p(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")
            
            
    async def make_post_request_openvino(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")

    def choose_endpoint(self, model, endpoint_type=None):
        if endpoint_type != None:
            this_endpoints = self.get_endpoints(model, endpoint_type)
        else:
            tei_endpoints = self.get_endpoints(model, endpoint_type="tei")
            libp2p_endpoints = self.get_endpoints(model, endpoint_type="libp2p")
            openvino_endpoints = self.get_endpoints(model, endpoint_type="openvino")
            local_endpoints = self.get_endpoints(model, endpoint_type="local")
            filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
            filtered_tei_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and tei_endpoints is not None and k in list(tei_endpoints.keys())}
            filtered_openvino_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and openvino_endpoints is not None and k in list(openvino_endpoints.keys())}
            filtered_local_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and local_endpoints is not None and k in list(local_endpoints.keys())}
            if not filtered_tei_endpoints and not filtered_libp2p_endpoints and not filtered_openvino_endpoints and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                if len(list(filtered_local_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_local_endpoints.keys()))
                if len(list(filtered_tei_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_tei_endpoints.keys()))
                if len(list(filtered_openvino_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_openvino_endpoints.keys()))
                elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_libp2p_endpoints.keys()))
                print("chosen endpoint for " + model + " is " + this_endpoint)
                return this_endpoint


    def get_endpoints(self, model, endpoint_type=None):
        if endpoint_type is None:
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "tei":
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "openvino":
            endpoints_dict = self.openvino_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "libp2p":
            endpoints_dict = self.libp2p_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "local":
            endpoints_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "cuda":
            endpoint_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoint_dict if "cuda" in endpoint and self.endpoint_status.get(endpoint, 0) >= 1]
        return filtered_endpoints

    async def async_generator(self, iterable):
        for item in iterable:
            yield item

    async def consumer(self, queue, column, batch_size, model_name, endpoint):
        print("consumer started for model " + model_name + " at endpoint " + endpoint)
        self.consumer_task_done[(model_name, endpoint)] = False
        batch = []
        if model_name not in self.caches.keys():
            self.caches[model_name] = {"items" : []}
        if model_name not in self.index.keys():
            self.index[model_name] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                for i in range(len(results)):
                    self.caches[model_name]["items"].append({"cid": batch[i]["cid"], "embedding": results[i]})
                batch = []  # Clear batch after sending
                self.saved = False
            queue.task_done()
            if self.producer_task_done and queue.empty():
                self.consumer_task_done[(model_name, endpoint)] = True
                break
        return None

    async def chunk_producer(self, dataset_stream, column, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None):
        chunk_tasks = []
        async for item in self.async_generator(dataset_stream):
            chunked_item = await self.chunk_item(item, column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model)
            if chunked_item["parent_cid"] not in self.cid_chunk_set:
                while self.cid_chunk_queue.full():
                    await asyncio.sleep(0.1)
                if not self.cid_chunk_queue.full():
                    self.cid_chunk_queue.put_nowait(chunked_item)
                    pass
        return None

    async def chunk_consumer(self, batch_size, model_name, endpoint):
        print("chunk consumer started for endpoint " + endpoint + " and model " + model_name)
        while True:
            test_ready = all([
                "cid_chunk_queue" in dir(self),
                "empty" in dir(self.cid_chunk_queue),
                not self.cid_chunk_queue
            ])
            while not test_ready:
                await asyncio.sleep(1)
                pass     
            chunked_item = await self.cid_chunk_queue.get()
            batch_results = []
            batch = []
            chunk_data = []
            if chunked_item is not None:
                for item in chunked_item["items"]:
                    batch.append(item)
                    chunk_data.append(item)
                    if len(batch) >= batch_size or len(batch) == len(chunked_item["items"]):
                        results = await self.send_batch_to_endpoint(batch, "content", model_name, endpoint)
                        for i in range(len(results)):
                            batch_results.append({"cid": batch[i]["cid"], "index": chunk_data[i]["index"], "content": chunk_data[i]["content"] , "embedding": results[i]})
                        batch = []
                        chunk_data = []
            if len(batch_results) > 0:
                self.chunk_cache[chunked_item["parent_cid"]] = {"items": batch_results, "parent_cid": chunked_item["parent_cid"]}
                self.cid_chunk_set.add(chunked_item["parent_cid"])
                self.cid_chunk_list.append(chunked_item["parent_cid"])
                self.cid_chunk_queue.task_done()
                self.saved = False

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
    
    async def process_chunk(self, chunk, column, model_name, endpoint):
        chunk_results = await self.send_batch_to_endpoint(chunk, column, model_name, endpoint)
        return chunk_results
    
    
    async def chunk_item(self, item, column=None, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None):
        # Assuming `item` is a dictionary with required data
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
            if embed_model not in list(self.tokenizer.keys()):
                self.tokenizer[embed_model] = {}
            if "cpu" not in self.tokenizer[embed_model].keys():                
                self.tokenizer[embed_model]["cpu"] = AutoTokenizer.from_pretrained(embed_model, device='cpu', use_fast=True)
            else:
                tokenizer = self.tokenizer[embed_model]["cpu"]
        if method is None:
            fixed_chunk_list = self.chunker.chunk(content, self.tokenizer[embed_model]["cpu"], "fixed", 512, 8, 256, self.metadata["models"][0]) 
            semantic_chunk_list = self.chunker.chunk(content, self.tokenizer[embed_model]["cpu"], "semantic", 512, 8, 256, self.metadata["models"][0])
            sentences_chunk_list = self.chunker.chunk(content, self.tokenizer[embed_model]["cpu"], "sentences", 512, 8, 256, self.metadata["models"][0] )
            sliding_window_chunk_list = self.chunker.chunk(content, self.tokenizer[embed_model]["cpu"], "sliding_window", 512, 8, 256, self.metadata["models"][0])
            content_chunks = fixed_chunk_list + semantic_chunk_list + sentences_chunk_list + sliding_window_chunk_list
        else:
            content_chunks = self.chunker.chunk(content, tokenizer, method, chunk_size, n_sentences, step_size, embed_model)
        parent_cid = item["items"]["cid"]
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
            cid_chunks = {"items" : [], "parent_cid": parent_cid}
            for chunk in content_chunks:
                chunk_index = chunk
                chunk_content = content_tokens[chunk[0]:chunk[1]]
                chunk_text = tokenizer.decode(chunk_content)
                child_cid = self.multiformats.get_cid(chunk_text)
                child_content = {"cid": child_cid, "index": chunk_index, "content": chunk_text}
                cid_chunks["items"].append(child_content)
        return cid_chunks
        
    async def process_item(self, item, column=None, queues=None):
        # Assuming `item` is a dictionary with required data
        if "new_dataset" not in list(self.caches.keys()):
            self.caches["new_dataset"] = {"items" : []}
        # print(f"Processing item with CID {index_cid(item[column])[0]}")
        if queues is None:
            queues = self.queues
        column_names = item.keys()
        if column is None:
            this_cid = self.index_cid(json.dumps(item))[0]
        elif column not in column_names:
            this_cid = self.index_cid(json.dumps(item))[0]
        else:
            this_cid = self.index_cid(item[column])[0]
        if "cid" not in column_names:
            item["cid"] = this_cid
        elif item["cid"] is None:
            item["cid"] = this_cid
        # Check if cid is in index
        if this_cid in self.cid_set:
            # print(f"CID {this_cid} already in index, skipping item.")
            return None
        else:
            self.cid_set.add(this_cid)
            if this_cid not in self.all_cid_set["new_dataset"]:
                self.caches["new_dataset"]["items"].append(item)
                self.saved = False
            models = self.queues.keys()
            for model, model_queues in queues.items():
                if len(model_queues) > 0:
                    if this_cid not in self.all_cid_set[model]:
                        model_queue_lengths = {k: v.qsize() for k, v in model_queues.items()}
                        ## if all model queues are empty, choose 
                        if all(value == 0 for value in model_queue_lengths.values()):
                            chosen_queue = random.choice(list(model_queues.keys()))
                        else:
                            chosen_queue = min(model_queue_lengths, key=model_queue_lengths.get)
                        queue = model_queues[chosen_queue]          
                        # endpoint, queue = min(model_queues.items(), key=lambda x: x[1].qsize())
                        while queue.full():
                            await asyncio.sleep(0.1)
                        queue.put_nowait(item)  # Non-blocking put
            return item

    async def send_batch_to_endpoint(self, batch, column, model_name, endpoint):
        if "cuda" not in endpoint and "cpu" not in endpoint:
            print(f"Sending batch of size {len(batch)} to model {model_name} at endpoint {endpoint}")
            model_context_length = self.tei_endpoints[model_name][endpoint]
            new_batch = []
            if model_name not in self.tokenizer.keys():
                self.tokenizer[model_name] = {}
            if "cpu" not in self.tokenizer[model_name].keys():
                self.tokenizer[model_name]["cpu"] = AutoTokenizer.from_pretrained(model_name, device='cpu')
            for item in batch:
                if column in list(item.keys()):
                    this_item_tokens = len(self.tokenizer[model_name]["cpu"].encode(item[column]))
                    if this_item_tokens > model_context_length:
                        encoded_item = self.tokenizer[model_name]["cpu"](item[column], return_tensors="pt")["input_ids"].tolist()[0]
                        truncated_encoded_item = encoded_item[:model_context_length]
                        unencode_item = self.tokenizer[model_name]["cpu"].decode(truncated_encoded_item)
                        new_batch.append(unencode_item)
                    else:
                        new_batch.append(item[column])
            results = None
            try:
                results = await self.index_knn(new_batch, model_name, endpoint)
            except Exception as e:
                print(e)
                pass
                # raise e
            if isinstance(results, ValueError):
                error = results.args[0]
                strerror = None
                if "strerror" in dir(error):
                    strerror = error.strerror
                if "status" in dir(error):
                    if error.status == 413:
                        if error.reason == "Payload Too Large":
                            error_content = error.content._buffer[0].decode("utf-8")
                            error_content = json.loads(error_content)
                            if "error" in error_content.keys() and "error_type" in error_content.keys():
                                if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
                                    expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
                                    given = int(error_content["error"].split("Given: ")[1])
                                    difference = given - expected
                                    self.tei_endpoints[model_name][endpoint] = model_context_length - difference
                                    for item in new_batch:
                                        index = new_batch.index(item)
                                        item = { column : item[:self.tei_endpoints[model_name][endpoint]] }
                                        new_batch[index] = item
                                    results = await self.send_batch_to_endpoint(new_batch, column, model_name, endpoint)
                                    return results
                                if "Validation" in error_content["error_type"] and "cannot be empty":
                                    print("error: " + error_content["error"])
                                    return None
                    elif error.status == 504 or error.status == 502 or  "can not write request body" in str(error):
                        # self.endpoint_status[endpoint] = 0
                        new_endpoint = self.choose_endpoint(model_name)
                        if new_endpoint:
                            # new_queue = self.queues[model_name][new_endpoint]
                            # for item in batch:
                            #     await new_queue.put(item)
                            return await self.send_batch_to_endpoint(batch, column, model_name, new_endpoint)
                        else:
                            return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                    elif error.status == 400 or error.status == 404:
                        return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                elif "Can not write request body" in error.strerror or "Timeout" in error.strerror:
                    # self.endpoint_status[endpoint] = 0
                    new_endpoint = self.choose_endpoint(model_name)
                    if new_endpoint:
                        # new_queue = self.queues[model_name][new_endpoint]
                        # for item in batch:
                        #     await new_queue.put(item)
                        return await self.send_batch_to_endpoint(batch, column, model_name, new_endpoint)
                    else:
                        return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                raise Exception(error) 
            else:
                if results is None:
                    return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                print(f"Received embeddings for {len(results)} items from model {model_name} at endpoint {endpoint}")
                return results
        else:
            print(f"Sending batch of size {len(batch)} to model {model_name} at endpoint {endpoint}")
            model_context_length = round(self.local_endpoints[model_name][endpoint].config.max_position_embeddings * 0.99)
            new_batch = []
            if model_name not in self.tokenizer.keys():
                self.tokenizer[model_name] = {}
            if endpoint not in self.tokenizer[model_name].keys():
                self.tokenizer[model_name][endpoint] = AutoTokenizer.from_pretrained(model_name, device=endpoint)
            for item in batch:
                if column in list(item.keys()):
                    this_item_tokens = len(self.tokenizer[model_name][endpoint].encode(item[column]))
                    if this_item_tokens > model_context_length:
                        encoded_item = self.tokenizer[model_name][endpoint](item[column], return_tensors="pt")["input_ids"].tolist()[0]
                        truncated_encoded_item = encoded_item[:model_context_length]
                        unencode_item = self.tokenizer[model_name][endpoint].decode(truncated_encoded_item)
                        new_batch.append(unencode_item)
                    else:
                        new_batch.append(item[column])
            results = await self.make_local_request(model_name, endpoint, new_batch)
            print(f"Received embeddings for {len(results)} items from model {model_name} at endpoint {endpoint}")
            return results

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

    def status(self):
        return self.endpoint_status

    def setStatus(self, endpoint, status):
        self.endpoint_status[endpoint] = status
        return None
    
    async def test_llama_cpp(self):
        test_llama_cpp_cmd = "pip show llama-cpp-python"
        test_results = {}
        try:
            test_llama_cpp = subprocess.check_output(test_llama_cpp_cmd, shell=True)
            test_llama_cpp = test_llama_cpp.decode("utf-8")
            test_results["llama_cpp"] = test_llama_cpp
        except Exception as e:
            print(e)
            raise ValueError(e)
        try:
            test_ollama = subprocess.check_output("ollama", shell=True)
            test_ollama = test_ollama.decode("utf-8")
            test_results["ollama"] = test_ollama
        except Exception as e:
            print(e)
            raise ValueError(e)
        
        test_pass = False
        test_pass = all(isinstance(value, str) for value in test_results.values() if not isinstance(value, ValueError))
        return test_pass
        
    
    async def test_local_openvino(self):
        test_openvino_cmd = "python3 -c 'import openvino; print(openvino.__version__)'"
        try:
            test_openvino = subprocess.check_output(test_openvino_cmd, shell=True).decode("utf-8")              
            if type(test_openvino) == str and type(test_openvino) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_ipex(self):        
        test_ipex_cmd = 'python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__);"'
        try:
            test_ipex = subprocess.check_output(test_ipex_cmd, shell=True)
            return test_ipex
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_cuda(self):
        try:
            gpus = torch.cuda.device_count()
            if type(gpus) == int and type(gpus) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
    
    async def install_openvino(self):
        from install_depends import install_depends_py
        this_install_package = install_depends_py(self.resources, self.metadata)
        try:
            return await this_install_package.install_package("openvino")
        except Exception as e:
            print(e)
            return ValueError(e)    
    
    async def install_ipex(self):
        from install_depends import install_depends_py
        this_install_package = install_depends_py(self.resources, self.metadata)
        try:
            return await this_install_package.install_package("ipex")
        except Exception as e:
            print(e)
            return ValueError(e)    
    
    async def install_cuda(self): 
        from install_depends import install_depends_py
        this_install_package = install_depends_py(self.metadata,self.resources)
        try:
            results = await this_install_package.install_package("cuda")
        except Exception as e:
            print(e)
            return ValueError(e)    
        return results
    
    async def install_llama_cpp(self):
        from install_depends import install_depends_py
        this_install_package = install_depends_py(self.metadata,self.resources)
        try:
            results = await this_install_package.install_package("llama_cpp")
        except Exception as e:
            print(e)
            return ValueError(e)    
        return results
        
    async def test_hardware(self):
        cuda_test = None
        openvino_test = None
        llama_cpp_test = None
        ipex_test = None
        cuda_install = None
        openvino_install = None
        llama_cpp_install = None
        ipex_install = None
        
        try:
            openvino_test = await self.test_local_openvino()
        except Exception as e:
            openvino_test = e
            print(e)
            try:
                openvino_install = await self.install_openvino()
                try:
                    openvino_test = await self.test_local_openvino()
                except Exception as e:
                    openvino_test = e
                    print(e)
            except Exception as e:
                openvino_install = e
                print(e)        
            pass
            
        try:
            llama_cpp_test = await self.test_llama_cpp()
        except Exception as e:
            llama_cpp_test = e
            try:
                llama_cpp_install = await self.install_llama_cpp()
                try:
                    llama_cpp_test = await self.test_llama_cpp()
                except:
                    llama_cpp_test = e
            except Exception as e:
                print(e)
                llama_cpp_install = e
            pass
        try:
            ipex_test = await self.test_ipex()
        except Exception as e:
            ipex_test = e
            print(e)
            try:
                ipex_install = await self.install_ipex()
                try:
                    ipex_test = await self.test_ipex()
                except Exception as e:
                    ipex_test = e
                    print(e)
            except Exception as e:
                ipex_install = e
                print(e)
            pass
        try:
            cuda_test = await self.test_cuda()
        except Exception as e:
            try:
                cuda_install = await self.install_cuda()
                try:
                    cuda_test = await self.test_cuda()
                except Exception as e:
                    cuda_test = e
                    print(e)                    
            except Exception as e:
                cuda_install = e
                print(e)
            pass
                
        print("local_endpoint_test")
        install_results = {
            "cuda": cuda_install,
            "openvino": openvino_install,
            "llama_cpp": llama_cpp_install,
            "ipex": ipex_install
        }
        print(install_results)
        test_results = {
            "cuda": cuda_test,
            "openvino": openvino_test,
            "llama_cpp": llama_cpp_test,
            "ipex": ipex_test
        }
        print(test_results)
        return test_results
    
    async def init_endpoints(self, models=None, endpoint_list=None):
        for endpoint_type in self.endpoint_types:
            if endpoint_type in resources.keys():
                for endpoint_info in resources[endpoint_type]:
                    model, endpoint, context_length = endpoint_info
                    await self.add_endpoint(model, endpoint, context_length, endpoint_type)    
        if "hwtest" not in dir(self):
            hwtest = await self.test_hardware()
            self.hwtest = hwtest
        for model in models:
            if model not in self.queues:
                self.queues[model] = {}
        if endpoint_list is None:    
            endpoints = self.get_endpoints(model)
            local = self.get_endpoints(model, "local")
            openvino = self.get_endpoints(model, "openvino")
            libp2p = self.get_endpoints(model, "libp2p")
            tei = self.get_endpoints(model, "tei")
            cuda = self.get_endpoints(model, "cuda")
            endpoints =  { "tei" : tei , "local" : local , "openvino": openvino , "libp2p": libp2p , "cuda": cuda }
            endpoints_set = set(endpoints["tei"] + endpoints["local"] + endpoints["openvino"] + endpoints["libp2p"] + endpoints["cuda"])
        else:
            endpoints = endpoint_list
            self.endpoints = endpoints
            endpoints_list = [ k for k in endpoints.keys() ]
            self.endpoints_list = endpoints_list
            self.endpoints = { k : v for k, v in enumerate(endpoints_list) }
            endpoints_set = set(endpoints_list)
            self.endpoint_set = endpoints_set
            local = [ endpoint for endpoint in endpoints["local_endpoints"] if "local" in endpoint or "cpu" in endpoint or "cuda" in endpoint or "openvino" in endpoint or "llama_cpp" in endpoint or "ipex" in endpoint]
            openvino = [endpoint for endpoint in endpoints["openvino_endpoints"] ]
            libp2p = []
            tei = [endpoint for endpoint in endpoints["tei_endpoints"]]
            pass
        if type(endpoint_list) == list:
            self.endpoints = { k : v for k, v in enumerate(endpoint_list) }
            endpoints = endpoint_list
            self.endpoint_list = endpoint_list
            endpoints_set = set(endpoints)
            self.endpoint_set = endpoints_set
        if type(endpoint_list) == dict:
            self.endpoints = endpoint_list
            endpoints_list = list(endpoint_list.keys())
            endpoints_set = set(endpoints_list)
            self.endpoint_set = endpoints_set
        if not endpoints_set:
            raise ValueError("No endpoints available for model " + model)
        cuda_test = self.hwtest["cuda"]
        openvino_test = self.hwtest["openvino"]
        llama_cpp_test = self.hwtest["llama_cpp"]
        ipex_test = self.hwtest["ipex"]
        cpus = os.cpu_count()
        cuda = torch.cuda.is_available()
        gpus = torch.cuda.device_count()
        for model in models:
            if model not in self.tokenizer:
                self.tokenizer[model] = {}
            if model not in self.local_endpoints:
                self.local_endpoints[model] = {}
            if model not in self.queues:    
                self.queues[model] = {}
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if "cpu" not in self.local_endpoints[model]:
                self.local_endpoints[model]["cpu"] = ""
            if "cpu" not in self.queues[model]:
                self.queues[model]["cpu"] = ""
            if "cpu" not in self.batch_sizes[model]:
                self.batch_sizes[model]["cpu"] = 1
        for model in models:
            if cuda and gpus > 0:
                if cuda_test and type(cuda_test) != ValueError:
                    self.local_endpoints[model] = {"cuda:" + str(gpu) : None for gpu in range(gpus) } if gpus > 0 else {"cpu": None}
                    for gpu in range(gpus):
                        self.tokenizer[model]["cuda:" + str(gpu)] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                        self.local_endpoints[model]["cuda:" + str(gpu)] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                        torch.cuda.empty_cache()
                        self.queues[model]["cuda:" + str(gpu)] = asyncio.Queue(64)
                        batch_size = await self.max_batch_size(model, "cuda:" + str(gpu))
                        self.local_endpoints[model]["cuda:" + str(gpu)] = batch_size
                        self.batch_sizes[model]["cuda:" + str(gpu)] = batch_size
                        self.endpoint_handler[(model, "cuda:" + str(gpu))] = ""
                        # consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.chunk_consumer(batch_size, model, "cuda:" + str(gpu)))
            if len(local) > 0 and cpus > 0:
                all_test_types = [ type(openvino_test), type(llama_cpp_test), type(ipex_test)]
                all_tests_ValueError = all(x is ValueError for x in all_test_types)
                all_tests_none = all(x is None for x in all_test_types)
                if all_tests_ValueError or all_tests_none:  
                    self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                    self.queues[model]["cpu"] = asyncio.Queue(4)
                    self.endpoint_handler[(model, "cpu")] = ""
                    # consumer_tasks[(model, "cpu")] = asyncio.create_task(self.chunk_consumer( 1, model, "cpu"))
                elif openvino_test and type(openvino_test) != ValueError:
                    ov_count = 0
                    for endpoint in local:
                        if "openvino" in endpoint[1]:
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
                                self.queues[model][endpoint_name] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint_name )] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                            ov_count = ov_count + 1
                elif llama_cpp_test and type(llama_cpp_test) != ValueError:
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
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                            llama_count = llama_count + 1
                elif ipex_test and type(ipex_test) != ValueError:
                    ipex_count = 0
                    for endpoint in local:
                        if "ipex" in endpoint:
                            endpoint_name = "ipex:"+str(ipex_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            ipex_count = ipex_count + 1
            if "openvino_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["openvino_endpoints"]) > 0 :
                    for endpoint in self.endpoints["openvino_endpoints"]:
                        batch_size = 0
                        if model not in self.batch_sizes:
                            self.batch_sizes[model] = {}
                        if model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][endpoint] = batch_size
                        if self.batch_sizes[model][endpoint] > 0:
                            self.queues[model][endpoint] = asyncio.Queue(64)  # Unbounded queue
                            self.endpoint_handler[(model, endpoint)] = ""
                            # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
            if "tei_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["tei_endpoints"]) > 0:
                    for endpoint in self.endpoints["tei_endpoints"]:
                        this_model = endpoint[0]
                        this_endpoint = endpoint[1]
                        context_length = endpoint[2]
                        batch_size = 0
                        if this_model not in self.batch_sizes:
                            self.batch_sizes[this_model] = {}
                        if this_model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][this_endpoint] = batch_size
                        if self.batch_sizes[model][this_endpoint] > 0:
                            self.queues[model][this_endpoint] = asyncio.Queue(64)  # Unbounded queue
                            self.endpoint_handler[(model, this_endpoint)] = ""
                            # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint)) 
            if "libp2p_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["libp2p_endpoints"]) > 0:
                    for endpoint in self.endpoints["libp2p_endpoints"]:
                        batch_size = 0
                        if model not in self.batch_sizes:
                            self.batch_sizes[model] = {}
                        if model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][endpoint] = batch_size
                        if self.batch_sizes[model][endpoint] > 0:
                            self.queues[model][endpoint] = asyncio.Queue(64)
        return None

    async def index_sparse_chunks(self, dataset, split, column, dst_path, models = None):
        self.queues = {}
        self.cid_set = set()
        self.all_cid_list = {}
        consumer_tasks = {}
        batch_sizes = {}
        resource_keys = list(self.resources.keys())
        endpoints = {}
        for resource in resource_keys:
            if "endpoints" in resource:
                endpoints[resource] = self.resources[resource]
        await self.load_clusters(dataset, split, dst_path)
        await self.init_endpoints(models, endpoints)
        if split is None:
            if "new_dataset" not in list(self.all_cid_set.keys()):
                self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
            else:
                self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
            columns = self.dataset.column_names
            columns.append("cid")
            await self.load_checkpoints( dataset, split, dst_path, models)
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        await self.load_checkpoints( dataset, split, dst_path, models)       
        for model, endpoint in self.endpoints:
            for endpoint in self.endpoints[model]:
                consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, self.batch_sizes[model][endpoint], model, endpoint))
            consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, self.batch_sizes[model][endpoint], model, endpoint))
        producer_task = asyncio.create_task(self.chunk_producer(self.dataset, column, self.queues))        
        save_task = asyncio.create_task(self.save_chunks_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, *consumer_tasks.values(), save_task)
        self.save_chunks_to_disk(dataset, dst_path, models)
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
    
    async def index_dataset_bak(self, dataset, split, column, dst_path, models = None):
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
        for model in models:
            endpoints = self.get_endpoints(model)
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if not endpoints:
                ## get gpus for local models
                if model not in self.tokenizer.keys():
                    self.tokenizer[model] = {}
                gpus = torch.cuda.device_count()
                self.local_endpoints[model] = {"cuda:" + str(gpu) : None for gpu in range(gpus) } if gpus > 0 else {"cpu": None}
                if gpus > 0:
                    for gpu in range(gpus):
                        self.tokenizer[model]["cuda:" + str(gpu)] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                        self.local_endpoints[model]["cuda:" + str(gpu)] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                        torch.cuda.empty_cache()  # Free up unused memory
                        self.queues[model]["cuda:" + str(gpu)] = asyncio.Queue(4)
                        batch_size = await self.max_batch_size(model, "cuda:" + str(gpu))
                        self.batch_sizes[model]["cuda:" + str(gpu)] = batch_size
                        consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.consumer(self.queues[model]["cuda:" + str(gpu)], column, batch_size, model, "cuda:" + str(gpu)))
                else:
                    self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                    self.queues[model]["cpu"] = asyncio.Queue()
                    consumer_tasks[(model, "cpu")] = asyncio.create_task(self.consumer(self.queues[model]["cpu"], column, 1, model, "cpu"))
            else:
                for endpoint in endpoints:
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
        # Compute commonn
        self.cid_set = set.intersection(*self.all_cid_set.values())
        producer_task = asyncio.create_task(self.producer(self.dataset, column, self.queues))        
        save_task = asyncio.create_task(self.save_checkpoints_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, *consumer_tasks.values(), save_task)
        self.save_checkpoints_to_disk(dataset, dst_path, models)
        return None 
    
    async def load_combined_checkpoints(self, dataset, split, dst_path, models):
        if "new_dataset" not in list(dir(self)):
            self.new_dataset = None
        if "all_cid_list" not in list(dir(self)):
            self.all_cid_list = {}
        if "all_cid_set" not in list(dir(self)):
            self.all_cid_set = {}
        for model in models:
            if model not in list(self.index.keys()):
                self.index[model] = None
        if self.new_dataset is None or isinstance(self.new_dataset, dict):
            new_dataset_dst_path = os.path.join(dst_path, "ipfs_" + dataset.replace("/","___") + ".parquet")
            if os.path.isfile(new_dataset_dst_path):
                self.new_dataset = load_dataset('parquet', data_files=new_dataset_dst_path)[split]
                self.all_cid_list["new_dataset"] = []
                self.all_cid_set["new_dataset"] = set()
            if os.path.exists(os.path.join(dst_path, "checkpoints")):
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                new_dataset_shards = [os.path.join(dst_path, "checkpoints", x) for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x ]
                if "new_dataset" not in list(self.all_cid_list.keys()):
                    self.all_cid_list["new_dataset"] = []
                if "new_dataset" not in list(self.all_cid_set.keys()):
                    self.all_cid_set["new_dataset"] = set()
                if "new_dataset" not in list(self.caches.keys()):
                    self.caches["new_dataset"] = {"items" : []}
                with multiprocessing.Pool() as pool:
                    args = [[new_dataset_shards[i], 'cids'] for i in range(len(new_dataset_shards))]
                    results = pool.map(self.process_new_dataset_shard, args)
                    if len(results) > 0:
                        # Initialize accumulators
                        total_cids = []
                        total_items = []
                        for res in results:
                            cid, items, schemas = (res + [None, None, None])[:3]
                            if cid is not None:
                                total_cids += cid
                            if items is not None:
                                total_items += items
                            if schemas is not None:
                                self.schemas["new_dataset"] = schemas
                        # Update the shared variables in bulk
                        
                        self.all_cid_list["new_dataset"] += total_cids
                        self.all_cid_set["new_dataset"].update(set(total_cids))
                        self.caches["new_dataset"]["items"] += total_items
                        
                if self.new_dataset is None or isinstance(self.new_dataset, dict):
                    if len(new_dataset_shards) > 0:
                        self.new_dataset = load_dataset('parquet', data_files=new_dataset_shards)[split]
                        
        for model in models:
            if model not in list(self.index.keys()):
                self.index[model] = None
            if model not in list(self.all_cid_list.keys()):
                self.all_cid_list[model] = []
            if model not in list(self.all_cid_set.keys()):
                self.all_cid_set[model] = set()
            if model not in list(self.caches.keys()):
                self.caches[model] = {"items" : []}
            model_dst_path = dst_path + "/" + model.replace("/","___") + ".parquet"
            if os.path.isfile(model_dst_path):
                self.caches[model] = {"items" : []}
                self.index[model] = load_dataset('parquet', data_files=model_dst_path, streaming=True)[split]
            if os.path.exists(os.path.join(dst_path, "checkpoints")):
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                this_model_shards = [os.path.join(dst_path, "checkpoints", x)  for x in ls_checkpoints if model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                with multiprocessing.Pool() as pool:
                    args = [[new_dataset_shards[i], 'cids'] for i in range(len(new_dataset_shards))]
                    results = pool.map(self.process_new_dataset_shard, args)
                    if len(results) > 0:
                        # Initialize accumulators
                        total_cids = []
                        total_items = []
                        for res in results:
                            cid, items, schemas = (res + [None, None, None])[:3]
                            if cid is not None:
                                total_cids += cid
                            if items is not None:
                                total_items += items
                            if schemas is not None:
                                self.schemas[model] = schemas
                        # Update the shared variables in bulk
                        self.all_cid_list[model] += total_cids
                        self.all_cid_set[model].update(set(total_cids))
                        self.caches[model]["items"] += total_items
                        
                if model not in list(self.index.keys()) or self.index[model] is None or isinstance(self.index[model], dict):
                    if len(this_model_shards) > 0:
                        self.index[model] = load_dataset('parquet', data_files=this_model_shards)[split]
                    else:
                        self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": [] })
                ls_chunks = []
                if os.path.exists(os.path.join(dst_path,"sparse_chunks", )):
                    ls_chunks = os.listdir(os.path.join(dst_path,"sparse_chunks", ))
                if len(ls_chunks) > 0:
                    for this_cid in ls_chunks:
                        this_cid_path = os.path.join(dst_path,"sparse_chunks", this_cid)
                        this_cid_dataset = load_dataset('parquet', data_files=this_cid_path)
                        if this_cid not in self.chunk_cache.keys():
                            self.chunk_cache[this_cid] = {"items": []}
                        self.chunk_cache[this_cid]["items"] += this_cid_dataset
                        self.cid_chunk_set.add(this_cid)
                        self.cid_chunk_list.append(this_cid)
        return None
        
    async def load_chunk_checkpoints(self, dataset, split, src_path, models):
        files = []
        if "doc_cid" not in list(dir(self)):
            self.chunks = {}
        if "doc_cid" not in list(dir(self)):
            self.chunk_cache_set = {}
        if os.path.isdir(src_path):
            files = os.listdir(src_path)
            files_by_models = [ [x for x in files if model.replace("/","___") in x and dataset in x and models in x ] for model in models]
        if len(files_by_models) > 0:
            with multiprocessing.Pool() as pool:
                results = pool.map(self.process_chunk_file, files_by_models)
                for result in results:
                    model, doc_cid, items = result
                    if model not in list(self.chunk_cache.keys()):
                        self.chunk_cache_set[model] = set()
                    if doc_cid not in list(self.chunk_cache[model].keys()):
                        self.chunk_cache[model][doc_cid] = {"items": []}
                    if doc_cid not in self.chunk_cache_set[model]:
                        self.chunk_cache_set[model].add(doc_cid)
                    self.doc_cid[model][doc_cid]["items"] += items
                    
        return None
    
    
    async def load_checkpoints(self, dataset, split, dst_path, models):
        if "new_dataset" not in list(dir(self)):
            self.new_dataset = None
        if "all_cid_list" not in list(dir(self)):
            self.all_cid_list = {}
        if "all_cid_set" not in list(dir(self)):
            self.all_cid_set = {}
        for model in models:
            if model not in list(self.index.keys()):
                self.index[model] = None
        if self.new_dataset is None or isinstance(self.new_dataset, dict):
            new_dataset_dst_path = os.path.join(dst_path, "ipfs_" + dataset.replace("/","___") + ".parquet")
            if os.path.isfile(new_dataset_dst_path):
                self.new_dataset = load_dataset('parquet', data_files=new_dataset_dst_path)[split]
                self.all_cid_list["new_dataset"] = []
                self.all_cid_set["new_dataset"] = set()
            if os.path.exists(os.path.join(dst_path, "checkpoints")):
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                new_dataset_shards = [os.path.join(dst_path, "checkpoints", x) for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x ]
                if "new_dataset" not in list(self.all_cid_list.keys()):
                    self.all_cid_list["new_dataset"] = []
                if "new_dataset" not in list(self.all_cid_set.keys()):
                    self.all_cid_set["new_dataset"] = set()
                if "new_dataset" not in list(self.caches.keys()):
                    self.caches["new_dataset"] = {"items" : []}
                with multiprocessing.Pool() as pool:
                    args = [[new_dataset_shards[i], 'cids'] for i in range(len(new_dataset_shards))]
                    results = pool.map(self.process_new_dataset_shard, args)
                    if len(results) > 0:
                        # Initialize accumulators
                        total_cids = []
                        total_items = []
                        for res in results:
                            cid, items, schemas = (res + [None, None, None])[:3]
                            if cid is not None:
                                total_cids += cid
                            if items is not None:
                                total_items += items
                            if schemas is not None:
                                self.schemas["new_dataset"] = schemas  # Assuming schemas won't conflict
                        # Update the shared variables in bulk
                        self.all_cid_list["new_dataset"] += total_cids
                        self.all_cid_set["new_dataset"].update(set(total_cids))
                        self.caches["new_dataset"]["items"] += total_items
                                            
                if self.new_dataset is None or isinstance(self.new_dataset, dict):
                    if len(new_dataset_shards) > 0:
                        self.new_dataset = load_dataset('parquet', data_files=new_dataset_shards)[split]
        
        for model in models:
            if model not in list(self.index.keys()):
                self.index[model] = None
            if model not in list(self.all_cid_list.keys()):
                self.all_cid_list[model] = []
            if model not in list(self.all_cid_set.keys()):
                self.all_cid_set[model] = set()
            if model not in list(self.caches.keys()):
                self.caches[model] = {"items" : []}
            model_dst_path = dst_path + "/" + model.replace("/","___") + ".parquet"
            if os.path.isfile(model_dst_path):
                self.caches[model] = {"items" : []}
                self.index[model] = load_dataset('parquet', data_files=model_dst_path, streaming=True)[split]
            if os.path.exists(os.path.join(dst_path, "checkpoints")):
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                this_model_shards = [os.path.join(dst_path, "checkpoints", x)  for x in ls_checkpoints if model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                with multiprocessing.Pool() as pool:
                    args = [[new_dataset_shards[i], 'cids'] for i in range(len(new_dataset_shards))]
                    results = pool.map(self.process_new_dataset_shard, args)
                    if len(results) > 0:
                        # Initialize accumulators
                        total_cids = []
                        total_items = []
                        for res in results:
                            cid, items, schemas = (res + [None, None, None])[:3]
                            if cid is not None:
                                total_cids += cid
                            if items is not None:
                                total_items += items
                            if schemas is not None:
                                self.schemas[model] = schemas  # Assuming schemas won't conflict
                        # Update the shared variables in bulk
                        self.all_cid_list[model] += total_cids
                        self.all_cid_set[model].update(set(total_cids))
                        self.caches[model]["items"] += total_items
        
                if model not in list(self.index.keys()) or self.index[model] is None or isinstance(self.index[model], dict):
                    if len(this_model_shards) > 0:
                        self.index[model] = load_dataset('parquet', data_files=this_model_shards)[split]
                    else:
                        self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": [] })
                ls_chunks = []
                if os.path.exists(os.path.join(dst_path,"sparse_chunks", )):
                    ls_chunks = os.listdir(os.path.join(dst_path, "sparse_chunks"))
                    for chunk in ls_chunks:
                        chunk_cid = chunk.replace(".parquet","")
                        if chunk.replace(".parquet","") not in self.cid_chunk_set:
                            self.cid_chunk_set.add(chunk_cid)
                            self.cid_chunk_list.append(chunk_cid)
                for chunk in ls_chunks:
                    chunk_cid = chunk.replace(".parquet","")
                    if chunk.replace(".parquet","") not in self.cid_chunk_set:
                        self.cid_chunk_set.add(chunk_cid)
                        self.cid_chunk_list.append(chunk_cid)
                del ls_chunks
                del this_model_shards
                del ls_checkpoints
        try:
            del new_dataset_shards
        except:
            pass
        self.cid_set = set.intersection(*self.all_cid_set.values())
        self.cid_list = list(self.cid_set)
        return None
    
    async def load_combined(self, dataset, split, dst_path, models):
        
        return None
    
    async def search_chunks(self, dataset, split, src_path, model, cids, query, endpoint=None, n=64):
        chunks = []
        results = []
        chunk_cid_list = []
        chunk_cid_set = set()
        if endpoint is None:
            endpoint = self.get_endpoints(model)
        if endpoint is None:
            raise ValueError("No endpoint available for model " + model)
        files = [ x for x in os.listdir(src_path) if model.replace("/","___") in x and dataset in x and cids in x ] 
        for chunk in files:
            chunk_cid = chunk.replace(".parquet","")
            if chunk_cid not in chunk_cid_set:
                chunk_cid_set.add(chunk_cid)
                if chunk_cid not in chunk_cid_list:
                    chunk_cid_list.append(chunk_cid)                
        for chunk_cid in chunk_cid_list:
            if chunk_cid not in self.chunk_cache.keys():
                self.chunk_cache[chunk_cid] = {"items": []}
        for chunk in chunk_cid_list:
            chunk_path = os.path.join(src_path, chunk)
            with multiprocessing.Pool() as pool:
                args = [[chunk_path]]
                results = pool.map(self.process_chunk_file, args)
                for result in results:
                    chunk_dataset = result
                    if "cid" in list(chunk_dataset.keys()):
                        if chunk_cid not in self.chunk_cache.keys():
                            self.chunk_cache[chunk_cid] = {"items": []}
                        self.chunk_cache[chunk_cid]["items"] += chunk_dataset["items"]
                    if "embeddings" in list(chunk_dataset.keys()):
                        if chunk_cid not in self.chunk_cache.keys():
                            self.chunk_cache[chunk_cid] = {"items": []}
                        self.chunk_cache[chunk_cid]["items"] += chunk_dataset["items"]
                    chunks.append(chunk_dataset)
            self.chunk_cache[chunk_cid]["items"] += chunk_dataset
        if "items" in list(chunks.keys()):
            vectors = [ x["items"]["embeddings"] for x in chunks if "embeddings" in list(x["items"].keys())]
            cids = [ x["items"]["cid"] for x in chunks if "cid" in list(x["items"].keys())]
            text = [ x["items"]["text"] for x in chunks if "text" in list(x["items"].keys())]
        else:
            vectors = [x["embeddings"] for x in chunks if "embeddings" in list(x.keys())]
            cids = [ x["cid"] for x in chunks if "cid" in list(x.keys())]
            text = [ x["text"] for x in chunks if "text" in list(x.keys())]
        if query is not None:
            query_vector = self.tokenizer[model][endpoint].encode(query)
        else:
            query_test = "the lazy dog jumped over the quick brown fox"
            query_vector = self.tokenizer[model][endpoint].encode(query_test)
            
        faiss.SearchParameters = faiss.SearchParameters()
        faiss.SearchParameters.init()
        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(np.array(vectors))
        D, I = index.search(np.array(query_vector), n)
        for i in I:
            results_keys = ["cid", "text", "embeddings"]
            result = {}
            for key in results_keys:
                if key == "cid":
                    result[key] = cids[i]
                elif key == "text":
                    result[key] = text[i]
                elif key == "embeddings":
                    result[key] = vectors[i]
            
            results.append(result)

        return results

            
    async def search_centroids(self, dataset, split, src_path, model, cids, query, endpoint=None, n=64):


        return None
    
    async def search_shards(self, dataset, split, src_path, models):
        
        
        return None
    
    async def autofaiss_chunks(self, dataset, split, src_path, models):
        
        return None
    
    async def autofaiss_shards(self, dataset, split, src_path, models):
        
        return None
    
    
    def demux_checkpoints_old4(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_item in this_dataset:
            item = this_item["items"]
            if "cid" in list(item.keys()):
                if item["cid"] not in self.unique_cid_set:
                    self.unique_cid_set.add(item["cid"])
                    self.unique_cid_list.append(item["cid"])
                    yield item
            else:
                continue
        
    def demux_checkpoints_old3(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_item in this_dataset:
            item = this_item["items"]
            if "cid" in list(item.keys()):
                del item["cid"]
            yield item
                
    def demux_checkpoints_old2(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_item in this_dataset:
            item = this_item["items"]
            if "cid" in list(item.keys()):
                if item["cid"] not in self.unique_cid_set:
                    del item["cid"]
                    item["cid"] = item["secondary_cid"]
                    del item["secondary_cid"]
                    self.unique_cid_set.add(item["cid"])
                    self.unique_cid_list.append(item["cid"])
                    yield item
            else:
                continue
            
    def demux_checkpoints_old(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_item in this_dataset:
            item = this_item["items"]
            if "cid" in list(item.keys()):
                if item["cid"] not in self.unique_cid_set:
                    del item["secondary_cid"]
                    self.unique_cid_set.add(item["cid"])
                    self.unique_cid_list.append(item["cid"])
                    yield item
            else:
                continue
            
    def demux_checkpoints4(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_cid in self.cid_list:
            alernate_index = self.new_dataset.select([this_cid])
            for this_item in alernate_index:
                item = this_item["items"]
                secondary_cid = item["secondary_cid"]

            dataset_index = self.all_cid_list[this_dataset].index(this_cid)
            dataset_item = self.index[this_dataset].select([dataset_index])
            for this_item in dataset_item:
                item = this_item["items"]
                item["cid"] = secondary_cid
                self.unique_cid_list.append(item["cid"])
                self.unique_cid_set.add(item["cid"])
                yield item
            
    def demux_checkpoints3(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_cid in self.cid_list:
            dataset_index = self.all_cid_list[this_dataset].index(this_cid)
            dataset_item = self.index[this_dataset].select([dataset_index])
            for this_item in dataset_item:
                item = this_item["items"]
                self.unique_cid_list.append(item["cid"])
                self.unique_cid_set.add(item["cid"])
                yield item
    
    def demux_checkpoints2(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_cid in self.cid_list:
            dataset_index = self.all_cid_list["new_dataset"].index(this_cid)
            dataset_item = self.new_dataset.select([dataset_index])
            for this_item in dataset_item:
                item = this_item["items"]
                self.unique_cid_list.append(item["cid"])
                self.unique_cid_set.add(item["cid"])
                del item["cid"]
                item["cid"] = item["secondary_cid"]
                del item["secondary_cid"]
                yield item
        
    def demux_checkpoints(self, this_dataset):
        self.unique_cid_set = set()
        self.unique_cid_list = []
        for this_cid in self.cid_list:
            dataset_index = self.all_cid_list["new_dataset"].index(this_cid)
            dataset_item = self.new_dataset.select([dataset_index])
            for this_item in dataset_item:
                item = this_item["items"]
                self.unique_cid_list.append(item["cid"])
                self.unique_cid_set.add(item["cid"])
                if "secondary_cid" in list(item.keys()):
                    del item["secondary_cid"]
                yield item
                
    async def combine_checkpoints(self, dataset, split, column, dst_path, models):
        await self.load_dataset(dataset, split)
        await self.load_checkpoints(dataset, split, dst_path, models)
        if not os.path.exists(os.path.join(dst_path, "combined")):
            os.makedirs(os.path.join(dst_path, "combined"))
        del self.dataset
        self.new_dataset_combined = {}
        self.embedding_datasets = {}
        ## get first row from self.new_datasets
        self.unique_cid_set = set()
        self.unique_cid_list = []
        if not os.path.exists(os.path.join(dst_path, "combined", "rm_secondary_cid_" + dataset.replace("/","___") + ".parquet")):
            self.new_dataset_combined = datasets.Dataset.from_generator(lambda: self.demux_checkpoints(self.new_dataset))            
            self.new_dataset_combined.to_parquet(os.path.join(dst_path, "combined",  "rm_secondary_cid_" + dataset.replace("/","___") + ".parquet"))
            combined_dataset_cids = datasets.Dataset.from_dict({"cids": self.unique_cid_list})
            combined_dataset_cids.to_parquet(os.path.join(dst_path, "combined", "rm_secondary_cid_" + "ipfs_" + dataset.replace("/","___") + "_cids.parquet"))

        if not os.path.exists(os.path.join(dst_path, "combined", "rm_cid_" + dataset.replace("/","___") + ".parquet")):
            self.new_dataset_combined = datasets.Dataset.from_generator(lambda: self.demux_checkpoints2(self.new_dataset))            
            self.new_dataset_combined.to_parquet(os.path.join(dst_path, "combined", "rm_cid_" + dataset.replace("/","___") + ".parquet"))
            combined_dataset_cids = datasets.Dataset.from_dict({"cids": self.unique_cid_list})
            combined_dataset_cids.to_parquet(os.path.join(dst_path, "combined", "rm_cid_" + "ipfs_" + dataset.replace("/","___") + "_cids.parquet"))

        for model in list(self.metadata["models"]):
            if not os.path.exists(os.path.join(dst_path, "combined", model.replace("/","___"))):
                combined_embedding_datasets = datasets.Dataset.from_generator(lambda: self.demux_checkpoints(self.index[model]))
                combined_embedding_datasets.to_parquet(os.path.join(dst_path, "combined", + dataset.replace("/","___") + model.replace("/","___") + ".parquet"))
                combined_embedding_datasets_cids = datasets.Dataset.from_dict({"cids": self.unique_cid_list})
                combined_embedding_datasets_cids.to_parquet(os.path.join(dst_path, "combined", dataset.replace("/","___") + model.replace("/","___") + "_cids.parquet"))
        
        for model in list(self.metadata["models"]):
            if not os.path.exists(os.path.join(dst_path, "combined", model.replace("/","___"))):
                combined_embedding_datasets = datasets.Dataset.from_generator(lambda: self.demux_checkpoints(self.index[model]))
                combined_embedding_datasets.to_parquet(os.path.join(dst_path, "secondary_combined", + dataset.replace("/","___") + model.replace("/","___") + ".parquet"))
                combined_embedding_datasets_cids = datasets.Dataset.from_dict({"cids": self.unique_cid_list})
                combined_embedding_datasets_cids.to_parquet(os.path.join(dst_path, "secondary_combined", dataset.replace("/","___") + model.replace("/","___") + "_cids.parquet"))
        return None              
    
    async def generate_clusters(self, dataset, split, dst_path):
        
        return None

    async def load_clusters(self, dataset, split, dst_path):
        ipfs_cid_clusters_list = []
        ipfs_cid_clusters_set = ()
        ipfs_cid_set = set()
        ipfs_cid_list = []
        cluster_cids_dataset = None
        try:
            if os.path.exists(os.path.join(dst_path, dataset.replace("/", "___") + "_cluster_cids.parquet")):
                cluster_cids_dataset = load_dataset('parquet', data_files=os.path.join(dst_path, dataset.replace("/", "___") + "_cluster_cids.parquet"))["train"]
                ipfs_cid_clusters_list = cluster_cids_dataset["cluster_cids"]
                ipfs_cid_clusters_set = [set(x) for x in ipfs_cid_clusters_list]
                ipfs_cid_list = [cid for sublist in ipfs_cid_clusters_list for cid in sublist]
                ipfs_cid_set = set([cid for sublist in ipfs_cid_clusters_list for cid in sublist])
            else:
                await self.generate_clusters(dataset, split, dst_path)
                pass
        except Exception as e:
            print(e)
            pass
        if cluster_cids_dataset is not None:
            self.cluster_cids_dataset = cluster_cids_dataset
        if ipfs_cid_clusters_list is not None:
            self.ipfs_cid_clusters_list = ipfs_cid_clusters_list
        if ipfs_cid_clusters_set is not None:
            self.ipfs_cid_clusters_set = ipfs_cid_clusters_set
        if ipfs_cid_list is not None:
            self.ipfs_cid_list = ipfs_cid_list
        if ipfs_cid_set is not None:
            self.ipfs_cid_set = ipfs_cid_set
        self.cid_set = self.ipfs_cid_set
        return cluster_cids_dataset, ipfs_cid_clusters_list, ipfs_cid_clusters_set, ipfs_cid_list, ipfs_cid_set
        
    async def kmeans_cluster_split(self, dataset, split, columns, dst_path, models, max_splits=None):
        await self.load_clusters(dataset, split, dst_path)
        await self.load_dataset(dataset, split)
        await self.load_checkpoints(dataset, split, dst_path, models)
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
    create_embeddings_batch = ipfs_embeddings_py(resources, metadata)
    # asyncio.run(create_embeddings_batch.index_dataset(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))    
    # asyncio.run(create_embeddings_batch.combine_checkpoints(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))
    # asyncio.run(create_embeddings_batch.kmeans_cluster_split(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"], 10))
    asyncio.run(create_embeddings_batch.index_sparse_chunks(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))