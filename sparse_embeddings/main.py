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
import numpy as np
from aiohttp import ClientSession, ClientTimeout
import multiprocessing
from multiprocessing import Pool
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
import ipfs_multiformats
from ipfs_multiformats import *
from chunker import Chunker
import time
import math



def process_new_dataset_shard_2(shard, datatype=None, split="train"):
    items = None
    cids = None
    schema = None
    if type(shard) is not str:
        if type(shard) is list:
            if len(shard) == 1:
                shard = shard[0]
            elif len(shard) == 2:
                shard, datatype = shard
            elif len(shard) == 3:
                shard, datatype, split = shard
        if type(shard) is dict:
            if "shard" in list(shard.keys()):
                shard = shard["shard"]
            if "datatype" in list(shard.keys()):
                datatype = shard["datatype"]
            if "split" in list(shard.keys()):
                split = shard["split"]
                
    if datatype is None:
        if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
            datatype = "cids"
        else:
            if os.path.exists(shard.replace(".parquet","")+".parquet"):
                datatype = "items"
            else:
                raise ValueError("No dataset found")      
    elif "cids" in datatype:
        if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
            tmp_new_dataset_cid_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet", streaming=True)[split]
            items = None
            schema = None
        else:
            tmp_new_dataset_items_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+".parquet", streaming=True)[split]
            tmp_new_dataset_cid_dataset = tmp_new_dataset_items_dataset.map(lambda x: {"cid": x["items"]["cid"]})
            tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet", batch_size=1024)
            tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
        cids = list(tmp_new_dataset_cid_dataset["cids"])
    elif "items" in datatype:
        if os.path.exists(shard.replace(".parquet", "")+".parquet"):
            tmp_new_dataset_items_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+".parquet")[split]
            if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
                tmp_new_dataset_cid_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")[split]
            else:
                tmp_new_dataset_cid_dataset = tmp_new_dataset_items_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                tmp_new_dataset_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_new_dataset_cid_dataset})
                tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
            cids = list(tmp_new_dataset_cid_dataset["cids"])
            items = {key: [item["items"][key] for item in tmp_new_dataset_items_dataset] for key in tmp_new_dataset_items_dataset[0]["items"].keys()}
            cids = list(tmp_new_dataset_cid_dataset["cids"])
            schema = None
            del tmp_new_dataset_cid_dataset
            del tmp_new_dataset_items_dataset
        else:
            return ValueError("No dataset found")
    else:
        return ValueError("datatype must be 'cids' or 'items' , received: '" + str(datatype) + "'")
            
    return [ cids , items, schema ]            



def process_new_dataset_shard(shard, datatype=None, split="train"):
    items = None
    cids = None
    schema = None
    if type(shard) is not str:
        if type(shard) is list:
            if len(shard) == 1:
                shard = shard[0]
            elif len(shard) == 2:
                shard, datatype = shard
            elif len(shard) == 3:
                shard, datatype, split = shard
        if type(shard) is dict:
            if "shard" in list(shard.keys()):
                shard = shard["shard"]
            if "datatype" in list(shard.keys()):
                datatype = shard["datatype"]
            if "split" in list(shard.keys()):
                split = shard["split"]
                
    if datatype is None:
        if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
            datatype = "cids"
        else:
            if os.path.exists(shard.replace(".parquet","")+".parquet"):
                datatype = "items"
            else:
                return ValueError("No dataset found")      
    elif "cids" in datatype:
        if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
            tmp_new_dataset_cid_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")[split]
            items = None
            schema = None
        else:
            tmp_new_dataset_items_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+".parquet")[split]
            tmp_new_dataset_cid_dataset = tmp_new_dataset_items_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
            tmp_new_dataset_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_new_dataset_cid_dataset})
            tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
        cids = list(tmp_new_dataset_cid_dataset["cids"])
    elif "items" in datatype:
        if os.path.exists(shard.replace(".parquet", "")+".parquet"):
            tmp_new_dataset_items_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+".parquet")[split]
            if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
                tmp_new_dataset_cid_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")[split]
            else:
                tmp_new_dataset_cid_dataset = tmp_new_dataset_items_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                tmp_new_dataset_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_new_dataset_cid_dataset})
                tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
            cids = list(tmp_new_dataset_cid_dataset["cids"])
            items = {key: [item["items"][key] for item in tmp_new_dataset_items_dataset] for key in tmp_new_dataset_items_dataset[0]["items"].keys()}
            cids = list(tmp_new_dataset_cid_dataset["cids"])
            schema = None
            del tmp_new_dataset_cid_dataset
            del tmp_new_dataset_items_dataset
        else:
            return ValueError("No dataset found")
    else:
        return ValueError("datatype must be 'cids' or 'items' , received: '" + str(datatype) + "'")
            
    return [ cids , items, schema ]            




def process_index_shard(shard, datatype=None, split="train"):
    items = None
    cids = None
    schema = None
    if type(shard) is not str:
        if type(shard) is list:
            if len(shard) == 1:
                shard = shard[0]
            elif len(shard) == 2:
                shard, datatype = shard
            elif len(shard) == 3:
                shard, datatype, split = shard
        if type(shard) is dict:
            if "shard" in list(shard.keys()):
                shard = shard["shard"]
            if "datatype" in list(shard.keys()):
                datatype = shard["datatype"]
            if "split" in list(shard.keys()):
                split = shard["split"]
                
    if datatype is None:
        if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
            datatype = "cids"
        else:
            if os.path.exists(shard.replace(".parquet","")+".parquet"):
                datatype = "items"
            else:
                return ValueError("No dataset found")      
    elif "cids" in datatype:
        if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
            tmp_new_dataset_cid_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")[split]
            items = None
            schema = None
        else:
            tmp_new_dataset_items_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+".parquet")[split]
            tmp_new_dataset_cid_dataset = tmp_new_dataset_items_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
            tmp_new_dataset_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_new_dataset_cid_dataset})
            tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
        cids = list(tmp_new_dataset_cid_dataset["cids"])
    elif "items" in datatype:
        if os.path.exists(shard.replace(".parquet", "")+".parquet"):
            tmp_new_dataset_items_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+".parquet")[split]
            if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
                tmp_new_dataset_cid_dataset = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")[split]
            else:
                tmp_new_dataset_cid_dataset = tmp_new_dataset_items_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                tmp_new_dataset_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_new_dataset_cid_dataset})
                tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
            cids = list(tmp_new_dataset_cid_dataset["cids"])
            items = {key: [item["items"][key] for item in tmp_new_dataset_items_dataset] for key in tmp_new_dataset_items_dataset[0]["items"].keys()}
            cids = list(tmp_new_dataset_cid_dataset["cids"])
            schema = None
            del tmp_new_dataset_cid_dataset
            del tmp_new_dataset_items_dataset
        else:
            return ValueError("No dataset found")
    else:
        return ValueError("datatype must be 'cids' or 'items' , received: '" + str(datatype) + "'")
            
    return [ cids , items, schema ]            



def process_model_shard(shard, split="train"):
    items = None
    cids = None
    if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
        tmp_model_cids = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")["train"]
        cids = list(tmp_model_cids["cids"])
        items = None
        del tmp_model_cids
    else:
        this_model_shard = load_dataset('parquet', data_files=shard)[split]
        tmp_model_cids = this_model_shard.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
        tmp_model_items = this_model_shard.map(lambda x: {"items": x["items"]})["items"]
        cids = list(tmp_model_cids)
        items = dict(tmp_model_items)
        tmp_model_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_model_cids})
        tmp_model_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
        del this_model_shard
        del tmp_model_cids
        del tmp_model_cid_dataset
    return [cids, items]

class ipfs_embeddings_py:
    def __init__(self, resources, metadata):
        self.multiformats = ipfs_multiformats_py(resources, metadata)
        self.datasets = datasets.Dataset
        self.chunker = Chunker(resources, metadata)
        self.process_model_shard = process_model_shard
        self.process_new_dataset_shard = process_new_dataset_shard
        self.process_index_shard = process_index_shard
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
        self.new_dataset = {}
        self.new_dataset_children = {}
        self.saved = False
        self.resources = resources
        self.metadata = metadata
        self.index_dataset = self.index_dataset
        self.add_tei_endpoint = self.add_tei_endpoint
        self.add_libp2p_endpoint = self.add_libp2p_endpoint
        self.add_openvino_endpoint = self.add_openvino_endpoint
        self.add_local_endpoint = self.add_local_endpoint
        self.rm_tei_endpoint = self.rm_tei_endpoint
        self.rm_libp2p_endpoint = self.rm_libp2p_endpoint
        self.rm_openvino_endpoint = self.rm_openvino_endpoint
        self.rm_local_endpoint = self.rm_local_endpoint
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
        self.status = self.status
        self.setStatus = self.setStatus
        self.index_cid = self.index_cid
        self.load_index = self.load_index
        self.async_generator = self.async_generator
        self.send_batch_to_endpoint = self.send_batch_to_endpoint
        # Initialize endpoints
        if "tei_endpoints" in resources.keys():
            for endpoint_info in resources.get('tei_endpoints', []):
                model, endpoint, context_length = endpoint_info
                self.add_tei_endpoint(model, endpoint, context_length)
        if "openvino_endpoints" in resources.keys():
            for endpoint_info in resources.get('openvino_endpoints', []):
                model, endpoint, context_length = endpoint_info
                self.add_openvino_endpoint(model, endpoint, context_length)
        if "libp2p_endpoints" in resources.keys():
            for endpoint_info in resources.get('libp2p_endpoints', []):
                model, endpoint, context_length = endpoint_info
                self.add_libp2p_endpoint(model, endpoint, context_length)
        if "local_endpoints" in resources.keys():
            for endpoint_info in resources.get('local_endpoints', []):
                model, endpoint, context_length = endpoint_info
                self.add_local_endpoint(model, endpoint, context_length)
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

    def add_tei_endpoint(self, model, endpoint, context_length):
        if model not in self.tei_endpoints:
            self.tei_endpoints[model] = {}
        self.tei_endpoints[model][endpoint] = context_length
        # Initialize endpoint status with context_length as max batch size
        self.endpoint_status[endpoint] = context_length
        return None
    
    def add_openvino_endpoint(self, model, endpoint, context_length):
        if model not in self.openvino_endpoints:
            self.openvino_endpoints[model] = {}
        self.openvino_endpoints[model][endpoint] = context_length
        self.endpoint_status[endpoint] = context_length
        return None

    def add_libp2p_endpoint(self, model, endpoint, context_length):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = {}
        self.libp2p_endpoints[model][endpoint] = context_length
        self.endpoint_status[endpoint] = context_length
        return None
    
    def add_local_endpoint(self, model, endpoint, context_length):
        if model not in self.local_endpoints:
            self.local_endpoints[model] = {}
        self.local_endpoints[model][endpoint] = context_length
        self.endpoint_status[endpoint] = context_length
        return None

    def rm_tei_endpoint(self, model, endpoint):
        if model in self.tei_endpoints and endpoint in self.tei_endpoints[model]:
            del self.tei_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None

    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            del self.libp2p_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None
    
    def rm_openvino_endpoint(self, model, endpoint):
        if model in self.openvino_endpoints and endpoint in self.openvino_endpoints[model]:
            del self.openvino_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None
    
    def rm_local_endpoint(self, model, endpoint):
        if model in self.local_endpoints and endpoint in self.local_endpoints[model]:
            del self.local_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
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

    def request_tei_endpoint(self, model, batch_size):
        if model in self.tei_endpoints:
            for endpoint in self.tei_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    def request_local_endpoint(self, model, batch_size):
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

    async def max_batch_size(self, model, endpoint=None):
        embed_fail = False
        exponent = 0
        batch = []
        batch_size = 2**exponent
        if "cuda" in endpoint or "cpu" in endpoint:
            token_length_size = round(self.local_endpoints[model][endpoint].config.max_position_embeddings * 0.99)
        else:
            token_length_size = round(self.tei_endpoints[model][endpoint] * 0.99)
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
            try:
                embeddings = await self.index_knn(test_batch, model, endpoint)
                if not isinstance(embeddings, list):
                    if isinstance(embeddings, ValueError):
                        fail_reason = embeddings.args[0]
                        if "413" in str(fail_reason):
                            error = fail_reason
                            if error.status == 413:
                                if error.reason == "Payload Too Large":
                                    error_content = error.content._buffer[0].decode("utf-8")
                                    error_content = json.loads(error_content)
                                    if "error" in error_content.keys() and "error_type" in error_content.keys():
                                        if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
                                            expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
                                            given = int(error_content["error"].split("Given: ")[1])
                                            difference = given - expected
                                            self.tei_endpoints[model][endpoint] = token_length_size - difference
                                            return await self.max_batch_size(model, endpoint)
                        if "502" in str(fail_reason):
                            self.endpoint_status[endpoint] = 0
                            return 0
                        if "504" in str(fail_reason):
                            self.endpoint_status[endpoint] = 0
                            return 0
                        if "400" in str(fail_reason):
                            return await self.max_batch_size(model, endpoint)
                    raise Exception(embeddings)
                exponent += 1
                batch_size = 2**exponent
            except Exception as e:
                fail_reason = e.args[0]
                embed_fail = True
                if isinstance(e, ValueError) or isinstance(e, Exception):
                    if "CUDA out of memory" in str(fail_reason):
                        if exponent == 0:
                            self.endpoint_status[endpoint] = 0
                            return 1
                        else:
                            self.endpoint_status[endpoint] = 2**(exponent-1)
                            return 2 ** (exponent-1)
                    if "413" in str(fail_reason):
                        error = fail_reason.args[0]
                        if error.status == 413:
                            if error.reason == "Payload Too Large":
                                error_content = error.content._buffer[0].decode("utf-8")
                                error_content = json.loads(error_content)
                                if "error" in error_content.keys() and "error_type" in error_content.keys():
                                    if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
                                        expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
                                        given = int(error_content["error"].split("Given: ")[1])
                                        difference = given - expected
                                        self.tei_endpoints[model][endpoint] = self.tei_endpoints[model][endpoint] - difference
                                        results = await self.max_batch_size(model, endpoint)
                                        return results
                        pass
                    if "504" in str(fail_reason):
                        self.endpoint_status[endpoint] = 0
                        return 0
                    if "502" in str(fail_reason):
                        self.endpoint_status[endpoint] = 0
                        return 0
                pass
        self.endpoint_status[endpoint] = 2**(exponent-1)
        if exponent == 0:
            return 1
        else:
            return 2**(exponent-1)

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
            else:
                try:
                    query_response = await self.make_post_request(chosen_endpoint, this_query)
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

    def choose_endpoint(self, model):
        tei_endpoints = self.get_https_endpoint(model)
        libp2p_endpoints = self.get_libp2p_endpoint(model)
        filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
        filtered_tei_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and tei_endpoints is not None and k in list(tei_endpoints.keys())}
        if not filtered_tei_endpoints and not filtered_libp2p_endpoints:
            return None
        else:
            this_endpoint = None
            if len(list(filtered_tei_endpoints.keys())) > 0:
                this_endpoint = random.choice(list(filtered_tei_endpoints.keys()))
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
        print("chunk consumer started")
        while True:
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

    async def kmeans_cluster_split(self, dataset, split, columns, dst_path, models, max_splits=None):
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
                for cluster_id in range(max_splits):
                    if cluster_id not in list(kmeans_embeddings_splits.keys()):
                        kmeans_embeddings_splits[cluster_id] = {}
                first_item = self.index[model][0]
                for key in first_item["items"].keys():
                    embedding_dim = len(first_item["items"]["embedding"])    
                    for cluster_id in range(max_splits):
                        if key not in list(kmeans_embeddings_splits[cluster_id].keys()):
                            if key == "embedding":
                                kmeans_embeddings_splits[cluster_id][key] = np.zeros((len(ipfs_cid_clusters_list[cluster_id]), embedding_dim))
                            else:
                                kmeans_embeddings_splits[cluster_id][key] = [ "" for _ in range(len(ipfs_cid_clusters_list[cluster_id]))]
                for item in self.index[model]:
                    if item["items"]["cid"] in ipfs_cid_set:
                        for cluster_id in range(max_splits):
                            if item["items"]["cid"] in ipfs_cid_clusters_set[cluster_id]:
                                cluders_id_index = ipfs_cid_clusters_list[cluster_id].index(item["items"]["cid"])
                                for key in item["items"].keys():
                                    if key == "embedding":
                                        kmeans_embeddings_splits[cluster_id][key][cluders_id_index] = np.array(item["items"][key])                                    
                                    else:
                                        kmeans_embeddings_splits[cluster_id][key][cluders_id_index] = item["items"][key]
                                break
                for cluster_id in range(max_splits):
                    if cluster_id not in list(kmeans_embeddings_splits.keys()):
                        continue
                    cluster_dataset = datasets.Dataset.from_dict(kmeans_embeddings_splits[cluster_id])
                    cluster_dataset.to_parquet(os.path.join(dst_path, dataset.replace("/", "___") + model.replace("/", "___") + "_clusters", f"cluster_{cluster_id}.parquet"))
        
        kmeans_embeddings_splits = {}
        if not os.path.exists(os.path.join(dst_path, dataset.replace("/", "___") + "_clusters")):
            os.makedirs(os.path.join(dst_path, dataset.replace("/", "___")  + "_clusters"))
        model_splits = os.listdir(os.path.join(dst_path, dataset.replace("/", "___") + "_clusters"))
        if len(model_splits) == max_splits:
            pass 
        else:
            for cluster_id in range(max_splits):
                if cluster_id not in list(kmeans_embeddings_splits.keys()):
                    kmeans_embeddings_splits[cluster_id] = {}
            first_item = self.new_dataset[0]
            for key in first_item["items"].keys():
                for cluster_id in range(max_splits):
                    if key not in list(kmeans_embeddings_splits[cluster_id].keys()):
                        kmeans_embeddings_splits[cluster_id][key] = [ "" for _ in range(len(ipfs_cid_clusters_list[cluster_id]))]    
            for item in self.new_dataset:
                if item["items"]["cid"] in ipfs_cid_set:
                    for cluster_id in range(max_splits):
                        if item["items"]["cid"] in ipfs_cid_clusters_set[cluster_id]:
                            cluders_id_index = ipfs_cid_clusters_list[cluster_id].index(item["items"]["cid"])
                            for key in item["items"].keys():
                                kmeans_embeddings_splits[cluster_id][key][cluders_id_index] = item["items"][key]
                            break
                    
            for cluster_id in range(max_splits):
                if cluster_id not in list(kmeans_embeddings_splits.keys()):
                    continue
                cluster_dataset = datasets.Dataset.from_dict(kmeans_embeddings_splits[cluster_id])
                cluster_dataset.to_parquet(os.path.join(dst_path, dataset.replace("/", "___"), f"cluster_{cluster_id}.parquet"))
                return None
    
    
if __name__ == "__main__":
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
        "dst_path": "/storage/teraflopai/tmp",
    }
    resources = {
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
    asyncio.run(create_embeddings_batch.index_dataset(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))    
    # asyncio.run(create_embeddings_batch.combine_checkpoints(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))
    # asyncio.run(create_embeddings_batch.kmeans_cluster_split(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"]))
