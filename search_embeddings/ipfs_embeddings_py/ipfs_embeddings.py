from .ipfs_multiformats import *
import requests
import subprocess
import json
import random
import datasets
import asyncio
from aiohttp import ClientSession, ClientTimeout
from datasets import load_dataset
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
import os
import sys
import subprocess
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
from multiprocessing import Pool

class ipfs_embeddings_py:
    def __init__(self, resources, metadata):
        self.multiformats = ipfs_multiformats_py(resources, metadata)
        self.https_endpoints = {}
        self.libp2p_endpoints = {}
        self.datasets = datasets.Dataset
        self.index =  {}
        self.queues = {}
        self.caches = {}
        self.batch_sizes = {}
        self.cid_list = set()
        self.cid_queue = iter([])
        self.knn_queue = iter([])
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.new_dataset = {}
        self.add_https_endpoint = self.add_https_endpoint
        self.add_libp2p_endpoint = self.add_libp2p_endpoint
        self.rm_https_endpoint = self.rm_https_endpoint
        self.rm_libp2p_endpoint = self.rm_libp2p_endpoint
        self.get_https_endpoint = self.get_https_endpoint
        self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.request_https_endpoint = self.request_https_endpoint
        self.index_knn = self.index_knn
        self.make_post_request = self.make_post_request
        self.choose_endpoint = self.choose_endpoint
        self.get_endpoints = self.get_endpoints
        self.max_batch_size = self.max_batch_size
        self.consumer = self.consumer
        self.producer = self.producer
        self.process_item = self.process_item
        self.save_to_disk = self.save_to_disk
        self.status = self.status
        self.setStatus = self.setStatus
        self.index_cid = self.index_cid
        self.load_index = self.load_index
        self.async_generator = self.async_generator
        self.send_batch_to_endpoint = self.send_batch_to_endpoint
        self.save_to_disk = self.save_to_disk
        self.index_dataset = self.index_dataset
        self.saved = False  # Added missing attribute
        # Initialize endpoints
        for endpoint_info in resources.get('https_endpoints', []):
            model, endpoint, context_length = endpoint_info
            self.add_https_endpoint(model, endpoint, context_length)
        return None

    def load_index(self, index):
        self.index = index
        return None 

    def add_https_endpoint(self, model, endpoint, context_length):
        if model not in self.https_endpoints:
            self.https_endpoints[model] = {}
        self.https_endpoints[model][endpoint] = context_length
        # Initialize endpoint status with context_length as max batch size
        self.endpoint_status[endpoint] = context_length
        if model not in self.queues:
            self.queues[model] = {}
        if endpoint not in list(self.queues[model].keys()):
            self.queues[model][endpoint] = asyncio.Queue()
        self.endpoint_status[endpoint] = context_length
        return None

    def add_libp2p_endpoint(self, model, endpoint, context_length):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = {}
        self.libp2p_endpoints[model][endpoint] = context_length
        self.endpoint_status[endpoint] = context_length
        return None

    def rm_https_endpoint(self, model, endpoint):
        if model in self.https_endpoints and endpoint in self.https_endpoints[model]:
            del self.https_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
            del self.queues[model][endpoint]
        return None

    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            del self.libp2p_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
            del self.queues[model][endpoint]
        return None

    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.https_endpoints and endpoint in self.https_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.https_endpoints:
            return self.https_endpoints[model]
        return None

    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_https_endpoint(self, model, batch_size):
        if model in self.https_endpoints:
            for endpoint in self.https_endpoints[model]:
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
        exponent = 1
        batch = []
        batch_size = 2**exponent
        token_length_size = round(self.https_endpoints[model][endpoint] * 0.99)
        test_tokens = []
        for i in range(token_length_size):
            test_tokens.append(random.randint(0,65536))
        if model not in self.tokenizer.keys():
            self.tokenizer[model] = AutoTokenizer.from_pretrained(model, device='cpu')
        test_text = self.tokenizer[model].decode(test_tokens)
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
                                            self.https_endpoints[model][endpoint] = token_length_size - difference
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
                                        self.https_endpoints[model][endpoint] = self.https_endpoints[model][endpoint] - difference
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
        return 2**(exponent-1)


    async def index_knn(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is iter:
            this_query = {"inputs": samples}
            try:
                query_response = self.make_post_request(chosen_endpoint, this_query)
            except Exception as e:
                raise Exception(e)
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        if type(samples) is list:
            this_query = {"inputs": samples}
            try:
                query_response = await self.make_post_request(chosen_endpoint, this_query)
            except Exception as e:
                if "413" in str(e):
                    return ValueError(e)
                raise Exception(e)
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        return knn_stack

    async def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    return ValueError(response)
                return await response.json()

    def choose_endpoint(self, model):
        https_endpoints = self.get_https_endpoint(model)
        libp2p_endpoints = self.get_libp2p_endpoint(model)
        filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
        filtered_https_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and https_endpoints is not None and k in list(https_endpoints.keys())}
        if not filtered_https_endpoints and not filtered_libp2p_endpoints:
            return None
        else:
            this_endpoint = None
            if len(list(filtered_https_endpoints.keys())) > 0:
                this_endpoint = random.choice(list(filtered_https_endpoints.keys()))
            elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                this_endpoint = random.choice(list(filtered_libp2p_endpoints.keys()))
            print("chosen endpoint for " + model + " is " + this_endpoint)
            return this_endpoint

    def get_endpoints(self, model):
        endpoints_dict = self.https_endpoints.get(model, {})
        filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        return filtered_endpoints

    async def async_generator(self, iterable):
        for item in iterable:
            yield item
    async def consumer(self, queue, column, batch_size, model_name, endpoint):
        print("consumer started for model " + model_name + " at endpoint " + endpoint)
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
        return None

    async def producer(self, dataset_stream, column, queues):
        tasks = []
        async for item in self.async_generator(dataset_stream):
            task = self.process_item(item, column, queues, self.index_cid, self.cid_list, self.new_dataset)
            tasks.append(task)
            if len(tasks) >= 1:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
        return None

    async def process_item(self, item, column, queues, index_cid, cid_list, new_dataset):
        # Assuming `item` is a dictionary with required data
        if "new_dataset" not in list(self.caches.keys()):
            self.caches["new_dataset"] = {"items" : []}
        # print(f"Processing item with CID {index_cid(item[column])[0]}")
        column_names = item.keys()
        this_cid = index_cid(item[column])[0]
        if "cid" not in column_names:
            item["cid"] = this_cid
        # Check if cid is in index
        if this_cid in cid_list:
            # print(f"CID {this_cid} already in index, skipping item.")
            pass
        else:
            cid_list.add(this_cid)
            if this_cid not in self.all_cid_list["new_dataset"]:
                self.caches["new_dataset"]["items"].append(item)
            # new_dataset = new_dataset.add_item(item)
            # print(f"Added item with CID {this_cid} to new_dataset.")
            models = self.queues.keys()
            for model, model_queues in queues.items():
                # Assign to the endpoint with the smallest queue
                # while len(model_queues) < 1:
                #     await asyncio.sleep(1)
                if len(model_queues) > 0:
                    if this_cid not in self.all_cid_list[model]:
                        endpoint, queue = min(model_queues.items(), key=lambda x: x[1].qsize())
                        queue.put_nowait(item)  # Non-blocking put

    async def send_batch_to_endpoint(self, batch, column, model_name, endpoint):
        print(f"Sending batch of size {len(batch)} to model {model_name} at endpoint {endpoint}")
        model_context_length = self.https_endpoints[model_name][endpoint]
        new_batch = []
        if model_name not in self.tokenizer.keys():
            self.tokenizer[model_name] = AutoTokenizer.from_pretrained(model_name, device='cpu')
        for item in batch:
            this_item_tokens = len(self.tokenizer[model_name].encode(item[column]))
            if this_item_tokens > model_context_length:
                encoded_item = self.tokenizer[model_name](item[column], return_tensors="pt")["input_ids"].tolist()[0]
                truncated_encoded_item = encoded_item[:model_context_length]
                unencode_item = self.tokenizer[model_name].decode(truncated_encoded_item)
                new_batch.append(unencode_item)
            else:
                new_batch.append(item[column])
        results = None
        try:
            results = await self.index_knn(new_batch, model_name, endpoint)
        except Exception as e:
            print(e)
            raise e
        if isinstance(results, ValueError):
            error = results.args[0]
            if error.status == 413:
                if error.reason == "Payload Too Large":
                    error_content = error.content._buffer[0].decode("utf-8")
                    error_content = json.loads(error_content)
                    if "error" in error_content.keys() and "error_type" in error_content.keys():
                        if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
                            expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
                            given = int(error_content["error"].split("Given: ")[1])
                            difference = given - expected
                            self.https_endpoints[model_name][endpoint] = model_context_length - difference
                            for item in new_batch:
                                index = new_batch.index(item)
                                item = { column : item[:self.https_endpoints[model_name][endpoint]] }
                                new_batch[index] = item
                            results = await self.send_batch_to_endpoint(new_batch, column, model_name, endpoint)
                            return results
                        if "Validation" in error_content["error_type"] and "cannot be empty":
                            print("error: " + error_content["error"])
                            return None
            elif error.status == 504 or error.status == 502:
                self.endpoint_status[endpoint] = 0
                new_endpoint = self.choose_endpoint(model_name)
                if new_endpoint:
                    new_queue = self.queues[model_name][new_endpoint]
                    for item in batch:
                        await new_queue.put(item)
                    return await self.send_batch_to_endpoint(batch, column, model_name, new_endpoint)
                else:
                    return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
            elif error.status == 400:
                return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
            raise Exception(error)
        else:
            print(f"Received embeddings for {len(results)} items")
            return results

    async def save_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(600)
            if self.saved == False:
                if self.caches["new_dataset"] and len(self.caches["new_dataset"]["items"]) > 0:
                    tmp_dataset = datasets.Dataset.from_dict(self.caches["new_dataset"])
                    self.caches["new_dataset"] = {"items" : []}
                    self.new_dataset = concatenate_datasets([self.new_dataset, tmp_dataset])
                    print("Saved "+ str(len(tmp_dataset)) + " items to disk for dataset " + dataset + " at " + dst_path)
                self.new_dataset.to_parquet(dst_path+"/"+dataset.replace("/","---")+".parquet")   
                for model in models:
                    if model in self.caches.keys():
                        if self.caches[model] and len(self.caches[model]["items"]) > 0:
                            tmp_dataset = datasets.Dataset.from_dict(self.caches[model])
                            self.caches[model] = {"items" : []}
                            self.index[model] = concatenate_datasets([self.index[model], tmp_dataset])
                            self.index[model].to_parquet(dst_path+"/"+model.replace("/","---")+".parquet")
                            print("Saved "+ str(len(tmp_dataset)) + " items to disk for model " + model + " at " + dst_path)
                self.saved = True
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
        self.cid_list = set()
        self.all_cid_list = {}
        consumer_tasks = {}
        batch_sizes = {}
        if models is None:
            models = list(self.https_endpoints.keys())
        for model in models:
            if model not in self.queues:
                self.queues[model] = {}
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        new_dataset_dst_path = dst_path+"/"+ dataset.replace("/","---") + ".parquet"
        self.all_cid_list["new_dataset"] = set()
        if os.path.isfile(new_dataset_dst_path):
            self.new_dataset = load_dataset('parquet', data_files=new_dataset_dst_path)['train']
            self.all_cid_list["new_dataset"] = set(self.new_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"])
        else:
            self.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })
        consumer_tasks = {}
        for model in models:
            endpoints = self.get_endpoints(model)
            if not endpoints:
                continue
            self.all_cid_list[model] = set()
            model_dst_path = dst_path + "/" + model.replace("/","---") + ".parquet"
            self.all_cid_list[model] = set()
            if os.path.isfile(model_dst_path):
                self.caches[model] = {"items" : []}
                self.index[model] = datasets.Dataset.from_parquet(model_dst_path)
                self.all_cid_list[model] = set(self.index[model].map(lambda x: {"cid": x["items"]["cid"]})["cid"])
            else:
                self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
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
        # Compute common cids
        self.cid_list = set.intersection(*self.all_cid_list.values())
        producer_task = asyncio.create_task(self.producer(self.dataset, column, self.queues))        
        save_task = asyncio.create_task(self.save_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, save_task, *consumer_tasks.values())
        return None 
