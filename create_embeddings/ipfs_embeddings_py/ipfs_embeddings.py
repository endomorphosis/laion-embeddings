from ipfs_multiformats import *
import requests
import subprocess
import json
import random
import datasets
import asyncio
from aiohttp import ClientSession
from datasets import load_dataset
import datasets
import os
import sys
import subprocess
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
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
        self.cid_list = set()
        self.cid_queue = iter([])
        self.knn_queue = iter([])
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.new_dataset = {}
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
        return None

    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            del self.libp2p_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
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
        if endpoint is None:
            endpoint = self.choose_endpoint(model)
        while not embed_fail:
            batch = ["Hello World"] * batch_size
            try:
                embeddings = await self.index_knn(batch, model, endpoint)
                if not isinstance(embeddings[0], list):
                    print("Embeddings not returned as list")
                    raise Exception("Embeddings not returned as list")
                exponent += 1
                batch_size = 2**exponent
            except Exception as e:
                embed_fail = True
                # Handle specific exceptions if needed
        self.endpoint_status[endpoint] = 2**(exponent-1)
        return 2**(exponent-1)

    async def index_knn(self, samples, model, chosen_endpoint=None):
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if samples is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
        this_query = {"inputs": samples}
        try:
            query_response = await self.make_post_request(chosen_endpoint, this_query)
        except Exception as e:
            raise e
        if isinstance(query_response, dict) and "error" in query_response.keys():
            raise Exception("error: " + query_response["error"])
        else:
            knn_stack = query_response
        return knn_stack

    async def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        async with ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    content = await response.text()
                    raise ValueError(f"HTTP {response.status}: {content}")
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
        batch = []
        if model_name not in self.index.keys():
            self.index[model_name] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                for i in range(len(results)):
                    self.index[model_name] = self.index[model_name].add_item({"cid": batch[i]["cid"], "embedding": results[i]})
                batch = []  # Clear batch after sending
                self.saved = False
            queue.task_done()
        return None

    async def producer(self, dataset_stream, column, queues):
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor() as executor:
            tasks = []
            async for item in self.async_generator(dataset_stream):
                tasks.append(loop.run_in_executor(executor, self.process_item, item, column, queues))
                # Limit the number of concurrent tasks
                if len(tasks) >= 1000:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)
        return None

    def process_item(self, item, column, queues):
        # Assuming `item` is a dictionary with required data
        column_names = item.keys()
        this_cid = self.index_cid(item[column])[0]
        if "cid" not in column_names:
            item["cid"] = this_cid
        # Check if cid is in index
        if this_cid in self.cid_list:
            return
        else:
            self.cid_list.add(this_cid)
            self.new_dataset = self.new_dataset.add_item(item)    
            for queue in queues.values():
                for q in queue.values():
                    q.put_nowait(item)  # Non-blocking put
        return None

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
            print(f"Received embeddings for {len(results)} items")
        except Exception as e:
            print(e)
            raise e
        return results

    async def save_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(600)
            if self.saved == False:
                self.new_dataset.to_parquet(dst_path+"/"+dataset.replace("/","---")+".parquet")   
                for model in models:
                    self.index[model].to_parquet(dst_path+"/"+model.replace("/","---")+".parquet")
                self.saved = True
        return None 

    def status(self):
        return self.endpoint_status

    def setStatus(self, endpoint, status):
        self.endpoint_status[endpoint] = status
        return None

    async def main(self, dataset, column, dst_path, models):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.queues = {}
        self.cid_list = set()
        self.all_cid_list = {}
        consumer_tasks = {}
        batch_sizes = {}
        self.dataset = load_dataset(dataset, split='train', streaming=True).shuffle(seed=42)
        columns = self.dataset.column_names
        columns.append("cid")
        new_dataset_dst_path = dst_path+"/"+ dataset.replace("/","---") + ".parquet"
        if os.path.isfile(new_dataset_dst_path):
            self.new_dataset = datasets.Dataset.from_parquet(new_dataset_dst_path)
            self.all_cid_list["new_dataset"] = set(self.new_dataset["cid"])
        else:
            self.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })
            self.all_cid_list["new_dataset"] = set()

        for model in models:
            endpoints = self.get_endpoints(model)
            if not endpoints:
                continue
            self.queues[model] = {}
            self.all_cid_list[model] = set()
            model_dst_path = dst_path + "/" + model.replace("/","---") + ".parquet"
            if os.path.isfile(model_dst_path):
                self.index[model] = datasets.Dataset.from_parquet(model_dst_path)
                self.all_cid_list[model] = set(self.index[model]["cid"])
            else:
                self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
                self.all_cid_list[model] = set()
            for endpoint in endpoints:
                batch_size = await self.max_batch_size(model, endpoint)
                self.queues[model][endpoint] = asyncio.Queue(maxsize=batch_size*10)
                consumer_tasks[(model, endpoint)] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))

        # Compute common cids
        common_cids = set(self.all_cid_list["new_dataset"])
        for cid_list in self.all_cid_list.values():
            common_cids.intersection_update(cid_list)
        self.cid_list = common_cids
        producer_task = asyncio.create_task(self.producer(self.dataset, column, self.queues))        
        save_task = asyncio.create_task(self.save_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, save_task, *consumer_tasks.values())
        return None 

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "models": [
            "BAAI/bge-m3",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "dunzhang/stella_en_1.5B_v5",
        ],
        "dst_path": "/storage/teraflopai/tmp"
    }
    resources = {
        "https_endpoints": [
            ["BAAI/bge-m3", "http://62.146.169.111:8080/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8080/embed-medium", 32768],
            ["dunzhang/stella_en_1.5B_v5", "http://62.146.169.111:8080/embed-large", 131072],
            ["BAAI/bge-m3", "http://62.146.169.111:8081/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8081/embed-medium", 32768],
            ["dunzhang/stella_en_1.5B_v5", "http://62.146.169.111:8081/embed-large", 131072],
            ["BAAI/bge-m3", "http://62.146.169.111:8082/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8082/embed-medium", 32768],
            ["dunzhang/stella_en_1.5B_v5", "http://62.146.169.111:8082/embed-large", 131072],
            ["BAAI/bge-m3", "http://62.146.169.111:8083/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8083/embed-medium", 32768],
            ["dunzhang/stella_en_1.5B_v5", "http://62.146.169.111:8083/embed-large", 131072],
        ]
    }
    create_embeddings_batch = ipfs_embeddings_py(resources, metadata)
    asyncio.run(create_embeddings_batch.main(metadata["dataset"], metadata["column"], metadata["dst_path"], metadata["models"]))    
