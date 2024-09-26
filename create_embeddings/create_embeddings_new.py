import asyncio
from aiohttp import ClientSession
from datasets import load_dataset, Dataset
import os
import sys
import datasets
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
        else:
            self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed-small", 8192)
            self.ipfs_embeddings_py.add_https_endpoint("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:80/embed-medium", 32768 )
        self.join_column = None
        self.tokenizer = {}

    async def main(self, dataset, column, dst_path, models):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.ipfs_embeddings_py.queues = {}
        self.ipfs_embeddings_py.cid_list = []
        self.all_cid_list = {}
        consumer_tasks = {}
        batch_sizes = {}
        self.dataset = load_dataset(dataset, split='train', streaming=True).shuffle(seed=random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        new_dataset_dst_path = dst_path+"/"+ dataset.replace("/","---") + ".parquet"
        if os.path.isfile(new_dataset_dst_path) == True:
            self.ipfs_embeddings_py.new_dataset = datasets.Dataset.from_parquet(new_dataset_dst_path)
            self.all_cid_list["new_dataset"] = self.ipfs_embeddings_py.new_dataset["cid"]
        else:
            self.ipfs_embeddings_py.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })
            self.all_cid_list["new_dataset"] = []

        for model in models:
            batch_size = await self.ipfs_embeddings_py.max_batch_size(model)
            model_dst_path = dst_path + "/" + model.replace("/","---") + ".parquet"
            if os.path.isfile(model_dst_path) == True:
                self.ipfs_embeddings_py.index[model] = datasets.Dataset.from_parquet(model_dst_path)
                self.all_cid_list[model] = self.ipfs_embeddings_py.index[model]["cid"]
            else:
                self.ipfs_embeddings_py.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
                self.all_cid_list[model] = []
            self.ipfs_embeddings_py.queues[model] = asyncio.Queue(batch_size)
            consumer_tasks[model] = asyncio.create_task(self.ipfs_embeddings_py.consumer(self.ipfs_embeddings_py.queues[model], column, batch_size, model))

        common_cids = set(self.all_cid_list["new_dataset"])
        for cid_list in self.all_cid_list.values():
            common_cids.intersection_update(cid_list)
        self.cid_list = common_cids
        self.ipfs_embeddings_py.cid_list = common_cids
        producer_task = asyncio.create_task(self.ipfs_embeddings_py.producer(self.dataset, column, self.ipfs_embeddings_py.queues))        
        save_task = asyncio.create_task(self.ipfs_embeddings_py.save_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, save_task, *consumer_tasks.values()) 
    
if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "models": [
            "BAAI/bge-m3",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            # "dunzhang/stella_en_1.5B_v5",
        ],
        "dst_path": "/storage/teraflopai"
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
    create_embeddings_batch = create_embeddings(resources, metadata)
    asyncio.run(create_embeddings_batch.main(metadata["dataset"], metadata["column"], metadata["dst_path"], metadata["models"]))    


