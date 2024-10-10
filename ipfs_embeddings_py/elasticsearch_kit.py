import os
import sys
import subprocess
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import asyncio
from elasticsearch import AsyncElasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError
import time

class elasticsearch_kit:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.index =  {}
        self.cid_list = []
        self.es = Elasticsearch("http://localhost:9200")
        if len(list(metadata.keys())) > 0:
            for key in metadata.keys():
                setattr(self, key, metadata[key])

    async def start_elasticsearch(self):

        ## detect if elasticsearch is already running on the host
        ps_command = ["sudo", "docker", "ps", "-q", "--filter", "ancestor=elasticsearch:8.15.2"]

        ps_result = subprocess.check_output(ps_command).decode("utf-8").strip()
        if len(ps_result) > 0:
            print("Elasticsearch container is already running")
            return None
        
        stopped_containers_command = ["sudo", "docker", "ps", "-a", "--filter", "status=exited", "--filter", "ancestor=elasticsearch:8.15.2", "--format", "{{.ID}}"]
        stopped_containers_result = subprocess.check_output(stopped_containers_command).decode("utf-8").strip()
        if stopped_containers_result:
            container_id = stopped_containers_result.split('\n')[0]
            print(f"Starting inactive Elasticsearch container with ID: {container_id}")
            start_command = ["sudo", "docker", "start", container_id]
            subprocess.run(start_command)
            return None
        
        else:

            command = [
                "sudo", "docker", "run", "-d",
                "--name", "elasticsearch", "-p", "9200:9200",
                "-e", "discovery.type=single-node",
                "-e", "xpack.security.enabled=false",
                "elasticsearch:8.10.2"
            ]

            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Failed to start Elasticsearch container: {result.stderr}")
                return None

            container_id = result.stdout.strip()
            print(f"Started Elasticsearch container with ID: {container_id}")
            return None
    
    async def create_elasticsearch_index(self, index_name):
        try:
            # Create index if it doesn't exist
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name)
                print(f"Created index '{index_name}'")
            else:
                print(f"Index '{index_name}' already exists")
        except ConnectionError as e:
            print(f"Could not connect to Elasticsearch: {e}")
            return None
        return None

    async def stop_elasticsearch(self):
        ps_command = ["sudo", "docker", "ps", "-q", "--filter", "ancestor=elasticsearch:8.15.2"]

        ps_result = subprocess.check_output(ps_command).decode("utf-8").strip()
        if len(ps_result) > 0:

            command = ["sudo","docker", "stop", ps_result]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Failed to stop Elasticsearch container: {result.stderr}")
                return None

            print(f"Stopped Elasticsearch container with ID: {ps_result}")
            return None
        else:
            print("Elasticsearch container is not running")
            return None

    async def send_item_to_elasticsearch(self, item, index_name):
        try:
            # Insert item into the index
            response = self.es.index(index=index_name, document=item)
            print(f"Indexed document {item['id']}: {response['result']}")
        except ConnectionError as e:
            print(f"Could not connect to Elasticsearch: {e}")
            return None
        return None

    async def send_batch_to_elasticsearch(self, batch, index_name):
        print("Sending batch to Elasticsearch")
        # Insert batch into the index
        actions = [
            {
                "index": {
                    "_index": index_name
                }
            }
            for doc in batch
        ]
        for i, doc in enumerate(batch):
            actions.insert(2 * i + 1, doc)
        self.es.bulk(body=actions)
        return None

    async def save_elasticsearch_snapshot(self, index_name, dst_path):
        print("Saving Elasticsearch snapshot")
        # Save the snapshot
        snapshot_name = f"{index_name}_snapshot"
        snapshot_path = os.path.join(dst_path, snapshot_name)

        #create folder if not exists
        os.makedirs(snapshot_path, exist_ok=True)
        #create folder in docker container if not exists
        command = ["sudo", "docker", "exec", "elasticsearch", "mkdir", "-p", f"/usr/share/elasticsearch/data/snapshot/{snapshot_name}"]
        subprocess.run(command, check=True)
        # Create snapshot repository if it doesn't exist
        if not self.es.snapshot.get(repository="snapshot", snapshot="_all"):
            self.es.snapshot.create_repository(
                repository="snapshot",
                body={
                    "type": "fs",
                    "settings": {
                        "location": "/usr/share/elasticsearch/data/snapshot"
                    }
                }
            )


        # Move the snapshot to the destination path
        os.makedirs(dst_path, exist_ok=True)
        command = ["sudo", "docker", "cp", f"elasticsearch:/usr/share/elasticsearch/data/snapshot/{snapshot_name}", snapshot_path]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Failed to save Elasticsearch snapshot: {result.stderr}")
            return None
        
        return None

    async def empty_elasticsearch_index(self, index_name):
        print("Emptying Elasticsearch index")
        try:
            # Delete the index
            self.es.indices.delete(index=index_name)
            print(f"Deleted index '{index_name}'")
        except ConnectionError as e:
            print(f"Could not connect to Elasticsearch: {e}")
            return None
        return None

    async def test(self):
        await self.start_elasticsearch()
        this_dataset = load_dataset("TeraflopAI/Caselaw_Access_Project", split="train", streaming=True)
        batch = []
        for item in this_dataset:
            batch.append(item)
            if len(batch) == 100:
                await self.create_elasticsearch_index("caselaw_access_project_train")
                await self.send_batch_to_elasticsearch(batch, "caselaw_access_project_train")
                await self.save_elasticsearch_snapshot("caselaw_access_project_train", "/storage/teraflopai/tmp")
                await self.empty_elasticsearch_index("caselaw_access_project_train")
                batch = []
        return None
    

    async def test2(self):
        await self.start_elasticsearch()
        this_dataset = load_dataset("TeraflopAI/Caselaw_Access_Project", split="train")
        this_dataset = this_dataset.add_elasticsearch_index("text", host="localhost", port=9200)

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "split": "train",
        "models": [
            "Alibaba-NLP/gte-large-en-v1.5",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            # "Alibaba-NLP/gte-Qwen2-7B-instruct",
        ],
        "dst_path": "/storage/teraflopai/tmp2"
    }
    resources = {
    }
    elasticsearch_kit = elasticsearch_kit(resources, metadata)
    asyncio.run(elasticsearch_kit.test2())
    