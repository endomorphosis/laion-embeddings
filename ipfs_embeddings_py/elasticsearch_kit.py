import os
import sys
import subprocess
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import asyncio
from elasticsearch import AsyncElasticsearch

class elasticsearch_kit:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.index =  {}
        self.cid_list = []
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
                "-p", "9200:9200", "-p", "9300:9300",
                "-e", "discovery.type=single-node",
                "-e", "ES_JAVA_OPTS=-Xms512m -Xmx512m",
                "elasticsearch:8.15.2"
            ]

            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Failed to start Elasticsearch container: {result.stderr}")
                return None

            container_id = result.stdout.strip()
            print(f"Started Elasticsearch container with ID: {container_id}")
            return container_id

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

    async def send_batch_to_elasticsearch(self, batch, columns, dataset, split):
        es = AsyncElasticsearch(hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

        index_name = f"{dataset.replace('/', '____')}_{split}".lower()

        try:
            # Create index if it doesn't exist
            if not await es.indices.exists(index=index_name):
                await es.indices.create(index=index_name)

            # Insert batch into the index
            await es.bulk(index=index_name, body=batch)

        except Exception as e:
            if "not Elasticsearch" in str(e):
                print("The client noticed that the server is not Elasticsearch and we do not support this unknown product.")
            else:
                print(f"An error occurred: {e}")
        finally:
            await es.close()

    async def save_elasticsearch_snapshot(self, dataset, split, columns, dst_path, models):

        return None

    async def empty_elasticsearch_index(self, dataset, split, columns, dst_path, models):
            
        return None

    async def test(self):
        await self.start_elasticsearch()

        this_dataset = load_dataset("TeraflopAI/Caselaw_Access_Project", split="train", streaming=True)
        batch = []
        for item in this_dataset:
            batch.append(item)
            if len(batch) == 100:
                await self.send_batch_to_elasticsearch(batch, ["text"], "TeraflopAI/Caselaw_Access_Project", "train")
                batch = []
        
        return None
    
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
    asyncio.run(elasticsearch_kit.test())
    