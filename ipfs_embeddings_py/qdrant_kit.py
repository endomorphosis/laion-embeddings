import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
import numpy as np
import random
import datasets
from datasets import Dataset
import json
import numpy as np
import os
import json
import pandas as pd
import subprocess
import asyncio
import hashlib
import random
import multiprocessing
from multiprocessing import Pool
class qdrant_kit_py:
    def __init__(self, resources: dict, metadata: dict):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
    
    def hash_chunk(chunk):
        this_hash_key = {}
        for column in chunk:
            this_hash_key[column] = chunk[column]
        return hashlib.sha256(json.dumps(this_hash_key).encode()).hexdigest()


    async def join_datasets(self, dataset, knn_index, join_column):
        dataset_iter = iter(dataset)
        knn_index_iter = iter(knn_index)
        while True:
            try:
                dataset_item = next(dataset_iter)
                knn_index_item = next(knn_index_iter)
                results = {}
                for key in dataset_item.keys():
                    results[key] = dataset_item[key]
                same = True
                for column in join_column:
                    if dataset_item[column] != knn_index_item[column]:
                        same = False
                        break
                if same:
                    for key in knn_index_item.keys():
                        results[key] = knn_index_item[key]
                else:
                    if not hasattr(self, 'knn_index_hash') or not hasattr(self, 'datasets_hash') or len(self.knn_index_hash) == 0 or len(self.datasets_hash) == 0:
                        cores = os.cpu_count() or 1
                        with Pool(processes=cores) as pool:
                            self.knn_index_hash = []
                            self.datasets_hash = []
                            chunk = []
                            async for item in self.ipfs_embeddings_py.async_generator(self.dataset):
                                chunk.append(item)
                                if len(chunk) == cores:
                                    self.datasets_hash.extend(pool.map(self.hash_chunk, chunk))
                                    chunk = []
                            if chunk:
                                self.datasets_hash.extend(pool.map(self.hash_chunk, chunk))
                            chunk = []
                            async for item in self.ipfs_embeddings_py.async_generator(self.knn_index):
                                chunk.append(item)
                                if len(chunk) == cores:
                                    self.knn_index_hash.extend(pool.map(self.hash_chunk, chunk))
                                    chunk = []
                            if chunk:
                                self.knn_index_hash.extend(pool.map(self.hash_chunk, chunk))
                    this_hash_key = {}
                    for column in join_column:
                        this_hash_key[column] = dataset_item[column]
                    this_hash_value = hashlib.md5(json.dumps(this_hash_key).encode()).hexdigest()
                    if this_hash_value in self.knn_index_hash and this_hash_value in self.datasets_hash:
                        knn_index_item = self.knn_index_hash.index(this_hash_value)
                        for key in self.knn_index[knn_index_item].keys():
                            results[key] = self.knn_index[knn_index_item][key]
                        dataset_item = self.datasets_hash.index(this_hash_value)
                        for key in self.dataset[dataset_item].keys():
                            results[key] = self.dataset[dataset_item][key]
                    else:
                        continue
                yield results
            except StopIteration:
                break
            except StopAsyncIteration:
                break

    async def load_qdrant_iter(self, dataset, knn_index, dataset_split= None, knn_index_split=None):
        self.knn_index_hash = []
        self.datasets_hash = []
        self.dataset_name = dataset
        self.knn_index_name = knn_index
        if dataset_split is not None:
            self.dataset = self.datasets.load_dataset(dataset, split=dataset_split, streaming=True)
            dataset_columns = self.dataset.column_names
        else:
            self.dataset = self.datasets.load_dataset(dataset, streaming=True)
            dataset_splits = list(self.dataset.keys())
            dataset_columns = self.dataset[dataset_splits[0]].column_names
            
        if knn_index_split is not None:
            self.knn_index = self.datasets.load_dataset(knn_index, split=knn_index_split, streaming=True)
            # knn_index_splits = list(self.knn_index.keys())
            # shared_splits = set(dataset_splits).intersection(set(knn_index_splits))
            # self.shared_splits = shared_splits
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            single_row = next(iter(self.knn_index.take(1)))
            self.embedding_size = len(single_row["embeddings"][0])
            self.knn_index = self.datasets.load_dataset(knn_index, split=knn_index_split, streaming=True)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            knn_columns = self.knn_index.column_names
        else:
            self.knn_index = self.datasets.load_dataset(knn_index, streaming=True)
            knn_index_splits = list(self.knn_index.keys())
            shared_splits = set(dataset_splits).intersection(set(knn_index_splits))
            self.shared_splits = shared_splits
            knn_columns = self.knn_index[knn_index_splits[0]].column_names
            if "Embeddings" in knn_columns:
                for split in knn_index_splits:
                    self.knn_index[split] = self.knn_index[split].rename_column("Embeddings", "embeddings")
            for split in knn_index_splits:
                single_row = next(iter(self.knn_index[split].take(1)))
                self.embedding_size = len(single_row["embeddings"][0])
            self.knn_index = self.datasets.load_dataset(knn_index, streaming=True)
            if "Embeddings" in knn_columns:
                for split in knn_index_splits:
                    self.knn_index[split] = self.knn_index[split].rename_column("Embeddings", "embeddings")
                                                     
        self.dataset_name = dataset
        common_columns = set(dataset_columns).intersection(set(knn_columns))
        self.join_column = common_columns

        if dataset_split is not None and knn_index_split is not None:
            self.joined_dataset = self.join_datasets(self.dataset, self.knn_index, self.join_column)
        elif dataset_splits is not None and knn_index_splits is not None:
            self.joined_dataset_splits = {}
            for split in shared_splits:
                self.joined_dataset_splits[split] = self.join_datasets(self.dataset[split], self.knn_index[split], self.join_column)  
        return None


    async def load_qdrant(self, dataset, knn_index, dataset_split= None, knn_index_split=None):
        self.dataset_name = dataset
        self.knn_index_name = knn_index
        if dataset_split is not None:  
            self.dataset = self.datasets.load_dataset(dataset, split=dataset_split).shuffle(seed=random.randint(0,65536))
            dataset_columns = self.dataset.column_names
        else:
            self.dataset = self.datasets.load_dataset(dataset).shuffle(seed=random.randint(0,65536))
            dataset_columns = self.dataset.column_names[list(self.dataset.column_names.keys())[0]]
            
        if knn_index_split is not None:
            self.knn_index = self.datasets.load_dataset(knn_index, split=knn_index_split)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            single_row = next(iter(self.knn_index.take(1)))
            self.knn_index = self.datasets.load_dataset(knn_index, split=knn_index_split)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            knn_columns = self.knn_index.column_names
        else:
            self.knn_index = self.datasets.load_dataset(knn_index)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            single_row = next(iter(self.knn_index.take(1)))
            self.embedding_size = len(single_row["embeddings"][0])
            self.embedding_size = len(self.knn_index[list(self.knn_index.keys())[0]].select([0])['Embeddings'][0][0])
            self.knn_index = self.datasets.load_dataset(knn_index)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            knn_columns = self.knn_index.column_names
            knn_columns = self.knn_index.column_names[list(self.knn_index.column_names.keys())[0]]

        # Check if the dataset has the same columns as the knn_index
        found = False
        common_columns = None
        for column in dataset_columns:
            if column in knn_columns:
                found = True
                common_columns = column
                self.join_column = common_columns
                break
        
        columns_to_keep = [common_columns, "Concat Abstract"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(dataset_columns)
        self.dataset = self.dataset.remove_columns(columns_to_remove)
        temp_dataset2 = self.knn_index.to_pandas()
        temp_dataset1 = self.dataset.to_pandas()
        self.joined_dataset = temp_dataset1.join(temp_dataset2.set_index(common_columns), on=common_columns)
        client = QdrantClient(url="http://localhost:6333")
        # Define the collection name
        collection_name = self.dataset_name.split("/")[1]
        embedding_size = len(self.knn_index[list(self.knn_index.keys())[0]].select([0])['Embeddings'][0][0])

        if (client.collection_exists(collection_name)):
            print("Collection already exists")
            return False
        else:
            print("Creating collection")        
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )

        # Chunk size for generating points
        chunk_size = 100
        knn_index_length = self.joined_dataset.shape[0]# Get the number of rows in the dataset
        # Prepare the points to be inserted in chunks
        print("start processing")
        for start in range(0, knn_index_length, chunk_size):
            end = min(start + chunk_size, knn_index_length)
            chunk_df = self.joined_dataset.iloc[start:end]
            points = []
            print(f"Processing chunk {start}:{end}")
            for index, row in chunk_df.iterrows():
                text = row["Concat Abstract"]
                embedding = row["Embeddings"][0]
                points.append(models.PointStruct(
                    id=index,
                    vector=embedding.tolist() if embedding is not None else None,  # Convert embedding to list if not None
                    payload={"text": text}
                ))

            client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        print("Data successfully ingested into Qdrant")
        print("All data successfully ingested into Qdrant from huggingface dataset")
        return True    
    
    async def ingest_qdrant_iter_bak(self, column_name):
        embedding_size = 0
        self.knn_index_length = 99999
        collection_name = self.dataset_name.split("/")[1]
        client = QdrantClient(url="http://localhost:6333")
        # Define the collection name
        collection_name = self.dataset_name.split("/")[1]
        if (client.collection_exists(collection_name)):
            print(collection_name + " Collection already exists")
        else:
            print("Creating collection " + collection_name)        
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
            )

        # Chunk size for generating points
        chunk_size = 100
        # Prepare the points to be inserted in chunks
        processed_rows = 0
        points = []
        if "joined_dataset_splits" in dir(self):
            for split in self.joined_dataset_splits:
                async for item in self.joined_dataset_splits[split]:
                    processed_rows += 1
                    points.append(models.PointStruct(
                        id=processed_rows,
                        vector=item["embeddings"][0],
                        payload={"text": item[column_name]}
                    ))
                    if len(points) == chunk_size:
                        print(f"Processing chunk {processed_rows-chunk_size} to {processed_rows}")
                        client.upsert(
                            collection_name=collection_name,
                            points=points
                        )
                        points = []
        elif "joined_dataset" in dir(self):
            async for item in self.joined_dataset:
                processed_rows += 1
                points.append(models.PointStruct(
                    id=processed_rows,
                    vector=item["embeddings"][0],
                    payload={"text": item[column_name]}
                ))
                if len(points) == chunk_size:
                    print(f"Processing chunk {processed_rows-chunk_size} to {processed_rows}")
                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    points = []        
            
        print("Data successfully ingested into Qdrant")
        print("All data successfully ingested into Qdrant from huggingface dataset")
        return True

    
    async def ingest_qdrant_iter(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]
        embedding_size = 0
        self.knn_index_length = 99999
        collection_name = self.dataset_name.split("/")[1]
        client = QdrantClient(url="http://localhost:6333")
        # Define the collection name
        collection_name = self.dataset_name.split("/")[1]
        if (client.collection_exists(collection_name)):
            print(collection_name + " Collection already exists")
        else:
            print("Creating collection " + collection_name)        
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
            )

        # Chunk size for generating points
        chunk_size = 100
        # Prepare the points to be inserted in chunks
        processed_rows = 0
        points = []
        if "joined_dataset_splits" in dir(self):
            for split in self.joined_dataset_splits:
                async for item in self.joined_dataset_splits[split]:
                    processed_rows += 1
                    payload = {}
                    for column_name in column_names:
                        payload[column_name] = item[column_name]

                    points.append(models.PointStruct(
                        id=processed_rows,
                        vector=item["embeddings"][0],
                        payload=payload
                    ))
                    if len(points) == chunk_size:
                        print(f"Processing chunk {processed_rows-chunk_size} to {processed_rows}")
                        client.upsert(
                            collection_name=collection_name,
                            points=points
                        )
                        points = []
        elif "joined_dataset" in dir(self):
            async for item in self.joined_dataset:
                processed_rows += 1
                payload = {}
                for column_name in column_names:
                    payload[column_name] = item[column_name]
                points.append(models.PointStruct(
                    id=processed_rows,
                    vector=item["embeddings"][0],
                    payload=payload
                ))
                if len(points) == chunk_size:
                    print(f"Processing chunk {processed_rows-chunk_size} to {processed_rows}")
                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    points = []        
            
        print("Data successfully ingested into Qdrant")
        print("All data successfully ingested into Qdrant from huggingface dataset")
        return True


    async def search_qdrant(self, collection_name, query_vector,  n=5):
        query_vector = np.array(query_vector[0])
        client = QdrantClient(url="http://localhost:6333")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=n
        )
        results = []
        for point in search_result:
            columns = point.payload.keys()
            result = {}
            for column in columns:
                result[column] = point.payload[column]
            result["score"] = point.score
            results.append({point.id: result})      
        return results

    def stop_qdrant(self):
        qdrant_port_ls = "sudo docker ps | grep 6333"
        qdrant_port_ls_results = os.system(qdrant_port_ls + " > /dev/null 2>&1")
        if qdrant_port_ls_results == 0:
            docker_ps = "sudo docker ps | grep 6333 | awk '{print $1}'"
            docker_ps_results = subprocess.check_output(docker_ps, shell=True).decode("utf-8").strip()  
            stop_qdrant_cmd = "sudo docker stop " + docker_ps_results
            os.system(stop_qdrant_cmd + " > /dev/null 2>&1")
        return None    

    def start_qdrant(self):
        docker_pull_cmd = "sudo docker pull qdrant/qdrant:latest"
        os.system(docker_pull_cmd + " > /dev/null 2>&1")
        docker_port_ls = "sudo docker ps | grep 6333"
        docker_port_ls_results = os.system(docker_port_ls + " > /dev/null 2>&1")
        docker_image_ls = "sudo docker images | grep qdrant/qdrant | grep latest | awk '{print $3}'"
        docker_image_ls_results = subprocess.check_output(docker_image_ls, shell=True).decode("utf-8").strip()
        docker_ps = "sudo docker ps | grep " + docker_image_ls_results
        try:
            docker_ps_results = subprocess.check_output(docker_ps, shell=True).decode("utf-8")
        except subprocess.CalledProcessError as e:
            docker_ps_results = e
            if docker_port_ls_results == 0:
                stop_qdrant_cmd = self.stop_qdrant()
            docker_stopped_ps = "sudo docker ps -a | grep " + docker_image_ls_results
            try:
                docker_stopped_ps_results = subprocess.check_output(docker_stopped_ps, shell=True).decode("utf-8")
                start_qdrant_cmd = "sudo docker start $(sudo docker ps -a -q --filter ancestor=qdrant/qdrant:latest --format={{.ID}})"
                os.system(start_qdrant_cmd + " > /dev/null 2>&1")
            except subprocess.CalledProcessError as e:
                docker_stopped_ps_results = e
                start_qdrant_cmd = "sudo docker run -d -p 6333:6333 -v /storage/qdrant:/qdrant/data qdrant/qdrant:latest"
                os.system(start_qdrant_cmd + " > /dev/null 2>&1")
        return 1
    