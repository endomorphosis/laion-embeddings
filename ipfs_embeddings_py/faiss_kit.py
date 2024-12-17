import os
import math
import faiss
import numpy as np
import datasets
import concurrent.futures
from datasets import Dataset,load_dataset, concatenate_datasets, load_from_disk
import multiprocessing
class faiss_kit_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.search_chunks = self.search_chunks
        self.autofaiss_chunks = self.autofaiss_chunks
        self.search_centroids = self.search_centroids
        self.search_shards = self.search_shards
        self.autofaiss_shards = self.autofaiss_shards
        self.kmeans_cluster_split_dataset = self.kmeans_cluster_split_dataset
        self.chunk_cache = {}
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
    
    async def kmeans_cluster_split_dataset(self, dataset, split, columns, dst_path, models, max_splits=None):
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
            hashed_dataset_download_size = self.hashed_dataset.dataset_size            
            embeddings_size = {}
            for model in self.metadata["models"]:
                embeddings_size[model] = self.index[model].dataset_size
            largest_embeddings_dataset = max(embeddings_size, key=embeddings_size.get)
            largest_embeddings_size = embeddings_size[max(embeddings_size, key=embeddings_size.get)]
            embeddings_size["hashed_dataset"] = hashed_dataset_download_size
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
                hashed_dataset_download_size = self.hashed_dataset.dataset_size            
                embeddings_size = {}
                for model in self.metadata["models"]:
                    embeddings_size[model] = self.index[model].dataset_size
                largest_embeddings_dataset = max(embeddings_size, key=embeddings_size.get)
                largest_embeddings_size = embeddings_size[max(embeddings_size, key=embeddings_size.get)]
                embeddings_size["hashed_dataset"] = hashed_dataset_download_size
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
                hashed_dataset_download_size = self.hashed_dataset.dataset_size            
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
            first_item = self.hashed_dataset[0]
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
                for item in self.hashed_dataset
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
    