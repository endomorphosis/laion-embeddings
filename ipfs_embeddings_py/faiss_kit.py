import os
import math
import faiss
import numpy as np
import datasets
import concurrent.futures
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
from datasets import load_from_disk

class faiss_kit_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        return None
    
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

    def __test__(self):
        results = {}
        test_faiss_kit_init = None
        test_faiss_kit = None
        test_faiss = self.faiss_kit.test()
        results = {"test_faiss_kit_init": test_faiss_kit_init, "test_faiss_kit": test_faiss_kit, "test_faiss": test_faiss}
        return results