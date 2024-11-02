import os
import datasets
import multiprocessing
from datasets import Dataset, concatenate_datasets, load_dataset

class ipfs_datasets:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        return None

    def test():    
        return None
    
    def __call__(self, *args, **kwds):
        return None
    
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
        
    
if __name__ == "main":
    metadata = {}
    resources = {}
    test_ipfs_datasets = ipfs_datasets(resources, metadata)
    test_ipfs_datasets.test()
    pass