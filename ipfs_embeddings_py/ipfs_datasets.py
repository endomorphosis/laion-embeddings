import os
import datasets
import multiprocessing
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

try:
    from .ipfs_parquet_to_car import ipfs_parquet_to_car_py
except Exception as e:
    try: 
        from ipfs_parquet_to_car import ipfs_parquet_to_car_py
    except Exception as e:
        pass
    pass

try:
    from .ipfs_multiformats import ipfs_multiformats_py
    from .ipfs_multiformats import *
except Exception as e:
    try:
        from ipfs_multiformats import ipfs_multiformats_py
        from ipfs_multiformats import *
    except Exception as e:
        try:
            import ipfs_multiformats
        except Exception as e:
            pass
    pass



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


class ipfs_datasets_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.load_dataset = load_dataset
        self.ipfs_cluster_name = None
        self.dataset = None
        self.caches = {}
        self.ipfs_parquet_to_car_py = ipfs_parquet_to_car_py(resources, metadata)
        self.combine_checkpoints = self.combine_checkpoints
        self.load_checkpoints = self.load_checkpoints
        self.generate_clusters = self.generate_clusters
        self.load_clusters = self.load_clusters
        self.process_new_dataset_shard = process_new_dataset_shard
        self.process_index_shard = self.process_index_shard
        self.cid_chunk_list = []
        self.cid_chunk_set = set()
        self.cid_list = []
        self.cid_set = set()
        self.index = {}
        self.new_dataset = None
        self.new_dataset_combined = None
        self.embedding_datasets = {}
        self.unique_cid_set = set()
        self.unique_cid_list = []
        self.cluster_cids_dataset = None
        self.ipfs_cid_clusters_list = []
        self.ipfs_cid_clusters_set = ()
        self.ipfs_cid_set = set()
        self.ipfs_cid_list = []
        self.all_cid_list = {}
        self.schemas = {} 
        return None
    
    async def load_combined_checkpoints(self, dataset, split, dst_path, models):
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
                                self.schemas["new_dataset"] = schemas
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
                                self.schemas[model] = schemas
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
                    ls_chunks = os.listdir(os.path.join(dst_path,"sparse_chunks", ))
                if len(ls_chunks) > 0:
                    for this_cid in ls_chunks:
                        this_cid_path = os.path.join(dst_path,"sparse_chunks", this_cid)
                        this_cid_dataset = load_dataset('parquet', data_files=this_cid_path)
                        if this_cid not in self.chunk_cache.keys():
                            self.chunk_cache[this_cid] = {"items": []}
                        self.chunk_cache[this_cid]["items"] += this_cid_dataset
                        self.cid_chunk_set.add(this_cid)
                        self.cid_chunk_list.append(this_cid)
        return None
        
    async def load_chunk_checkpoints(self, dataset, split, src_path, models):
        files = []
        if "doc_cid" not in list(dir(self)):
            self.chunks = {}
        if "doc_cid" not in list(dir(self)):
            self.chunk_cache_set = {}
        if os.path.isdir(src_path):
            files = os.listdir(src_path)
            files_by_models = [ [x for x in files if model.replace("/","___") in x and dataset in x and models in x ] for model in models]
        if len(files_by_models) > 0:
            with multiprocessing.Pool() as pool:
                results = pool.map(self.process_chunk_file, files_by_models)
                for result in results:
                    model, doc_cid, items = result
                    if model not in list(self.chunk_cache.keys()):
                        self.chunk_cache_set[model] = set()
                    if doc_cid not in list(self.chunk_cache[model].keys()):
                        self.chunk_cache[model][doc_cid] = {"items": []}
                    if doc_cid not in self.chunk_cache_set[model]:
                        self.chunk_cache_set[model].add(doc_cid)
                    self.doc_cid[model][doc_cid]["items"] += items
                    
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
                    results = pool.map(process_new_dataset_shard, args)
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
        
    def test():    
        return None

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

    def process_chunk_files(path, datatype="cids"):
        cids = None
        items = None
        schema = None
        
        if type(path) is not str:
            if type(path) is list:
                if len(path) == 1:
                    path = path[0]
                elif len(path) == 2:
                    path, datatype = path
            if type(path) is dict:
                if "file" in list(path.keys()):
                    path = path["file"]
                if "type" in list(path.keys()):
                    datatype = path["type"]
        
        if datatype == "cids":
            if os.path.exists(path):
                cid_path = path.replace(".parquet","")+"_cids.parquet"
                if os.path.exists(cid_path):
                    cids = load_dataset('parquet', data_files=cid_path)["cids"]
            else:
                return ValueError("No dataset found") 
            
        elif datatype == "items":
            cid_path = path.replace(".parquet","")+"_cids.parquet"
            chunk_dataset = None
            cids = None
            if os.path.exists(cid_path):
                cids = load_dataset('parquet', data_files=cid_path)["cids"]
            else:
                if os.path.exists(path):
                    chunk_dataset = load_dataset('parquet', data_files=path)
                    cids = [ item["items"]["cid"] for item in chunk_dataset ]
                    tmp_dataset = datasets.Dataset.from_dict({"cids": cids})
                    tmp_dataset.to_parquet(cid_path)
                else:
                    return ValueError("No dataset found")
            if chunk_dataset is None:
                chunk_dataset = load_dataset('parquet', data_files = path)   
                if cids is None and os.path.exists(cid_path):
                    cids = load_dataset('parquet', data_files = cid_path)["cids"]
                else:
                    cids = [ item["items"]["cid"] for item in chunk_dataset ]
                    tmp_dataset = datasets.Dataset.from_dict({"cids": cids})
                    tmp_dataset.to_parquet(cid_path)
                pass
            items = {key: [item["items"][key] for item in chunk_dataset] for key in chunk_dataset[0]["items"].keys()}
            schema = None        
        elif datatype == "schema":
            schema = None

        return [ cids , items, schema ]

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "namespace": "TeraflopAI/Caselaw_Access_Project",
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
        "local_endpoints": [
            ["thenlper/gte-small", "cpu", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cpu", 32768],
            ["thenlper/gte-small", "cuda:0", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cuda:0", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:0", 32768],
            ["thenlper/gte-small", "cuda:1", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cuda:1", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:1", 32768],
            ["thenlper/gte-small", "openvino", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "openvino", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "openvino", 32768],
            ["thenlper/gte-small", "llama_cpp", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "llama_cpp", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "llama_cpp", 32768],
            ["thenlper/gte-small", "ipex", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "ipex", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "ipex", 32768],
        ],
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
    test_ipfs_datasets = ipfs_datasets_py(resources, metadata)
    test_ipfs_datasets.test()
    pass