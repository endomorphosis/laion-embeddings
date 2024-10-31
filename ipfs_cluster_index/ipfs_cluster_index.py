
try:
    from ipfs_kit_py import ipfs_kit_py
except:
    try:    
        from ipfs_kit_py import *
    except:
        pass
    
import datasets
from datasets import *
try:
    from ..ipfs_embeddings_py import ipfs_embeddings_py
except Exception as e:
    try:
        from ipfs_embeddings_py import ipfs_embeddings_py
    except Exception as e:  
        pass
class ipfs_cluster_index:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_kit_py = ipfs_kit_py.ipfs_kit(resources, metadata)
        self.storacha_kit_py = ipfs_kit_py.storacha_kit(resources, metadata)
        self.parquet_to_car_py = ipfs_embeddings_py.ipfs_parquet_to_car(resources, metadata)
        return None
    
    def export_cid_list(self, dst_path):
        results = {}
        cid_list = []
        cid_set = set()
        dtype = {}
        contents = {}
        metadata = {}

        try:
            cid_list = self.ipfs_kit_py.ipfs_get_pinset()
            cid_set = set(cid_list)
            results = {"cid_list": cid_list, "cid_set": cid_set}
        except Exception as e:
            print(e)
            raise e
        
        for cid in cid_list:
            cid_index = cid_list.index(cid)
            try:
                dtype = {{ cid : "" } for cid in cid_list }
                contents = {{ cid : "" } for cid in cid_list }
                metadata = {{ cid : "" } for cid in cid_list }
                cid_data = self.ipfs_kit_py.ipfs_get(cid)
                dtype = type(cid_data)
                dtype[cid] = dtype
                contents[cid] = cid_data
                metadata[cid] = cid_data.metadata
            except Exception as e:  
                print(e)
                raise e         
        
        parquet_data = {"cid": {}}
        for key in list(contents.keys()):
            parquet_data[key] = contents[key]
        for key in list(metadata.keys()):
            parquet_data[key] = metadata[key]
        for key in list(dtype.keys()):
            parquet_data[key] = dtype[key]
        
        for key in parquet_data.keys():
            if key == "cid":
                parquet_data["cid"] = [cid for cid in cid_list]
            elif key in list(contents.keys()):
                parquet_data[key] = [contents[key] for key in contents.keys()]
            elif key in list(metadata.keys()):
                parquet_data[key] = [metadata[key] for key in metadata.keys()]
            elif key in list(dtype.keys()):
                parquet_data[key] = [dtype[key] for key in dtype.keys()]
            else:
                pass
            
        for item in cid_list:
            item_cid = item
            index = cid_list.index(item)
            parquet_data["cid"][index] = item_cid
        for item in contents:
            content_index = contents.index(item)
            for key in list(item.keys()):
                parquet_data[key][content_index] = item[key]
        for item in metadata:
            metadata_index = metadata.index(item)
            for key in list(item.keys()):
                parquet_data[key][metadata_index] = item[key]
        for item in dtype:
            dtype_index = dtype.index(item)
            for key in list(item.keys()):
                parquet_data[key][dtype_index] = item[key]
                
        cid_data_dataset = datasets.Dataset.from_dict(parquet_data)
        cid_data_dataset.to_parquet("ipfs_cluster_cid_data.parquet")          
        del parquet_data
        return cid_data_dataset
    
    def parquet_to_car(self, src_path, dst_path):
        return self.ipfs_kit_py.ipfs_parquet_to_car(src_path, dst_path)
    
    def upload_to_storacha(self, src_path, dst_path):
        self.storacha_kit_py.__init__(self.resources, self.metadata)
        self.storacha_kit_py.install()
        
    
    def get_cid(self, cid):
        return self.ipfs_kit_py.ipfs_get_pinset(cid)
    
    def test(self):
        self.export_cid_list("ipfs_cluster_cid_data.parquet")
        return None
    
if __name__ == "main":
    resources = {}
    metadata = {}
    test_ipfs_cluster_index = ipfs_cluster_index(resources, metadata)
    test_ipfs_cluster_index.test()