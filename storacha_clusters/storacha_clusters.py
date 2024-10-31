import ipfs_kit_py
from ipfs_kit_py import *
from ..ipfs_embeddings_py import ipfs_embeddings_py
from ipfs_embeddings_py import *
from ..ipfs_embeddings_py import ipfs_parquet_to_car

class storacha_clusters:
    def _init(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_kit_py = ipfs_kit_py.ipfs_kit(resources, metadata)
        self.ipfs_embeddings_py = ipfs_embeddings_py.ipfs_embeddings(resources, metadata)
        self.ipfs_parquet_to_car = ipfs_embeddings_py.ipfs_parquet_to_car(resources, metadata)
        self.kmeans_cluster_split = ipfs_embeddings_py.kmeans_cluster_split()
        return None
    
    def test(self):
        results = {}
        test_ipfs_kit_init = None
        test_ipfs_kit = None
        test_ipfs_parquet_to_car = None
        test_storacha_clusters = None
        try:
            test_ipfs_kit_init = self.ipfs_kit.init()
        except Exception as e:
            test_ipfs_kit_init = e
            print(e)
            raise e 
        
        try:
            test_ipfs_kit = self.ipfs_kit.test()
        except Exception as e:
            test_ipfs_kit = e
            print(e)
            raise e
        try:
            test_ipfs_parquet_to_car = self.ipfs_parquet_to_car.test()
        except Exception as e:
            test_ipfs_parquet_to_car = e
            print(e)
            raise e
        try:
            test_storacha_clusters = self.ipfs_kit_py.storacha_kit_py.test()
        except Exception as e:
            test_storacha_clusters = e
            print(e)
            raise e
        

    
if __name__ == "main":
    metadata = {}
    resources = {}
    test_storacha_clusters = storacha_clusters(resources, metadata)
    test_storacha_clusters.test()