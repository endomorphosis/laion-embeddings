from aiohttp import ClientSession, ClientTimeout
from multiprocessing import Pool
from transformers import AutoTokenizer
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
try:    
    from ..ipfs_embeddings_py import ipfs_embeddings_py
except:
    try:
        from ipfs_embeddings_py import ipfs_embeddings_py
    except:
         import ipfs_embeddings_py
class sparse_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        self.ipfs_multiformats_py = self.ipfs_embeddings_py.ipfs_multiformats_py(resources, metadata)
        self.index_dataset = self.index_dataset
        self.index_sparse_embeddings = self.index_sparse_embeddings        
        return None

    def index_dataset(self, dataset):
        return self.ipfs_embeddings_py.index_dataset(dataset)
    
    def index_sparse_embeddings(self, embeddings):
        return self.ipfs_embeddings_py.index_sparse_embeddings(embeddings)
    
    def test(self):
        results = {}
        test_ipfs_kit_init = None
        test_ipfs_kit = None
        test_sparse_embeddings = None
        try:
            test_ipfs_embeddings_py = self.ipfs_embeddings_py.__init__(self.resources, self.metadata)
        except Exception as e:
            test_ipfs_embeddings_py = e
            print(e)
        
        try:
            test_sparse_embeddings = self.ipfs_embeddings_py.index_sparse_embeddings()  
        except Exception as e:
            test_sparse_embeddings = e
            print(e)
        
        results = {"test_ipfs_kit_init": test_ipfs_kit_init, "test_ipfs_kit": test_ipfs_kit, "test_sparse_embeddings": test_sparse_embeddings}
        return results
    
        
        

if __name__ == "main":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
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
        "dst_path": "/storage/teraflopai/tmp2",
    }
    resources = {
        "https_endpoints": [
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8080/embed-small", 8192],
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
    test_sparse_embeddings = sparse_embeddings(resources, metadata)
    results = test_sparse_embeddings.test()
    print(results)   