from aiohttp import ClientSession, ClientTimeout
from multiprocessing import Pool
from transformers.models.auto.tokenization_auto import AutoTokenizer
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset

# Corrected imports
from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
from ipfs_embeddings_py import ipfs_multiformats

class sparse_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        # Instantiate the ipfs_embeddings class correctly
        self.ipfs_embeddings_instance = ipfs_embeddings_py(resources, metadata)
        self.ipfs_multiformats_py = ipfs_multiformats.ipfs_multiformats_py(resources, metadata)
        # No need to reassign self.index_dataset and self.index_sparse_embeddings to themselves
        # The methods will be called

    def index_dataset(self, dataset):
        return self.ipfs_embeddings_instance.index_dataset(dataset)
    
    def index_sparse_embeddings(self, embeddings):
        return self.ipfs_embeddings_instance.index_sparse_embeddings(embeddings)
    
    def test(self):
        results = {}
        test_ipfs_embeddings_instance_init = None
        test_sparse_embeddings_result = None
        try:
            # Assuming ipfs_embeddings_instance has an __init__ method that can be called for testing
            # This might need adjustment based on the actual ipfs_embeddings.ipfs_embeddings class structure
            test_ipfs_embeddings_instance_init = self.ipfs_embeddings_instance.__init__(self.resources, self.metadata)
        except Exception as e:
            test_ipfs_embeddings_instance_init = e
            print(e)
        
        try:
            # Call the actual method on the instantiated object
            test_sparse_embeddings_result = self.ipfs_embeddings_instance.index_sparse_embeddings()  
        except Exception as e:
            test_sparse_embeddings_result = e
            print(e)
        
        results = {
            "test_ipfs_embeddings_instance_init": test_ipfs_embeddings_instance_init,
            "test_sparse_embeddings_result": test_sparse_embeddings_result
        }
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
