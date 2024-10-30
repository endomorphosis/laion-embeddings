from ..search_embeddings import search_embeddings
from ..create_embeddings import create_embeddings
from ..sparse_embeddings import sparse_embeddings
from ..shard_embeddings import shard_embeddings
from ..storacha_clusters import storacha_clusters

class test_search_embeddings:
    def __init__(self, resources, metadata):
        self.search_embeddings = search_embeddings.search_embeddings(resources, metadata)
        return None
    
    def __call__(self, request):
        return self.search_embeddings.search(request)
    
    def __test__(self):
        test_text = "Hello World"
        test_search = self(test_text)
        print(test_search)
        return None

class test_create_embeddings:
    def __init__(self, resources, metadata):
        self.create_embeddings = create_embeddings.create_embeddings(resources, metadata)
        return None
    
    def __call__(self, request):
        return self.create_embeddings.index_dataset(request)
    
    def __test__(self):
        test_dataset = "laion/Wikipedia-X-Concat"
        test_faiss_index = "laion/Wikipedia-M3"
        test_model = "BAAI/bge-m3"
        test_create = self.create_embeddings(test_dataset, test_faiss_index, test_model)
        print(test_create)
        return None 
    
class test_shard_dataset:
    def __init__(self, resources, metadata):
        self.shard_embeddings = shard_embeddings.shard_embeddings(resources, metadata)
        return None
    
    def __call__(self, request):
        return self.shard_embeddings.kmeans_cluster_split(request)

    def __test__(self):
        test_dataset = "laion/Wikipedia-X-Concat"
        test_faiss_index = "laion/Wikipedia-M3"
        test_model = "BAAI/bge-m3"
        test_shard_dataset = self(test_dataset, test_faiss_index, test_model)
        print(test_shard_dataset)
        return None
    
class test_sparse_embeddings:
    def __init__(self, resources, metadata):
        self.sparse_embeddings = sparse_embeddings.sparse_embeddings(resources, metadata)
        return None
    
    def __call__(self, request):
        return self.sparse_embeddings.index_sparse_embeddings(request)
    
    def __test__(self):
        test_dataset = "laion/Wikipedia-X-Concat"
        test_faiss_index = "laion/Wikipedia-M3"
        test_model = "BAAI/bge-m3"
        test_sparse_embeddings= self(test_dataset, test_faiss_index, test_model)
        print(test_sparse_embeddings)
        return None

class test_storacha_clusters:
    def __init__(self, resources, metadata):
        self.storacha_clusters = storacha_clusters.storacha_clusters(resources, metadata)
        return None
    
    def __call__(self, request):
        return self.storacha_clusters._init(request)
    
    def __test__(self):
        results = {}
        test_ipfs_kit_init = None
        test_ipfs_kit = None
        test_storacha = self.storacha_clusters.test()
        results = {"test_ipfs_kit_init": test_ipfs_kit_init, "test_ipfs_kit": test_ipfs_kit, "test_storacha": test_storacha}
        return self()

    
def test(metadata, resources):
    results = {}
    test_search = test_search_embeddings(resources, metadata)
    test_search.__test__()
    results["search"] = test_search
    test_create = test_create_embeddings(resources, metadata)
    test_create.__test__()
    results["create"] = test_create
    test_shard = test_shard_dataset(resources, metadata)
    test_shard.__test__()
    results["shard"] = test_shard
    test_sparse = test_sparse_embeddings(resources, metadata)
    test_sparse.__test__()
    results["sparse"] = test_sparse
    test_storacha_clusters = test_storacha_clusters(resources, metadata)
    test_storacha_clusters.__test__()
    results["storacha_clusters"] = test_storacha_clusters
    return results
    
if __name__ == '__main__':
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "split": "train",
        "models": [
            "thenlper/gte-small",
            "Alibaba-NLP/gte-large-en-v1.5",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
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
    results = test(metadata, resources)
    print (results)