from ..ipfs_embeddings_py import ipfs_embeddings_py

class shard_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_embeddings_py = ipfs_embeddings_py.ipfs_embeddings_py(resources, metadata)
        return None
    
    async def __call__(self, metadata=None):
        results = None
        try: 
            if metadata is None and metadata in list(dir(self)):
                metadata = self.metadata
            return await self.ipfs_embeddings_py.kmeans_cluster_split(metadata["dataset"], metadata["split"], metadata["column"], metadata["dst_path"], metadata["models"])
        except Exception as e:
            print(e)
            raise e
        
    def test (self):
        results = None
        try:
            results = self()
            return results
        except Exception as e:
            print(e)
            raise e
    
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

    shard_embeddings_test = shard_embeddings(resources, metadata)
    results = None
    try:
        results = shard_embeddings_test.test()
        print("shard_embeddings_test passed")
        print(results)
        exit(0)
    except Exception as e:
        print("shard_embeddings_test failed")
        print(e)
        exit(1)