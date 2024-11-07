import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class test_ipfs_embeddings_py:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        # self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        if "ipfs_embeddings_py" not in globals():
            import ipfs_embeddings
            from ipfs_embeddings_py import ipfs_embeddings_py
        if "install_depends_py" not in globals():
            import install_depends 
            from install_depends import install_depends_py
        if "ipfs_accelerate_py" not in globals():
            import ipfs_embeddings
            from .ipfs_embeddings import ipfs_embeddings_py
        
    
    async def test_dependencencies(self):
        
        return
    

    async def test_hardware(self):
        cuda_test = None
        openvino_test = None
        llama_cpp_test = None
        ipex_test = None
        cuda_install = None
        openvino_install = None
        llama_cpp_install = None
        ipex_install = None
        
        try:
            openvino_test = await self.ipfs_accelerate_py.test_local_openvino()
        except Exception as e:
            openvino_test = e
            print(e)
            try:
                openvino_install = await self.install_depends.install_openvino()
                try:
                    openvino_test = await self.ipfs_accelerate_py.test_local_openvino()
                except Exception as e:
                    openvino_test = e
                    print(e)
            except Exception as e:
                openvino_install = e
                print(e)        
            pass
            
        try:
            llama_cpp_test = await self.ipfs_accelerate_py.test_llama_cpp()
        except Exception as e:
            llama_cpp_test = e
            try:
                llama_cpp_install = await self.install_depends.install_llama_cpp()
                try:
                    llama_cpp_test = await self.ipfs_accelerate_py.test_llama_cpp()
                except:
                    llama_cpp_test = e
            except Exception as e:
                print(e)
                llama_cpp_install = e
            pass
        try:
            ipex_test = await self.ipfs_accelerate_py.test_ipex()
        except Exception as e:
            ipex_test = e
            print(e)
            try:
                ipex_install = await self.install_depends.install_ipex()
                try:
                    ipex_test = await self.ipfs_accelerate_py.test_ipex()
                except Exception as e:
                    ipex_test = e
                    print(e)
            except Exception as e:
                ipex_install = e
                print(e)
            pass
        try:
            cuda_test = await self.ipfs_accelerate_py.test_cuda()
        except Exception as e:
            try:
                cuda_install = await self.install_depends.install_cuda()
                try:
                    cuda_test = await self.ipfs_accelerate_py.test_cuda()
                except Exception as e:
                    cuda_test = e
                    print(e)                    
            except Exception as e:
                cuda_install = e
                print(e)
            pass
                
        print("local_endpoint_test")
        install_results = {
            "cuda": cuda_install,
            "openvino": openvino_install,
            "llama_cpp": llama_cpp_install,
            "ipex": ipex_install
        }
        print(install_results)
        test_results = {
            "cuda": cuda_test,
            "openvino": openvino_test,
            "llama_cpp": llama_cpp_test,
            "ipex": ipex_test
        }
        print(test_results)
        return test_results
        
    def test(self):
        results = {}
        test_hardware = None
        test_dependencies = None
        try:
            test_hardware = self.test_hardware()
        except Exception as e:
            test_hardware = e
            print(e)
            raise e
        try:
            test_dependencies = self.test_dependencies()
        except Exception as e:
            test_dependencies = e
            print(e)
            raise e
        results = {"test_hardware": test_hardware, "test_dependencies": test_dependencies}
        
        test_ipfs_embeddings = None
        try:
            test_ipfs_embeddings = self.ipfs_embeddings_py.test()
        except Exception as e:
            test_ipfs_embeddings = e
            print(e)
            raise e
        results["test_ipfs_embeddings"] = test_ipfs_embeddings
        return results
    
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
    test = test_ipfs_embeddings_py(resources, metadata)
    results = test.test()
    print(results)
