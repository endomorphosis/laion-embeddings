import torch
import requests
import json
import os
import sys
import asyncio
import random
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from transformers import AutoTokenizer
from transformers import AutoModel

class ipfs_accelerate_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.tei_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.local_endpoints = {}
        self.endpoint_status = {}
        self.endpoint_handler = {}
        self.batch_sizes = {}
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.init_endpoints = self.init_endpoints
        self.get_https_endpoint = self.get_https_endpoint
        self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.request_tei_endpoint = self.request_tei_endpoint
        self.request_libp2p_endpoint = self.request_libp2p_endpoint
        self.request_openvino_endpoint = self.request_openvino_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        self.test_tei_https_endpoint = self.test_tei_https_endpoint
        self.test_libp2p_endpoint = self.test_libp2p_endpoint
        self.test_openvino_endpoint = self.test_openvino_endpoint
        self.test_local_endpoint = self.test_local_endpoint               
        return None
    
    def init_endpoints(self):
        return None
    
    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_endpoints and endpoint in self.tei_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False
    
    def test_openvino_endpoint(self, model, endpoint):
        if model in self.openvino_endpoints and endpoint in self.openvino_endpoints[model]:
            return True
        return False
    
    def test_local_endpoint(self, model, endpoint):
        if model in self.local_endpoints and endpoint in self.local_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.tei_endpoints:
            return self.tei_endpoints[model]
        return None

    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_tei_endpoint(self, model, endpoint, endpoint_type, batch):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    async def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model, batch_size):
        if model in self.llama_cpp_endpoints:
            for endpoint in self.llama_cpp_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    async def make_local_request(self, model, endpoint, endpoint_type, data):
        device = torch.device(endpoint)
        inputs = self.tokenizer[model][endpoint](data, return_tensors="pt", padding=True, truncation=True).to(device)
        self.local_endpoints[model][endpoint].to(device).eval()
        with torch.no_grad():
            outputs = self.local_endpoints[model][endpoint](**inputs)
            query_response = outputs.last_hidden_state.mean(dim=1).tolist()  # Use mean of token embeddings
            results = query_response  # Return the entire batch of results
            del inputs, outputs  # Unallocate inputs and outputs
            torch.cuda.synchronize()  # Ensure all operations are complete
            torch.cuda.empty_cache()  # Free up GPU memory
        # self.local_endpoints[model][endpoint].to('cpu')  # Move model back to CPU
        torch.cuda.empty_cache()  # Free up GPU memory again
        return results

    
    async def add_endpoint(self, model, endpoint, context_length, endpoint_type):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if endpoint_type not in list(dir(self)):
                    self.__dict__[endpoint_type]= {}
                if model not in list(self.__dict__[endpoint_type].keys()):
                    self.__dict__[endpoint_type][model] = {}
                if endpoint not in list(self.__dict__[endpoint_type][model].keys()):
                    self.__dict__[endpoint_type][model][endpoint] = context_length
                self.endpoint_status[endpoint] = context_length
                success = True
            except Exception as e:
                print(e)
                pass
            return success        
        return None
    
    async def rm_endpoint(self, model, endpoint, endpoint_type):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if model in self.__dict__[endpoint_type] and endpoint in self.__dict__[endpoint_type][model]:
                    del self.__dict__[endpoint_type][model][endpoint]
                if endpoint in self.endpoint_status:
                    del self.endpoint_status[endpoint]
                success = True
            except Exception as e:
                print(e)
                pass
            return success
        return None
    
    
    def __test__(self):
        results = {}
        test_ipfs_accelerate_init = None
        test_ipfs_accelerate = None
        test_ipfs_accelerate = self.ipfs_accelerate.test()
        results = {"test_ipfs_accelerate_init": test_ipfs_accelerate_init, "test_ipfs_accelerate": test_ipfs_accelerate}
        return results
    
    
    async def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model, batch_size):
        if model in self.llama_cpp_endpoints:
            for endpoint in self.llama_cpp_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_endpoints and endpoint in self.tei_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False
    
    def test_openvino_endpoint(self, model, endpoint):
        if model in self.openvino_endpoints and endpoint in self.openvino_endpoints[model]:
            return True
        return False
    
    def test_local_endpoint(self, model, endpoint):
        if model in self.local_endpoints and endpoint in self.local_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.tei_endpoints:
            return self.tei_endpoints[model]
        return None

    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_tei_endpoint(self, model, endpoint, endpoint_type, batch):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    async def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")

    async def make_post_request_libp2p(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")
            
            
    async def make_post_request_openvino(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")

 
    async def choose_endpoint(self, model, endpoint_type=None):
        if endpoint_type != None:
            this_endpoints = self.get_endpoints(model, endpoint_type)
        else:
            tei_endpoints = self.get_endpoints(model, endpoint_type="tei")
            libp2p_endpoints = self.get_endpoints(model, endpoint_type="libp2p")
            openvino_endpoints = self.get_endpoints(model, endpoint_type="openvino")
            local_endpoints = self.get_endpoints(model, endpoint_type="local")
            filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
            filtered_tei_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and tei_endpoints is not None and k in list(tei_endpoints.keys())}
            filtered_openvino_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and openvino_endpoints is not None and k in list(openvino_endpoints.keys())}
            filtered_local_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and local_endpoints is not None and k in list(local_endpoints.keys())}
            if not filtered_tei_endpoints and not filtered_libp2p_endpoints and not filtered_openvino_endpoints and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                if len(list(filtered_local_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_local_endpoints.keys()))
                if len(list(filtered_tei_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_tei_endpoints.keys()))
                if len(list(filtered_openvino_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_openvino_endpoints.keys()))
                elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_libp2p_endpoints.keys()))
                print("chosen endpoint for " + model + " is " + this_endpoint)
                return this_endpoint


    async def get_endpoints(self, model, endpoint_type=None):
        if endpoint_type is None:
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "tei":
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "openvino":
            endpoints_dict = self.openvino_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "libp2p":
            endpoints_dict = self.libp2p_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "local":
            endpoints_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "cuda":
            endpoint_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoint_dict if "cuda" in endpoint and self.endpoint_status.get(endpoint, 0) >= 1]
        return filtered_endpoints
    
    
    async def async_generator(self, iterable):
        for item in iterable:
            yield item

ipfs_accelerate_py = ipfs_accelerate_py

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
    ipfs_accelerate_py = ipfs_accelerate_py(resources, metadata)
    results = ipfs_accelerate_py.__test__()
    
