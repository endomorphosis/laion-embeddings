import subprocess
from qdrant_kit import qdrant_kit_py
from ipfs_parquet_to_car import ipfs_parquet_to_car_py

class install_depends_py():
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.results = {}
        self.stdout = {}
        self.stderr = {}
        self.install_results = {}
        return None
    
    async def install(self, resources=None):        
        if resources is None:
            if self.resources is not None and len(list(self.resources.keys())) != 0:
                resources = self.resources
            else:
                resources["packagess"] = ["faiss", "faiss-cuda", "faiss-amx", "qdrant", "elasticsearch"]
            pass
        for package in self.resources["packages"]:
            try:
                self.stdout[package] = await self.install_package(package)
            except Exception as e:
                self.stderr[package] = e
                print(e)
            install_results = [ stdout if stdout else stderr for stdout, stderr in zip(self.stdout, self.stderr) ] 
        return install_results

    async def install_package(self, package):
        if package == "faiss":
            return await self.install_faiss()
        elif package == "faiss-cuda":
            return await self.install_faiss_cuda()
        elif package == "faiss-amx":
            return await self.install_faiss_amx()
        elif package == "qdrant":
            return await self.install_qdrant()
        elif package == "elasticsearch":
            return await self.install_elasticsearch()
        elif package == "openvino":
            return await self.install_openvino()
        elif package == "ipex":
            return await self.install_ipex()
        elif package == "tortch":
            return await self.install_torch()
        elif package == "storacha":
            return await self.install_storacha()
        elif package == "ipfs":
            return await self.install_ipfs_kit()
        else:
            return None
    
    async def install_ipfs_kit(self):
        return None    
    
    async def install_storacha(self):
        return None
    
    async def install_torch(self):
        return None
            
    async def install_openvino(self):
        return None
    
    async def install_ipex(self):
        return None
    
    async def install_cuda(self):
        install_results = {}
        install_cuda_cmd = ["apt-get", "install", "nvidia-cuda-toolkit"]
        try:
            install_results["install_cuda"] = subprocess.run(install_cuda_cmd, check=True)
        except Exception as e:
            install_results["install_cuda"] = e
            print(e)
        try:
            import torch
            torch.cuda.is_available()
            install_results["install_cuda"] = True
        except Exception as e:
            install_results["install_cuda"] = e
            print(e)
            
        install_results["install_cuda"] = None
        
        return None
    
    async def install_faiss(self):
        return None
    
    async def install_faiss_cuda(self):
        return None
    
    async def install_faiss_amx(self):
        return None
    
    async def install_qdrant(self):
        return None
    
    async def install_elasticsearch(self):
        return None
