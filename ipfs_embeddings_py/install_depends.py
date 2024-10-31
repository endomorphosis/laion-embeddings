class install_depends():
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.results = {}
        self.stdout = {}
        self.stderr = {}
        return None
    
    def install(self, resources=None):        
        if resources is None:
            if self.resources is not None and len(list(self.resources.keys())) != 0:
                resources = self.resources
            else:
                resources["packagess"] = ["faiss", "faiss-cuda", "faiss-amx", "qdrant", "elasticsearch"]
            pass
        for package in self.resources["packages"]:
            try:
                self.stdout[package] = self.install_package(package)
            except Exception as e:
                self.stderr[package] = e
                print(e)
            install_results = [ stdout if stdout else stderr for stdout, stderr in zip(self.stdout, self.stderr) ] 
        return install_results

    def install_package(self, package):
        if package == "faiss":
            return self.install_faiss()
        elif package == "faiss-cuda":
            return self.install_faiss_cuda()
        elif package == "faiss-amx":
            return self.install_faiss_amx()
        elif package == "qdrant":
            return self.install_qdrant()
        elif package == "elasticsearch":
            return self.install_elasticsearch()
        else:
            return None
    
    def install_cuda(self):
        return None
    
    def install_faiss(self):
        return None
    
    def install_faiss_cuda(self):
        return None
    
    def install_faiss_amx(self):
        return None
    
    def install_qdrant(self):
        return None
    
    def install_elasticsearch(self):
        return None
