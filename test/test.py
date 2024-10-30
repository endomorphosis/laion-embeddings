from ..search_embeddings import search_embeddings
from ..create_embeddings import create_embeddings

class test_search_embeddings:
    def __init__(self, resources, metadata):
        self.search_embeddings = search_embeddings.search_embeddings(resources, metadata)
        self.create_embeddings = create_embeddings.create_embeddings(resources, metadata)
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
        test_create = self(test_dataset, test_faiss_index, test_model)
        print(test_create)
        return None 
    
class test_index_dataset:
    def __init__(self, resources, metadata):
        self.create_embeddings = create_embeddings.create_embeddings(resources, metadata)
        return None
    
    def __call__(self, request):
        return self.create_embeddings.index_dataset(request)
    
    def __test__(self):
        test_dataset = "laion/Wikipedia-X-Concat"
        test_faiss_index = "laion/Wikipedia-M3"
        test_model = "BAAI/bge-m3"
        test_create = self(test_dataset, test_faiss_index, test_model)
        print(test_create)
        return None
    
class test_search_dataset:
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
    
if __name__ == '__main__':
    metadata = {
        "dataset": "laion/Wikipedia-X-Concat",
        "faiss_index": "laion/Wikipedia-M3",
        "model": "BAAI/bge-m3"
    }
    resources = {
        "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",1]]
    }
    test_search = test_search_embeddings(resources, metadata)
    test_search.__test__()

    test_create = test_create_embeddings(resources, metadata)
    test_create.__test__()
    