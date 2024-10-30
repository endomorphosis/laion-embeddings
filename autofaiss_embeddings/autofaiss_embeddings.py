from ..ipfs_embeddings_py import ipfs_embeddings_py
 
class autofaiss_embeddings:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        return None
        
    def test(self):
        results = {}
        test_auto_faiss_chunks = None
        test_auto_faiss_shards = None
        try:
            test_auto_faiss = self.ipfs_embeddings_py.autofaiss_chunks()
        except Exception as e:
            test_auto_faiss = e
            print(e)
            raise e
        
        try:
            test_auto_faiss_shards = self.ipfs_embeddings_py.autofaiss_shards()
        except Exception as e:
            test_auto_faiss_shards = e
            print(e)
            raise e
        results = {"test_auto_faiss_chunks": test_auto_faiss_chunks, "test_auto_faiss_shards": test_auto_faiss_shards}
        print(results)
        return results
    
if __name__ == "main":
    metadata = {}
    resources = {}
    test_auto_faiss = autofaiss_embeddings(resources, metadata)
    test_auto_faiss.test()
 
    