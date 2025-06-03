from elasticsearch import Elasticsearch

# # Create an Elasticsearch client instance
# es = Elasticsearch("http://localhost:9200")

# # Test the connection
# if es.ping():
#     print("Connected to Elasticsearch")
# else:
#     print("Could not connect to Elasticsearch")
#     exit()

# Define the index name
# index_name = 'test-index'

# # Create an index if it doesn't exist
# if not es.indices.exists(index=index_name):
#     es.indices.create(index=index_name)
#     print(f"Created index '{index_name}'")
# else:
#     print(f"Index '{index_name}' already exists")

# # Sample documents
# documents = [
#     {'id': 1, 'title': 'First Document', 'content': 'This is the first document.'},
#     {'id': 2, 'title': 'Second Document', 'content': 'This is the second document.'},
#     {'id': 3, 'title': 'Third Document', 'content': 'This is the third document.'}
# ]

# # Index the documents
# for doc in documents:
#     response = es.index(index=index_name, id=doc['id'], document=doc)
#     print(f"Indexed document {doc['id']}: {response['result']}")

# # Refresh the index to make sure documents are searchable
# es.indices.refresh(index=index_name)

# # Search for documents
# response = es.search(index=index_name, query={"match_all": {}})
# print(f"Found {response['hits']['total']['value']} documents:")
# for hit in response['hits']['hits']:
#     print(hit['_source'])
