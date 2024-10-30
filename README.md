IPFS Embeddings Search Engine

To start the endpoint:
./run.sh

this runs a python command 

```
python3 -m fastapi run main.py
```

This is an example of how to  to pull the qdrant docker container and load the embeddings into it
./load.sh

This runs a curl command that will import a K nearest neighbors index, and a dataset index, where the tables are joined on the column "column" by requesting the api endpoint to do so.

```
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "knn_index":"laion/Wikipedia-M3", "dataset_split": "enwiki_concat", "knn_index_split": "enwiki_embed", "column": "Concat Abstract"}' \
    -H 'Content-Type: application/json'
```
NOTE: THAT THIS WILL TAKE HOURS TO DOWNLOAD / INGEST FOR LARGE DATASETS
NOTE: FAST API IS UNAVAILABLE WHILE THIS RUNS

Then to search the index there is an example in this file
./search.sh 

this runs a curl command, which queries the index with the text

```
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "Wikipedia-X-Concat"}' \
    -H 'Content-Type: application/json'
```

To create an index from a dataset, which the ouputs will be stored in the "checkpoints" directory

```
./create.sh
```

this creates a curl command which queries the endpoint, to download a huggingface dataset, and to generate dense embeddings from it

```
curl 127.0.0.1:/create \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

To index the ipfs_cluster that this node is running on

```
./index_cluster.sh
```

This runs a curl request against the api, to send a command to the ipfs node to output its cid's list, so that the embedding models can index the cids, and output the results in the "checkpoints" directory

```
#!/bin/bash
curl 127.0.0.1:/index_cluster \
    -X POST \
    -d '["loclhost", "cloudkit_storage", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

To create an sparse index from a dataset, which the ouputs will be stored in the "sparse_checkpoints" directory

```
./create_sparse.sh
```

this creates a curl command which queries the endpoint, to download a huggingface dataset, and to generate sparse embeddings from it

```
curl 127.0.0.1:/create_sparse \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

To shard the indexes using K means clustering

```
./shard_cluster.sh
```

this creates a curl command which queries the endpoint, to shard the clusters into sizes no larger than 50MB or 4096 rows.

```
curl 127.0.0.1:/shard_cluster \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```

To upload the indices that you have sharded to storacha use the following command

```
./storacha.sh
```

this runs a curl command that makes the package convert all the parquet files into car files, and uploads it to the storacha network.

```
curl 127.0.0.1:/storacha \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'
```
