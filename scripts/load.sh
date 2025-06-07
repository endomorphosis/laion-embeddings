curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "knn_index":"laion/Wikipedia-M3", "dataset_split": "enwiki_concat", "knn_index_split": "enwiki_embed", "columns": "Concat Abstract"}' \
    -H 'Content-Type: application/json'