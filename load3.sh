curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/German-ConcatX-Abstract", "knn_index":"laion/German-ConcatX-M3", "dataset_split": "train", "knn_index_split": "train", "columns": ["Concat Abstract","Title","URL"] }' \
    -H 'Content-Type: application/json'