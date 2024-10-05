curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/English-ConcatX-Abstract", "knn_index":"laion/English-ConcatX-M3", "dataset_split": "train", "knn_index_split": "train", "column": "Concat Abstract"}' \
    -H 'Content-Type: application/json'