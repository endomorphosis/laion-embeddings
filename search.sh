curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "English-ConcatX-Abstract"}' \
    -H 'Content-Type: application/json'

