curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "German-ConcatX-Abstract", "n": 10}' \
    -H 'Content-Type: application/json'

