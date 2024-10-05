metadata='{
    "dataset": "TeraflopAI/Caselaw_Access_Project",
    "column": "text",
    "models": [
        "BAAI/bge-m3",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    ],
    "dst_path": "/storage/teraflopai"
}'
resources='{
    "https_endpoints": [
        ["BAAI/bge-m3", "http://127.0.0.1:8080/embed", 8188],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://127.0.0.1:8081/embed", 32766],
        ["dunzhang/stella_en_1.5B_v5", "http://127.0.0.1:8082/embed", 131072]
    ]
}'

# model name, endpoint name, context length

curl 127.0.0.1:9999/create \
    -X POST \
    -d "{\"resources\": $resources, \"metadata\": $metadata}" \
    -H 'Content-Type: application/json'

