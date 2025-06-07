#!/bin/bash
curl 127.0.0.1:/create_sparse \
    -X POST \
    -d '["TeraflopAI/Caselaw_Access_Project", "train", "text", "/storage/teraflopai/tmp", ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]]' \
    -H 'Content-Type: application/json'