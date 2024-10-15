## assumes 2 3090 GPUs
hf_token=YOUR_HF_TOKEN
volume=/storage/hf_models
model=Alibaba-NLP/gte-large-en-v1.5
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model  --max-batch-tokens 8192 --payload-limit 32000000 &
docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8081:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model  --max-batch-tokens 8192 --payload-limit 32000000 &
volume=/storage/hf_models
model=Alibaba-NLP/gte-Qwen2-1.5B-instruct
sleep 15 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8082:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 --payload-limit 32000000 &
sleep 15 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8083:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 --payload-limit 32000000 &
volume=/storage/hf_models
model=thenlper/gte-small
sleep 30 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8084:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 512 --payload-limit 32000000 &
sleep 30 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8085:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 512 --payload-limit 32000000 &
volume=/storage/hf_models
# model=Alibaba-NLP/gte-Qwen2-7B-instruct
# sleep 30 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8084:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 --payload-limit 32000000 --api-key $hf_token &
# sleep 30 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8085:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 --payload-limit 32000000 --api-key $hf_token &
