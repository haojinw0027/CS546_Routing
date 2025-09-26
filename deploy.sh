CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 16384 \
    --port 8088
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen3-8B-Base \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 16384 \
    --port 8089