NUM_RANKS=1
CUDA_VISIBLE_DEVICES=6
vllm serve facebook/cwm \
    --tensor-parallel-size=$NUM_RANKS \
    --gpu-memory-utilization 0.9 \
    --port 8888