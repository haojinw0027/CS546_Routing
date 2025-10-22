python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark aime_2025 \
    --port 8089 \
    --system-prompt-type always_cot \
    --max-tokens 2048 \
    --max-sample 30 & 
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark aime_2025 \
    --port 8089 \
    --system-prompt-type adaptive_cot \
    --max-tokens 2048 \
    --max-sample 30 &
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark aime_2025 \
    --port 8089 \
    --system-prompt-type direct_answer \
    --max-tokens 2048 \
    --max-sample 30 &
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark arc_challenge \
    --port 8089 \
    --system-prompt-type direct_answer \
    --max-tokens 2048 \
    --max-sample 1000 &
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark arc_challenge \
    --port 8089 \
    --system-prompt-type adaptive_cot \
    --max-tokens 2048 \
    --max-sample 1000 &
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark arc_challenge \
    --port 8089 \
    --system-prompt-type always_cot \
    --max-tokens 2048 \
    --max-sample 1000
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark math_500 \
    --port 8089 \
    --system-prompt-type always_cot \
    --max-tokens 2048 \
    --max-sample 500 &
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark math_500 \
    --port 8089 \
    --system-prompt-type adaptive_cot \
    --max-tokens 2048 \
    --max-sample 500 &
python baseline.py --model Qwen/Qwen3-8B-Base \
    --benchmark math_500 \
    --port 8089 \
    --system-prompt-type direct_answer \
    --max-tokens 2048 \
    --max-sample 500