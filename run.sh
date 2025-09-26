python baseline.py --model Qwen/Qwen3-8B \
    --benchmark aime_2025 \
    --system-prompt-type always_cot \
    --max-tokens 2048 \
    --max-sample 30 & 
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark aime_2025 \
    --system-prompt-type adaptive_cot \
    --max-tokens 2048 \
    --max-sample 30 &
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark aime_2025 \
    --system-prompt-type direct_answer \
    --max-tokens 2048 \
    --max-sample 30 &
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark arc_challenge \
    --system-prompt-type direct_answer \
    --max-tokens 2048 \
    --max-sample 1000 &
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark arc_challenge \
    --system-prompt-type adaptive_cot \
    --max-tokens 2048 \
    --max-sample 1000 &
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark arc_challenge \
    --system-prompt-type always_cot \
    --max-tokens 2048 \
    --max-sample 1000