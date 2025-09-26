python baseline.py --model Qwen/Qwen3-8B \
    --benchmark aime_2025 \
    --system-prompt-type always_cot \
    --max-sample 30 & 
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark aime_2025 \
    --system-prompt-type adaptive_cot \
    --max-sample 30 &
python baseline.py --model Qwen/Qwen3-8B \
    --benchmark aime_2025 \
    --system-prompt-type direct_answer \
    --max-sample 30