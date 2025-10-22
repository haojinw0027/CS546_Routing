#!/bin/bash
# Configuration examples for different reflection models

# Example 1: Using OpenAI GPT-4 (Best performance, higher cost)
# export OPENAI_API_KEY="sk-..."
# export REFLECTION_MODEL="openai/gpt-4"
# ./scripts/run_gepa_real.sh

# Example 2: Using OpenAI GPT-4 Mini (Good balance)
# export OPENAI_API_KEY="sk-..."
# export REFLECTION_MODEL="openai/gpt-4-mini"
# export MAX_CALLS=200
# ./scripts/run_gepa_real.sh

# Example 3: Using Anthropic Claude 3.5 Sonnet (Great performance)
# export ANTHROPIC_API_KEY="sk-ant-..."
# export REFLECTION_MODEL="anthropic/claude-3-5-sonnet-20241022"
# ./scripts/run_gepa_real.sh

# Example 4: Using OpenRouter with Qwen 80B (Cost-effective)
# export OPENROUTER_API_KEY="sk-or-..."
# export REFLECTION_MODEL="openrouter/qwen/qwen3-next-80b-a3b-thinking"
# ./scripts/run_gepa_real.sh

# Example 5: Quick test with small budget
# export OPENAI_API_KEY="sk-..."
# export REFLECTION_MODEL="openai/gpt-4-mini"
# export MAX_CALLS=50
# ./scripts/run_gepa_real.sh

# Example 6: Using custom dataset
# export OPENAI_API_KEY="sk-..."
# export DATASET="./data/my_custom_dataset.json"
# export SEED_PROMPTS="./system_prompts/custom_seeds.yaml"
# ./scripts/run_gepa_real.sh

# ================================================
# Quick Start Commands (Copy and modify these)
# ================================================

# Step 1: Set your API key
echo "Step 1: Set your API key"
echo "Choose ONE of the following:"
echo ""
echo "For OpenAI:"
echo "  export OPENAI_API_KEY='sk-...'"
echo ""
echo "For Anthropic:"
echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "For OpenRouter:"
echo "  export OPENROUTER_API_KEY='sk-or-...'"
echo ""

# Step 2: Start vLLM server
echo "Step 2: Start vLLM server (in another terminal)"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model Qwen/Qwen3-8B \\"
echo "    --port 8088"
echo ""

# Step 3: Run optimization
echo "Step 3: Run optimization"
echo "  ./scripts/run_gepa_real.sh"
echo ""
echo "Or with custom settings:"
echo "  REFLECTION_MODEL='openai/gpt-4-mini' MAX_CALLS=100 ./scripts/run_gepa_real.sh"
