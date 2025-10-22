#!/bin/bash
# Quick start script for GEPA optimization with real reflection model

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}GEPA Real Optimization Runner${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if GEPA is installed
if ! python -c "import gepa" 2>/dev/null; then
    echo -e "${RED}ERROR: GEPA library not found!${NC}"
    echo "Please install GEPA first:"
    echo "  pip install gepa"
    echo "  OR: pip install git+https://github.com/gepa-ai/gepa.git"
    exit 1
fi

# Default configuration
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
PORT=${PORT:-8088}
HOST=${HOST:-"localhost"}
REFLECTION_MODEL=${REFLECTION_MODEL:-"openai/gpt-4"}
MAX_CALLS=${MAX_CALLS:-150}
DATASET=${DATASET:-"./data/mixed_math_arc.json"}
SEED_PROMPTS=${SEED_PROMPTS:-"./system_prompts/initial.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results/gepa_optimization_real"}

# Check for API keys
echo -e "${YELLOW}Checking configuration...${NC}"
echo "Task Model: $MODEL"
echo "Reflection Model: $REFLECTION_MODEL"
echo ""

if [[ $REFLECTION_MODEL == *"openai"* ]] || [[ $REFLECTION_MODEL == *"gpt"* ]]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}ERROR: OPENAI_API_KEY not set!${NC}"
        echo "Please set your OpenAI API key:"
        echo "  export OPENAI_API_KEY='sk-...'"
        exit 1
    fi
    echo -e "${GREEN}✓ OpenAI API key found${NC}"
elif [[ $REFLECTION_MODEL == *"anthropic"* ]] || [[ $REFLECTION_MODEL == *"claude"* ]]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${RED}ERROR: ANTHROPIC_API_KEY not set!${NC}"
        echo "Please set your Anthropic API key:"
        echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
        exit 1
    fi
    echo -e "${GREEN}✓ Anthropic API key found${NC}"
fi

# Check if vLLM server is running
echo ""
echo -e "${YELLOW}Checking vLLM server...${NC}"
if curl -s http://$HOST:$PORT/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ vLLM server is running on $HOST:$PORT${NC}"
else
    echo -e "${RED}ERROR: vLLM server not responding on $HOST:$PORT${NC}"
    echo "Please start your vLLM server first:"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $MODEL \\"
    echo "    --port $PORT"
    exit 1
fi

# Check if files exist
echo ""
echo -e "${YELLOW}Checking files...${NC}"
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}ERROR: Dataset not found: $DATASET${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset found: $DATASET${NC}"

if [ ! -f "$SEED_PROMPTS" ]; then
    echo -e "${RED}ERROR: Seed prompts not found: $SEED_PROMPTS${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Seed prompts found: $SEED_PROMPTS${NC}"

# Run optimization
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting GEPA Optimization${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Configuration:"
echo "  Task Model: $MODEL"
echo "  vLLM Server: $HOST:$PORT"
echo "  Reflection Model: $REFLECTION_MODEL"
echo "  Max Metric Calls: $MAX_CALLS"
echo "  Dataset: $DATASET"
echo "  Seed Prompts: $SEED_PROMPTS"
echo "  Output Directory: $OUTPUT_DIR"
echo ""
echo -e "${YELLOW}This may take several minutes to hours depending on max_calls...${NC}"
echo ""

python src/gepa_optimize_real.py \
  --model "$MODEL" \
  --port $PORT \
  --host "$HOST" \
  --reflection-model "$REFLECTION_MODEL" \
  --dataset "$DATASET" \
  --seed-prompts "$SEED_PROMPTS" \
  --max-metric-calls $MAX_CALLS \
  --output-dir "$OUTPUT_DIR"

# Check if optimization succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Optimization Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Check the optimized prompt in the output directory"
    echo "  2. Evaluate it on your test set"
    echo "  3. Compare with baseline prompts"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}Optimization Failed!${NC}"
    echo -e "${RED}================================${NC}"
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  - API key not set or invalid"
    echo "  - vLLM server crashed or unreachable"
    echo "  - GEPA library not installed"
    exit 1
fi
