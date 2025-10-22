#!/bin/bash
# Example script to run GEPA optimization with custom Claude API endpoint

# Configuration
TASK_MODEL="Qwen/Qwen3-8B"
TASK_PORT=8088
REFLECTION_API="http://172.31.13.66:8080/reward"
DATASET="./data/mixed_math_arc.json"
SEED_PROMPTS="./system_prompts/initial.yaml"
MAX_CALLS=150
OUTPUT_DIR="./results/gepa_optimization_custom_claude"

echo "=================================================="
echo "GEPA Optimization with Custom Claude API"
echo "=================================================="
echo "Task Model: $TASK_MODEL (port $TASK_PORT)"
echo "Reflection API: $REFLECTION_API"
echo "Dataset: $DATASET"
echo "Max Metric Calls: $MAX_CALLS"
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="
echo ""

# Run the optimization
python src/gepa_optimize_real.py \
    --model "$TASK_MODEL" \
    --port $TASK_PORT \
    --reflection-api-endpoint "$REFLECTION_API" \
    --dataset "$DATASET" \
    --seed-prompts "$SEED_PROMPTS" \
    --max-metric-calls $MAX_CALLS \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "=================================================="
echo "Optimization Complete!"
echo "=================================================="
echo "Check results in: $OUTPUT_DIR"
