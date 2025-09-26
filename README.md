# System Prompt Optimization

A tool for optimizing and evaluating different system prompts on language models. Compare the effectiveness of various prompting strategies across different benchmarks.

## Requirements

```bash
pip install requests pyyaml datasets
```

## System Prompt Types

This tool supports various system prompt strategies for different reasoning approaches:

### Available Prompt Types

Defined in `system_prompts/initial.yaml`:

- **`always_cot`**: Always require step-by-step reasoning with explicit verification
- **`adaptive_cot`**: Adaptive reasoning that triggers based on problem complexity
- **`direct_answer`**: Direct answers with minimal reasoning for efficiency

### Prompt Strategy Details

#### Always CoT (`always_cot`)
Forces the model to:
1. Think step by step with numbered reasoning
2. Verify assumptions and check edge cases
3. Provide a clear final answer
4. Show all intermediate calculations

Best for: Complex reasoning tasks, mathematical problems, multi-step analysis

#### Adaptive CoT (`adaptive_cot`)
Intelligently decides when to show reasoning:
- Direct answers for simple lookup/definition questions
- Concise reasoning for multi-step problems
- Full reasoning only when uncertainty is high

Best for: Mixed-complexity benchmarks, general-purpose evaluation

#### Direct Answer (`direct_answer`)
Optimized for efficiency:
- Minimal visible reasoning process
- Concise 1-3 sentence responses
- Brief justification only when absolutely necessary

Best for: Large-scale evaluation, time-sensitive tasks, simple Q&A

## Usage

### Basic Usage

```bash
python baseline.py --model <model_name> --benchmark <benchmark_name> --system-prompt-type <prompt_type>
```

### System Prompt Parameters
- `--system-prompt-type`: Choose prompt strategy (always_cot, adaptive_cot, direct_answer)
- `--system-prompt-yaml`: Path to system prompt YAML file (default: ./system_prompts/initial.yaml)
- `--system`: Custom system prompt (overrides YAML settings)

### System Prompt Comparison Examples

#### Compare Different Strategies on AIME 2025
```bash
# Test step-by-step reasoning approach
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark aime_2025 --system-prompt-type always_cot

# Test adaptive reasoning approach
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark aime_2025 --system-prompt-type adaptive_cot

# Test direct answer approach
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark aime_2025 --system-prompt-type direct_answer
```

#### Compare on ARC Challenge
```bash
# For science reasoning tasks
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark arc_challenge --system-prompt-type always_cot
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark arc_challenge --system-prompt-type adaptive_cot
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark arc_challenge --system-prompt-type direct_answer
```

#### Quick Testing with Limited Samples
```bash
# Test different prompts on smaller sample for fast iteration
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark aime_2025 --system-prompt-type always_cot --max-sample 5
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark aime_2025 --system-prompt-type adaptive_cot --max-sample 5
```

### Custom System Prompts

#### Creating Custom Prompts
```bash
# Use a completely custom system prompt
python baseline.py \
  --model meta-llama/Llama-3.2-3B \
  --benchmark aime_2025 \
  --system "You are a math expert. Always show your work step by step and verify your answer."
```

#### Adding New Prompt Types
Edit `system_prompts/initial.yaml` to add new prompt strategies:

```yaml
prompts:
  - id: my_custom_prompt
    title: "My Custom Strategy"
    description: "A custom prompting approach"
    content: |-
      You are a helpful assistant with expertise in problem-solving.
      Always approach problems methodically and explain your reasoning.
    tags: ["custom", "reasoning"]
    version: "1.0"
```

Then use it:
```bash
python baseline.py --model meta-llama/Llama-3.2-3B --benchmark aime_2025 --system-prompt-type my_custom_prompt
```

## Results and Analysis

### Output File Naming
Results are automatically saved with descriptive names:
```
./results/benchmark/{system_prompt_type}_{model_short}_{benchmark_name}.json
```

Examples:
- `always_cot_llama3.2_3B_AIME_2025_default.json`
- `adaptive_cot_llama3.2_3B_ARC_Challenge_test.json`

### Comparing System Prompt Performance

#### Key Metrics to Compare
1. **Accuracy**: Overall correctness rate
2. **Response Length**: Token efficiency
3. **Reasoning Quality**: Clarity of explanation
4. **Error Patterns**: Types of mistakes made

#### Performance Analysis
Compare different system prompts by examining:

```json
{
  "evaluation": {
    "accuracy": 0.167,
    "correct": 5,
    "total": 30,
    "detailed_results": [
      {
        "problem_id": 1,
        "prediction": "Step 1: ...\nStep 2: ...\nFinal Answer: 42",
        "correct": true
      }
    ]
  }
}
```

### Expected Performance Patterns

| Prompt Type | Math Tasks | Reasoning Tasks | Simple Q&A | Token Usage |
|-------------|------------|-----------------|------------|-------------|
| `always_cot` | High accuracy | High accuracy | Over-detailed | High |
| `adaptive_cot` | Good balance | Good balance | Appropriate | Medium |
| `direct_answer` | Lower accuracy | Lower accuracy | Efficient | Low |

### Optimization Tips

1. **For Mathematical Problems**: Use `always_cot` for highest accuracy
2. **For Mixed Benchmarks**: Use `adaptive_cot` for best balance
3. **For Large-Scale Evaluation**: Use `direct_answer` for efficiency
4. **For Custom Tasks**: Create task-specific prompts in YAML file

### Model Deployment Setup

To use this tool, ensure you have a model deployed via vLLM:

```bash
# Example deployment
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B \
    --served-model-name meta-llama/Llama-3.2-3B
```