# CS546 System Prompt

This repository contains two approaches for improving reasoning in language models.

## 1. System Prompt Reasoning Routing

**File:** `src/gepa_traj.py`

This approach uses GEPA (Genetic Prompt Algorithm) to automatically optimize system prompts for mathematical reasoning tasks.

### How It Works

- Uses a task model (vLLM server) to solve math problems
- Uses a reflection model (Claude via AWS Bedrock) to analyze performance and suggest improvements
- Iteratively evolves system prompts based on feedback
- Logs detailed trajectories of the optimization process

### Usage

1. Start a vLLM server on port 8088:
```bash
vllm serve Qwen/Qwen3-14B --port 8088
```

2. Start the Claude reflection service:
```bash
python web.py  # Should run on port 8080
```

3. Run the GEPA training:
```bash
python src/gepa_traj.py
```

### Configuration

Edit the following in `gepa_traj.py`:
- `train_file`: Path to training data (JSON format)
- `output_file`: Where to save trajectory logs
- `seed_prompt`: Initial system prompt (loaded from YAML)
- `max_metric_calls`: Number of evaluation iterations

### Output

The script generates a JSON file containing:
- Iteration-by-iteration improvements
- Reflection prompts and responses
- Query-answer pairs with scores
- Final optimized system prompt

## 2. Routing through MentorCollab

**File:** `mentorcollab/mentorcollab.py`

This approach uses token-level routing between a generator model and a mentor model during inference.

### How It Works

- **Generator**: Base model that generates most tokens
- **Mentor**: More capable model that provides guidance at decision points
- During generation, the system probabilistically queries the mentor
- When generator and mentor disagree, generator self-reflects to choose the better path

### Usage

```python
from mentorcollab import MentorCollab

# Initialize with generator and mentor models
mentor_collab = MentorCollab(
    generator="meta-llama/Llama-3.1-8B",
    mentor="meta-llama/Llama-3.1-8B-Instruct",
    generator_devices="cuda:0",
    mentor_devices="cuda:1",
    decision_proportion=25,  # Query mentor 25% of the time
    patch_size=16            # Lookahead tokens for comparison
)

# Generate text with mentor guidance
output = mentor_collab.generate(
    prompt="Your question here",
    max_new_tokens=100
)
```

### Key Parameters

- `decision_proportion`: How often to consult mentor (1-100)
- `patch_size`: Number of tokens to look ahead when comparing options

## Results

More detailed experimental results will be published after ARR submission.

## Requirements

- PyTorch
- Transformers
- vLLM
- AWS Bedrock (for Claude access)
- GEPA framework

## Acknowledgements

Our first approach (System Prompt Reasoning Routing) is based on and extends the [GEPA framework](https://github.com/gepa-ai/gepa). We adapted their genetic prompt optimization algorithm to work with our custom reflection pipeline and trajectory logging system.
