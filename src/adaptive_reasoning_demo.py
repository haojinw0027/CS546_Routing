"""
Adaptive Reasoning Length Optimization Demo

Goal: Teach the model to adjust reasoning depth based on problem difficulty
- Simple ARC problems → Short reasoning
- Complex MATH problems → Long reasoning
"""

import json
import dspy
import requests
from adaptive_reasoning_metric import (
    metric_with_adaptive_reasoning_feedback,
    metric_simple_adaptive
)

# Reuse ClaudeBedrock class from dem.py
import sys
sys.path.append('/home/haojinw2/efs/haojin/vlaa/system_prompt/src')
from dem import ClaudeBedrock


class LocalLLM(dspy.LM):
    """
    DSPy Language Model wrapper for local LLM via vLLM API.
    Supports prompt logging for debugging.
    """

    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 8088,
        temperature: float = 1.0,
        max_tokens: int = 8000,
        log_prompts: bool = True,
        **kwargs
    ):
        """
        Initialize Local LLM.

        Args:
            model: Model name
            host: Server host
            port: Server port
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            log_prompts: Whether to print prompts before calling LLM
        """
        self.model = model
        self.host = host
        self.port = port
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_prompts = log_prompts
        self.kwargs = kwargs
        self.call_count = 0

        # Call parent constructor
        super().__init__(model=model)

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Call the local LLM via completions API.

        Args:
            prompt: Text prompt
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            List of generated responses
        """
        self.call_count += 1

        # Merge kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        temperature = merged_kwargs.get('temperature', self.temperature)
        max_tokens = merged_kwargs.get('max_tokens', self.max_tokens)

        # Convert messages to single prompt if needed
        if messages is not None:
            # Combine system and user messages
            full_prompt = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    full_prompt += f"{content}\n\n"
                elif role == "user":
                    full_prompt += f"Question: {content}\n\nAnswer:"
            prompt = full_prompt

        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")

        # Log prompt if enabled
        if self.log_prompts:
            print("\n" + "=" * 80)
            print(f"[LocalLLM Call #{self.call_count}] {self.model}")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")

        # Call completions API
        url = f"http://{self.host}:{self.port}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ['<|endoftext|>', '\nQuestion']
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["text"].strip()

            if self.log_prompts:
                print(f"[LocalLLM Response #{self.call_count}]")
                print(text[:500] + ("..." if len(text) > 500 else ""))
                print("\n")

            return [text]
        except Exception as e:
            print(f"Error calling local LLM: {e}")
            return [""]

    def basic_request(self, prompt: str, **kwargs):
        """Basic request interface for DSPy compatibility."""
        return self.__call__(prompt=prompt, **kwargs)


class ClaudeBedrockWithLogging(ClaudeBedrock):
    """
    Wrapper around ClaudeBedrock that logs prompts.
    """

    def __init__(self, *args, log_prompts: bool = True, model_label: str = "Reflection", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_prompts = log_prompts
        self.model_label = model_label
        self.call_count = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Call with logging."""
        self.call_count += 1

        # Log prompt if enabled
        if self.log_prompts:
            # Reconstruct full prompt for display
            display_prompt = ""
            if messages is not None:
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        display_prompt += f"[SYSTEM]\n{content}\n\n"
                    elif role == "user":
                        display_prompt += f"[USER]\n{content}\n\n"
            elif prompt is not None:
                display_prompt = prompt

            print("\n" + "=" * 80)
            print(f"[{self.model_label} Call #{self.call_count}] {self.model}")
            print("=" * 80)
            print(display_prompt)
            print("=" * 80 + "\n")

        # Call parent
        result = super().__call__(prompt=prompt, messages=messages, **kwargs)

        if self.log_prompts and result:
            print(f"[{self.model_label} Response #{self.call_count}]")
            response_text = result[0] if isinstance(result, list) else str(result)
            print(response_text[:500] + ("..." if len(response_text) > 500 else ""))
            print("\n")

        return result


def load_mixed_data(file_path):
    """Load mixed dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        examples.append(
            dspy.Example({
                'question': item['formatted_prompt'],
                'answer': item['answer'],
                'task_type': item['task_type'],
            }).with_inputs('question')
        )

    return examples


class AdaptiveReasoning(dspy.Signature):
    """You are a helpful assistant."""
    question = dspy.InputField()
    answer = dspy.OutputField()


def main():
    print("=" * 80)
    print("Adaptive Reasoning Length Optimization Experiment")
    print("=" * 80)

    # 1. Initialize model
    print("\n[1/4] Initializing model...")

    # Use local Qwen3-8B model on port 8088
    lm = LocalLLM(
        model="Qwen/Qwen3-8B",
        host="localhost",
        port=8088,
        temperature=1.0,
        max_tokens=8000,
        log_prompts=True
    )

    # Use Claude Sonnet 4 for reflection with logging
    reflection_lm = ClaudeBedrockWithLogging(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-west-2",
        temperature=1.0,
        max_tokens=32000,
        log_prompts=True,
        model_label="Reflection Model"
    )

    dspy.configure(lm=lm)
    print(f"  Main model: {lm.model} (Local on port {lm.port})")
    print(f"  Reflection model: {reflection_lm.model}")

    # 2. Load data
    print("\n[2/4] Loading mixed dataset...")
    data_path = "/home/haojinw2/efs/haojin/vlaa/system_prompt/data/mixed_math_arc_200.json"
    all_examples = load_mixed_data(data_path)

    # Split data
    train_size = int(len(all_examples) * 0.5)
    val_size = int(len(all_examples) * 0.25)

    train_set = all_examples[:train_size]
    val_set = all_examples[train_size:train_size + val_size]
    test_set = all_examples[train_size + val_size:]

    print(f"  Training set: {len(train_set)} questions")
    print(f"  Validation set: {len(val_set)} questions")
    print(f"  Test set: {len(test_set)} questions")

    # Count task type distribution
    task_counts = {}
    for ex in train_set:
        task_type = ex.task_type
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"  Training set distribution: {task_counts}")

    # 3. Define program
    print("\n[3/4] Defining Chain of Thought program...")
    program = dspy.ChainOfThought(AdaptiveReasoning)

    # 4. Evaluate unoptimized program
    print("\n[4/4] Evaluating unoptimized program...")
    evaluate = dspy.Evaluate(
        devset=test_set[:5],  # Use small sample for testing
        metric=metric_simple_adaptive,
        num_threads=1,
        display_table=True,
        display_progress=True
    )

    unoptimized_score = evaluate(program)
    print(f"\n  Unoptimized score: {unoptimized_score}")

    # 5. Optimize with GEPA
    print("\n[5/4] Optimizing prompt with GEPA...")
    print("  This will teach the model to adjust reasoning depth based on problem difficulty...")

    optimizer = dspy.GEPA(
        metric=metric_with_adaptive_reasoning_feedback,  # Use detailed feedback
        num_threads=1,
        track_stats=True,
        reflection_minibatch_size=2,  # Let reflection model analyze 2 questions per batch
        reflection_lm=reflection_lm,
        max_metric_calls=50
    )

    print("\n  Starting optimization...")
    optimized_program = optimizer.compile(
        program,
        trainset=train_set[:10],  # Use small sample for optimization
        valset=val_set[:5],
    )

    print("\n  Optimization complete!")

    # 6. Evaluate optimized program
    print("\n[6/4] Evaluating optimized program...")
    optimized_score = evaluate(optimized_program)
    print(f"\n  Optimized score: {optimized_score}")

    # 7. Display optimized prompt
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPT:")
    print("=" * 80)
    print(optimized_program.predict.signature.instructions)
    print("=" * 80)

    # 8. Comparative analysis
    print("\n" + "=" * 80)
    print("Optimization Effect Analysis:")
    print("=" * 80)
    print(f"Unoptimized score: {unoptimized_score}")
    print(f"Optimized score: {optimized_score}")
    print(f"Improvement: {float(optimized_score) - float(unoptimized_score):.2%}")
    print("=" * 80)

    # 9. Test a few examples to see reasoning length changes
    print("\n[Test] Checking optimized reasoning length...")
    test_cases = [
        ("ARC simple question", test_set[0]),
        ("MATH question", test_set[1]) if len(test_set) > 1 else None,
    ]

    for name, example in test_cases:
        if example is None:
            continue
        print(f"\n{name}:")
        print(f"  Question: {example.question[:100]}...")

        # Unoptimized version
        pred_before = program(question=example.question)
        reasoning_before = getattr(pred_before, 'rationale', '')
        print(f"  Unoptimized reasoning length: {len(reasoning_before.split())} words")

        # Optimized version
        pred_after = optimized_program(question=example.question)
        reasoning_after = getattr(pred_after, 'rationale', '')
        print(f"  Optimized reasoning length: {len(reasoning_after.split())} words")

        if len(reasoning_after.split()) < len(reasoning_before.split()):
            print(f"  ✓ Reasoning became shorter, more efficient!")
        else:
            print(f"  → Reasoning length change: {len(reasoning_after.split()) - len(reasoning_before.split())} words")


if __name__ == "__main__":
    main()
