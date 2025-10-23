#!/usr/bin/env python3
"""
Test script for evaluating the optimized prompt on mixed dataset.

This script:
1. Loads the optimized system prompt from file
2. Tests on the mixed ARC/MATH dataset
3. Evaluates accuracy and reasoning length
4. Saves detailed results
"""

import json
import os
import requests
from typing import Optional
from tqdm import tqdm
import argparse


def load_optimized_prompt(prompt_file: str) -> str:
    """Load optimized system prompt from file."""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Optimized prompt file not found: {prompt_file}")

    with open(prompt_file, 'r') as f:
        content = f.read()

    # Extract the prompt between the === markers
    lines = content.split('\n')
    prompt_lines = []
    in_prompt = False

    for line in lines:
        if '=' * 40 in line:
            if in_prompt:
                break
            in_prompt = True
            continue
        if in_prompt and line.strip() and not line.startswith('Unoptimized') and not line.startswith('Optimized') and not line.startswith('Improvement'):
            prompt_lines.append(line)

    prompt = '\n'.join(prompt_lines).strip()

    if not prompt:
        # If extraction failed, just use the whole file
        prompt = content

    return prompt


def load_mixed_dataset(data_path: str):
    """Load mixed ARC/MATH dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['examples']


def prompt_model(
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    host: str = "localhost",
    port: int = 8088
) -> Optional[str]:
    """Send a prompt to the model and return the response."""
    chat_url = f"http://{host}:{port}/v1/completions"

    # Combine system prompt and user prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nReasoning:"
    else:
        full_prompt = f"Question: {question}\n\nReasoning:"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        'stop': ['<|endoftext|>', '\nQuestion']
    }
    print(full_prompt)
    try:
        response = requests.post(chat_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        generated_text = result["choices"][0]["text"].strip()
        return generated_text
    except requests.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing response: {e}")
        return None


def extract_answer(response: str, task_type: str) -> str:
    """Extract answer from model response."""
    if not response:
        return "NO_ANSWER"

    # Try to find the last occurrence of "Answer:" or similar patterns
    response_lower = response.lower()

    if task_type == 'arc':
        # For ARC, look for single letter A, B, C, D
        import re
        # Try to find answer pattern like "Answer: A" or "The answer is A"
        patterns = [
            r'answer\s*(?:is|:)\s*([A-D])',
            r'\b([A-D])\s*(?:is|\.)',
            r'^([A-D])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Fallback: find any single capital letter A-D
        for char in ['A', 'B', 'C', 'D']:
            if char in response:
                return char

        return "NO_ANSWER"
    else:
        # For MATH, try to extract integer
        import re
        # 首先找有没有 \boxed{...}，优先匹配 {} 里的内容（一般是数字）
        boxed_match = re.search(r'\\boxed\{([^\}]*)\}', response)
        if boxed_match:
            inner = boxed_match.group(1)
            # 尝试只提取数字部分
            digit_match = re.search(r'([-+]?\d+\.?\d*)', inner)
            if digit_match:
                return digit_match.group(1)
            return inner.strip()
        return "NO_ANSWER"


def check_correctness(predicted: str, correct: str, task_type: str) -> bool:
    """Check if prediction is correct."""
    if predicted == "NO_ANSWER":
        return False

    try:
        if task_type == 'arc':
            return predicted.strip().upper() == str(correct).strip().upper()
        else:
            return int(predicted) == int(correct)
    except:
        return False


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(description="Test optimized prompt on mixed dataset")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--port", "-p", type=int, default=8088, help="vLLM server port")
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--prompt-file", default="/home/haojinw2/efs/haojin/vlaa/system_prompt/results/optimized_prompt.txt",
                        help="Path to optimized prompt file")
    parser.add_argument("--data-file", default="/home/haojinw2/efs/haojin/vlaa/system_prompt/data/mixed_math_arc_200.json",
                        help="Path to mixed dataset")
    parser.add_argument("--output-dir", default="/home/haojinw2/efs/haojin/vlaa/system_prompt/results",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to test")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--use-baseline", action="store_true", help="Test with baseline prompt instead")

    args = parser.parse_args()

    print("=" * 80)
    print("Testing Optimized Prompt on Mixed Dataset")
    print("=" * 80)

    # Load optimized prompt
    if args.use_baseline:
        system_prompt = "You are a helpful assistant."
        print("\n[Using Baseline Prompt]")
    else:
        print(f"\n[1/4] Loading optimized prompt from: {args.prompt_file}")
        system_prompt = load_optimized_prompt(args.prompt_file)

    print(f"\nSystem Prompt (first 200 chars):")
    print(f"{system_prompt[:200]}...")
    print()

    # Load dataset
    print(f"[2/4] Loading dataset from: {args.data_file}")
    examples = load_mixed_dataset(args.data_file)

    if args.max_samples:
        examples = examples[:args.max_samples]

    print(f"  Total examples: {len(examples)}")

    # Count task types
    task_counts = {}
    for ex in examples:
        task_type = ex['task_type']
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"  Task distribution: {task_counts}")

    # Setup result file
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_type = "baseline" if args.use_baseline else "optimized"
    result_file = os.path.join(args.output_dir, f"test_results_{prompt_type}.json")

    # Try to load existing results for resume
    start_index = 0
    results = []
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            start_index = len(results)
            print(f"\n[CHECKPOINT] Resuming from index {start_index} (completed {len(results)} questions)")
        except:
            print(f"\n[WARNING] Failed to load existing results, starting from scratch")
            results = []

    # Run evaluation
    print(f"\n[3/4] Running evaluation...")

    for i, example in enumerate(tqdm(examples[start_index:], desc="Testing", initial=start_index, total=len(examples))):
        actual_index = start_index + i
        question = example['formatted_prompt']
        correct_answer = example['answer']
        task_type = example['task_type']

        # Get model response
        response = prompt_model(
            model=args.model,
            question=question,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=2048,
            host=args.host,
            port=args.port
        )

        # Extract and check answer
        extracted_model_answer = extract_answer(response, task_type)
        extracted_correct_answer = extract_answer(correct_answer, task_type) if task_type == 'math' else correct_answer

        # Save result in simple format
        results.append({
            'query': question,
            'correct_answer': correct_answer,
            'model_answer': response,
            'extracted_correct_answer': extracted_correct_answer,
            'extracted_model_answer': extracted_model_answer,
            'token_count': count_words(response) if response else 0
        })

        # Save checkpoint after each question
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculate metrics from results
    correct_count = 0
    total_reasoning_words = 0
    task_stats = {'arc': {'correct': 0, 'total': 0, 'words': []},
                  'math': {'correct': 0, 'total': 0, 'words': []}}

    for i, result in enumerate(results):
        example = examples[i]
        task_type = example['task_type']

        is_correct = check_correctness(result['extracted_model_answer'], result['extracted_correct_answer'], task_type)

        if is_correct:
            correct_count += 1
            task_stats[task_type]['correct'] += 1

        task_stats[task_type]['total'] += 1
        task_stats[task_type]['words'].append(result['token_count'])
        total_reasoning_words += result['token_count']

    accuracy = correct_count / len(results) if results else 0
    avg_words = total_reasoning_words / len(results) if results else 0

    print(f"\n[4/4] Results:")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"Average Reasoning Length: {avg_words:.1f} words")
    print()

    # Per-task stats
    for task_type in ['arc', 'math']:
        if task_stats[task_type]['total'] > 0:
            task_acc = task_stats[task_type]['correct'] / task_stats[task_type]['total']
            task_avg_words = sum(task_stats[task_type]['words']) / len(task_stats[task_type]['words'])
            print(f"{task_type.upper()} Accuracy: {task_acc:.2%} ({task_stats[task_type]['correct']}/{task_stats[task_type]['total']})")
            print(f"{task_type.upper()} Avg Reasoning Length: {task_avg_words:.1f} words")
            print()

    print(f"Results saved to: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
