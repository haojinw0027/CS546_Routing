#!/usr/bin/env python3
"""
Baseline script for prompting models through vLLM with benchmark support.
Includes model name verification and configurable parameters.
"""

import argparse
import requests
import json
import sys
import yaml
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from utils import get_model_short_name, is_valid_model, get_available_models


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark prompts"""
    name: str
    prompts: List[str]
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000


def get_available_models(host: str = "localhost", port: int = 8000) -> List[str]:
    """Get list of available models from the server"""
    models_url = f"http://{host}:{port}/v1/models"
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
    except requests.RequestException as e:
        print(f"Error getting models from server: {e}")
        return []


def verify_model(expected_model: str, host: str = "localhost", port: int = 8000) -> bool:
    """Verify that the expected model is available on the server"""
    available_models = get_available_models(host, port)
    if not available_models:
        print("Warning: Could not retrieve model list from server")
        return False

    if expected_model in available_models:
        print(f"✓ Model '{expected_model}' verified on server")
        return True
    else:
        print(f"✗ Model '{expected_model}' not found on server")
        print(f"Available models: {', '.join(available_models)}")
        return False


def prompt_model(model: str,
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: float = 0.0,
                max_tokens: int = 1000,
                host: str = "localhost",
                port: int = 8000) -> Optional[str]:
    """Send a prompt to the model and return the response"""
    chat_url = f"http://{host}:{port}/v1/completions"

    # Combine system prompt and user prompt for completions API
    full_prompt = ""
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
    else:
        full_prompt = prompt

    payload = {
        "model": model,
        "prompt": full_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        'stop': ['<|endoftext|>', '\nAnswer']
    }
    try:
        response = requests.post(chat_url,
                               json=payload,
                               timeout=60)
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
        print(f"Response content: {result}")
        return None


def load_system_prompt_from_yaml(yaml_path: str, prompt_type: str = "default") -> Optional[str]:
    """Load system prompt from YAML file"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle different YAML structures
        if "prompts" in data:
            # Structure with prompts array
            for prompt in data["prompts"]:
                if prompt.get("id") == prompt_type:
                    return prompt.get("content", "")

            # If not found, show available prompt IDs
            available_types = [prompt.get("id", "unknown") for prompt in data["prompts"]]
            print(f"Error: System prompt type '{prompt_type}' not found in {yaml_path}")
            print(f"Available types: {', '.join(available_types)}")
            return None
        elif prompt_type in data:
            # Direct key-value structure
            return data[prompt_type]
        else:
            # Show available keys for direct structure
            available_types = list(data.keys())
            print(f"Error: System prompt type '{prompt_type}' not found in {yaml_path}")
            print(f"Available types: {', '.join(available_types)}")
            return None
    except FileNotFoundError:
        print(f"Error: System prompt file not found: {yaml_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return None


def format_arc_question(question: str, choices: List[str], labels: List[str]) -> str:
    """Format ARC question as multiple choice prompt"""
    formatted = f"Question: {question}\n"
    for label, choice in zip(labels, choices):
        formatted += f"{label}. {choice}\n"
    return formatted[:-1]


def load_arc_challenge_dataset(split: str = "test") -> Optional[BenchmarkConfig]:
    """Load ARC Challenge dataset from Hugging Face"""
    if not HF_DATASETS_AVAILABLE:
        print("Error: datasets library not available. Install with: pip install datasets")
        return None

    try:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]

        prompts = []
        answers = []

        for item in dataset:
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]

            # Format as multiple choice question
            formatted_question = format_arc_question(question, choices, labels)
            prompts.append(formatted_question)
            answers.append(answer_key)

        benchmark = BenchmarkConfig(
            name=f"ARC_Challenge_{split}",
            prompts=prompts,
            system_prompt=None,
            temperature=0.0,
            max_tokens=500
        )

        # Add ground truth answers and dataset type
        benchmark.ground_truth_answers = answers
        benchmark.dataset_type = "arc"

        return benchmark

    except Exception as e:
        print(f"Error loading ARC Challenge dataset: {e}")
        return None


def load_aime_2025_dataset(split: str = "default") -> Optional[BenchmarkConfig]:
    """Load AIME 2025 dataset from Hugging Face"""
    if not HF_DATASETS_AVAILABLE:
        print("Error: datasets library not available. Install with: pip install datasets")
        return None

    try:
        if split == "default":
            dataset = load_dataset("yentinglin/aime_2025")["train"]
        else:
            dataset = load_dataset("yentinglin/aime_2025", split)["train"]

        prompts = [item["problem"] for item in dataset]
        answers = [item["answer"] for item in dataset]

        # Store answers for evaluation
        benchmark = BenchmarkConfig(
            name=f"AIME_2025_{split}",
            prompts=prompts,
            system_prompt=None,
            temperature=0.0,
            max_tokens=2000
        )

        # Add ground truth answers and dataset type
        benchmark.ground_truth_answers = answers
        benchmark.dataset_type = "aime"

        return benchmark

    except Exception as e:
        print(f"Error loading AIME 2025 dataset: {e}")
        return None


def load_math_500_dataset(split: str = "test") -> Optional[BenchmarkConfig]:
    """Load MATH-500 dataset from Hugging Face"""
    if not HF_DATASETS_AVAILABLE:
        print("Error: datasets library not available. Install with: pip install datasets")
        return None

    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500")[split]

        prompts = []
        answers = []

        for item in dataset:
            problem = item["problem"]
            solution = item["solution"]

            prompts.append(problem)
            answers.append(solution)

        benchmark = BenchmarkConfig(
            name=f"MATH_500_{split}",
            prompts=prompts,
            system_prompt=None,
            temperature=0.0,
            max_tokens=2000
        )

        # Add ground truth answers and dataset type
        benchmark.ground_truth_answers = answers
        benchmark.dataset_type = "math"

        return benchmark

    except Exception as e:
        print(f"Error loading MATH-500 dataset: {e}")
        return None


def load_benchmark_config(config_path: str) -> Optional[BenchmarkConfig]:
    """Load benchmark configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        return BenchmarkConfig(
            name=data["name"],
            prompts=data["prompts"],
            system_prompt=data.get("system_prompt"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 1000)
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading benchmark config: {e}")
        return None


def run_benchmark(model: str,
                 benchmark: BenchmarkConfig,
                 host: str = "localhost",
                 port: int = 8000,
                 verbose: bool = False,
                 system_prompt: Optional[str] = None,
                 max_sample: Optional[int] = None,
                 output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run benchmark prompts against the model"""
    # Load existing results if checkpoint file exists
    existing_results = None
    completed_indices = set()

    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_results = json.load(f)

            # Get indices of completed prompts
            for i, response in enumerate(existing_results.get("responses", [])):
                if response.get("response") is not None:
                    completed_indices.add(i)

            print(f"Found existing results: {len(completed_indices)} prompts already completed")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_path}: {e}")
            existing_results = None

    # Apply max_sample limit if specified
    prompts_to_run = benchmark.prompts
    ground_truth_subset = None

    if max_sample and max_sample < len(benchmark.prompts):
        prompts_to_run = benchmark.prompts[:max_sample]
        if hasattr(benchmark, 'ground_truth_answers'):
            ground_truth_subset = benchmark.ground_truth_answers[:max_sample]

    # Initialize results structure
    if existing_results and existing_results.get("benchmark") == benchmark.name and existing_results.get("model") == model:
        # Continue from existing results
        results = existing_results
        results["evaluated"] = len(prompts_to_run)

        # Ensure responses list is the right length
        while len(results["responses"]) < len(prompts_to_run):
            results["responses"].append({"prompt": "", "response": None})
    else:
        # Start fresh
        results = {
            "benchmark": benchmark.name,
            "model": model,
            "responses": [],
            "total_available": len(benchmark.prompts),
            "evaluated": len(prompts_to_run)
        }
        # Initialize empty responses
        for i, prompt in enumerate(prompts_to_run):
            results["responses"].append({"prompt": prompt, "response": None})

    print(f"\nRunning benchmark: {benchmark.name}")
    print(f"Model: {model}")
    print(f"Total prompts available: {len(benchmark.prompts)}")
    remaining_prompts = len(prompts_to_run) - len(completed_indices)
    print(f"Evaluating: {len(prompts_to_run)} prompts")
    print(f"Already completed: {len(completed_indices)} prompts")
    print(f"Remaining: {remaining_prompts} prompts")
    if max_sample and max_sample < len(benchmark.prompts):
        print(f"Limited by max_sample: {max_sample}")
    print("-" * 50)

    for i, prompt in tqdm(enumerate(prompts_to_run), total=len(prompts_to_run)):
        # Skip if already completed
        if i in completed_indices:
            if verbose:
                print(f"Prompt {i+1}/{len(prompts_to_run)} - SKIPPED (already completed)")
            continue

        print(f"Prompt {i+1}/{len(prompts_to_run)}")

        if verbose:
            print(f"Input: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
        response = prompt_model(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=benchmark.temperature,
            max_tokens=benchmark.max_tokens,
            host=host,
            port=port
        )

        # Prepare response data
        response_data = {
            "prompt": prompt,
            "response": response,
            'full_prompt': full_prompt
        }

        # Add gold answer immediately if available
        if hasattr(benchmark, 'ground_truth_answers') and i < len(benchmark.ground_truth_answers):
            if ground_truth_subset is not None and i < len(ground_truth_subset):
                response_data["gold_answer"] = ground_truth_subset[i]
            elif i < len(benchmark.ground_truth_answers):
                response_data["gold_answer"] = benchmark.ground_truth_answers[i]

        if response:
            results["responses"][i] = response_data

            if verbose:
                print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")

            # Save progress after each completion
            if output_path:
                save_results(results, output_path)
        else:
            print(f"Failed to get response for prompt {i+1}")
            response_data["response"] = None
            results["responses"][i] = response_data

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def get_auto_output_path(prompt_type: str, model_name: str, benchmark_name: str = None, max_tokens: int = 1000) -> str:
    """Generate automatic output path based on prompt type, model, and benchmark"""
    model_short = get_model_short_name(model_name)
    filename = f"{prompt_type}.json"
    bench_safe = benchmark_name.replace(' ', '_').lower()
    return os.path.join(f"./results/{bench_safe}/{model_short}/{max_tokens}", filename)


def main():
    parser = argparse.ArgumentParser(description="Baseline script for prompting models through vLLM")
    parser.add_argument("--model", "-m", required=True, help="Model name to use")
    parser.add_argument("--port", "-p", type=int, default=8088, help="vLLM server port (default: 8000)")
    parser.add_argument("--host", default="localhost", help="vLLM server host (default: localhost)")
    parser.add_argument("--benchmark", "-b", required=True, choices = ["aime_2025", "arc_challenge", "math_500"],
                       help="Benchmark dataset to use (aime_2025, arc_challenge, math_500) or path to JSON config file")
    parser.add_argument("--split", default=None, help="Dataset split (for AIME: default/part1/part2, for ARC: train/validation/test, for MATH-500: test/train)")
    parser.add_argument("--max-sample", type=int, help="Maximum number of samples to evaluate from the benchmark")
    parser.add_argument("--system", help="System prompt to use (overrides YAML)")
    parser.add_argument("--system-prompt-yaml", default="./system_prompts/initial.yaml",
                       help="Path to YAML file containing system prompts (default: ./system_prompts/initial.yaml)")
    parser.add_argument("--system-prompt-type", default="always_cot",
                       help="Type of system prompt to use from YAML file (default: always_cot)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Verify model availability
    verify_model(args.model, args.host, args.port)
    # Determine system prompt to use
    system_prompt = args.system
    if not system_prompt and os.path.exists(args.system_prompt_yaml):
        system_prompt = load_system_prompt_from_yaml(args.system_prompt_yaml, args.system_prompt_type)
        if system_prompt and args.verbose:
            print(f"Loaded system prompt type '{args.system_prompt_type}' from {args.system_prompt_yaml}")

    # Load benchmark
    if args.benchmark == "aime_2025":
        split = args.split if args.split else "default"
        benchmark = load_aime_2025_dataset(split)
        if not benchmark:
            sys.exit(1)
    elif args.benchmark == "arc_challenge":
        split = args.split if args.split else "test"
        benchmark = load_arc_challenge_dataset(split)
        if not benchmark:
            sys.exit(1)
    elif args.benchmark == "math_500":
        split = args.split if args.split else "test"
        benchmark = load_math_500_dataset(split)
        if not benchmark:
            sys.exit(1)
    else:
        # Treat as JSON config file path
        benchmark = load_benchmark_config(args.benchmark)
        if not benchmark:
            sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = get_auto_output_path(args.system_prompt_type, args.model, benchmark.name, args.max_tokens)

    # Run benchmark
    results = run_benchmark(args.model, benchmark, args.host, args.port, args.verbose, system_prompt, args.max_sample, output_path)

    # Save final results
    save_results(results, output_path)


if __name__ == "__main__":
    main()