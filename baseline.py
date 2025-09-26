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


class VLLMClient:
    """Client for interacting with vLLM OpenAI-compatible API"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.models_url = f"http://{host}:{port}/v1/models"
        self.chat_url = f"http://{host}:{port}/v1/completions"

    def get_available_models(self) -> List[str]:
        """Get list of available models from the server"""
        try:
            response = requests.get(self.models_url, timeout=10)
            response.raise_for_status()
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except requests.RequestException as e:
            print(f"Error getting models from server: {e}")
            return []

    def verify_model(self, expected_model: str) -> bool:
        """Verify that the expected model is available on the server"""
        available_models = self.get_available_models()
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

    def prompt_model(self,
                    model: str,
                    prompt: str,
                    system_prompt: Optional[str] = None,
                    temperature: float = 0.0,
                    max_tokens: int = 1000) -> Optional[str]:
        """Send a prompt to the model and return the response"""
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
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.chat_url,
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


def normalize_aime_answer(answer: str) -> str:
    """Normalize AIME answer to 3-digit format"""
    # Extract first number from the answer string
    numbers = re.findall(r'\d+', str(answer))
    if numbers:
        return f"{int(numbers[0]):03d}"
    return "000"


def extract_answer_choice(response: str) -> str:
    """Extract answer choice (A, B, C, D) from model response"""
    # Look for patterns like "A)", "A.", "A:", or standalone "A"
    patterns = [
        r'\b([A-E])\)',  # A)
        r'\b([A-E])\.',  # A.
        r'\b([A-E]):',   # A:
        r'\b([A-E])\b',  # standalone A
        r'answer.*?([A-E])',  # "answer is A"
        r'([A-E]).*?correct',  # "A is correct"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()

    return ""


def evaluate_arc_predictions(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    """Evaluate ARC multiple choice predictions"""
    correct = 0
    total = len(predictions)
    detailed_results = []

    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        extracted_answer = extract_answer_choice(pred)
        is_correct = extracted_answer == truth.upper()

        if is_correct:
            correct += 1

        detailed_results.append({
            "question_id": i + 1,
            "prediction": pred,
            "extracted_answer": extracted_answer,
            "ground_truth": truth,
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "detailed_results": detailed_results
    }


def evaluate_aime_predictions(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    """Evaluate AIME predictions against ground truth answers"""
    correct = 0
    total = len(predictions)
    detailed_results = []

    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        pred_normalized = normalize_aime_answer(pred)
        truth_normalized = normalize_aime_answer(truth)

        is_correct = pred_normalized == truth_normalized
        if is_correct:
            correct += 1

        detailed_results.append({
            "problem_id": i + 1,
            "prediction": pred,
            "prediction_normalized": pred_normalized,
            "ground_truth": truth,
            "ground_truth_normalized": truth_normalized,
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "aime_score": f"{correct}/15",
        "detailed_results": detailed_results
    }


def format_arc_question(question: str, choices: List[str], labels: List[str]) -> str:
    """Format ARC question as multiple choice prompt"""
    formatted = f"Question: {question}\n"
    for label, choice in zip(labels, choices):
        formatted += f"{label}. {choice}\n"
    formatted += "Answer:"
    return formatted


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


def run_benchmark(client: VLLMClient,
                 model: str,
                 benchmark: BenchmarkConfig,
                 verbose: bool = False,
                 system_prompt: Optional[str] = None,
                 max_sample: Optional[int] = None) -> Dict[str, Any]:
    """Run benchmark prompts against the model"""
    # Apply max_sample limit if specified
    prompts_to_run = benchmark.prompts
    ground_truth_subset = None

    if max_sample and max_sample < len(benchmark.prompts):
        prompts_to_run = benchmark.prompts[:max_sample]
        if hasattr(benchmark, 'ground_truth_answers'):
            ground_truth_subset = benchmark.ground_truth_answers[:max_sample]

    results = {
        "benchmark": benchmark.name,
        "model": model,
        "responses": [],
        "total_available": len(benchmark.prompts),
        "evaluated": len(prompts_to_run)
    }

    print(f"\nRunning benchmark: {benchmark.name}")
    print(f"Model: {model}")
    print(f"Total prompts available: {len(benchmark.prompts)}")
    print(f"Evaluating: {len(prompts_to_run)} prompts")
    if max_sample and max_sample < len(benchmark.prompts):
        print(f"Limited by max_sample: {max_sample}")
    print("-" * 50)

    for i, prompt in enumerate(prompts_to_run, 1):
        print(f"Prompt {i}/{len(prompts_to_run)}")

        if verbose:
            print(f"Input: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"System prompt: {system_prompt}")
        response = client.prompt_model(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=benchmark.temperature,
            max_tokens=benchmark.max_tokens
        )

        if response:
            results["responses"].append({
                "prompt": prompt,
                "response": response
            })

            if verbose:
                print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        else:
            print(f"Failed to get response for prompt {i}")
            results["responses"].append({
                "prompt": prompt,
                "response": None
            })

    # Add evaluation if ground truth answers are available
    if hasattr(benchmark, 'ground_truth_answers'):
        predictions = [resp["response"] for resp in results["responses"] if resp["response"]]

        # Use the subset if max_sample was applied
        if ground_truth_subset is not None:
            ground_truth = ground_truth_subset[:len(predictions)]
        else:
            ground_truth = benchmark.ground_truth_answers[:len(predictions)]

        if hasattr(benchmark, 'dataset_type') and benchmark.dataset_type == "arc":
            evaluation = evaluate_arc_predictions(predictions, ground_truth)
            results["evaluation"] = evaluation

            print(f"\nARC Evaluation Results:")
            print(f"Accuracy: {evaluation['accuracy']:.3f}")
            print(f"Correct: {evaluation['correct']}/{evaluation['total']}")

        elif hasattr(benchmark, 'dataset_type') and benchmark.dataset_type == "aime":
            evaluation = evaluate_aime_predictions(predictions, ground_truth)
            results["evaluation"] = evaluation

            print(f"\nAIME Evaluation Results:")
            print(f"Score: {evaluation['aime_score']}")
            print(f"Accuracy: {evaluation['accuracy']:.3f}")
            print(f"Correct: {evaluation['correct']}/{evaluation['total']}")

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


def get_auto_output_path(prompt_type: str, model_name: str, benchmark_name: str = None) -> str:
    """Generate automatic output path based on prompt type, model, and benchmark"""
    model_short = get_model_short_name(model_name)
    filename = f"{prompt_type}.json"
    bench_safe = benchmark_name.replace(' ', '_').lower()
    return os.path.join(f"./results/{bench_safe}/{model_short}", filename)


def main():
    parser = argparse.ArgumentParser(description="Baseline script for prompting models through vLLM")
    parser.add_argument("--model", "-m", required=True, help="Model name to use")
    parser.add_argument("--port", "-p", type=int, default=8088, help="vLLM server port (default: 8000)")
    parser.add_argument("--host", default="localhost", help="vLLM server host (default: localhost)")
    parser.add_argument("--benchmark", "-b", required=True,
                       help="Benchmark dataset to use (aime_2025, arc_challenge) or path to JSON config file")
    parser.add_argument("--split", default=None, help="Dataset split (for AIME: default/part1/part2, for ARC: train/validation/test)")
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

    # Initialize client
    client = VLLMClient(host=args.host, port=args.port)
    client.verify_model(args.model)
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
    else:
        # Treat as JSON config file path
        benchmark = load_benchmark_config(args.benchmark)
        if not benchmark:
            sys.exit(1)

    # Run benchmark
    results = run_benchmark(client, args.model, benchmark, args.verbose, system_prompt, args.max_sample)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = get_auto_output_path(args.system_prompt_type, args.model, benchmark.name)

    save_results(results, output_path)


if __name__ == "__main__":
    main()