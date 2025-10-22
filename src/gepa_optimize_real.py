#!/usr/bin/env python3
"""
Real GEPA Optimization using the official GEPA library.

This script integrates with the real GEPA library to optimize system prompts
using LLM-based reflection instead of template-based mutations.

Key differences from gepa_optimize.py:
1. Uses real GEPA library's optimize() function
2. Uses a reflection_lm (e.g., GPT-4, Claude) for intelligent prompt evolution
3. Leverages GEPA's built-in evolutionary search and reflection mechanisms
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import gepa
    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False
    print("WARNING: GEPA library not installed. Install with: pip install gepa")

from gepa_adapter_mixed import MixedDatasetAdapter


class GEPAAdapterWrapper:
    """
    Wrapper to make our MixedDatasetAdapter compatible with GEPA's API.

    GEPA expects:
    - evaluate(candidate, minibatch) -> (score, traces)
    - extract_traces_for_reflection(traces) -> reflective data
    """

    def __init__(
        self,
        adapter: MixedDatasetAdapter,
        trainset: List[Dict],
        valset: List[Dict]
    ):
        self.adapter = adapter
        self.trainset = trainset
        self.valset = valset

    def evaluate(
        self,
        candidate: Dict[str, str],
        minibatch: Optional[List[Dict]] = None
    ) -> tuple:
        """
        Evaluate a candidate prompt on a minibatch.

        Args:
            candidate: Dict with 'system_prompt' key
            minibatch: List of examples to evaluate on (if None, use full trainset)

        Returns:
            (score, traces): Overall score and execution traces
        """
        system_prompt = candidate.get('system_prompt', '')
        examples = minibatch if minibatch is not None else self.trainset

        # Evaluate using our adapter
        evaluation = self.adapter.evaluate(system_prompt, examples, capture_traces=True)

        score = evaluation['overall_score']
        traces = {
            'results': evaluation['results'],
            'trajectories': evaluation['trajectories'],
            'math_score': evaluation['math_score'],
            'arc_score': evaluation['arc_score'],
            'overall_score': score
        }

        return score, traces

    def extract_traces_for_reflection(
        self,
        traces: Dict[str, Any],
        component_name: str = "system_prompt"
    ) -> str:
        """
        Extract and format traces for reflection by the reflection LM.

        Returns a formatted string that helps the reflection model understand
        what went wrong and how to improve the prompt.
        """
        results = traces['results']

        # Separate errors and successes
        errors = [r for r in results if r.score == 0.0]
        successes = [r for r in results if r.score == 1.0]

        # Build reflection text
        reflection_text = []

        reflection_text.append("=== EVALUATION SUMMARY ===")
        reflection_text.append(f"Overall Score: {traces['overall_score']:.2%}")
        reflection_text.append(f"MATH Score: {traces['math_score']:.2%}")
        reflection_text.append(f"ARC Score: {traces['arc_score']:.2%}")
        reflection_text.append(f"Errors: {len(errors)}/{len(results)}")
        reflection_text.append("")

        # Show error patterns
        reflection_text.append("=== ERROR ANALYSIS ===")

        math_errors = [r for r in errors if r.task_type == "math"]
        arc_errors = [r for r in errors if r.task_type == "arc"]

        if math_errors:
            reflection_text.append(f"\nMATH Errors ({len(math_errors)}):")
            for i, result in enumerate(math_errors[:3], 1):  # Show first 3
                traj = result.trajectory
                reflection_text.append(f"\n  Example {i}:")
                reflection_text.append(f"    Question: {traj['question']}")
                reflection_text.append(f"    Expected: {result.gold_answer}")
                reflection_text.append(f"    Got: {traj.get('extracted_answer', 'nothing')}")
                reflection_text.append(f"    Has Reasoning: {traj.get('has_reasoning', False)}")
                if not traj.get('has_reasoning'):
                    reflection_text.append(f"    Issue: Missing step-by-step reasoning for math problem")

        if arc_errors:
            reflection_text.append(f"\nARC Errors ({len(arc_errors)}):")
            for i, result in enumerate(arc_errors[:3], 1):
                traj = result.trajectory
                reflection_text.append(f"\n  Example {i}:")
                reflection_text.append(f"    Question: {traj['question']}")
                reflection_text.append(f"    Expected: {result.gold_answer}")
                reflection_text.append(f"    Got: {traj.get('extracted_answer', 'nothing')}")
                reflection_text.append(f"    Response Length: {traj.get('response_length', 0)} chars")
                if traj.get('response_length', 0) > 500:
                    reflection_text.append(f"    Issue: Response too verbose for simple multiple choice")

        # Show successful patterns
        if successes:
            reflection_text.append(f"\n=== SUCCESSFUL PATTERNS ===")
            math_success = [r for r in successes if r.task_type == "math"]
            arc_success = [r for r in successes if r.task_type == "arc"]

            if math_success:
                reflection_text.append(f"MATH successes: {len(math_success)} (with detailed reasoning)")
            if arc_success:
                reflection_text.append(f"ARC successes: {len(arc_success)} (direct answers)")

        return "\n".join(reflection_text)


def load_seed_prompts(yaml_path: str) -> List[str]:
    """Load initial seed prompts from YAML"""
    import yaml

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    prompts = [p["content"] for p in data["prompts"]]
    return prompts


def run_gepa_optimization(
    adapter_wrapper: GEPAAdapterWrapper,
    seed_prompts: List[str],
    task_lm: str,
    reflection_lm: str,
    max_metric_calls: int = 150,
    output_dir: str = "./results/gepa_optimization"
) -> Dict[str, Any]:
    """
    Run GEPA optimization using the real GEPA library.

    Args:
        adapter_wrapper: Our custom adapter wrapper
        seed_prompts: List of initial system prompts to start with
        task_lm: Model being optimized (e.g., "openai/gpt-4-mini" or vLLM endpoint)
        reflection_lm: Model for reflection (e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet")
        max_metric_calls: Budget for evaluations
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results
    """
    if not GEPA_AVAILABLE:
        raise ImportError("GEPA library is not installed. Install with: pip install gepa")

    print(f"\n{'='*60}")
    print("STARTING REAL GEPA OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Task Model: {task_lm}")
    print(f"Reflection Model: {reflection_lm}")
    print(f"Max Metric Calls: {max_metric_calls}")
    print(f"Seed Prompts: {len(seed_prompts)}")
    print(f"{'='*60}\n")

    # Prepare seed candidate (GEPA expects a dict)
    seed_candidate = {
        "system_prompt": seed_prompts[0]  # Start with first seed prompt
    }

    # Run GEPA optimization
    # Note: GEPA will handle the evolutionary search, reflection, and mutation
    gepa_result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=adapter_wrapper.trainset,
        valset=adapter_wrapper.valset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        adapter=adapter_wrapper,  # Our custom adapter
    )

    print(f"\n{'='*60}")
    print("GEPA OPTIMIZATION COMPLETE")
    print(f"{'='*60}\n")

    best_prompt = gepa_result.best_candidate['system_prompt']

    # Final validation
    print("Running final validation...")
    val_score, val_traces = adapter_wrapper.evaluate(
        gepa_result.best_candidate,
        adapter_wrapper.valset
    )

    print(f"\nFinal Validation Results:")
    print(f"  Overall: {val_score:.2%}")
    print(f"  MATH: {val_traces['math_score']:.2%}")
    print(f"  ARC: {val_traces['arc_score']:.2%}")

    results = {
        "best_prompt": best_prompt,
        "train_score": gepa_result.best_score,
        "val_score": val_score,
        "val_math_score": val_traces['math_score'],
        "val_arc_score": val_traces['arc_score'],
        "optimization_history": gepa_result.history if hasattr(gepa_result, 'history') else [],
        "config": {
            "task_lm": task_lm,
            "reflection_lm": reflection_lm,
            "max_metric_calls": max_metric_calls,
            "seed_prompts": len(seed_prompts)
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Real GEPA optimization for mixed dataset")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-8B",
                       help="Task model to optimize for")
    parser.add_argument("--port", "-p", type=int, default=8088,
                       help="vLLM server port for task model")
    parser.add_argument("--host", default="localhost",
                       help="vLLM server host")
    parser.add_argument("--reflection-model", default="openai/gpt-4",
                       help="Reflection model (e.g., openai/gpt-4, anthropic/claude-3-5-sonnet)")
    parser.add_argument("--reflection-api-key", default=None,
                       help="API key for reflection model (or set via env var)")
    parser.add_argument("--dataset", default="./data/mixed_math_arc.json",
                       help="Path to mixed dataset JSON")
    parser.add_argument("--seed-prompts", default="./system_prompts/initial.yaml",
                       help="Path to seed prompts YAML")
    parser.add_argument("--max-metric-calls", type=int, default=150,
                       help="Budget for metric evaluations")
    parser.add_argument("--output-dir", default="./results/gepa_optimization_real",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    if not GEPA_AVAILABLE:
        print("ERROR: GEPA library is not installed.")
        print("Please install with: pip install gepa")
        print("Or: pip install git+https://github.com/gepa-ai/gepa.git")
        return

    # Set API key for reflection model if provided
    if args.reflection_api_key:
        if "openai" in args.reflection_model:
            os.environ["OPENAI_API_KEY"] = args.reflection_api_key
        elif "anthropic" in args.reflection_model or "claude" in args.reflection_model:
            os.environ["ANTHROPIC_API_KEY"] = args.reflection_api_key

    # Load dataset
    print("Loading mixed dataset...")
    with open(args.dataset, 'r') as f:
        data = json.load(f)

    examples = data["examples"]

    # Split into train/val
    random.seed(args.seed)
    random.shuffle(examples)
    split_idx = int(0.7 * len(examples))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"  Total examples: {len(examples)}")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")

    # Load seed prompts
    print(f"\nLoading seed prompts from {args.seed_prompts}...")
    seed_prompts = load_seed_prompts(args.seed_prompts)
    print(f"  Loaded {len(seed_prompts)} seed prompts")

    # Initialize adapter
    print(f"\nInitializing adapter for model {args.model}...")
    adapter = MixedDatasetAdapter(
        model=args.model,
        host=args.host,
        port=args.port
    )

    # Wrap adapter for GEPA compatibility
    adapter_wrapper = GEPAAdapterWrapper(
        adapter=adapter,
        trainset=train_examples,
        valset=val_examples
    )

    # Construct task_lm string
    # For local vLLM, we might need to use a special format or custom adapter
    # For now, we'll assume the model name works
    task_lm = args.model

    # Run GEPA optimization
    try:
        results = run_gepa_optimization(
            adapter_wrapper=adapter_wrapper,
            seed_prompts=seed_prompts,
            task_lm=task_lm,
            reflection_lm=args.reflection_model,
            max_metric_calls=args.max_metric_calls,
            output_dir=args.output_dir
        )

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"optimization_{timestamp}.json")

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_path}")
        print(f"\nBest Prompt:")
        print("-" * 60)
        print(results["best_prompt"])
        print("-" * 60)
        print(f"\nFinal Scores:")
        print(f"  Train: {results['train_score']:.2%}")
        print(f"  Val: {results['val_score']:.2%}")
        print(f"  Val MATH: {results['val_math_score']:.2%}")
        print(f"  Val ARC: {results['val_arc_score']:.2%}")

        # Save best prompt to YAML
        prompt_output = os.path.join(args.output_dir, f"optimized_prompt_{timestamp}.yaml")
        import yaml
        with open(prompt_output, 'w') as f:
            yaml.dump({
                "prompts": [{
                    "id": "gepa_optimized_real",
                    "title": "Real GEPA Optimized Routing Prompt",
                    "description": f"Optimized using real GEPA with {args.reflection_model}. Val score: {results['val_score']:.2%}",
                    "content": results["best_prompt"],
                    "tags": ["gepa", "optimized", "routing", "adaptive", "real-llm-reflection"],
                    "version": "2.0",
                    "optimization_date": timestamp,
                    "reflection_model": args.reflection_model
                }]
            }, f)

        print(f"\nBest prompt also saved to: {prompt_output}")

    except Exception as e:
        print(f"\nERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure your vLLM server is running on the specified port")
        print("2. Ensure you have set the API key for the reflection model:")
        print(f"   - For OpenAI: export OPENAI_API_KEY=your_key")
        print(f"   - For Anthropic: export ANTHROPIC_API_KEY=your_key")
        print("3. Check that GEPA is installed: pip install gepa")


if __name__ == "__main__":
    main()
