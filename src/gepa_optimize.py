#!/usr/bin/env python3
"""
GEPA Optimization for Mixed MATH/ARC Dataset

This script uses GEPA-inspired optimization to evolve a system prompt that:
1. Recognizes MATH problems and provides detailed reasoning
2. Recognizes ARC problems and provides direct answers
3. Routes automatically between the two strategies

Note: This is a simplified GEPA-style implementation that doesn't require
the full GEPA library installation. It implements the core ideas:
- Evolutionary search over prompts
- LLM-based reflection on failures
- Iterative improvement
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from gepa_adapter_mixed import MixedDatasetAdapter


class SimpleGEPAOptimizer:
    """
    Simplified GEPA-style optimizer for system prompts.

    Uses:
    - Population-based search with mutations
    - LLM-based reflection on failures to guide mutations
    - Multi-objective optimization (MATH accuracy + ARC accuracy)
    """

    def __init__(
        self,
        adapter: MixedDatasetAdapter,
        train_examples: List[Dict],
        val_examples: List[Dict],
        population_size: int = 4,
        n_iterations: int = 10,
        reflection_model: str = "gpt-4",  # For generating improved prompts
        seed: int = 42
    ):
        self.adapter = adapter
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.reflection_model = reflection_model
        self.seed = seed
        random.seed(seed)

        self.population: List[Dict[str, Any]] = []
        self.best_prompt: Optional[str] = None
        self.best_score: float = 0.0
        self.history: List[Dict] = []

    def initialize_population(self, seed_prompts: List[str]):
        """Initialize population with seed prompts"""
        print(f"\n{'='*60}")
        print("INITIALIZING POPULATION")
        print(f"{'='*60}")

        for i, prompt in enumerate(seed_prompts[:self.population_size]):
            print(f"\nEvaluating seed prompt {i+1}/{len(seed_prompts[:self.population_size])}...")
            evaluation = self.adapter.evaluate(prompt, self.train_examples)

            candidate = {
                "id": f"seed_{i}",
                "prompt": prompt,
                "score": evaluation["overall_score"],
                "math_score": evaluation["math_score"],
                "arc_score": evaluation["arc_score"],
                "evaluation": evaluation,
                "generation": 0
            }

            self.population.append(candidate)

            print(f"  Overall: {evaluation['overall_score']:.2%}")
            print(f"  MATH: {evaluation['math_score']:.2%}")
            print(f"  ARC: {evaluation['arc_score']:.2%}")

            # Track best
            if candidate["score"] > self.best_score:
                self.best_score = candidate["score"]
                self.best_prompt = prompt

        # Sort by score
        self.population.sort(key=lambda x: x["score"], reverse=True)

        print(f"\n{'='*60}")
        print(f"Initial best score: {self.best_score:.2%}")
        print(f"{'='*60}\n")

    def reflect_and_mutate(self, candidate: Dict[str, Any], generation: int) -> str:
        """
        Generate improved prompt based on error analysis.

        In full GEPA, this would use an LLM to reflect on failures.
        Here we use template-based mutations with some heuristics.
        """
        evaluation = candidate["evaluation"]
        reflective_data = self.adapter.make_reflective_dataset(evaluation)

        # Analyze failure patterns
        math_failures = [r for r in evaluation["results"] if r.task_type == "math" and r.score == 0.0]
        arc_failures = [r for r in evaluation["results"] if r.task_type == "arc" and r.score == 0.0]

        current_prompt = candidate["prompt"]

        # Heuristic-based mutation strategies
        mutations = []

        # If MATH problems are failing
        if math_failures and evaluation["math_score"] < 0.5:
            mutations.append(self._emphasize_math_reasoning)

        # If ARC problems are failing
        if arc_failures and evaluation["arc_score"] < 0.5:
            mutations.append(self._emphasize_arc_directness)

        # If both are doing poorly, try rebalancing
        if evaluation["math_score"] < 0.5 and evaluation["arc_score"] < 0.5:
            mutations.append(self._rebalance_prompt)

        # If one is good but other is bad, emphasize distinction
        if abs(evaluation["math_score"] - evaluation["arc_score"]) > 0.3:
            mutations.append(self._emphasize_distinction)

        # Apply random mutation
        if mutations:
            mutation_fn = random.choice(mutations)
            new_prompt = mutation_fn(current_prompt, math_failures, arc_failures)
        else:
            # Small random variation
            new_prompt = self._small_variation(current_prompt)

        return new_prompt

    def _emphasize_math_reasoning(self, prompt: str, math_failures: List, arc_failures: List) -> str:
        """Strengthen MATH reasoning requirements"""
        additions = [
            "\n\nFor mathematical problems with calculations or proofs, you MUST show detailed step-by-step work.",
            "\n\nIMPORTANT: Multi-step math problems require explicit reasoning with numbered steps.",
            "\n\nWhen you see equations, variables, or calculations, always provide comprehensive reasoning."
        ]
        return prompt + random.choice(additions)

    def _emphasize_arc_directness(self, prompt: str, math_failures: List, arc_failures: List) -> str:
        """Strengthen ARC directness"""
        additions = [
            "\n\nFor simple multiple choice questions, answer directly without lengthy explanation.",
            "\n\nIMPORTANT: Science or general knowledge multiple choice questions should be answered concisely.",
            "\n\nWhen you see A/B/C/D choices, provide a direct answer without excessive reasoning."
        ]
        return prompt + random.choice(additions)

    def _rebalance_prompt(self, prompt: str, math_failures: List, arc_failures: List) -> str:
        """Try to rebalance the prompt structure"""
        # Simplify and restructure
        base = """You are an adaptive problem solver. Follow these rules:

1. IDENTIFY the question type first:
   - Is it a math problem requiring calculations? → Use detailed reasoning
   - Is it a multiple choice question? → Answer directly

2. RESPOND accordingly:
   - Math problems: Show step-by-step work, verify, conclude with **Final Answer:**
   - Multiple choice: State the correct letter with brief justification

Be precise and adapt your verbosity to the question type."""
        return base

    def _emphasize_distinction(self, prompt: str, math_failures: List, arc_failures: List) -> str:
        """Emphasize the distinction between task types"""
        addition = """

CRITICAL: Distinguish between:
- Complex math problems (algebra, geometry, calculus) → DETAILED reasoning required
- Multiple choice questions (A/B/C/D format) → DIRECT answers required

Your response length should match the question complexity."""
        return prompt + addition

    def _small_variation(self, prompt: str) -> str:
        """Make small random variation"""
        variations = [
            prompt.replace("must", "MUST"),
            prompt.replace("Final Answer", "FINAL ANSWER"),
            prompt + "\n\nBe precise and clear.",
            prompt.replace("step-by-step", "step by step"),
        ]
        return random.choice(variations)

    def evolve_generation(self, generation: int):
        """Evolve one generation"""
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}")
        print(f"{'='*60}")

        # Generate new candidates by mutating current population
        new_candidates = []

        # Keep best candidate
        elite = self.population[0]
        new_candidates.append(elite)
        print(f"\nElite (kept): Score {elite['score']:.2%}")

        # Mutate top candidates
        n_to_mutate = self.population_size - 1
        parents = self.population[:max(2, self.population_size // 2)]

        for i in range(n_to_mutate):
            parent = random.choice(parents)
            print(f"\nMutating candidate {i+1}/{n_to_mutate} (parent score: {parent['score']:.2%})...")

            new_prompt = self.reflect_and_mutate(parent, generation)
            evaluation = self.adapter.evaluate(new_prompt, self.train_examples)

            candidate = {
                "id": f"gen{generation}_mut{i}",
                "prompt": new_prompt,
                "score": evaluation["overall_score"],
                "math_score": evaluation["math_score"],
                "arc_score": evaluation["arc_score"],
                "evaluation": evaluation,
                "generation": generation,
                "parent_id": parent["id"]
            }

            new_candidates.append(candidate)

            print(f"  Overall: {evaluation['overall_score']:.2%}")
            print(f"  MATH: {evaluation['math_score']:.2%}")
            print(f"  ARC: {evaluation['arc_score']:.2%}")

            # Track best
            if candidate["score"] > self.best_score:
                self.best_score = candidate["score"]
                self.best_prompt = new_prompt
                print(f"  🎉 NEW BEST SCORE: {self.best_score:.2%}")

        # Replace population
        self.population = sorted(new_candidates, key=lambda x: x["score"], reverse=True)

        # Log generation
        self.history.append({
            "generation": generation,
            "best_score": self.population[0]["score"],
            "best_math_score": self.population[0]["math_score"],
            "best_arc_score": self.population[0]["arc_score"],
            "avg_score": sum(c["score"] for c in self.population) / len(self.population)
        })

        print(f"\n{'='*60}")
        print(f"Generation {generation} complete")
        print(f"  Best score: {self.population[0]['score']:.2%}")
        print(f"  Best MATH: {self.population[0]['math_score']:.2%}")
        print(f"  Best ARC: {self.population[0]['arc_score']:.2%}")
        print(f"{'='*60}\n")

    def optimize(self, seed_prompts: List[str]) -> Dict[str, Any]:
        """Run full optimization"""
        self.initialize_population(seed_prompts)

        for generation in range(1, self.n_iterations + 1):
            self.evolve_generation(generation)

        # Final validation
        print(f"\n{'='*60}")
        print("FINAL VALIDATION")
        print(f"{'='*60}\n")

        best_candidate = self.population[0]
        val_evaluation = self.adapter.evaluate(best_candidate["prompt"], self.val_examples)

        print(f"Validation Results:")
        print(f"  Overall: {val_evaluation['overall_score']:.2%}")
        print(f"  MATH: {val_evaluation['math_score']:.2%}")
        print(f"  ARC: {val_evaluation['arc_score']:.2%}")

        return {
            "best_prompt": best_candidate["prompt"],
            "train_score": best_candidate["score"],
            "val_score": val_evaluation["overall_score"],
            "val_math_score": val_evaluation["math_score"],
            "val_arc_score": val_evaluation["arc_score"],
            "history": self.history,
            "final_evaluation": val_evaluation
        }


def load_seed_prompts(yaml_path: str) -> List[str]:
    """Load initial seed prompts from YAML"""
    import yaml

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    prompts = [p["content"] for p in data["prompts"]]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="GEPA-style optimization for mixed dataset")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-8B", help="Model to optimize for")
    parser.add_argument("--port", "-p", type=int, default=8088, help="vLLM server port")
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--dataset", default="./data/mixed_math_arc.json",
                       help="Path to mixed dataset JSON")
    parser.add_argument("--seed-prompts", default="./system_prompts/initial.yaml",
                       help="Path to seed prompts YAML")
    parser.add_argument("--population-size", type=int, default=3,
                       help="Population size for evolution")
    parser.add_argument("--n-iterations", type=int, default=5,
                       help="Number of optimization iterations")
    parser.add_argument("--output-dir", default="./results/gepa_optimization",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

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

    # Initialize optimizer
    print("\nInitializing optimizer...")
    optimizer = SimpleGEPAOptimizer(
        adapter=adapter,
        train_examples=train_examples,
        val_examples=val_examples,
        population_size=args.population_size,
        n_iterations=args.n_iterations,
        seed=args.seed
    )

    # Run optimization
    print("\n" + "="*60)
    print("STARTING OPTIMIZATION")
    print("="*60)

    results = optimizer.optimize(seed_prompts)

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
                "id": "gepa_optimized",
                "title": "GEPA Optimized Routing Prompt",
                "description": f"Optimized for mixed MATH/ARC dataset. Val score: {results['val_score']:.2%}",
                "content": results["best_prompt"],
                "tags": ["gepa", "optimized", "routing", "adaptive"],
                "version": "1.0",
                "optimization_date": timestamp
            }]
        }, f)

    print(f"\nBest prompt also saved to: {prompt_output}")


if __name__ == "__main__":
    main()
