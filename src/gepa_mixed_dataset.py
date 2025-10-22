#!/usr/bin/env python3
"""
GEPA optimization for mixed MATH500 and ARC Challenge dataset.
Goal: Optimize a single system prompt that can route between:
  - Detailed reasoning for MATH problems
  - Direct answers for ARC problems
"""

import argparse
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: datasets library not available")


@dataclass
class MixedExample:
    """A single example with question, answer, and task type"""
    question: str
    answer: str
    task_type: str  # "math" or "arc"
    formatted_prompt: str


def load_math_500_samples(n_samples: int = 15, split: str = "test", seed: int = 42) -> List[MixedExample]:
    """Load n_samples from MATH-500 dataset"""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library required")

    dataset = load_dataset("HuggingFaceH4/MATH-500")[split]

    # Sample randomly
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    examples = []
    for idx in indices:
        item = dataset[idx]
        examples.append(MixedExample(
            question=item["problem"],
            answer=item["solution"],
            task_type="math",
            formatted_prompt=item["problem"]
        ))

    return examples


def load_arc_samples(n_samples: int = 15, split: str = "test", seed: int = 42) -> List[MixedExample]:
    """Load n_samples from ARC Challenge dataset"""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library required")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]

    # Sample randomly
    random.seed(seed + 1)  # Different seed to avoid overlap
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    examples = []
    for idx in indices:
        item = dataset[idx]
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        # Format as multiple choice question
        formatted = f"Question: {question}\n"
        for label, choice in zip(labels, choices):
            formatted += f"{label}. {choice}\n"
        formatted = formatted.rstrip()

        examples.append(MixedExample(
            question=formatted,
            answer=answer_key,
            task_type="arc",
            formatted_prompt=formatted
        ))

    return examples


def create_mixed_dataset(
    n_math: int = 15,
    n_arc: int = 15,
    split: str = "test",
    seed: int = 42,
    shuffle: bool = True
) -> List[MixedExample]:
    """Create a mixed dataset with MATH and ARC examples"""
    math_examples = load_math_500_samples(n_math, split, seed)
    arc_examples = load_arc_samples(n_arc, split, seed)

    mixed = math_examples + arc_examples

    if shuffle:
        random.seed(seed)
        random.shuffle(mixed)

    return mixed


def save_mixed_dataset(examples: List[MixedExample], output_path: str):
    """Save mixed dataset to JSON"""
    data = {
        "total": len(examples),
        "math_count": sum(1 for e in examples if e.task_type == "math"),
        "arc_count": sum(1 for e in examples if e.task_type == "arc"),
        "examples": [
            {
                "question": e.question,
                "answer": e.answer,
                "task_type": e.task_type,
                "formatted_prompt": e.formatted_prompt
            }
            for e in examples
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} examples to {output_path}")
    print(f"  - MATH: {data['math_count']}")
    print(f"  - ARC: {data['arc_count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed MATH500 + ARC Challenge dataset for GEPA optimization"
    )
    parser.add_argument("--n-math", type=int, default=15, help="Number of MATH examples")
    parser.add_argument("--n-arc", type=int, default=15, help="Number of ARC examples")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", default="./data/mixed_math_arc.json",
                       help="Output path for mixed dataset")
    parser.add_argument("--no-shuffle", action="store_true",
                       help="Don't shuffle the mixed dataset")

    args = parser.parse_args()

    print("Creating mixed dataset...")
    print(f"  MATH-500: {args.n_math} examples")
    print(f"  ARC-Challenge: {args.n_arc} examples")
    print(f"  Split: {args.split}")
    print(f"  Seed: {args.seed}")

    mixed_dataset = create_mixed_dataset(
        n_math=args.n_math,
        n_arc=args.n_arc,
        split=args.split,
        seed=args.seed,
        shuffle=not args.no_shuffle
    )

    save_mixed_dataset(mixed_dataset, args.output)


if __name__ == "__main__":
    main()
