"""
Labeling script for MATH benchmark data.
This script loads MATH benchmark data, samples 200 questions randomly,
and uses Claude (via AWS Bedrock) to classify each question as easy (0) or hard (1).
"""

import json
import os
import random
import sys
from typing import Dict, List
from datasets import load_dataset
import boto3

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def classify_difficulty(bedrock_client, question: str, answer: str, model_id: str) -> int:
    """
    Use Claude (via AWS Bedrock) to classify a question as easy (0) or hard (1).

    Args:
        bedrock_client: AWS Bedrock runtime client
        question: The math question
        answer: The answer to the question
        model_id: The model ID to use

    Returns:
        0 for easy, 1 for hard
    """
    prompt = f"""Please analyze the following math problem and classify it as either EASY or HARD.

Question: {question}

Answer: {answer}

Definition:
- EASY: Problems that a model can solve correctly with short or surface-level reasoning steps. These typically involve direct computation, basic formulas, or simple logical deductions.
- HARD: Problems that require the model to perform deep, multi-step, or rigorous reasoning to arrive at the correct solution. These often involve abstract concepts, complex relationships, or non-trivial reasoning chains.

Please respond with only one word: either "EASY" or "HARD"."""

    try:
        # Prepare the request body for Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })

        # Invoke the model
        response = bedrock_client.invoke_model(
            body=body,
            modelId=model_id
        )

        # Parse response
        response_body = json.loads(response.get("body").read())
        response_text = response_body.get("content", [{}])[0].get("text", "").strip().upper()
        print(response_text)
        # Return 0 for easy, 1 for hard
        if "EASY" in response_text:
            return 0
        elif "HARD" in response_text:
            return 1
        else:
            # Default to hard if uncertain
            print(f"Unexpected response: {response_text}, defaulting to HARD")
            return 1

    except Exception as e:
        print(f"Error calling Claude via Bedrock: {e}")
        # Default to hard on error
        return 1


def main():
    """Main function to load MATH data, classify, and save results."""

    # Initialize AWS Bedrock client
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2"
    )

    # Use Claude Sonnet 4 (via inference profile)
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    output_path = "data/MATH_adaptive_demo/data.json"

    # Load existing data if available (for resuming)
    if os.path.exists(output_path):
        print(f"Found existing file at {output_path}, loading...")
        with open(output_path, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)
        print(f"Loaded {len(labeled_data)} existing labeled examples")
    else:
        labeled_data = []

    print("Loading MATH dataset...")
    # Load MATH dataset from Hugging Face
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="train")
    except:
        # If train split doesn't exist, try test split
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Get total number of examples
    total_examples = len(dataset)
    print(f"Total examples in MATH dataset: {total_examples}")

    # Randomly sample 200 examples
    sample_size = 200
    if total_examples < sample_size:
        print(f"Warning: Dataset has only {total_examples} examples, using all of them")
        sample_size = total_examples

    # Create random indices
    random.seed(42)  # For reproducibility
    sampled_indices = random.sample(range(total_examples), sample_size)

    print(f"Sampling {sample_size} examples...")
    print(f"Using model: {model_id}")
    print(f"Already labeled: {len(labeled_data)}, Remaining: {sample_size - len(labeled_data)}")

    # Process each sampled example
    for i, idx in enumerate(sampled_indices):
        # Skip if already processed
        if i < len(labeled_data):
            continue

        example = dataset[idx]

        # Extract question and answer
        question = example.get("problem", "")
        answer = example.get("solution", "")

        print(f"\nProcessing {i+1}/{sample_size}...")
        print(f"Question preview: {question[:100]}...")

        # Classify difficulty using Claude via Bedrock
        difficulty_type = classify_difficulty(bedrock_client, question, answer, model_id)

        print(f"Classified as: {'EASY' if difficulty_type == 0 else 'HARD'}")

        # Create labeled example in Hugging Face format
        labeled_example = {
            "problem": question,
            "solution": answer,
            "level": example.get("level", ""),
            "type": difficulty_type,
            "difficulty_type": difficulty_type  # 0 for easy, 1 for hard
        }

        labeled_data.append(labeled_example)

        # Write to file immediately after each labeling
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labeled_data, f, indent=2, ensure_ascii=False)

        print(f"Saved progress: {len(labeled_data)}/{sample_size} examples")

    # Print final statistics
    easy_count = sum(1 for item in labeled_data if item["difficulty_type"] == 0)
    hard_count = sum(1 for item in labeled_data if item["difficulty_type"] == 1)

    print(f"\nLabeling complete!")
    print(f"Total examples: {len(labeled_data)}")
    print(f"Easy questions: {easy_count} ({easy_count/len(labeled_data)*100:.1f}%)")
    print(f"Hard questions: {hard_count} ({hard_count/len(labeled_data)*100:.1f}%)")
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    main()
