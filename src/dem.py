"""
GEPA Optimization for AIME Math Problems using Claude via AWS Bedrock

This script optimizes a Chain of Thought program for solving AIME math problems
using the GEPA optimizer with Claude models through AWS Bedrock (boto3).
"""

import boto3
import json
import dspy
from datasets import load_dataset
import random
from typing import Any, Dict


class ClaudeBedrock(dspy.LM):
    """
    DSPy Language Model wrapper for Claude models via AWS Bedrock using boto3.
    """

    def __init__(
        self,
        model: str,
        region_name: str = "us-west-2",
        temperature: float = 1.0,
        max_tokens: int = 32000,
        **kwargs
    ):
        """
        Initialize Claude Bedrock LM.

        Args:
            model: Model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0:28k")
            region_name: AWS region
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Initialize bedrock runtime client
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )

        # Call parent constructor
        super().__init__(model=model)

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Call the Claude model via Bedrock.

        Args:
            prompt: Text prompt (will be converted to messages format)
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            List of generated responses
        """
        # Merge kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        temperature = merged_kwargs.get('temperature', self.temperature)
        max_tokens = merged_kwargs.get('max_tokens', self.max_tokens)

        # Convert prompt to messages format if needed
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]

        # Extract system messages and convert to system parameter
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                # Concatenate system messages
                if system_prompt is None:
                    system_prompt = msg.get("content", "")
                else:
                    system_prompt += "\n\n" + msg.get("content", "")
            else:
                filtered_messages.append(msg)

        # Prepare request body
        body_dict = {
            "max_tokens": max_tokens,
            "messages": filtered_messages,
            "temperature": temperature,
            "anthropic_version": "bedrock-2023-05-31"
        }

        # Add system prompt if present
        if system_prompt:
            body_dict["system"] = system_prompt

        body = json.dumps(body_dict)

        # Invoke model
        response = self.client.invoke_model(
            body=body,
            modelId=self.model
        )

        # Parse response
        response_body = json.loads(response.get("body").read())

        # Extract text from response
        content = response_body.get("content", [])
        if isinstance(content, list) and len(content) > 0:
            text = content[0].get("text", "")
        else:
            text = str(content)

        return [text]

    def basic_request(self, prompt: str, **kwargs):
        """Basic request interface for DSPy compatibility."""
        return self.__call__(prompt=prompt, **kwargs)


def init_dataset():
    """
    Load and prepare AIME datasets.

    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    # Load training/validation data from previous years (2022-2024)
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]

    # Shuffle with fixed seed for reproducibility
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    # Load test data (AIME 2025)
    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]

    # Split train/val 50/50
    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]

    # Repeat test set 5 times for statistical stability
    test_set = test_split * 5

    return train_set, val_set, test_set


class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Simple evaluation metric: exact match on answer.

    Args:
        example: Ground truth example
        prediction: Model prediction

    Returns:
        1 if correct, 0 otherwise
    """
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Optimization metric with detailed feedback for GEPA.

    This metric provides:
    - Score (0 or 1)
    - Feedback text explaining correctness
    - Full solution when available

    Args:
        example: Ground truth example
        prediction: Model prediction

    Returns:
        dspy.Prediction with score and feedback
    """
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')

    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. "
            f"You responded with '{prediction.answer}', which couldn't be parsed as a python integer. "
            f"Please ensure your answer is a valid integer without any additional text or formatting."
        )
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += (
                f" Here's the full step-by-step solution:\n{written_solution}\n\n"
                f"Think about what takeaways you can learn from this solution to improve your future answers "
                f"and approach to similar problems and ensure your final answer is a valid integer."
            )
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += (
            f" Here's the full step-by-step solution:\n{written_solution}\n\n"
            f"Think about what takeaways you can learn from this solution to improve your future answers "
            f"and approach to similar problems."
        )

    return dspy.Prediction(score=score, feedback=feedback_text)


def main():
    """Main execution function."""

    print("=" * 80)
    print("GEPA Optimization for AIME Math Problems")
    print("Using Claude models via AWS Bedrock")
    print("=" * 80)

    # Step 1: Initialize language models
    print("\n[1/6] Initializing language models...")

    # Main LM: Claude 3.5 Sonnet v2
    lm = ClaudeBedrock(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-west-2",
        temperature=1.0,
        max_tokens=32000
    )

    # Reflection LM: Claude Sonnet 4 (cross-region inference profile)
    reflection_lm = ClaudeBedrock(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-west-2",
        temperature=1.0,
        max_tokens=32000
    )

    # Configure DSPy with main LM
    dspy.configure(lm=lm)
    print(f"  Main LM: {lm.model}")
    print(f"  Reflection LM: {reflection_lm.model}")

    # Step 2: Load datasets
    print("\n[2/6] Loading AIME datasets...")
    train_set, val_set, test_set = init_dataset()
    print(f"  Train set size: {len(train_set)}")
    print(f"  Validation set size: {len(val_set)}")
    print(f"  Test set size: {len(test_set)}")

    # Step 3: Define program
    print("\n[3/6] Defining Chain of Thought program...")
    program = dspy.ChainOfThought(GenerateResponse)
    print("  Program: dspy.ChainOfThought(GenerateResponse)")

    # Step 4: Evaluate unoptimized program
    print("\n[4/6] Evaluating unoptimized program on test set...")
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=1,
        display_table=False,
        display_progress=True
    )

    unoptimized_score = evaluate(program)
    print(f"  Unoptimized score: {unoptimized_score:.2%}")

    # Step 5: Optimize with GEPA
    print("\n[5/6] Optimizing with GEPA...")
    print("  This may take a while...")

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        auto="light",
        num_threads=1,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=reflection_lm
    )

    optimized_program = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )

    print("  Optimization complete!")

    # Step 6: Evaluate optimized program
    print("\n[6/6] Evaluating optimized program on test set...")
    optimized_score = evaluate(optimized_program)
    print(f"  Optimized score: {optimized_score:.2%}")

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Unoptimized performance: {unoptimized_score:.2%}")
    print(f"Optimized performance:   {optimized_score:.2%}")
    improvement = optimized_score - unoptimized_score
    print(f"Improvement:             {improvement:+.2%}")
    print("=" * 80)

    # Display optimized prompt
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPT INSTRUCTIONS")
    print("=" * 80)
    print(optimized_program.predict.signature.instructions)
    print("=" * 80)

    return optimized_program


if __name__ == "__main__":
    main()
