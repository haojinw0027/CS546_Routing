#!/usr/bin/env python3
"""
Custom GEPA Adapter for mixed MATH/ARC dataset optimization.
This adapter evaluates how well a system prompt performs on both task types.
"""

import re
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of evaluating one example"""
    prompt: str
    response: str
    gold_answer: str
    task_type: str
    score: float  # 0.0 or 1.0
    trajectory: Dict[str, Any]  # For reflection


class MixedDatasetAdapter:
    """
    GEPA Adapter for mixed MATH/ARC optimization.

    This adapter:
    1. Evaluates a system prompt on both MATH and ARC examples
    2. Scores based on correctness (exact match for ARC, answer extraction for MATH)
    3. Provides detailed trajectories for GEPA's reflection
    """

    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 8088,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ):
        self.model = model
        self.host = host
        self.port = port
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"http://{host}:{port}/v1/completions"

    def prompt_model(
        self,
        user_prompt: str,
        system_prompt: str
    ) -> Optional[str]:
        """Send a prompt to the model via vLLM API"""
        full_prompt = f"{system_prompt}\n\nQuestion: {user_prompt}\n\nAnswer:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": ['<|endoftext|>', '\nQuestion:']
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error calling model: {e}")
            return None

    def extract_math_answer(self, response: str) -> Optional[str]:
        """
        Extract final answer from MATH response.
        Looks for patterns like "Final Answer: X" or boxed answers.
        """
        # Try to find "Final Answer:" pattern
        final_answer_match = re.search(r'\*\*Final Answer:\*\*\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()

        # Try to find boxed answer
        boxed_match = re.search(r'\\boxed\{(.+?)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Fallback: last line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return None

    def extract_arc_answer(self, response: str) -> Optional[str]:
        """
        Extract answer from ARC response.
        Looks for single letter answers (A, B, C, D).
        """
        # Try to find "Final Answer:" pattern with letter
        final_answer_match = re.search(r'\*\*Final Answer:\*\*\s*([A-D])', response, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).upper()

        # Look for standalone letter answers
        letter_match = re.search(r'\b([A-D])\b', response)
        if letter_match:
            return letter_match.group(1).upper()

        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        # Remove common formatting
        answer = re.sub(r'\\boxed\{(.+?)\}', r'\1', answer)
        answer = answer.strip()
        # Remove dollar signs
        answer = answer.replace('$', '')
        return answer

    def check_math_correctness(self, predicted: str, gold: str) -> bool:
        """
        Check if MATH answer is correct.
        Uses string matching (can be enhanced with symbolic math later).
        """
        if not predicted:
            return False

        pred_norm = self.normalize_answer(predicted)
        gold_norm = self.normalize_answer(gold)

        # Direct match
        if pred_norm == gold_norm:
            return True

        # Try to extract final answer from gold if it's a full solution
        gold_final_match = re.search(r'\\boxed\{(.+?)\}', gold)
        if gold_final_match:
            gold_final = self.normalize_answer(gold_final_match.group(1))
            if pred_norm == gold_final:
                return True

        return False

    def check_arc_correctness(self, predicted: str, gold: str) -> bool:
        """Check if ARC answer is correct (exact match)"""
        if not predicted:
            return False
        return predicted.upper() == gold.upper()

    def evaluate_single(
        self,
        example: Dict[str, Any],
        system_prompt: str
    ) -> EvaluationResult:
        """Evaluate a single example with the given system prompt"""
        question = example["formatted_prompt"]
        gold_answer = example["answer"]
        task_type = example["task_type"]

        # Get model response
        response = self.prompt_model(question, system_prompt)

        if response is None:
            return EvaluationResult(
                prompt=question,
                response="",
                gold_answer=gold_answer,
                task_type=task_type,
                score=0.0,
                trajectory={
                    "error": "Model did not respond",
                    "task_type": task_type
                }
            )

        # Extract and check answer based on task type
        if task_type == "math":
            extracted = self.extract_math_answer(response)
            correct = self.check_math_correctness(extracted or "", gold_answer)
        else:  # arc
            extracted = self.extract_arc_answer(response)
            correct = self.check_arc_correctness(extracted or "", gold_answer)

        score = 1.0 if correct else 0.0

        # Build trajectory for reflection
        trajectory = {
            "task_type": task_type,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "response": response,
            "extracted_answer": extracted,
            "gold_answer": gold_answer,
            "correct": correct,
            "response_length": len(response),
            "has_reasoning": "**Reasoning:**" in response or "Step" in response
        }

        return EvaluationResult(
            prompt=question,
            response=response,
            gold_answer=gold_answer,
            task_type=task_type,
            score=score,
            trajectory=trajectory
        )

    def evaluate(
        self,
        system_prompt: str,
        examples: List[Dict[str, Any]],
        capture_traces: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate system prompt on a batch of examples.
        Returns results compatible with GEPA's expectations.
        """
        results = []

        for example in examples:
            result = self.evaluate_single(example, system_prompt)
            results.append(result)

        # Compute statistics
        math_results = [r for r in results if r.task_type == "math"]
        arc_results = [r for r in results if r.task_type == "arc"]

        math_score = sum(r.score for r in math_results) / len(math_results) if math_results else 0.0
        arc_score = sum(r.score for r in arc_results) / len(arc_results) if arc_results else 0.0
        overall_score = sum(r.score for r in results) / len(results) if results else 0.0

        return {
            "results": results,
            "scores": [r.score for r in results],
            "overall_score": overall_score,
            "math_score": math_score,
            "arc_score": arc_score,
            "math_count": len(math_results),
            "arc_count": len(arc_results),
            "trajectories": [r.trajectory for r in results] if capture_traces else None
        }

    def make_reflective_dataset(
        self,
        evaluation: Dict[str, Any],
        component_name: str = "system_prompt"
    ) -> List[Dict[str, Any]]:
        """
        Create reflective dataset for GEPA's instructor.
        Focuses on failures and highlights task-specific patterns.
        """
        results = evaluation["results"]

        # Focus on errors, especially cross-task patterns
        error_examples = [r for r in results if r.score == 0.0]
        success_examples = [r for r in results if r.score == 1.0]

        reflective_data = []

        # Add error cases with feedback
        for result in error_examples:
            feedback = self._generate_feedback(result, is_error=True)
            reflective_data.append({
                "Inputs": {
                    "task_type": result.task_type,
                    "question": result.prompt[:200] + "..."
                },
                "Generated Outputs": result.response[:300] + "..." if len(result.response) > 300 else result.response,
                "Feedback": feedback
            })

        # Add a few success cases for contrast
        for result in success_examples[:2]:
            feedback = self._generate_feedback(result, is_error=False)
            reflective_data.append({
                "Inputs": {
                    "task_type": result.task_type,
                    "question": result.prompt[:200] + "..."
                },
                "Generated Outputs": result.response[:300] + "..." if len(result.response) > 300 else result.response,
                "Feedback": feedback
            })

        return reflective_data

    def _generate_feedback(self, result: EvaluationResult, is_error: bool) -> str:
        """Generate human-readable feedback for reflection"""
        task_type = result.task_type
        trajectory = result.trajectory

        if is_error:
            if task_type == "math":
                if not trajectory.get("has_reasoning"):
                    return (f"MATH problem requires detailed step-by-step reasoning but "
                           f"response lacks structured reasoning. Expected answer: {result.gold_answer}, "
                           f"Got: {trajectory.get('extracted_answer', 'nothing')}")
                else:
                    return (f"MATH problem has reasoning but wrong answer. "
                           f"Expected: {result.gold_answer}, Got: {trajectory.get('extracted_answer')}")
            else:  # arc
                if trajectory.get("has_reasoning"):
                    return (f"ARC question is simple multiple choice - detailed reasoning unnecessary "
                           f"and leads to confusion. Expected: {result.gold_answer}, "
                           f"Got: {trajectory.get('extracted_answer', 'nothing')}")
                else:
                    return (f"ARC answer is incorrect. Expected: {result.gold_answer}, "
                           f"Got: {trajectory.get('extracted_answer')}")
        else:
            if task_type == "math":
                return f"MATH problem solved correctly with proper reasoning. Answer: {result.gold_answer}"
            else:
                return f"ARC question answered correctly and concisely. Answer: {result.gold_answer}"


def test_adapter():
    """Test the adapter with sample data"""
    print("Testing MixedDatasetAdapter...")

    # Load mixed dataset
    with open("./data/mixed_math_arc.json", 'r') as f:
        data = json.load(f)

    examples = data["examples"][:5]  # Test on first 5

    adapter = MixedDatasetAdapter(
        model="Qwen/Qwen3-8B",
        host="localhost",
        port=8088
    )

    # Test with adaptive_cot prompt
    test_prompt = """You are an efficient problem solver. Analyze each question and choose the appropriate response strategy:

For MATH problems (multi-step calculations, proofs, algebraic manipulations):
- Provide detailed step-by-step reasoning
- Show all intermediate calculations
- Verify your work
- End with **Final Answer:**

For simple multiple choice questions:
- Answer directly and concisely
- End with **Final Answer:** [letter]

Adapt your response depth to the question complexity."""

    print("\nEvaluating with test prompt...")
    evaluation = adapter.evaluate(test_prompt, examples)

    print(f"\nResults:")
    print(f"  Overall Score: {evaluation['overall_score']:.2%}")
    print(f"  MATH Score: {evaluation['math_score']:.2%} ({evaluation['math_count']} examples)")
    print(f"  ARC Score: {evaluation['arc_score']:.2%} ({evaluation['arc_count']} examples)")

    print("\nReflective dataset:")
    reflective = adapter.make_reflective_dataset(evaluation)
    print(json.dumps(reflective[0], indent=2))


if __name__ == "__main__":
    test_adapter()
