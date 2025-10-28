#!/usr/bin/env python3
"""
GEPA training script for MATH dataset with trajectory logging.
Uses local gepa/ code, AWS Bedrock Claude for reflection, and local vLLM for task model.
"""

import sys
import os
import json
import requests
from typing import Any, Dict, List, TypedDict
from pathlib import Path

# Add gepa to path - use local gepa/ directory instead of pip installed version
GEPA_PATH = Path(__file__).parent.parent / "gepa" / "src"
sys.path.insert(0, str(GEPA_PATH))

# Now import from local gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.core.state import GEPAState
import gepa


# Data structures for our adapter
class MathDataInst(TypedDict):
    problem: str
    answer: str
    solution: str  # Ground truth solution for additional context


class MathTrajectory(TypedDict):
    data: MathDataInst
    full_response: str
    system_prompt: str


class MathRolloutOutput(TypedDict):
    full_response: str


class TrajectoryLogger:
    """Logs detailed GEPA trajectories with immediate writing"""

    def __init__(self, output_file=None):
        self.iterations = []
        self.output_file = output_file

        # Initialize file with empty structure if output file is specified
        if self.output_file:
            self._initialize_file()

    def _initialize_file(self):
        """Initialize the output file with empty structure"""
        with open(self.output_file, 'w') as f:
            json.dump({
                'iterations': [],
                'total_iterations': 0
            }, f, indent=2)
        print(f"Initialized trajectory file: {self.output_file}")

    def log_iteration(self, iteration_data: Dict[str, Any]):
        """Log data from one GEPA iteration and immediately write to file"""
        self.iterations.append(iteration_data)

        # Immediately write to file if output file is configured
        if self.output_file:
            self._append_to_file(iteration_data)

    def _append_to_file(self, iteration_data: Dict[str, Any]):
        """Append new iteration data to file"""
        try:
            # Read existing data
            with open(self.output_file, 'r') as f:
                data = json.load(f)

            # Add new iteration
            data['iterations'].append(iteration_data)
            data['total_iterations'] = len(data['iterations'])

            # Write back to file
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[Iteration {iteration_data.get('iteration', '?')}] Saved to {self.output_file}")

        except Exception as e:
            print(f"Warning: Failed to write iteration data: {e}")
            # Don't raise exception to avoid interrupting main loop

    def save(self, filepath: str):
        """Final save (for compatibility, now mostly redundant if using immediate writes)"""
        # If already writing continuously, this just ensures data is complete
        if filepath != self.output_file or self.output_file is None:
            with open(filepath, 'w') as f:
                json.dump({
                    'iterations': self.iterations,
                    'total_iterations': len(self.iterations)
                }, f, indent=2)
        print(f"Final save complete: {filepath}")


# Global trajectory logger (will be initialized in main() with output file)
traj_logger = None


class MathAdapter(GEPAAdapter[MathDataInst, MathTrajectory, MathRolloutOutput]):
    """
    Custom adapter for MATH dataset that:
    - Uses vLLM on port 8088 for task execution
    - Logs all questions, answers, and reflection prompts
    """

    def __init__(
        self,
        vllm_model: str = "Qwen/Qwen3-8B",
        vllm_host: str = "localhost",
        vllm_port: int = 8088,
        max_tokens: int = 2048,
    ):
        self.vllm_model = vllm_model
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.max_tokens = max_tokens
        self.vllm_url = f"http://{vllm_host}:{vllm_port}/v1/completions"

    def call_vllm(self, prompt: str, system_prompt: str = "") -> str:
        """Call vLLM server"""
        # Combine system and user prompt
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:" if system_prompt else prompt

        payload = {
            "model": self.vllm_model,
            "prompt": full_prompt,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stop": ["<|endoftext|>", "\nQuestion:", "Question:"]
        }

        try:
            response = requests.post(self.vllm_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            return ""

    def evaluate(
        self,
        batch: List[MathDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MathTrajectory, MathRolloutOutput]:
        """Evaluate candidate on batch of math problems"""
        outputs: List[MathRolloutOutput] = []
        scores: List[float] = []
        trajectories: List[MathTrajectory] | None = [] if capture_traces else None

        # Get system prompt from candidate
        system_prompt = next(iter(candidate.values()))

        for data in batch:
            # Call vLLM to get response
            response = self.call_vllm(data["problem"], system_prompt)

            # Score based on whether answer appears in response
            # Normalize both for comparison
            expected_answer = data["answer"].strip()
            score = 1.0 if expected_answer in response else 0.0

            output = {"full_response": response}
            outputs.append(output)
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "data": data,
                    "full_response": response,
                    "system_prompt": system_prompt,
                })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[MathTrajectory, MathRolloutOutput],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build reflective dataset from evaluation results.
        This is what gets fed to the reflection LM.
        """
        ret_d: Dict[str, List[Dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        items: List[Dict[str, Any]] = []
        trace_instances = list(zip(
            eval_batch.trajectories,
            eval_batch.scores,
            eval_batch.outputs,
            strict=False
        ))

        for traj, score, _ in trace_instances:
            data = traj["data"]
            generated_output = traj["full_response"]

            if score > 0.0:
                feedback = (
                    f"✓ CORRECT: The response correctly includes the answer '{data['answer']}'. "
                    f"The solution approach is valid."
                )
            else:
                feedback = (
                    f"✗ INCORRECT: The response does not contain the correct answer '{data['answer']}'. "
                    f"\n\nGround truth solution:\n{data['solution']}\n\n"
                    f"The model should learn from this solution approach to improve its problem-solving strategy."
                )

            item = {
                "Problem": data["problem"],
                "Generated Answer": generated_output,
                "Expected Answer": data["answer"],
                "Feedback": feedback,
                "Score": score,
            }
            items.append(item)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for reflection.")

        return ret_d


def create_claude_reflection_lm(claude_host: str = "http://172.31.13.66:8080"):
    """
    Create reflection LM function using local Claude HTTP service.
    Returns a function that takes a prompt and returns LM response.

    Args:
        claude_host: URL of the local Claude service (running web.py)
    """
    claude_url = f"{claude_host}/reward"

    def reflection_lm(prompt: str) -> str:
        """Call Claude via local HTTP service for reflection"""
        # Log the reflection prompt
        print(f"\n{'='*80}")
        print("REFLECTION PROMPT:")
        print(f"{'='*80}")
        print(prompt)
        print(f"{'='*80}\n")

        payload = {
            "prompt": prompt,
            "max_tokens": 4096,
            "temperature": 1.0,
        }

        try:
            response = requests.post(claude_url, json=payload, timeout=300)
            response.raise_for_status()
            result_json = response.json()

            # Extract text from Bedrock response format
            if "content" in result_json and len(result_json["content"]) > 0:
                result = result_json["content"][0].get("text", "")
            else:
                result = ""

            print(f"\n{'='*80}")
            print("REFLECTION RESPONSE:")
            print(f"{'='*80}")
            print(result)
            print(f"{'='*80}\n")

            return result

        except Exception as e:
            print(f"Error calling Claude: {e}")
            import traceback
            traceback.print_exc()
            return ""

    return reflection_lm


class LoggingReflectionLM:
    """Wrapper around reflection LM that logs all prompts and responses"""

    def __init__(self, base_lm):
        self.base_lm = base_lm
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1

        # Log to trajectory
        call_data = {
            'call_number': self.call_count,
            'reflection_prompt': prompt,
        }

        # Call base LM
        response = self.base_lm(prompt)
        call_data['reflection_response'] = response

        # Store in current iteration (will be added by custom proposer)
        if hasattr(self, '_current_iteration_data'):
            if 'reflection_calls' not in self._current_iteration_data:
                self._current_iteration_data['reflection_calls'] = []
            self._current_iteration_data['reflection_calls'].append(call_data)

        return response


def load_math_dataset(filepath: str) -> List[MathDataInst]:
    """Load MATH dataset from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def main():
    """Main training loop"""
    global traj_logger

    print("="*80)
    print("GEPA Training for MATH Dataset")
    print("="*80)

    # Configuration
    train_file = "data/MATH_adaptive_demo/train.json"
    output_file = "data/MATH_adaptive_demo/traj.json"

    # Initialize trajectory logger with output file for immediate writes
    traj_logger = TrajectoryLogger(output_file=output_file)

    # Load dataset
    print(f"\nLoading dataset from {train_file}...")
    dataset = load_math_dataset(train_file)
    print(f"Loaded {len(dataset)} problems")

    # Split into train/val (use small subset for testing)
    # For full training, adjust these numbers
    trainset = dataset[:200]  # Use first 20 for training
    valset = dataset[200:300]  # Use next 10 for validation

    print(f"Training set: {len(trainset)} problems")
    print(f"Validation set: {len(valset)} problems")

    # Initial seed prompt
    seed_prompt = {
        "system_prompt": """You are a helpful math problem solver.

When given a math problem:
1. Read the problem carefully and identify what is being asked
2. Break down the problem into steps
3. Show your work clearly
4. Provide the final answer in the exact format requested

Your answer should be clear and complete."""
    }

    print("\n" + "="*80)
    print("Initial Prompt:")
    print("="*80)
    print(seed_prompt["system_prompt"])
    print("="*80 + "\n")

    # Create adapter
    adapter = MathAdapter(
        vllm_model="Qwen/Qwen3-8B",
        vllm_host="localhost",
        vllm_port=8088,
        max_tokens=2048,
    )

    # Create reflection LM with logging
    base_reflection_lm = create_claude_reflection_lm()
    reflection_lm = LoggingReflectionLM(base_reflection_lm)

    # Custom logger to capture trajectories
    class CustomLogger:
        def __init__(self, traj_logger):
            self.traj_logger = traj_logger
            self.current_iteration = None

        def log(self, message: str):
            print(message)
            if self.current_iteration is not None:
                if 'logs' not in self.current_iteration:
                    self.current_iteration['logs'] = []
                self.current_iteration['logs'].append(message)

    custom_logger = CustomLogger(traj_logger)

    # Monkey-patch to capture iteration data
    original_propose = None

    def wrapped_propose(self, state: GEPAState):
        # Create iteration data structure
        iteration_data = {
            'iteration': state.i + 1,
            'total_evals': state.total_num_evals,
            'candidates_so_far': len(state.program_candidates),
        }

        # Set up for logging
        reflection_lm._current_iteration_data = iteration_data
        custom_logger.current_iteration = iteration_data

        result = None
        try:
            # Call original propose (original_propose is an instance method, needs self)
            result = original_propose(self, state)

            # Log iteration data
            if result is not None:
                iteration_data['new_candidate'] = result.candidate
                iteration_data['subsample_scores_before'] = result.subsample_scores_before
                iteration_data['subsample_scores_after'] = result.subsample_scores_after
                iteration_data['improved'] = sum(result.subsample_scores_after) > sum(result.subsample_scores_before)

        except Exception as e:
            # Log the error but re-raise to maintain original behavior
            iteration_data['error'] = str(e)
            raise

        finally:
            # Always log the iteration, even if there was an error
            try:
                traj_logger.log_iteration(iteration_data)
            except Exception as log_error:
                print(f"Warning: Failed to log iteration: {log_error}")

            custom_logger.current_iteration = None

        return result

    print("\nStarting GEPA optimization...")
    print("="*80)

    result = None  # Initialize to None in case of exception
    try:
        # Monkey-patch the propose method to capture data
        from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
        original_propose = ReflectiveMutationProposer.propose
        ReflectiveMutationProposer.propose = wrapped_propose

        # Run GEPA optimization
        result = gepa.optimize(
            seed_candidate=seed_prompt,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            max_metric_calls=1000,  # Small number for testing; increase for real training
            reflection_minibatch_size=3,
            logger=custom_logger,
            candidate_selection_strategy="pareto",
            skip_perfect_score=True,
            display_progress_bar=True,
            seed=42,
        )

        # Restore original method
        ReflectiveMutationProposer.propose = original_propose

    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()

    # Save trajectory
    print("\n" + "="*80)
    print("Saving trajectory...")
    traj_logger.save(output_file)

    # Print final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    if result is not None:
        print(f"\nBest candidate prompt:")
        print("-" * 80)
        print(result.best_candidate['system_prompt'])
        print("-" * 80)
        print(f"\nBest validation score: {result}")
        print(f"Total iterations: {len(traj_logger.iterations)}")
        print(f"Trajectory saved to: {output_file}")
    else:
        print("\nOptimization failed - no result available.")
        print(f"Trajectory saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
