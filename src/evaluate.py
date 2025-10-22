#!/usr/bin/env python3
"""
Script to evaluate JSON responses by extracting Final Answer numbers and comparing with gold answers.
Adds extracted_answer and is_correct fields to the original JSON structure.
"""

import json
import re
import sys
import os
import boto3
import argparse
from typing import Optional
from tqdm import tqdm

def extract_final_answer(response_text):
    """
    Extract the answer after **Final Answer:** until \n
    Supports both numbers and letter choices (A, B, C, D)

    Args:
        response_text (str): The response text containing the final answer

    Returns:
        str: The extracted answer (number or letter), or None if not found
    """
    # First try to match **Final Answer:** followed by any text until newline
    pattern1 = r'\*\*Final Answer:\*\*\s*([^\n\r]*)'
    if not response_text:
        return None
    match1 = re.search(pattern1, response_text)
    
    if match1:
        answer_text = match1.group(1).strip()

        # First try to extract letter choices (A, B, C, D) - case insensitive
        letter_pattern = r'\b[ABCD]\b'
        letter_match = re.search(letter_pattern, answer_text, re.IGNORECASE)

        if letter_match:
            return letter_match.group(0).upper()

        # If no letter found, extract numbers from the answer text
        # This handles cases like "16", "16.", "$16$", etc.
        number_pattern = r'[-+]?\d+\.?\d*'
        number_match = re.search(number_pattern, answer_text)

        if number_match:
            return number_match.group(0)
    
    # Try alternative pattern: Final Answer: (without bold formatting)
    pattern2 = r'Final Answer:\s*([^\n\r]*)'
    match2 = re.search(pattern2, response_text)
    
    if match2:
        answer_text = match2.group(1).strip()

        # First try to extract letter choices (A, B, C, D) - case insensitive
        letter_pattern = r'\b[ABCD]\b'
        letter_match = re.search(letter_pattern, answer_text, re.IGNORECASE)

        if letter_match:
            return letter_match.group(0).upper()

        # If no letter found, extract numbers from the answer text
        number_pattern = r'[-+]?\d+\.?\d*'
        number_match = re.search(number_pattern, answer_text)

        if number_match:
            return number_match.group(0)
    
    # Try to find the last occurrence of a standalone answer at the end of the response
    # This might catch cases where the response was truncated but the final answer is still there
    lines = response_text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Check for standalone letter choices first
        if line and re.match(r'^[ABCD]$', line, re.IGNORECASE):
            return line.upper()
        # Check for standalone numbers
        if line and re.match(r'^\d+$', line):
            return line
    
    return None


class ClaudeMathEvaluator:
    """Claude Sonnet 4 evaluator for math_500 benchmark"""

    def __init__(self, model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-west-2"):
        """
        Initialize Claude evaluator

        Args:
            model_id: Bedrock model ID for Claude Sonnet 4
            region_name: AWS region name
        """
        self.model_id = model_id
        self.region_name = region_name

        try:
            self.bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to AWS Bedrock. Please ensure your AWS credentials are configured: {e}")

    def evaluate_math_answer(self, prompt: str, model_response: str, gold_answer: str) -> dict:
        """
        Use Claude Sonnet 4 to evaluate if model response matches gold answer

        Args:
            prompt: The original math problem/prompt
            model_response: The model's complete response
            gold_answer: The ground truth answer

        Returns:
            dict: {"is_correct": bool, "reasoning": str}
        """
        system_prompt = f"""Please evaluate whether the model's answer is correct given the question and the gold answer.

        ## Question:
        {prompt}

        ## Model's answer:
        {model_response}

        ## Gold answer:
        {gold_answer}

        ## Evaluation
        - Determine the correctness of the model's answer based on whether the final answer is equivalent to the gold answer.
        - Do not evaluate the intermediate steps or the rationale in the model's answer.

        IMPORTANT: When outputting JSON, make sure to properly escape backslashes. Use double backslashes (\\\\) for any mathematical expressions containing backslashes.

        Now, please output your scores and a short rationale below in valid JSON format:
        {{
            "reason": "your rationale (escape backslashes as \\\\)",
            "correct": "1 if correct, 0 if incorrect"
        }}
        """
        body = {
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": system_prompt}
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": 0,
            "system": system_prompt
        }
        try:
            response = self.bedrock.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id
            )
            response_body = json.loads(response.get("body").read())
            response_text = response_body.get("content")[0].get("text", "").strip()

            # Parse JSON response from Claude
            try:
                evaluation_result = json.loads(response_text)

                # Handle both string and integer values for "correct"
                correct_value = evaluation_result.get("correct", False)
                if isinstance(correct_value, str):
                    is_correct = correct_value.strip() == "1" or correct_value.strip().lower() == "true"
                elif isinstance(correct_value, int):
                    is_correct = bool(correct_value)
                else:
                    is_correct = bool(correct_value)

                return {
                    "is_correct": is_correct,
                    "reasoning": evaluation_result.get("reason", "No reasoning provided")
                }
            except json.JSONDecodeError as e:
                return {
                    "is_correct": None,
                    "reasoning": 'Json decode error'
                }

        except Exception as e:
            print(f"Warning: Claude evaluation failed: {e}")
            exit()


def process_json_file(input_file, output_file=None, benchmark=None, sample_count=None):
    """
    Process a JSON file to extract final answers and compare with gold answers.

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional, defaults to input_file)
        benchmark (str): Benchmark type (aime_2025, math_500, arc_challenge)
    """
    if output_file is None:
        output_file = input_file

    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    evaluate_sample_count = sample_count if sample_count else len(data.get('responses', []))
    # Auto-detect benchmark if not provided
    if not benchmark:
        raise ValueError("Benchmark must be provided")
    claude_evaluator = None
    if benchmark == 'math_500':
        try:
            claude_evaluator = ClaudeMathEvaluator()
            print("Claude Sonnet 4 evaluator initialized for math_500")
        except Exception as e:
            print(f"Warning: Failed to initialize Claude evaluator: {e}")
            print("Falling back to basic evaluation")
            exit()
    processed_ids = []
    print(f"Evaluating {evaluate_sample_count} samples")
    for response_item in tqdm(data.get('responses', [])[:evaluate_sample_count]):
        if 'is_correct' in response_item:
            processed_ids.append(response_item['prompt'])
            continue
        # Get response data
        response_text = response_item.get('response', '')
        prompt = response_item.get('prompt', '')
        gold_answer = str(response_item.get('gold_answer', '')).strip()

        if benchmark == 'math_500' and claude_evaluator:
            # Use Claude Sonnet 4 for math_500 evaluation
            evaluation_result = claude_evaluator.evaluate_math_answer(prompt, response_text, gold_answer)
            if evaluation_result['is_correct'] is None:
                response_item['is_correct'] = None
                response_item['claude_reasoning'] = 'Json decode error'
                continue
            response_item['is_correct'] = 1 if evaluation_result['is_correct'] == 1 else evaluation_result['is_correct']
            response_item['claude_reasoning'] = evaluation_result['reasoning']
        else:
            # Use basic evaluation for aime_2025 and arc_challenge
            extracted_answer = extract_final_answer(response_text)
            is_correct = 0
            if extracted_answer is not None and gold_answer:
                # Convert both to strings for comparison, handling potential formatting differences
                extracted_clean = str(extracted_answer).strip()
                gold_clean = str(gold_answer).strip()
                # Try exact match first
                if extracted_clean == gold_clean:
                    is_correct = 1
                else:
                    # Try numeric comparison for cases like "16.0" vs "16"
                    try:
                        extracted_num = float(extracted_clean)
                        gold_num = float(gold_clean)
                        if abs(extracted_num - gold_num) < 1e-9:
                            is_correct = 1
                    except (ValueError, TypeError):
                        pass
            # Add fields to response item
            response_item['extracted_answer'] = extracted_answer
            response_item['is_correct'] = is_correct
        processed_ids.append(response_item['prompt'])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate JSON responses by extracting Final Answer numbers and comparing with gold answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input_file",
        help="Path to input JSON file containing responses to evaluate"
    )

    parser.add_argument(
        "--output", "-o",
        dest="output_file",
        help="Path to output JSON file (default: modifies input file in place)"
    )

    parser.add_argument(
        "--benchmark", "-b",
        required=True,
        choices=['aime_2025', 'math_500', 'arc_challenge'],
        help="Benchmark type for evaluation method selection"
    )
    
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="Benchmark type for evaluation method selection"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    output_file = args.output_file if args.output_file else args.input_file

    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")
    print(f"Benchmark: {args.benchmark}")

    process_json_file(args.input_file, output_file, args.benchmark, args.sample_count)

if __name__ == "__main__":
    main()
