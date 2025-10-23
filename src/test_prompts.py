import yaml
import datasets
import random
import re
from statistics import mean
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
from datetime import datetime
import concurrent.futures
import requests

# Global configuration
model_name = 'Qwen/Qwen3-14B'  # Local vllm model name
api_base = 'http://localhost:8088/v1'  # vllm service address
max_workers = 10  # Number of parallel API calls
num_samples = 250  # Number of samples per dataset
num_prompts = 4  # Number of prompts to test, default 10
seed = 42  # Random seed for fixed sampling

# Load tokenizer for fallback token calculation
fallback_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)

# Load system prompts from YAML
with open('./system_prompts/prompts.yaml', 'r') as f:
    prompts_data = yaml.safe_load(f)

# Load datasets
data_path = "./data"
math_ds = datasets.load_dataset("HuggingFaceH4/MATH-500", cache_dir=data_path)['test']
arc_ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=data_path)['test']

# Create mixed test set (sample num_samples from each dataset and mix)
random.seed(seed)
math_samples = random.sample(range(len(math_ds)), num_samples)
arc_samples = random.sample(range(len(arc_ds)), num_samples)

# Helper function: extract content from \boxed{} (handle nested braces)
def extract_boxed_answer(text):
    """Extract content from \\boxed{...} in text, supporting nested braces"""
    match = re.search(r'\\boxed{', text)
    if not match:
        return None

    start = match.end()
    brace_count = 1
    end = start

    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1

    if brace_count == 0:
        return text[start:end-1]
    return None

mixed_test = []
for idx in math_samples:
    sample = math_ds[idx]
    answer = extract_boxed_answer(sample.get('solution', '')) if 'solution' in sample else sample.get('final_answer', '')
    mixed_test.append({'type':'math', 'question':sample['problem'], 'answer': answer if answer else ''})
for idx in arc_samples:
    sample = arc_ds[idx]
    choices = '\n'.join([f"{label}: {text}" for label, text in zip(sample['choices']['label'], sample['choices']['text'])])
    question = f"{sample['question']}\n{choices}\nSelect the correct choice letter."
    mixed_test.append({'type':'arc', 'question':question, 'answer':sample['answerKey']})
random.shuffle(mixed_test)

# Save mixed test set to JSON
output_dir = './results/template_test'
os.makedirs(output_dir, exist_ok=True)
test_size = len(mixed_test)
mixed_test_filename = f"mixed_test_{test_size}_{seed}.json"
with open(os.path.join(output_dir, mixed_test_filename), 'w') as f:
    json.dump(mixed_test, f, indent=4, ensure_ascii=False)

# Evaluation functions
def evaluate_math(gold, pred):
    return gold.strip() == pred.strip()

def evaluate_arc(gold, pred):
    return gold.strip() == pred.strip().upper()  # ARC answers are A/B/C/D

# Configure vllm API parameters
api_url = f"{api_base}/chat/completions"
request_config = {
    'model': model_name,
    'max_tokens': 8088,
    'temperature': 0.7
}

# Test function
results = {}
results[model_name] = {}

# Create results directory (without timestamp)
run_dir = os.path.join(output_dir, model_name.replace('/', '_'))
os.makedirs(run_dir, exist_ok=True)

prompt_items = list(prompts_data.items())[:num_prompts]

all_accuracies = []
all_token_usages = []

for prompt_key, prompt_info in prompt_items:
    system_prompt = prompt_info['content']
    accuracies = []
    token_usages = []
    is_first = True  # Mark the first example

    # Prepare JSON output file
    results_filename = f"results_{prompt_key}_{test_size}_{seed}.json"
    results_filepath = os.path.join(run_dir, results_filename)

    # Resume from checkpoint: check if results file already exists
    completed_questions = set()
    if os.path.exists(results_filepath):
        try:
            with open(results_filepath, 'r') as f:
                existing_data = json.load(f)

            # Extract completed questions (use question text as unique identifier)
            # Only consider as completed when model_output is not empty
            for case in existing_data.get('cases', []):
                if case.get('model_output', '').strip():
                    completed_questions.add(case['question'])

            print(f"\n[Resume] Found existing results for {prompt_key}: {len(completed_questions)}/{len(mixed_test)} completed")

            # If all questions are already completed, skip this prompt
            if len(completed_questions) >= len(mixed_test):
                print(f"[Skip] All questions already completed for {prompt_key}")
                # Calculate summary
                for case in existing_data['cases']:
                    accuracies.append(1 if case['correct'] else 0)
                    token_usages.append(case['total_tokens'])
                avg_accuracy = mean(accuracies) * 100
                avg_tokens = mean(token_usages)
                results[model_name][prompt_key] = {'accuracy': avg_accuracy, 'avg_tokens': avg_tokens}
                all_accuracies.append(avg_accuracy)
                all_token_usages.extend(token_usages)
                continue
        except Exception as e:
            print(f"[Warning] Failed to load existing results: {e}. Starting fresh.")
            completed_questions = set()
    else:
        # Initialize JSON file structure
        initial_data = {
            'summary': {
                'accuracy': 0.0,
                'avg_tokens': 0.0
            },
            'cases': []
        }
        with open(results_filepath, 'w') as f:
            json.dump(initial_data, f, indent=4, ensure_ascii=False)

    def process_example(example):
        user_prompt = example['question'] + "\nPlease box your final answer within \\boxed{}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # Call vLLM API directly
        payload = {
            **request_config,
            'messages': messages
        }

        try:
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()

            pred = response_data['choices'][0]['message']['content']
            total_tokens = response_data['usage']['prompt_tokens'] + response_data['usage']['completion_tokens']
        except Exception as e:
            print(f"API call failed: {e}")
            # Fallback: use tokenizer to calculate
            pred = ""
            full_prompt = system_prompt + "\n" + user_prompt
            prompt_tokens = len(fallback_tokenizer.encode(full_prompt))
            completion_tokens = 0
            total_tokens = prompt_tokens + completion_tokens

        # Extract final answer
        pred_answer = extract_boxed_answer(pred)
        if pred_answer is None:
            # If no \boxed{}, fallback
            if example['type'] == 'math':
                pred_answer = pred.split()[-1] if pred.strip() else ''
            else:
                pred_answer = pred.strip()[0].upper() if pred.strip() else ''

        correct = (example['answer'].strip() == pred_answer.strip())
        return {
            'question': example['question'],
            'type': example['type'],
            'ground_truth': example['answer'],
            'model_output': pred,
            'extracted_answer': pred_answer,
            'correct': correct,
            'total_tokens': total_tokens
        }

    # Filter out completed questions
    remaining_test = [ex for ex in mixed_test if ex['question'] not in completed_questions]

    if len(remaining_test) == 0:
        print(f"[Info] No remaining questions for {prompt_key}")
    else:
        print(f"[Info] Processing {len(remaining_test)} remaining questions for {prompt_key}")

    # Use thread pool to process remaining questions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_example, example) for example in remaining_test]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Testing {prompt_key}"):
            result = future.result()

            # Append each result to the cases array in JSON file in real-time
            with open(results_filepath, 'r') as f:
                data = json.load(f)
            data['cases'].append(result)
            with open(results_filepath, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            accuracies.append(1 if result['correct'] else 0)
            token_usages.append(result['total_tokens'])
            if is_first:
                print(f"\nFull prompt for {prompt_key} first example:")
                print(f"System: {system_prompt}")
                print(f"User: {result['question']} + boxed instruction")
                is_first = False

    # Read final complete results (including previously completed + newly completed)
    with open(results_filepath, 'r') as f:
        final_data = json.load(f)

    # Calculate final accuracy and avg_tokens (based on all cases)
    all_cases_accuracies = [1 if case['correct'] else 0 for case in final_data['cases']]
    all_cases_tokens = [case['total_tokens'] for case in final_data['cases']]

    avg_accuracy = mean(all_cases_accuracies) * 100 if all_cases_accuracies else 0
    avg_tokens = mean(all_cases_tokens) if all_cases_tokens else 0

    results[model_name][prompt_key] = {'accuracy': avg_accuracy, 'avg_tokens': avg_tokens}
    print(f"{model_name} with {prompt_key}: Accuracy {avg_accuracy:.2f}%, Avg Tokens {avg_tokens:.2f}")

    # Update summary to JSON file
    final_data['summary'] = {
        'accuracy': avg_accuracy,
        'avg_tokens': avg_tokens
    }
    with open(results_filepath, 'w') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    all_accuracies.append(avg_accuracy)
    all_token_usages.extend(all_cases_tokens)

# Generate total summary JSON
total_accuracy = mean(all_accuracies) if all_accuracies else 0
total_tokens = sum(all_token_usages) if all_token_usages else 0

summary_data = {
    'total_accuracy': total_accuracy,
    'total_tokens': total_tokens
}
with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
    json.dump(summary_data, f, indent=4, ensure_ascii=False)

# Output results
print("Test Results:")
for model, prompt_results in results.items():
    print(f"\nModel: {model}")
    for prompt, metrics in prompt_results.items():
        print(f"  Prompt: {prompt} - Accuracy: {metrics['accuracy']:.2f}%, Avg Tokens: {metrics['avg_tokens']:.2f}")
        