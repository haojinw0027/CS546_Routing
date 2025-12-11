#!/usr/bin/env python3
"""
Multi-benchmark inference with triple model approach using vLLM:
Supports TruthfulQA, MMLU-STEM, and StrategyQA datasets.

1. Base model generates token with confidence
2. If confidence >= threshold, use base model token
3. If confidence < threshold, judge model (vLLM) decides:
   - Use base model token
   - Use expert model token  
   - Generate its own token
"""

import json
import os
import argparse
import time
import torch
from datasets import load_dataset, load_from_disk
import requests
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Dict, Any
from utils import apply_instruct_template, MODEL_NAME_DICT, INSTRUCT_MODEL_LIST
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml

# Random seed for reproducibility (identical to single_inference.py)
RANDOM_SEED = 609
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def get_dataset_arc_challenge(split='test',
                        num_sample=None,
                        input_key='question',
                        output_key='answerKey',
                        **kwargs):
    """Load ARC Challenge dataset from TIGER-Lab/ARC-Challenge"""
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    dataset = arc_challenge[split]
    
    input_data = []
    output_data = []
    for example in dataset:
        choices = example["choices"]
        label = choices["label"]
        text = choices["text"]
        choice_str = ''
        for i in range(len(label)):
            choice_str += f"{label[i]}. {text[i]}\n"
        question = example[input_key] + "\n" + choice_str
        input_data.append({
            "id": example["id"],
            "input": question
        })
        output_data.append(example[output_key])
    if num_sample is not None and num_sample < len(input_data):
        input_data = input_data[:num_sample]
        output_data = output_data[:num_sample]
    return input_data, output_data, input_key, output_key

def get_dataset_mmlu_pro(split='test',
                        num_sample=None,
                        input_key='question',
                        output_key='answer',
                        subject=None,
                        few_shot='yes',
                        **kwargs):
    """Load MMLU-Pro dataset from TIGER-Lab/MMLU-Pro"""
    mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro")
    validation_dataset = mmlu_pro['validation']  # Load validation set for few-shot examples
    
    if split == 'test':
        dataset = load_from_disk('data/mmlu_pro_test')
    else:
        dataset = load_from_disk('data/mmlu_pro_train')
    # fix random seed for reproducibility
    if num_sample is not None and num_sample < len(dataset):
        dataset = dataset.select(range(num_sample))
    input_data = []
    output_data = []
    for example in dataset:
        current_example = {
            "question": example['question'],
            "options": [opt for opt in example['options'] if opt != "N/A"],  # Filter out N/A options
            "answer": example['answer'],
            "category": example.get('category', 'unknown')
        }
        prompt = current_example['question'] + "\n"
        for i, choice in enumerate(current_example['options']):
            prompt += f"{chr(ord('A') + i)}. {choice}\n"
        
        input_data.append({
            "context": "",
            "input": prompt,
            "category": current_example["category"],
            "original_question": current_example["question"],
            "options": current_example["options"]
        })
        
        # Answer is the letter (A, B, C, D, etc.)
        answer_idx = example['answer']
        if isinstance(answer_idx, str):
            # If answer is already a letter, use it directly
            if len(answer_idx) == 1 and answer_idx.isalpha():
                output_data.append(answer_idx.upper())
            else:
                # Try to convert to int
                try:
                    answer_idx = int(answer_idx)
                    output_data.append(chr(65 + answer_idx))
                except ValueError:
                    output_data.append('A')  # Default fallback
        else:
            output_data.append(chr(65 + int(answer_idx)))
    
    return input_data, output_data, input_key, output_key

def get_dataset_com_hard_intervention(split='test',
                        num_sample=None,
                        input_key='Q',
                        output_key='A',
                        **kwargs):
    dataset = []
    if split == 'test':
        with open('Com2/benckmark/com2/hard.json', 'r') as f:
            dataset = json.load(f)
    else:
        with open('Com2/benckmark/train.json', 'r') as f:
            dataset = json.load(f)
    input_data = []
    output_data = []
    if split == 'test':
        for example in dataset:
            if example['type'] == "intervention":
                input_data.append({
                    "crime": example['crime'],
                    "facts": example['facts'],
                    "question": example['Q'],
                    "options": example['O']
                })
                output_data.append(example['A'])
            else:
                continue
    else:
        for example in dataset:
            if example['type'] == "intervention":
                input_data.append({
                    "question": example['question'],
                    "options": example['options']
                })
                output_data.append(example['answer'])
            else:
                continue
    if num_sample is not None and num_sample < len(input_data):
        input_data = input_data[:num_sample]
        output_data = output_data[:num_sample]
    return input_data, output_data, input_key, output_key


def get_dataset_minerva(split='train',
                        num_sample=None,
                        input_key='problem',
                        output_key='solution',
                        **kwargs):
    """Load Minerva dataset from TIGER-Lab/Minerva"""
    minerva = load_dataset("knoveleng/Minerva-Math")
    dataset = minerva['train']
    np.random.seed(RANDOM_SEED)
    if num_sample is not None and num_sample < len(dataset):
        random_indexes = np.random.choice(len(dataset), num_sample, replace=False)
        dataset = dataset.select(random_indexes)
    
    input_data = []
    output_data = []
    for example in dataset:
        input_data.append({
            "context": "",
            "input": example[input_key]
        })
        output_data.append(example[output_key])
    
    return input_data, output_data, input_key, output_key

def get_dataset_math(split='test',
                        num_sample=None,
                        input_key='problem',
                        output_key='solution',
                        **kwargs):
    """Load MATH dataset from nlile/hendrycks-MATH-benchmark"""
    math = load_dataset("nlile/hendrycks-MATH-benchmark")
    dataset = math[split]
    
    # fix random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    if num_sample is not None and num_sample < len(dataset):
        random_indexes = np.random.choice(len(dataset), num_sample, replace=False)
        dataset = dataset.select(random_indexes)
    
    input_data = []
    output_data = []
    for example in dataset:
        input_data.append({
            "context": "",
            "input": example[input_key]
        })
        output_data.append(example[output_key])
    
    return input_data, output_data, input_key, output_key

def get_dataset_supergpqa(split='test',
                         num_sample=None,
                         input_key='question',
                         output_key='answer',
                         subject=None,
                         **kwargs):
    """Load SuperGPQA dataset from m-a-p/SuperGPQA"""
    supergpqa = load_dataset("m-a-p/SuperGPQA")
    dataset = supergpqa['train']

    # Filter by subject if specified (unless subject is "mixed")
    if subject is not None and subject != "mixed":
        print(f"Filtering SuperGPQA dataset for subject: {subject}")

        # Check if subject exists in the dataset
        available_subjects = set(example.get('field', 'unknown') for example in dataset)
        if subject not in available_subjects:
            print(f"Available subjects: {sorted(available_subjects)}")
            raise ValueError(f"Subject '{subject}' not found in SuperGPQA dataset. Available subjects: {sorted(available_subjects)}")

        # Filter dataset by subject
        filtered_data = []
        for example in dataset:
            if example.get('field') == subject:
                filtered_data.append(example)

        print(f"Found {len(filtered_data)} questions for subject '{subject}'")
        dataset = filtered_data
    else:
        # If subject is None or "mixed", use all questions
        if subject == "mixed":
            print(f"Using mixed subjects - randomly sampling from all subjects in SuperGPQA dataset")
        dataset = list(dataset)
    
    np.random.seed(RANDOM_SEED)
    # Sample based on split: test -> from end, train -> from beginning
    if num_sample is not None and num_sample < len(dataset):
        if split == 'test':
            # For test split, select from the end
            dataset = dataset[-num_sample:]
            print(f"Selected last {num_sample} samples from test split")
        else:
            # For train split (or other splits), select from the beginning
            dataset = dataset[:num_sample]
            print(f"Selected first {num_sample} samples from {split} split")
    
    # fix random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    if num_sample is not None and num_sample < len(dataset):
        if split == 'test':
            # For test split, select from the end
            dataset = dataset[-num_sample:]
            print(f"Selected last {num_sample} samples from test split")
        else:
            # For train split (or other splits), select from the beginning
            dataset = dataset[:num_sample]
            print(f"Selected first {num_sample} samples from {split} split")

    
    input_data = []
    output_data = []
    for example in dataset:
        # Format question with multiple choice options
        question = example.get('question', '')
        options = example.get('options', [])
        
        # Build formatted question with options
        formatted_question = f"{question}\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, etc.
            formatted_question += f"{letter}) {option}\n"
        
        input_data.append({
            "context": "",
            "input": formatted_question.strip(),
            "field": example.get('field', 'unknown'),
            "discipline": example.get('discipline', 'unknown'),
            "subfield": example.get('subfield', 'unknown')
        })
        
        # Get correct answer - handle both string and integer formats
        answer = example.get('answer_letter', 0)
        output_data.append(answer)
    
    return input_data, output_data, input_key, output_key

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run TruthfulQA, HaluEval, or MMLU-Pro inference with triple model approach using vLLM")
    # Dataset selection
    parser.add_argument("--benchmark", type=str, choices=["mmlu_pro", "MATH", "supergpqa", "minerva", "com_hard_intervention", "arc_challenge"], default="mmlu_pro",
                        help="Benchmark to evaluate (truthfulqa, halueval, mmlu_pro, MATH, lastletter, justlogic, aime2025, supergpqa, or minerva)")
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name/path (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--expert_model", type=str, required=True,
                        help="Expert model name/path (e.g., meta-llama/Llama-3.2-3B)")
    parser.add_argument("--confidence_threshold", type=float, default=0.40,
                        help="Confidence threshold for activating judge model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--base_port", type=int, default=8000,
                        help="Port for base model vLLM serve endpoint")
    parser.add_argument("--expert_port", type=int, default=8001,
                        help="Port for expert model vLLM serve endpoint")
    parser.add_argument("--hf_token", type=str, default='',
                        help="HuggingFace token for accessing restricted models")
    parser.add_argument("--consult_model", type=str,default = 'base')
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to process")
    parser.add_argument("--split", type=str, default='test',
                        help="Split to process")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k tokens to consider for entropy calculation")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject to filter for MMLU-Pro and SuperGPQA datasets (e.g., 'math', 'physics' for MMLU-Pro; 'Clinical Medicine', 'Biology' for SuperGPQA)")
    parser.add_argument("--few-shot", type=str, default='yes', choices=['yes', 'no'],
                        help="Whether to use few-shot prompting")
    parser.add_argument("--complete_tokens", type=int, default=16,
                        help="Number of tokens to complete")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of parallel workers for processing (default: 1, sequential processing)")
    return parser.parse_args()

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load existing results from a JSON file, or initialize if not present."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return []

def save_results(results: List[Dict[str, Any]], file_path: str):
    """Save the updated results to the JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

def formulate_prompt_com_hard_intervention(crime, facts, question, options, model_name):
    """
    Formulate prompt for Com Hard Intervention questions using 5-shot examples from com_hard_intervention.yaml.
    """
    if crime is not None and facts is not None:
        with open('./config/com_hard_intervention.yaml', 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
        prompt_template = yaml_content['prompt_format'][0]
        full_prompt = prompt_template.format(crime=crime, facts=facts, question=question, options=options)
    else:
        with open('./config/com_hard_intervention_train.yaml', 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
        prompt_template = yaml_content['prompt_format'][0]
        full_prompt = prompt_template.format(question=question, options=options)
    return full_prompt

def formulate_prompt_arc_challenge(question, model_name):
    """
    Formulate prompt for ARC Challenge questions using 5-shot examples from arc_challenge.yaml.
    """
    system_prompt = "Answer the following multiple choice question. There is only one correct answer. Give a detailed reasoning process before giving your answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D.\n"
    instruct_prompt = f"Question:\n{question}"
    response_prompt = "Answer:\n"
    return apply_instruct_template(
        model_name=model_name,
        system_prompt=system_prompt,
        instruct_prompt=instruct_prompt,
        response_prompt=response_prompt
    )

def formulate_prompt_mmlu_pro(question, model_name):
    """
    Formulate prompt for MMLU-Pro questions using 5-shot CoT format.
    The question parameter already contains the full CoT prompt with few-shot examples.
    """
    # For MMLU-Pro with CoT, the question already contains the complete prompt
    # including few-shot examples and the target question, so we return it directly
    system_prompt = "Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I or J.\n"
    instruct_prompt = f"Question:\n{question}"
    response_prompt = "Answer:\nLet's think step by step. "
    return apply_instruct_template(
        model_name=model_name,
        system_prompt=system_prompt,
        instruct_prompt=instruct_prompt,
        response_prompt=response_prompt
    )
    return question

def formulate_prompt_minerva(question, model_name):
    """
    Formulate prompt for Minerva questions using 5-shot CoT format.
    """
    with open('./config/minerva.yaml', 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    prompt_template = yaml_content['prompt_format'][0]
    full_prompt = prompt_template.format(question)
    system_prompt = "Think step by step."
    return apply_instruct_template(
        model_name=model_name,
        system_prompt=system_prompt,
        instruct_prompt=full_prompt,
        response_prompt=""
    )

def formulate_prompt_math(question: str, model_name: str = "") -> str:
    """
    Formulate prompt for MATH questions using few-shot examples from train split.
    Uses 4 examples from the training set as demonstrations.
    """
    # Load few-shot examples from train split (cached for efficiency)
    if not hasattr(formulate_prompt_math, '_few_shot_examples'):
        try:
            math_train = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
            # Select first 4 examples for consistency
            formulate_prompt_math._few_shot_examples = list(math_train.select(range(4)))
        except Exception as e:
            print(f"Warning: Could not load MATH train examples: {e}")
            formulate_prompt_math._few_shot_examples = []
    
    # Build few-shot prompt
    few_shot_text = ""
    for i, example in enumerate(formulate_prompt_math._few_shot_examples):
        few_shot_text += f"Problem: {example['problem']}\n"
        few_shot_text += f"Solution: {example['solution']}\n\n"
    
    # Add current question
    full_prompt = few_shot_text + f"Problem: {question}\nSolution: "
    
    if model_name:
        system_prompt = "Think step by step."
        return apply_instruct_template(
            model_name=model_name,
            system_prompt=system_prompt,
            instruct_prompt=full_prompt,
            response_prompt="",
            add_bos=True
        )
    else:
        # Fallback to simple formatting if no model name provided
        return full_prompt

def formulate_prompt_supergpqa(question: str, model_name: str = "") -> str:
    """
    Format question for SuperGPQA using 5-shot examples from super_gpqa.yaml.
    """
    import yaml
    
    # Load prompt template from super_gpqa.yaml
    try:
        with open('./config/super_gpqa.yaml', 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
        
        # Extract the prompt template
        prompt_template = yaml_content['prompt_format'][0]
        
        # Format with the current question
        full_prompt = prompt_template.format(question)
    except FileNotFoundError:
        # Fallback to inline template if yaml file not found
        print("Warning: super_gpqa.yaml not found, using fallback prompt")
        full_prompt = f"""Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

Question: 
{question}

Answer: Let's think step by step. """
    
    if model_name:
        system_prompt = "You are an expert in science who answers multiple choice questions step by step."
        return apply_instruct_template(
            model_name=model_name,
            system_prompt=system_prompt,
            instruct_prompt=full_prompt,
            response_prompt="",
            add_bos=True
        )
    else:
        # Fallback to simple formatting if no model name provided
        return full_prompt

class TripleModelGenerator:
    def __init__(self, base_model_name: str, expert_model_name: str, 
                 hf_token: str, base_port: int = 8000, expert_port: int = 8001, consult_model: str = 'base', complete_tokens: int = 16):
        
        self.base_model_name = base_model_name
        self.expert_model_name = expert_model_name
        self.base_port = base_port
        self.expert_port = expert_port
        self.confidence_threshold = 0.40
        self.consult_model = consult_model
        self.complete_tokens = complete_tokens
        # Test connections to vLLM serve endpoints
        self._test_connections()
        print(f"Using base model at port {base_port}")
        print(f"Using expert model at port {expert_port}")
        
        # Load tokenizers
        print("Loading tokenizers...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, token=hf_token, trust_remote_code=True
        )
        self.expert_tokenizer = AutoTokenizer.from_pretrained(
            expert_model_name, token=hf_token, trust_remote_code=True
        )
        
        # Set pad tokens
        for tokenizer in [self.base_tokenizer, self.expert_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        print("Tokenizers loaded successfully!")
        print("Triple model generator initialized!")
    
    def _test_connections(self):
        """Test connections to vLLM serve endpoints and verify model names"""
        print(f"Testing connection to base model endpoint at port {self.base_port}...")
        try:
            test_url = f"http://localhost:{self.base_port}/v1/models"
            response = requests.get(test_url, timeout=10)
            response.raise_for_status()
            models_info = response.json()
            
            # Extract deployed model name
            deployed_model = None
            if 'data' in models_info and len(models_info['data']) > 0:
                deployed_model = models_info['data'][0]['id']
            
            # Verify model name matches
            if deployed_model:
                # Extract model name from full path if needed
                deployed_model_name = deployed_model.split('/')[-1] if '/' in deployed_model else deployed_model
                expected_model_name = self.base_model_name.split('/')[-1] if '/' in self.base_model_name else self.base_model_name
                
                if deployed_model_name.lower() != expected_model_name.lower():
                    raise Exception(f"Model name mismatch at port {self.base_port}: expected '{self.base_model_name}' but found '{deployed_model}'")
                
                print(f"✓ Base model endpoint connected successfully (verified: {deployed_model})")
            else:
                print(f"✓ Base model endpoint connected successfully (model name verification skipped - no model info)")
                
        except Exception as e:
            raise Exception(f"Failed to connect to base model endpoint at port {self.base_port}: {e}")
        
        print(f"Testing connection to expert model endpoint at port {self.expert_port}...")
        try:
            test_url = f"http://localhost:{self.expert_port}/v1/models"
            response = requests.get(test_url, timeout=10)
            response.raise_for_status()
            models_info = response.json()
            
            # Extract deployed model name
            deployed_model = None
            if 'data' in models_info and len(models_info['data']) > 0:
                deployed_model = models_info['data'][0]['id']
            
            # Verify model name matches
            if deployed_model:
                # Extract model name from full path if needed
                deployed_model_name = deployed_model.split('/')[-1] if '/' in deployed_model else deployed_model
                expected_model_name = self.expert_model_name.split('/')[-1] if '/' in self.expert_model_name else self.expert_model_name
                
                if deployed_model_name.lower() != expected_model_name.lower():
                    raise Exception(f"Model name mismatch at port {self.expert_port}: expected '{self.expert_model_name}' but found '{deployed_model}'")
                
                print(f"✓ Expert model endpoint connected successfully (verified: {deployed_model})")
            else:
                print(f"✓ Expert model endpoint connected successfully (model name verification skipped - no model info)")
                
        except Exception as e:
            raise Exception(f"Failed to connect to expert model endpoint at port {self.expert_port}: {e}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for activating judge model."""
        self.confidence_threshold = threshold
    
    def generate_sequence(self, port: int, prompt: str, max_tokens: int = 16) -> str:
        """Generate a sequence of tokens using vLLM serve endpoint"""
        url = f"http://localhost:{port}/v1/completions"

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["choices"] and result["choices"][0]["text"]:
                return result["choices"][0]["text"]
            else:
                return ""

        except Exception as e:
            print(f"Error calling vLLM serve endpoint: {e}")
            return ""

    def get_next_token_with_logits(self, port: int, prompt: str, tokenizer, k=5) -> Tuple[int, float]:
        """Get next token and its probability using vLLM serve endpoint"""
        url = f"http://localhost:{port}/v1/completions"

        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": k
        }
        response = requests.post(url, json=payload, timeout=30).json()
        top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
        return top_logprobs
    
    def ensemble_decoding(self, base_logprobs: dict, expert_logprobs: dict) -> str:
        merged_probs = {}
        all_tokens = set(base_logprobs.keys()) | set(expert_logprobs.keys())
        for tok in all_tokens:
            lp_base = base_logprobs.get(tok, None)
            lp_expert = expert_logprobs.get(tok, None)
            p_base = math.exp(lp_base) if lp_base is not None else 0.0
            p_expert = math.exp(lp_expert) if lp_expert is not None else 0.0
            merged_probs[tok] = 0.5 * (p_base + p_expert)
        return max(merged_probs, key=merged_probs.get)
    
    def generate_with_triple_model(self, prompt: str, max_new_tokens: int = 64, benchmark: str = "truthfulqa") -> Dict[str, Any]:
        """Generate text using triple model approach."""
        generated_tokens = []
        token_choices = []
        base_model_count = 0
        expert_model_count = 0
        judge_decisions = 0
        next_tokens = []
        
        current_prompt = prompt
        original_prompt = prompt
        generated_text = ""
        original_question = prompt.split("Question: ")[1].split("\n")[0] if "Question: " in prompt else ""
        for _ in range(max_new_tokens):
            base_distribution = self.get_next_token_with_logits(self.base_port, current_prompt, self.base_tokenizer)
            expert_distribution = self.get_next_token_with_logits(self.expert_port, current_prompt, self.expert_tokenizer)
            next_token = self.ensemble_decoding(base_distribution, expert_distribution)
            token_choices.append({
                "selected_token_text": next_token,
                "base_token_text": base_distribution,
                "expert_token_text": expert_distribution,
                "model_used": "",
                "confidence": 0.0,
                "base_confidence": 0.0,
                "expert_confidence": 0.0,
                "Prev_entropy": 0.0,
                "Next_entropy": 0.0,
                "base_consult": "",
                "expert_consult": "",
            })
            if next_token == "":
                break
            if len(next_tokens) < 3:
                next_tokens.append(next_token)
            else:
                next_tokens.pop(0)
                next_tokens.append(next_token)
            if len(next_tokens) == 3 and next_tokens[0] == next_tokens[1] == next_tokens[2]:
                break
            # Update prompt and generated text
            generated_text += next_token
            current_prompt = prompt + generated_text
            
            # Check for end conditions
            if any(stop.lower() in generated_text.lower() for stop in ['\n\nPremises:', '\n\nProblem:', '\n\nQuestion:', '<|endoftext|>', '</s>']):
                break
        newly_generated_text = current_prompt[len(original_prompt):]
        newly_generated_length = len(self.base_tokenizer.encode(newly_generated_text, add_special_tokens=False))
        return {
            "generated_text": generated_text.strip(),
            "token_choices": token_choices,
            "base_model_tokens": base_model_count,
            "total_tokens": newly_generated_length,
            "expert_model_tokens": expert_model_count,
            "judge_decisions": judge_decisions,
            "base_usage_ratio": base_model_count / len(token_choices) if len(token_choices) else 0,
            "expert_usage_ratio": expert_model_count / len(token_choices) if len(token_choices) else 0,
        }

def process_single_item(item: Dict[str, Any], generator: TripleModelGenerator, args, benchmark: str) -> Dict[str, Any]:
    """
    Process a single item through the triple model generator.

    Args:
        item: The data item to process
        generator: The TripleModelGenerator instance
        args: Command line arguments
        benchmark: The benchmark name

    Returns:
        Dictionary containing the processing result
    """
    # Generate prompt based on benchmark
    if benchmark == "mmlu_pro":
        prompt = formulate_prompt_mmlu_pro(item["question"], args.base_model)
        item_id = item["id"]
        question = item.get("original_question", item["question"])
    elif benchmark == "MATH":
        prompt = formulate_prompt_math(item["question"], args.base_model)
        item_id = item["id"]
        question = item["question"]
    elif benchmark == "supergpqa":
        prompt = formulate_prompt_supergpqa(item["question"], args.base_model)
        item_id = item["id"]
        question = item["question"]
    elif benchmark == "minerva":
        prompt = formulate_prompt_minerva(item["question"], args.base_model)
        item_id = item["id"]
        question = item["question"]
    elif benchmark == "com_hard_intervention":
        if args.split == 'test':
            prompt = formulate_prompt_com_hard_intervention(item["crime"], item["facts"], item["question"], item["options"], args.base_model)
        else:
            prompt = formulate_prompt_com_hard_intervention(None, None, item["question"], item["options"], args.base_model)
        item_id = item["id"]
        question = item["question"]
    elif benchmark == "arc_challenge":
        prompt = formulate_prompt_arc_challenge(item["question"], args.base_model)
        item_id = item["id"]
        question = item["question"]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Generate response using triple model approach
    generation_result = generator.generate_with_triple_model(prompt, args.max_new_tokens, benchmark)

    # Build result dictionary
    result = {
        "id": item_id,
        "question": question,
        "prompt": prompt,
        "answer": generation_result["generated_text"],
        "base_model_tokens": generation_result["base_model_tokens"],
        "expert_model_tokens": generation_result["expert_model_tokens"],
        "total_tokens": generation_result["total_tokens"],
        "judge_decisions": generation_result["judge_decisions"],
        "base_usage_ratio": generation_result["base_usage_ratio"],
        "expert_usage_ratio": generation_result["expert_usage_ratio"],
        "token_choices": generation_result["token_choices"],
        "base_model": args.base_model,
        "expert_model": args.expert_model,
        "confidence_threshold": args.confidence_threshold,
        "benchmark": benchmark,
        "base_port": args.base_port,
        "expert_port": args.expert_port,
        "correct_answer": item['gold_answer'] if benchmark == "tinymmlu" else "",
        "choices": item['choices'] if benchmark == "tinymmlu" else "",
        "subject": item['subject'] if benchmark == "tinymmlu" else ""
    }

    # Add benchmark-specific fields
    if benchmark == "mmlu_pro":
        result["correct_answer"] = item["correct_answer"]
        result["category"] = item.get("category", "unknown")
        result["options"] = item.get("options", [])
    elif benchmark == "MATH":
        result["correct_answer"] = item["correct_answer"]
    elif benchmark == "supergpqa":
        result["correct_answer"] = item["correct_answer"]
        result["field"] = item.get("field", "unknown")
        result["discipline"] = item.get("discipline", "unknown")
        result["subfield"] = item.get("subfield", "unknown")
    elif benchmark == "minerva":
        result["correct_answer"] = item["correct_answer"]
    elif benchmark == "com_hard_intervention":
        result["correct_answer"] = item["correct_answer"]
    elif benchmark == "arc_challenge":
        result["correct_answer"] = item["correct_answer"]
    return result

def main():
    args = parse_arguments()
    if args.base_model not in MODEL_NAME_DICT.keys():
        raise ValueError(f"Base model {args.base_model} not found in MODEL_NAME_DICT")
    if args.expert_model not in MODEL_NAME_DICT.keys():
        raise ValueError(f"Expert model {args.expert_model} not found in MODEL_NAME_DICT")
    
    # Set HuggingFace token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    
    print(f"Base model endpoint: http://localhost:{args.base_port}")
    print(f"Expert model endpoint: http://localhost:{args.expert_port}")
    
    if args.benchmark == "mmlu_pro":
        print("Loading MMLU-Pro dataset...")
        input_data, output_data, input_key, output_key = get_dataset_mmlu_pro(
            split=args.split,
            num_sample=args.num_samples,
            subject=args.subject,
            few_shot=args.few_shot
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out,
                'category': inp.get('category', 'unknown'),
                'original_question': inp.get('original_question', ''),
                'options': inp.get('options', [])
            })
    elif args.benchmark == "com_hard_intervention":
        print("Loading Com Hard Intervention dataset...")
        input_data, output_data, input_key, output_key = get_dataset_com_hard_intervention(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            if args.split == 'test':
                test_set.append({
                    'id': i,
                    'crime': inp['crime'],
                    'facts': inp['facts'],
                    'question': inp['question'],
                    'options': inp['options'],
                    'correct_answer': out
                })
            else:
                test_set.append({
                    'id': i,
                    'question': inp['question'],
                    'options': inp['options'],
                    'correct_answer': out
                })
    elif args.benchmark == "MATH":
        print("Loading MATH dataset...")
        input_data, output_data, input_key, output_key = get_dataset_math(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out
            })
    elif args.benchmark == "supergpqa":
        print("Loading SuperGPQA dataset...")
        input_data, output_data, input_key, output_key = get_dataset_supergpqa(
            split=args.split,
            num_sample=args.num_samples,
            subject=args.subject
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out,
                'field': inp.get('field', 'unknown'),
                'discipline': inp.get('discipline', 'unknown'),
                'subfield': inp.get('subfield', 'unknown')
            })
    elif args.benchmark == "minerva":
        print("Loading Minerva dataset...")
        input_data, output_data, input_key, output_key = get_dataset_minerva(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out
            })
    elif args.benchmark == "arc_challenge":
        print("Loading ARC Challenge dataset...")
        input_data, output_data, input_key, output_key = get_dataset_arc_challenge(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out
            })
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # Initialize triple model generator
    generator = TripleModelGenerator(
        base_model_name=args.base_model,
        expert_model_name=args.expert_model,
        hf_token=args.hf_token,
        base_port=args.base_port,
        expert_port=args.expert_port,
        consult_model=args.consult_model
    )
    generator.set_confidence_threshold(args.confidence_threshold)
    
    # Create output file path
    # Handle subject-specific output directory for MMLU-Pro and SuperGPQA
    if (args.benchmark == 'mmlu_pro' or args.benchmark == 'supergpqa') and args.subject:
        dataset_dir = f'{args.benchmark}_{args.subject}_results'
    else:
        dataset_dir = f'{args.benchmark}_results'
    
    if args.consult_model == 'base':
        if args.few_shot == 'yes':
            output_file = f'./result/{dataset_dir}/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/ensemble_official.json'
        else:
            output_file = f'./result/{dataset_dir}/zero_shot/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/ensemble_official.json'
    if args.consult_model == 'expert':
        if args.few_shot == 'yes':
            output_file = f'./result/{dataset_dir}/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/ensemble_official.json'
        else:
            output_file = f'./result/{dataset_dir}/zero_shot/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/ensemble_official.json'
    
    print(f"Output file: {output_file}")
    print(f"Configuration:")
    print(f"  - Benchmark: {args.benchmark}")
    print(f"  - Base model: {args.base_model}")
    print(f"  - Expert model: {args.expert_model}")
    print(f"  - Confidence threshold: {args.confidence_threshold}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - Base model port: {args.base_port}")
    print(f"  - Expert model port: {args.expert_port}")
    
    # Load existing results
    results = load_results(output_file)
    processed_ids = {result['id'] for result in results}
    
    # Prepare unprocessed items based on benchmark
    unprocessed_items = []
    if args.benchmark == "mmlu_pro":
        for i, item in enumerate(test_set):
            # Create unique ID for mmlu_pro
            item_id = f"mmlu_pro###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "MATH":
        for i, item in enumerate(test_set):
            # Create unique ID for MATH
            item_id = f"MATH###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "supergpqa":
        for i, item in enumerate(test_set):
            # Create unique ID for supergpqa
            item_id = f"supergpqa###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "minerva":
        for i, item in enumerate(test_set):
            # Create unique ID for minerva
            item_id = f"minerva###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "com_hard_intervention":
        for i, item in enumerate(test_set):
            # Create unique ID for com_hard_intervention
            item_id = f"com_hard_intervention###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "arc_challenge":
        for i, item in enumerate(test_set):
            # Create unique ID for arc_challenge
            item_id = item["id"]
            if item_id not in processed_ids:
                unprocessed_items.append(item)
    if not unprocessed_items:
        print("All items have been processed already!")
        return
    
    print(f"Processing {len(unprocessed_items)} unprocessed items...")
    print(f"Using {args.max_workers} parallel worker(s)")

    # Thread-safe lock for saving results
    save_lock = threading.Lock()

    # Process items in parallel or sequentially based on max_workers
    if args.max_workers == 1:
        # Sequential processing (original behavior)
        for item in tqdm(unprocessed_items, desc="Processing questions"):
            result = process_single_item(item, generator, args, args.benchmark)
            results.append(result)
            save_results(results, output_file)
    else:
        # Parallel processing using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_single_item, item, generator, args, args.benchmark): item
                for item in unprocessed_items
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_item), total=len(unprocessed_items), desc="Processing questions"):
                try:
                    result = future.result()
                    # Thread-safe append and save
                    with save_lock:
                        results.append(result)
                        save_results(results, output_file)
                except Exception as e:
                    item = future_to_item[future]
                    item_id = item.get('id', item.get('Source', 'unknown'))
                    print(f"\nError processing item {item_id}: {e}")
                    # Continue processing other items

    # Calculate statistics
    total_base_tokens = sum(result["base_model_tokens"] for result in results)
    total_expert_tokens = sum(result["expert_model_tokens"] for result in results)
    total_tokens = total_base_tokens + total_expert_tokens
    
    print(f"\nInference completed! Results saved to {output_file}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Total responses generated: {len(results)}")
    
    print(f"\nToken usage statistics:")
    print(f"  - Base model tokens: {total_base_tokens} ({total_base_tokens/total_tokens*100:.1f}%)")
    print(f"  - Expert model tokens: {total_expert_tokens} ({total_expert_tokens/total_tokens*100:.1f}%)")
    print(f"  - Total tokens: {total_tokens}")
    
    # Calculate confidence statistics
    all_base_confidences = []
    all_expert_confidences = []
    
    for result in results:
        for token_choice in result.get("token_choices", []):
            all_base_confidences.append(token_choice.get("base_confidence", 0))
            all_expert_confidences.append(token_choice.get("expert_confidence", 0))
    
    if all_base_confidences:
        print(f"\nConfidence statistics:")
        print(f"  - Average base confidence: {sum(all_base_confidences)/len(all_base_confidences):.3f}")
        if all_expert_confidences:
            print(f"  - Average expert confidence (when consulted): {sum(all_expert_confidences)/len(all_expert_confidences):.3f}")

if __name__ == "__main__":
    main()
