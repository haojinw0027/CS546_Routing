#!/usr/bin/env python3
"""
Multi-Benchmark Test Script with Nudge Logic
Supports MMLU-STEM, TruthfulQA, StrategyQA, and JustEval benchmarks
Features: fp8 quantization, CUDA device specification, and Llama prompt formatting
"""

import json
import os
import argparse
import torch
from datasets import load_dataset, load_from_disk
import requests
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk
import random
from typing import Tuple, List, Dict, Any
import time
from utils import MODEL_NAME_DICT
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import completion_with_nudging

# Random seed for reproducibility (identical to main.py)
RANDOM_SEED = 609
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class VLLMClientWrapper:
    """Wrapper to make vLLM serve endpoint compatible with OpenAI API client interface"""

    def __init__(self, port: int, model_name: str):
        self.port = port
        self.model_name = model_name
        self.completions = self

    def create(self, model: str, prompt: str, max_tokens: int, temperature: float,
               logprobs: int = None, top_p: float = 1.0, **kwargs):
        """Create completion compatible with OpenAI API format"""
        url = f"http://localhost:{self.port}/v1/completions"

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if logprobs is not None:
            payload["logprobs"] = logprobs

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            # Convert to OpenAI-compatible format
            class Choice:
                def __init__(self, choice_data):
                    self.text = choice_data.get("text", "")
                    self.finish_reason = choice_data.get("finish_reason", "")

                    # Handle logprobs
                    if "logprobs" in choice_data and choice_data["logprobs"]:
                        logprobs_data = choice_data["logprobs"]
                        self.logprobs = type('obj', (object,), {
                            'tokens': logprobs_data.get("tokens", []),
                            'token_logprobs': logprobs_data.get("token_logprobs", []),
                            'top_logprobs': logprobs_data.get("top_logprobs", []),
                        })()
                    else:
                        self.logprobs = None

            class Response:
                def __init__(self, result_data):
                    self.choices = [Choice(c) for c in result_data.get("choices", [])]

            return Response(result)

        except Exception as e:
            print(f"Error calling vLLM endpoint at port {self.port}: {e}")
            # Return empty response
            class EmptyChoice:
                def __init__(self):
                    self.text = ""
                    self.finish_reason = "error"
                    self.logprobs = None

            class EmptyResponse:
                def __init__(self):
                    self.choices = [EmptyChoice()]

            return EmptyResponse()


def apply_instruct_template(model_name, system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """Apply instruction template based on model type (from utils.py)"""
    model_name = model_name.lower()
    if not system_prompt:
        return instruct_prompt
    return f"{system_prompt}\n{instruct_prompt}\n{response_prompt}"

def formulate_prompt_com_hard_intervention(crime, facts, question, options, model_name):
    """
    Formulate prompt for Com Hard Intervention questions using 5-shot examples from com_hard_intervention.yaml.
    """
    with open('./config/com_hard_intervention.yaml', 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    prompt_template = yaml_content['prompt_format'][0]
    full_prompt = prompt_template.format(crime=crime, facts=facts, question=question, options=options)
    return full_prompt

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

def format_question_math(question: str, use_instruct_template: bool = True, model_name: str = "") -> str:
    """
    Format question for MATH with few-shot examples from train split.
    Uses 4 examples from the training set as demonstrations (matching single_inference.py).
    """
    
    # Load few-shot examples from train split (cached for efficiency)
    if not hasattr(format_question_math, '_few_shot_examples'):
        try:
            math_train = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
            # Select first 4 examples for consistency
            format_question_math._few_shot_examples = list(math_train.select(range(4)))
        except Exception as e:
            print(f"Warning: Could not load MATH train examples: {e}")
            format_question_math._few_shot_examples = []
    
    # Build few-shot prompt
    few_shot_text = ""
    for i, example in enumerate(format_question_math._few_shot_examples):
        few_shot_text += f"Problem: {example['problem']}\n"
        few_shot_text += f"Solution: {example['solution']}\n\n"
    
    # Add current question
    full_prompt = few_shot_text + f"Problem: {question}\nSolution: "
    
    if use_instruct_template and model_name:
        system_prompt = "Think step by step."
        return apply_instruct_template(
            model_name=model_name,
            system_prompt=system_prompt,
            instruct_prompt=full_prompt,
            response_prompt=""
        )
    else:
        # Fallback to simple formatting if no model name provided
        return full_prompt

def format_question_supergpqa(question: str, use_instruct_template: bool = True, model_name: str = "") -> str:
    """Format question for SuperGPQA using 5-shot examples from super_gpqa.yaml"""
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
    
    if use_instruct_template and model_name:
        system_prompt = "You are an expert in science who answers multiple choice questions step by step."
        return apply_instruct_template(
            model_name=model_name,
            system_prompt=system_prompt,
            instruct_prompt=full_prompt,
            response_prompt=""
        )
    else:
        return full_prompt

class NudgeMMSLGenerator:
    """MMLU-STEM Generator with Nudge Logic, fp8 support, and flexible CUDA device management"""

    def __init__(self, base_model_name: str, expert_model_name: str, hf_token: str,
                 base_port: int = 8000, expert_port: int = 8001,
                 confidence_threshold: float = 0.3):

        self.base_model_name = base_model_name
        self.expert_model_name = expert_model_name
        self.confidence_threshold = confidence_threshold
        self.base_port = base_port
        self.expert_port = expert_port

        print(f"Initializing Nudge MMLU-STEM Generator...")
        print(f"Base model: {base_model_name}")
        print(f"Expert model: {expert_model_name}")
        print(f"Base port: {base_port}")
        print(f"Expert port: {expert_port}")
        print(f"Confidence threshold: {confidence_threshold}")

        # Test connections to vLLM serve endpoints
        self._test_connections()

        # Load tokenizers for token counting
        self._init_tokenizers(hf_token)

        # Create vLLM client wrappers compatible with OpenAI API
        self.client_base = VLLMClientWrapper(port=base_port, model_name=base_model_name)
        self.client_nudging = VLLMClientWrapper(port=expert_port, model_name=expert_model_name)

        print("All connections verified successfully!")
    
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
    
    def _init_tokenizers(self, hf_token: str):
        """Initialize tokenizers for token counting"""
        print(f"Loading tokenizers...")
        
        # Load base tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, token=hf_token, trust_remote_code=True
        )
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Load expert tokenizer
        self.expert_tokenizer = AutoTokenizer.from_pretrained(
            self.expert_model_name, token=hf_token, trust_remote_code=True
        )
        if self.expert_tokenizer.pad_token is None:
            self.expert_tokenizer.pad_token = self.expert_tokenizer.eos_token
        
        print(f"✓ Tokenizers loaded successfully")
    
    def _completion_with_nudging(self, system_prompt_base: str, system_prompt_expert: str,
                               question: str, context: str = "", question_prompt: str = "",
                               answer_start_prompt_base: str = "", answer_start_prompt_expert: str = "",
                               completion_token_num: int = 16, completion_token_num_expert: int = 16,
                               max_token_total: int = 256, max_round: int = 150,
                               print_intermediate_output: bool = False) -> Dict[str, Any]:
        """
        Complete text with nudging logic using nudging.utils.completion_with_nudging
        """
        # Call the completion_with_nudging function from nudging/utils.py
        result = completion_with_nudging(
            base_model=self.base_model_name,
            nudging_model=self.expert_model_name,
            system_prompt_base=system_prompt_base,
            system_prompt_nudging=system_prompt_expert,
            question=question,
            context=context,
            question_prompt=question_prompt,
            answer_start_prompt_base=answer_start_prompt_base,
            answer_start_prompt_nudging=answer_start_prompt_expert,
            completion_token_num=completion_token_num,
            completion_token_num_nudging=completion_token_num_expert,
            max_token_total=max_token_total,
            print_intermediate_output=print_intermediate_output,
            client_base=self.client_base,
            client_nudging=self.client_nudging,
            max_round=max_round,
            nudging_temperature=0.0,
            base_temperature=0.0,
            nudging_method='top_prob',
            top_prob_thres=self.confidence_threshold,
            top_p=0.9,
        )

        # Count tokens for compatibility with existing code
        total_nudging_tokens = 0
        total_base_tokens = 0

        # Estimate token counts from all_nudging_words and all_completions
        if "all_nudging_words" in result:
            for nudging_word in result["all_nudging_words"]:
                total_nudging_tokens += len(self.expert_tokenizer.encode(nudging_word, add_special_tokens=False))

        if "all_completions" in result:
            for completion in result["all_completions"]:
                total_base_tokens += len(self.base_tokenizer.encode(completion, add_special_tokens=False))

        # Adjust base tokens (subtract nudging tokens since all_completions includes both)
        total_base_tokens = max(0, total_base_tokens - total_nudging_tokens)

        # Add additional fields for compatibility
        result.update({
            "system_prompt_expert": system_prompt_expert,
            "full_prefix_expert": result.get("full_prefix_nudging", ""),
            "all_expert_words": result.get("all_nudging_words", []),
            "rounds": len(result.get("all_completions", [])),
            "total_expert_tokens": total_nudging_tokens,
            "total_base_tokens": total_base_tokens,
            "total_generated_tokens": total_nudging_tokens + total_base_tokens,
        })

        return result

def get_dataset_math(split='test',
                    num_sample=None,
                    input_key='problem',
                    output_key='solution',
                    **kwargs):
    """Load MATH dataset from nlile/hendrycks-MATH-benchmark (matching single_inference.py)"""
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

def get_dataset_com_hard_intervention(split='test',
                        num_sample=None,
                        input_key='Q',
                        output_key='A',
                        **kwargs):
    """Load Com Hard Intervention dataset from Com2/benckmark/com2/hard.json"""
    dataset = []
    with open('Com2/benckmark/com2/hard.json', 'r') as f:
        dataset = json.load(f)
    input_data = []
    output_data = []
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
    if num_sample is not None and num_sample < len(input_data):
        if split == 'test':
            input_data = input_data[-num_sample:]
            output_data = output_data[-num_sample:]
        else:
            input_data = input_data[:num_sample]
            output_data = output_data[:num_sample]
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


def get_dataset(dataset_name, 
                split=None, 
                num_sample=None,
                few_shot='yes',
                subject=None,
                **kwargs):
    """Get dataset (supporting justeval, csqa, halueval, math, mmlu_pro, lastletter, justlogic, aime2025, and supergpqa)"""
    if dataset_name == 'MATH':
        dataset = get_dataset_math(split, num_sample, **kwargs)
    elif dataset_name == 'supergpqa':
        # Extract subject from kwargs to avoid duplicate parameter
        subject = kwargs.pop('subject', None)
        dataset = get_dataset_supergpqa(split, num_sample, subject=subject, **kwargs)
    elif dataset_name == 'minerva':
        dataset = get_dataset_minerva(split, num_sample, **kwargs)
    elif dataset_name == 'com_hard_intervention':
        dataset = get_dataset_com_hard_intervention(split, num_sample, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test MMLU-STEM, TruthfulQA, StrategyQA, JustEval, or MATH with Nudge Logic")
    
    # Model configurations
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name/path")
    parser.add_argument("--expert_model", type=str, required=True,
                        help="Expert/aligned model name/path")
    parser.add_argument("--base_port", type=int, default=8000,
                        help="Port for base model vLLM serve endpoint")
    parser.add_argument("--expert_port", type=int, default=8001,
                        help="Port for expert model vLLM serve endpoint")
    parser.add_argument("--hf_token", type=str, default='',
                        help="HuggingFace token")
    
    # Generation settings
    parser.add_argument("--confidence_threshold", type=float, default=0.4,
                        help="Confidence threshold for nudging")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--completion_token_num", type=int, default=16,
                        help="Tokens per completion round")
    
    # Dataset settings
    parser.add_argument("--benchmark", type=str, choices=['MATH', 'supergpqa', 'minerva', 'com_hard_intervention'], default="MATH",
                        help="Benchmark to test (MATH, supergpqa, minerva, or com_hard_intervention)")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Maximum samples per subject/total for MATH")
    parser.add_argument("--print_intermediate", action="store_true",
                        help="Print intermediate generation steps")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Output directory for results")
    parser.add_argument("--force_restart", action="store_true",
                        help="Force restart from beginning, ignoring existing results")
    parser.add_argument("--split", type=str, default='test',
                        help="Split to process")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of parallel workers for processing (default: 1, sequential processing)")
    parser.add_argument("--subject", type=str, default='mixed',
                        help="Subject to process")

    return parser.parse_args()

def load_existing_results(output_file: str) -> Dict[str, Dict]:
    """Load existing results to support resume functionality"""
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        # Use subject + question/instruction as key for uniqueness
                        if 'instruction' in result:
                            # For JustEval
                            key = f"{result['subject']}###{result['instruction']}"
                        else:
                            # For MMLU, TruthfulQA, StrategyQA
                            key = f"{result['subject']}###{result['question']}"
                        existing_results[key] = result
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return existing_results

def append_result_to_file(result: Dict[str, Any], output_file: str):
    """Append a single result to the output file (JSONL format)"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing result to file: {e}")

def process_single_item(generator: NudgeMMSLGenerator, item: Dict[str, Any], existing_results: Dict[str, Dict],
                       benchmark: str, args, output_file: str, save_lock: threading.Lock = None) -> Dict[str, Any]:
    """
    Process a single item through the nudge generator.

    Args:
        generator: The NudgeMMSLGenerator instance
        item: The data item to process
        existing_results: Dictionary of already processed results
        benchmark: The benchmark name
        args: Command line arguments
        output_file: Path to output file
        save_lock: Thread-safe lock for file writing (None for sequential processing)

    Returns:
        Dictionary containing the processing result or existing result if already processed
    """
    question = item['question']
    if benchmark == "com_hard_intervention":
        crime = item['crime']
        facts = item['facts']
        options = item['options']
    else:
        crime = None
        facts = None
        options = None
    item_id = str(item['id'])
    correct_answer = item['correct_answer']

    # Create unique key for this question
    question_key = f"{benchmark}###{question}"

    # Check if already processed
    if question_key in existing_results:
        return existing_results[question_key]

    # Process new question
    if benchmark == "com_hard_intervention":
        result = test_single_question(
            generator, question, None,
            correct_answer, benchmark, item_id, args, crime, facts, options
        )
    else:
        result = test_single_question(
            generator, question, None,
            correct_answer, benchmark, item_id, args
        )

    # Write result to file (with lock if parallel processing)
    if save_lock:
        with save_lock:
            append_result_to_file(result, output_file)
    else:
        append_result_to_file(result, output_file)

    return result

def test_single_question(generator: NudgeMMSLGenerator, question: str, choices: List[str] = None,
                        correct_answer: str = None, subject: str = "", item_id: str = "", args = None, crime: str = None, facts: str = None, options: str = None) -> Dict[str, Any]:
    """Test a single question (MMLU, TruthfulQA, StrategyQA, or JustEval)"""
    
    # Format question based on benchmark type
    if args.benchmark == "MATH":
        formatted_question = format_question_math(
            question, use_instruct_template=True,
            model_name=generator.expert_model_name
        )
        # System prompts for MATH (matching single_inference.py)
        system_prompt_base = "Think step by step."
        system_prompt_expert = "Think step by step."
    elif args.benchmark == "supergpqa":
        formatted_question = format_question_supergpqa(
            question, use_instruct_template=True,
            model_name=generator.expert_model_name
        )
        # System prompts for SuperGPQA (matching single_inference.py)
        system_prompt_base = "You are an expert in science who answers multiple choice questions step by step."
        system_prompt_expert = "You are an expert in science who answers multiple choice questions step by step."
    elif args.benchmark == "minerva":
        formatted_question = formulate_prompt_minerva(question, generator.expert_model_name)
        system_prompt_base = "Think step by step."
        system_prompt_expert = "Think step by step."
    elif args.benchmark == "com_hard_intervention":
        formatted_question = formulate_prompt_com_hard_intervention(crime, facts, question, options, generator.expert_model_name)
        system_prompt_base = ""
        system_prompt_expert = ""
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    # Generate answer with nudging
    start_time = time.time()
    result = generator._completion_with_nudging(
        system_prompt_base=system_prompt_base,
        system_prompt_expert=system_prompt_expert,
        question=formatted_question,
        context="",
        completion_token_num=args.completion_token_num,
        completion_token_num_expert=args.completion_token_num,
        max_token_total=args.max_new_tokens,
        print_intermediate_output=args.print_intermediate
    )
    generation_time = time.time() - start_time
    
    # Create base result dictionary
    result_dict = {
        "benchmark": args.benchmark,
        "question": question,
        "formatted_question": formatted_question,
        "generated_answer": result["raw_answer"],
        "all_expert_words": result["all_expert_words"],
        "all_completions": result["all_completions"],
        "stop_reason": result["stop_reason"],
        "rounds": result["rounds"],
        "generation_time": generation_time,
        "base_model": generator.base_model_name,
        "expert_model": generator.expert_model_name,
        "confidence_threshold": generator.confidence_threshold,
        "base_port": generator.base_port,
        "expert_port": generator.expert_port,
        "total_expert_tokens": result["total_expert_tokens"],
        "total_base_tokens": result["total_base_tokens"],
        "total_generated_tokens": result["total_generated_tokens"],
        "correct_answer": correct_answer if args.benchmark == "tinymmlu" else "",
        "choices": choices if args.benchmark == "tinymmlu" else "",
        "subject": subject if args.benchmark == "tinymmlu" else ""
    }
    
    # Add benchmark-specific fields
    if args.benchmark == "MATH":
        result_dict.update({
            "id": item_id,
            "subject": "MATH",  # Use consistent subject name
            "correct_answer": correct_answer
        })
    elif args.benchmark == "supergpqa":
        result_dict.update({
            "id": item_id,
            "subject": "supergpqa",  # Use consistent subject name
            "correct_answer": correct_answer
        })
    elif args.benchmark == "minerva":
        result_dict.update({
            "id": item_id,
            "subject": "minerva",  # Use consistent subject name
            "correct_answer": correct_answer
        })
    elif args.benchmark == "com_hard_intervention":
        result_dict.update({
            "id": item_id,
            "subject": "com_hard_intervention",  # Use consistent subject name
            "correct_answer": correct_answer
        })
    return result_dict

def process_dataset_with_workers(generator: NudgeMMSLGenerator, dataset, existing_results: Dict[str, Dict],
                                benchmark: str, args, output_file: str) -> List[Dict[str, Any]]:
    """
    Process dataset with optional parallel workers.

    Args:
        generator: The NudgeMMSLGenerator instance
        dataset: The dataset to process
        existing_results: Dictionary of already processed results
        benchmark: The benchmark name
        args: Command line arguments
        output_file: Path to output file

    Returns:
        List of all results (including existing and newly processed)
    """
    all_results = []

    # Convert dataset to list if needed
    dataset_list = list(dataset)

    print(f"Processing {len(dataset_list)} items...")
    print(f"Using {args.max_workers} parallel worker(s)")

    # Thread-safe lock for file writing and results list
    save_lock = threading.Lock()

    if args.max_workers == 1:
        # Sequential processing (original behavior with detailed printing)
        for i, item in enumerate(tqdm(dataset_list, desc=f"Testing {benchmark}")):
            result = process_single_item(generator, item, existing_results, benchmark, args, output_file, save_lock=None)
            all_results.append(result)

            # Extract question for printing
            if benchmark == "truthfulqa":
                question = item['Question']
                item_id = item['Source']
            else:
                question = item['question']
                item_id = str(item['id'])

            # Print detailed info if result was newly generated
            question_key = f"{benchmark}###{question}"
            if question_key not in existing_results:
                print(f"\nQuestion {i+1}: COMPLETED")
                print(f"Q: {question[:100]}...")
                if 'correct_answer' in item and benchmark != "truthfulqa":
                    print(f"Correct: {item['correct_answer']}")
                print(f"Generated: {result['generated_answer']}")
                print(f"Rounds: {result['rounds']}, Time: {result['generation_time']:.2f}s")
                print(f"Stop reason: {result['stop_reason']}")
                print(f"Tokens - Expert: {result['total_expert_tokens']}, Base: {result['total_base_tokens']}, Total: {result['total_generated_tokens']}")
                print(f"✓ Result saved to {output_file}")
            else:
                print(f"\nQuestion {i+1}: SKIPPED (already processed)")
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_single_item, generator, item, existing_results, benchmark, args, output_file, save_lock): item
                for item in dataset_list
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_item), total=len(dataset_list), desc=f"Testing {benchmark}"):
                try:
                    result = future.result()
                    # Thread-safe append
                    with save_lock:
                        all_results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    if benchmark == "truthfulqa":
                        question = item['Question']
                    else:
                        question = item['question']
                    print(f"\nError processing question: {question[:100]}... - {e}")
                    # Continue processing other items

    return all_results

def main():
    args = parse_arguments()
    if args.base_model not in MODEL_NAME_DICT.keys():
        raise ValueError(f"Base model {args.base_model} not found in MODEL_NAME_DICT")
    if args.expert_model not in MODEL_NAME_DICT.keys():
        raise ValueError(f"Expert model {args.expert_model} not found in MODEL_NAME_DICT")
    
    # Set environment variables
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    
    print("="*80)
    print("Multi-Benchmark Nudge Test Script")
    print("="*80)
    print(f"Benchmark: {args.benchmark}")
    print(f"Base model: {args.base_model}")
    print(f"Expert model: {args.expert_model}")
    print(f"Base port: {args.base_port}")
    print(f"Expert port: {args.expert_port}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    if args.benchmark == "truthfulqa":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "halueval":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "MATH":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "mmlu_pro":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "lastletter":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "justlogic":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "aime2025":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "supergpqa":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "TheoremQA":
        print(f"Max total samples: {args.num_samples}")
    elif args.benchmark == "com_hard_intervention":
        print(f"Max total samples: {args.num_samples}")
    print(f"Force restart: {args.force_restart}")
    print("="*80)
    
    # Initialize generator
    generator = NudgeMMSLGenerator(
        base_model_name=args.base_model,
        expert_model_name=args.expert_model,
        hf_token=args.hf_token,
        base_port=args.base_port,
        expert_port=args.expert_port,
        confidence_threshold=args.confidence_threshold
    )
    
    # Load dataset based on benchmark type
    if args.benchmark == "MATH":
        print("Loading MATH dataset...")
        try:
            input_data, output_data, input_key, output_key = get_dataset(
                dataset_name='MATH',
                split=args.split,
                num_sample=args.num_samples
            )
            # Convert to format similar to other datasets
            full_dataset = []
            for i, (inp, out) in enumerate(zip(input_data, output_data)):
                full_dataset.append({
                    'id': i,
                    'question': inp['input'],
                    'correct_answer': out
                })
            print(f"Loaded {len(full_dataset)} total samples")
        except Exception as e:
            print(f"Error loading MATH dataset: {e}")
            return
    elif args.benchmark == "minerva":
        print("Loading Minerva dataset...")
        try:
            input_data, output_data, input_key, output_key = get_dataset(
                dataset_name='minerva',
                split=args.split,
                num_sample=args.num_samples
            )
            # Convert to format similar to other datasets
            full_dataset = []
            for i, (inp, out) in enumerate(zip(input_data, output_data)):
                full_dataset.append({
                    'id': i,
                    'question': inp['input'],
                    'correct_answer': out
                })
            print(f"Loaded {len(full_dataset)} total samples")
        except Exception as e:
            print(f"Error loading Minerva dataset: {e}")
            return
    elif args.benchmark == "supergpqa":
        print("Loading SuperGPQA dataset...")
        try:
            input_data, output_data, input_key, output_key = get_dataset(
                dataset_name='supergpqa',
                split=args.split,
                num_sample=args.num_samples,
                subject=args.subject
            )
            # Convert to format similar to other datasets
            full_dataset = []
            for i, (inp, out) in enumerate(zip(input_data, output_data)):
                full_dataset.append({
                    'id': i,
                    'question': inp['input'],
                    'correct_answer': out,
                    'field': inp.get('field', 'unknown'),
                    'discipline': inp.get('discipline', 'unknown'),
                    'subfield': inp.get('subfield', 'unknown')
                })
            print(f"Loaded {len(full_dataset)} total samples")
        except Exception as e:
            print(f"Error loading SuperGPQA dataset: {e}")
            return
    elif args.benchmark == "com_hard_intervention":
        print("Loading Com Hard Intervention dataset...")
        try:
            input_data, output_data, input_key, output_key = get_dataset(
                dataset_name='com_hard_intervention',
                split=args.split,
                num_sample=args.num_samples
            )
        except Exception as e:
            print(f"Error loading Com Hard Intervention dataset: {e}")
            return
        # Convert to format similar to other datasets
        full_dataset = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            full_dataset.append({
                'id': i,
                'crime': inp['crime'],
                'facts': inp['facts'],
                'question': inp['question'],
                'options': inp['options'],
                'correct_answer': out
            })
            print(f"Loaded {len(full_dataset)} total samples")
    # Handle subject-specific output directory for MMLU-Pro and SuperGPQA
    if (args.benchmark == 'mmlu_pro' or args.benchmark == 'supergpqa') and args.subject:
        dataset_dir = f'{args.benchmark}_{args.subject}_results'
    else:
        dataset_dir = f'{args.benchmark}_results'
    
    args.output_dir = f"./result/{dataset_dir}/{MODEL_NAME_DICT[args.base_model]}"
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup output file (JSONL format for incremental writing)
    output_file = os.path.join(args.output_dir, f"{args.split}/{MODEL_NAME_DICT[args.expert_model]}/official_nudge.jsonl")
    
    # Handle force restart
    if args.force_restart and os.path.exists(output_file):
        print(f"Force restart enabled. Removing existing results file: {output_file}")
        os.remove(output_file)
    
    # Load existing results for resume functionality
    existing_results = load_existing_results(output_file)
    
    if existing_results:
        print(f"Resume mode: Found {len(existing_results)} completed questions")
        print("Will skip already processed questions...")
    else:
        print("Starting fresh - no existing results found")
    
    # Test based on benchmark type
    print(f"\n{'='*60}")
    print(f"Testing {args.benchmark}")
    print(f"{'='*60}")

    # Limit total samples if needed
    test_data = full_dataset
    if args.num_samples and args.benchmark in ["MATH", "supergpqa", "minerva"]:
        if hasattr(test_data, 'select'):
            test_data = test_data.select(range(min(args.num_samples, len(test_data))))
        else:
            test_data = test_data[:args.num_samples]
    # Process all items with optional parallelism
    if args.benchmark == "MATH":
        all_results = process_dataset_with_workers(generator, test_data, existing_results, "MATH", args, output_file)
    elif args.benchmark == "supergpqa":
        all_results = process_dataset_with_workers(generator, test_data, existing_results, "supergpqa", args, output_file)
    elif args.benchmark == "minerva":
        all_results = process_dataset_with_workers(generator, test_data, existing_results, "minerva", args, output_file)
    elif args.benchmark == "com_hard_intervention":
        all_results = process_dataset_with_workers(generator, test_data, existing_results, "com_hard_intervention", args, output_file)
    # Print summary (results already saved incrementally)
    total_questions = len(all_results)
    total_time = sum(r['generation_time'] for r in all_results)
    avg_rounds = sum(r['rounds'] for r in all_results) / total_questions if total_questions > 0 else 0
    
    # Token statistics
    total_expert_tokens = sum(r['total_expert_tokens'] for r in all_results)
    total_base_tokens = sum(r['total_base_tokens'] for r in all_results)
    total_generated_tokens = total_expert_tokens + total_base_tokens
    avg_expert_tokens = total_expert_tokens / total_questions if total_questions > 0 else 0
    avg_base_tokens = total_base_tokens / total_questions if total_questions > 0 else 0
    expert_token_ratio = total_expert_tokens / total_generated_tokens if total_generated_tokens > 0 else 0
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions tested: {total_questions}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Average time per question: {total_time/total_questions:.2f}s")
    print(f"Average rounds per question: {avg_rounds:.1f}")
    print(f"")
    print(f"TOKEN STATISTICS:")
    print(f"Total expert tokens: {total_expert_tokens}")
    print(f"Total base tokens: {total_base_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Average expert tokens per question: {avg_expert_tokens:.1f}")
    print(f"Average base tokens per question: {avg_base_tokens:.1f}")
    print(f"Expert token ratio: {expert_token_ratio:.2%}")
    print(f"Base token ratio: {1-expert_token_ratio:.2%}")
    print(f"")
    print(f"Results saved incrementally to: {output_file}")

if __name__ == "__main__":
    main() 
