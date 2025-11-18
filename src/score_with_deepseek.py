# score_with_deepseek.py

import json
import os
from openai import OpenAI
import re
import argparse
import concurrent.futures

client = OpenAI(
    api_key="sk-cb9c6cee5c11422c94128da465b93cac",  # Updated API key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# The evaluation prompt template
EVAL_PROMPT = """
You are a strict evaluator of mathematical reasoning. Your task is to evaluate a `model_output` based on a gold-standard `solution`. Your **core focus** is not on whether the `model_output` is correct, but on its performance regarding "Overthinking".

**Core Instruction:**
Treat the `solution` as the **"5-point perfect"** baseline. The closer the `model_output`'s reasoning process is to the `solution`, the higher its score. If the `model_output` is more verbose or uses more convoluted steps than the `solution`, its score must be lowered.

### Evaluation Metrics
**Overthinking:** Does the solution include obvious, superfluous, or unnecessary steps?
    * 1 = A lot of overthinking (e.g., defining unnecessary variables, stating common knowledge like "1+1=2")
    * 2 = Obvious overthinking
    * 3 = Slight overthinking
    * 4 = Almost no overthinking
    * 5 = No overthinking (no redundant steps, just like the `solution`)

### Evaluation Process
1.  **Read the `question`** to understand the task.
2.  **Closely study the `solution`**: Use it as the "perfect score" standard for conciseness and lack of overthinking.
3.  **Evaluate the `model_output`**: Compare it against the `solution` baseline.
4.  **Provide scores and justification**: Based on the metrics above, provide your scores and detailed reasoning in the `evaluation_output` format.

### Data to be Evaluated
{data_json}

### Evaluation Output (Please use this format)
```json
{{
  "overthinking_score": <Enter score 1-5>,
  "justification": "<Provide a detailed rationale for your scores here, specifically contrasting the model_output against the 'solution' baseline.>"
}}
```
"""

def evaluate_case(question, solution, model_output):
    data = {
        "question": question,
        "solution": solution,
        "model_output": model_output
    }
    data_json = json.dumps(data)
    prompt = EVAL_PROMPT.format(data_json=data_json)
    
    response = client.chat.completions.create(
        model="deepseek-r1", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  
        max_tokens=8000 
    )
    
    eval_output = response.choices[0].message.content.strip()
    
    # Extract JSON block
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', eval_output)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = eval_output 
    
    try:
        eval_dict = json.loads(json_str)
        return eval_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing evaluation output: {eval_output}\nError: {e}")
        return None

max_workers = 10  # Set maximum parallel API calls

def process_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cases = data.get("cases", [])
    easy_total = 0
    easy_count = 0
    hard_total = 0
    hard_count = 0
    total_overthinking = 0
    num_cases = len(cases)
    
    temp_file = os.path.join(os.path.dirname(json_path), f"temp_{os.path.basename(json_path)}")
    processed_questions = set()
    processed_results = []
    
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            processed_results = json.load(f)
        processed_questions = {r['question'] for r in processed_results}
    
    remaining_cases = [case for case in cases if case['question'] not in processed_questions]
    
    def evaluate_single_case(case):
        eval_result = evaluate_case(
            case["question"],
            case["solution"],
            case["model_output"]
        )
        if eval_result and "overthinking_score" in eval_result:
            case["evaluation"] = eval_result  # Already includes score and justification
            return case, eval_result["overthinking_score"]
        return case, None
    
    print(f"共 {num_cases} 个 cases，开始评估...", flush=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_case, case) for case in remaining_cases]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), len(processed_results) + 1):
            print(f"  处理进度: {idx}/{num_cases} ({idx*100//num_cases}%)", end='\r', flush=True)
            updated_case, score = future.result()
            processed_results.append(updated_case)
            
            # Save progress to temp file
            with open(temp_file, 'w') as f:
                json.dump(processed_results, f, indent=4)
    
    print() 
    
    # Update original cases with evaluations
    case_dict = {c['question']: c for c in processed_results}
    for case in cases:
        if case['question'] in case_dict and 'evaluation' in case_dict[case['question']]:
            case['evaluation'] = case_dict[case['question']]['evaluation']
    
    # Compute totals from all processed
    for case in processed_results:
        if "evaluation" in case and "overthinking_score" in case["evaluation"]:
            score = case["evaluation"]["overthinking_score"]
            total_overthinking += score
            if case.get("difficulty_type") == 0:
                easy_total += score
                easy_count += 1
            elif case.get("difficulty_type") == 1:
                hard_total += score
                hard_count += 1
    
    # Compute averages
    avg_overthinking = total_overthinking / num_cases if num_cases > 0 else 0
    easy_avg_overthinking = easy_total / easy_count if easy_count > 0 else 0
    hard_avg_overthinking = hard_total / hard_count if hard_count > 0 else 0
    
    # Update summary
    if "summary" in data:
        data["summary"]["avg_overthinking"] = avg_overthinking
        data["summary"]["easy_avg_overthinking"] = easy_avg_overthinking
        data["summary"]["hard_avg_overthinking"] = hard_avg_overthinking
    
    # Write back
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Remove temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"Processed {json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Path to folder containing JSON files to process")
    args = parser.parse_args()
    
    folder = args.folder_path
    if not os.path.isabs(folder):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(script_dir, folder)
    
    json_files = [f for f in os.listdir(folder) if f.endswith('.json') and f != 'summary.json']
    for json_file in json_files:
        full_path = os.path.join(folder, json_file)
        process_json_file(full_path)
    
    print("All files processed.")

if __name__ == "__main__":
    main()
