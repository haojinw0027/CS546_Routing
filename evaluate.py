#!/usr/bin/env python3
"""
Script to evaluate JSON responses by extracting Final Answer numbers and comparing with gold answers.
Adds extracted_answer and is_correct fields to the original JSON structure.
"""

import json
import re
import sys
import os

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

def process_json_file(input_file, output_file=None):
    """
    Process a JSON file to extract final answers and compare with gold answers.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional, defaults to input_file)
    """
    if output_file is None:
        output_file = input_file
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each response
    correct_count = 0
    total_count = 0
    
    for response_item in data.get('responses', []):
        total_count += 1
        
        # Extract the final answer from response
        response_text = response_item.get('response', '')
        extracted_answer = extract_final_answer(response_text)
        
        # Get gold answer
        gold_answer = str(response_item.get('gold_answer', '')).strip()
        
        # Compare answers
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
        
        # Add new fields to the response item
        response_item['extracted_answer'] = extracted_answer
        response_item['is_correct'] = is_correct
        
        if is_correct:
            correct_count += 1
        
        # Print progress
        print(f"Item {total_count}: Gold={gold_answer}, Extracted={extracted_answer}, Correct={is_correct}")
    
    # Add summary statistics
    accuracy = correct_count / total_count if total_count > 0 else 0
    data['evaluation_summary'] = {
        'total_responses': total_count,
        'correct_responses': correct_count,
        'accuracy': accuracy
    }
    
    # Write the updated JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete!")
    print(f"Total responses: {total_count}")
    print(f"Correct responses: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_file}")

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <input_json_file> [output_json_file]")
        print("If output_json_file is not provided, the input file will be modified in place.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    process_json_file(input_file, output_file)

if __name__ == "__main__":
    main()
