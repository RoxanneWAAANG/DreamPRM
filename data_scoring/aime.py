#!/usr/bin/env python3
"""
Simple AIME Meta Dataset Generator for DreamPRM
"""

import pandas as pd
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
AIME_CSV_PATH = "/workspace/DreamPRM/data/AIME_Dataset_1983_2024.csv"
OUTPUT_PATH = "data/meta_aime.json"
NUM_ATTEMPTS = 6 
MAX_PROBLEMS = 10

def load_model():
    """Load Qwen2.5-Math model"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded!")
    return model, tokenizer

def generate_solution(problem, model, tokenizer):
    """Generate one reasoning chain for a problem"""
    prompt = f"""<|im_start|>system
        You are an expert mathematician solving AIME competition problems. Provide step-by-step reasoning and end with a numerical answer between 0 and 999.
        <|im_end|>
        <|im_start|>user
        Problem: {problem}

        Solve this step by step:
        1. Restate and understand the problem
        2. Identify key information and constraints  
        3. Choose your mathematical approach
        4. Execute the solution with detailed steps
        5. Verify your answer

        Provide your final answer as: Final answer: [number]
        <|im_end|>
        <|im_start|>assistant
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_length = len(inputs.input_ids[0])  # Store length before moving to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()

def extract_answer(text):
    """Extract final numerical answer from reasoning text"""
    patterns = [
        r"[Ff]inal\s+[Aa]nswer\s*:?\s*(\d+)",
        r"[Aa]nswer\s*:?\s*(\d+)", 
        r"[Tt]herefore,?\s*(\d+)",
        r"(?:^|\n)\s*(\d+)\s*$"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1]  # Take last match
            if answer.isdigit() and 0 <= int(answer) <= 999:
                return answer
    return None

def create_meta_dataset():
    """Main function to create meta.json"""
    # Load model
    model, tokenizer = load_model()
    
    # Load AIME data
    print(f"Loading AIME data from {AIME_CSV_PATH}")
    df = pd.read_csv(AIME_CSV_PATH)
    
    # Auto-detect column names
    problem_col = None
    answer_col = None
    for col in df.columns:
        if 'problem' in col.lower() or 'question' in col.lower():
            problem_col = col
        elif 'answer' in col.lower():
            answer_col = col
    
    print(f"Using columns: Problem='{problem_col}', Answer='{answer_col}'")
    
    # Limit problems if specified
    if MAX_PROBLEMS:
        df = df.head(MAX_PROBLEMS)
        print(f"Processing first {MAX_PROBLEMS} problems")
    
    meta_samples = []
    sample_id = 0
    
    # Process each problem
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating solutions"):
        problem = str(row[problem_col]).strip()
        correct_answer = str(row[answer_col]).strip()
        
        # Generate multiple solutions
        for attempt in range(NUM_ATTEMPTS):
            solution = generate_solution(problem, model, tokenizer)
            if not solution:
                continue
                
            extracted_answer = extract_answer(solution)
            if not extracted_answer:
                continue
            
            # Check if answer is correct
            is_correct = (extracted_answer == correct_answer)
            
            # Create meta sample
            full_input = f"Question: {problem}\n\n{solution}"
            meta_sample = {
                "id": sample_id,
                "true_false": is_correct,
                "input": full_input,
                "image_path": None
            }
            
            meta_samples.append(meta_sample)
            sample_id += 1
    
    # Save results
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(meta_samples, f, indent=2)
    
    # Print stats
    total = len(meta_samples)
    correct = sum(1 for s in meta_samples if s['true_false'])
    
    print(f"\nGenerated {total} samples")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_meta_dataset()