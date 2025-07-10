import torch
import json
from model import QwenMath_RM

def load_aime_data(file_path):
    """Load AIME 2025 JSONL data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_candidate(model, device, problem, steps):
    """Evaluate a single reasoning candidate using PRM"""
    if not steps:
        return 0.0
    
    # Combine all steps into full response
    full_response = " ".join(steps)
    
    # Format for PRM evaluation
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": full_response + "<extra_0>"}
    ]
    
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    inputs = model.tokenizer(
        text=[text],
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    with torch.no_grad():
        score = model(inputs['input_ids'], inputs['attention_mask'])
        return score.item()

def evaluate_aime_performance(model_path, aime_file_path):
    """Evaluate trained model on AIME 2025 dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = QwenMath_RM(device, model_path)
    model.eval()
    
    # Load AIME data
    aime_data = load_aime_data(aime_file_path)
    
    correct = 0
    total = 0
    
    for item in aime_data:
        problem = item.get('question') or item.get('problem', '')
        correct_answer = item.get('answers')
        reasoning_candidates = item.get('reasoning_candidates', [])
        
        if not problem or correct_answer is None or not reasoning_candidates:
            continue
        
        # Evaluate each candidate
        best_score = -float('inf')
        best_answer = None
        
        for candidate in reasoning_candidates:
            if not candidate or 'steps' not in candidate:
                continue
                
            steps = candidate['steps']
            final_answer = candidate.get('final')
            
            if not steps:
                continue
            
            score = evaluate_candidate(model, device, problem, steps)
            
            if score > best_score:
                best_score = score
                best_answer = final_answer
        
        if best_answer is not None:
            total += 1
            if best_answer == correct_answer:
                correct += 1
                print(f"✓ Problem {total}: Score={best_score:.3f}, Answer={best_answer}")
            else:
                print(f"✗ Problem {total}: Score={best_score:.3f}, Got={best_answer}, Expected={correct_answer}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nResults: {correct}/{total} correct, Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    MODEL_PATH = "outputs/qwen_math_prm"  
    AIME_FILE = "data/aime2025.jsonl"     
    
    evaluate_aime_performance(MODEL_PATH, AIME_FILE)