import torch
from model import QwenMath_RM
from transformers import AutoTokenizer

def test_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenMath_RM(device, "outputs/qwen_math_prm/qwen_math_prm")
    model.eval()
    
    test_cases = [
        {
            "input": "Solve for x: 2x + 5 = 11",
            "response": "2x + 5 = 11\n2x = 11 - 5\n2x = 6\nx = 3",
            "expected_good": True
        },
        {
            "input": "Solve for x: 2x + 5 = 11", 
            "response": "2x + 5 = 11\n2x = 11 + 5\n2x = 16\nx = 8",  # bad solution
            "expected_good": False
        }
    ]
    
    for i, case in enumerate(test_cases):
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": case["input"]},
            {"role": "assistant", "content": case["response"] + "<extra_0>"}
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
        
        print(f"Test {i+1}: Score = {score.item():.3f}, Expected: {'Good' if case['expected_good'] else 'Bad'}")

if __name__ == "__main__":
    test_trained_model()
