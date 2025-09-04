import json
from PIL import Image
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info

from tqdm import tqdm


def split_step(s_id, response):
    """    
    Split the response string to extract the assistant's response for a specific step.
    Args:
        s_id: Step number to extract.
        response: Full response string containing multiple steps.
    """
    s = f"Step {s_id}"
    s_next = f"Step {s_id+1}"
    if s_next in response:
        assistant = response.split(s_next)[0]
    elif "Final answer" in response and s in response:
        assistant = response.split("Final answer")[0]
    else:
        assistant = ""
    return assistant


def find_max_step(response):
    """
    Find the maximum step number in a response string containing steps.

    Args:
        response: String containing steps in formats like "Step 1: ...", "Step 2: ...", etc.

    Returns:
        Integer representing the highest step number found. Returns 0 if no steps are found.
    """
    # Find all occurrences of step patterns (case-insensitive)
    # Matches: "Step 1", "STEP 2", "step3", "Step: 4", etc.
    step_numbers = re.findall(r'Step[\s:]*(\d+)', response, re.IGNORECASE)

    # Return 0 if no step numbers found
    if not step_numbers:
        return 0

    # Convert found numbers from strings to integers
    step_numbers = [int(num) for num in step_numbers]

    # Return the maximum step number
    return max(step_numbers)


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


class MyDataset_QwenMath(Dataset):
    """
    Single-step PRM training examples for QwenMath.
    Each sample provides one <extra_0> reasoning step.
    Pre-processes all data during initialization for efficiency.
    """
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
        print(f"Loaded {len(records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        
        prompt = item['input']
        add = item['add']
        label = float(item.get('accuracy', item.get('score', 0)))
        dataset = item.get('dataset', str(item.get('id', idx)))

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": add}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,  # False because we have the complete conversation
            tokenize=True,                # Direct tokenization
            return_dict=True,             # Returns dict with input_ids, attention_mask
            return_tensors="pt",          # Use PyTorch tensors
            truncation=True,              # Truncate to max length
            max_length=512,               # Set max length here
            padding="max_length"          # Pad to max length
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float32),
            'dataset': dataset
        }
        

class MyMetaDataset_QwenMath(Dataset):
    """
    Multi-step meta-learning examples. The `input` field contains all steps
    joined by <extra_0>. We split and re-wrap each step individually.
    Pre-processes all data during initialization for efficiency.
    """
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
        print(f"Loaded {len(records)} meta samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        torch.cuda.empty_cache()
        
        record = self.records[idx]
        input_text = record['input']
        label = float(record['true_false'])
        dataset = record.get('id', str(idx))
        # print(f"Processing dataset: {dataset}")

        # Create a dictionary to hold the cumulative content
        r_dict = {}
        # Find max step
        step_num = find_max_step(input_text)
        # print(f"Max step number found: {step_num}")
        for index in range(step_num):
            step = split_step(index+1, input_text)
            # print(f"Processing step {index+1}: {step.strip()}")

            messages = [
                {
                    "role": "user", 
                    "content": f"{step.strip()}\n\n"
                }
            ]

            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # True because we want model to generate response
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )

            # Store the inputs in the dictionary
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
            }
            # print(f"Processed step {index+1}: {r_dict[f'{index+1}']['input_ids']}")
            # print(r_dict.keys())
        
        r_dict["labels"] = torch.tensor(label, dtype=torch.float32)
        r_dict["dataset"] = dataset
        
        return r_dict

import random

def build_dataloader(
    processor_path,
    train_json_file,
    meta_json_file,
    train_batch_size,
    meta_batch_size,
    max_meta_samples=None
):
    tokenizer = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=True)
    
    # Load train data
    train_records = read_json(train_json_file)
    train_dataset = MyDataset_QwenMath(train_records, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    # Load meta data
    if meta_json_file:
        meta_records = read_json(meta_json_file)

        if max_meta_samples is not None and len(meta_records) > max_meta_samples:
            random.seed(42)
            # meta_records = meta_records[:max_meta_samples]  # Take first N samples
            meta_records = random.sample(meta_records, max_meta_samples)
            print(f"Limited meta samples from {len(read_json(meta_json_file))} to {max_meta_samples}")

        meta_dataset = MyMetaDataset_QwenMath(meta_records, tokenizer)
        meta_loader = DataLoader(
            meta_dataset,
            batch_size=meta_batch_size,
            shuffle=True,
        )
    else:
        meta_loader = None

    print(f"Train samples: {len(train_dataset)}")
    print(f"Meta samples: {len(meta_dataset) if meta_loader else 0}")
    return train_loader, meta_loader


if __name__ == "__main__":
    processor_path = "/workspace/weights/qwen2.5-math-1.5b"
    train_json_file = "data/train_prm800k.json"
    meta_json_file = "data/meta_aime2025.json"
    
    train_loader, meta_loader = build_dataloader(
        processor_path,
        train_json_file,
        meta_json_file,
        train_batch_size=1,
        meta_batch_size=1
    )
    
    for batch in tqdm(train_loader):
        # print(batch) 
        pass
    
    if meta_loader:
        for batch in tqdm(meta_loader):
            # print(batch)
            pass
    else:
        print("No meta loader provided.")  

    print("Data loading test completed.") # logits: [batch_size, seq_len, vocab_size] 
