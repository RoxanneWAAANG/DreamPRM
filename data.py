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


def resize_image_if_needed(img, max_size=512):
    """
    Resize an image proportionally if either width or height exceeds max_size.
    Maintains the original aspect ratio while scaling down the longest side to max_size.

    :param img: PIL.Image object to be resized
    :param max_size: Maximum allowed length for the longest side (default: 512)
    :return: Resized PIL.Image object
    """
    width, height = img.size
    # Check if the longest dimension exceeds max_size
    if max(width, height) > max_size:
        # Calculate scaling ratio while maintaining aspect ratio
        scale_ratio = max_size / float(max(width, height))
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        # Resize image using LANCZOS resampling for high quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

class MyMetaDataset_QwenMath(Dataset):
    """
    Multi-step meta-learning examples. The `input` field contains all steps
    joined by <extra_0>. We split and re-wrap each step individually.
    Pre-processes all data during initialization for efficiency.
    """
    def __init__(self, records, tokenizer):
        self.data_js = []
        self.tokenizer = tokenizer
        
        pbar = tqdm(records, desc="Processing meta data")
        for record in pbar:
            input_text = record['input']
            label = float(record['true_false'])
            
            # Get dataset identifier
            dataset = record.get('id', str(len(self.data_js)))
            
            # Split into steps by separator
            raw_steps = input_text.split("\n\n<extra_0>")
            steps = [s.strip() for s in raw_steps if s.strip()]
            
            r_dict = {}
            for step_idx, step in enumerate(steps, start=1):
                messages = [
                    {"role": "user", "content": step}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    text,  # String format, not list
                    return_tensors="pt"
                    # Remove .to("cuda") - let DataLoader handle device movement
                )
                r_dict[str(step_idx)] = {
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze()
                }
            
            r_dict['labels'] = torch.tensor(label, dtype=torch.float32)
            r_dict['dataset'] = dataset
            self.data_js.append(r_dict)

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        return self.data_js[idx]


class MyDataset_QwenMath(Dataset):
    """
    Single-step PRM training examples for QwenMath.
    Each sample provides one <extra_0> reasoning step.
    Pre-processes all data during initialization for efficiency.
    """
    def __init__(self, records, tokenizer):
        self.data_js = []
        self.tokenizer = tokenizer
        
        pbar = tqdm(records, desc="Processing train data")
        for item in pbar:
            prompt = item['input']
            add = item['add']
            label = float(item.get('accuracy', item.get('score', 0)))
            # Use the actual dataset field (which contains problem id) for instance reweighting
            dataset = item.get('dataset', str(item.get('id', len(self.data_js))))

            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": add}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.tokenizer(
                text,  # String format, not list
                return_tensors="pt"
                # Remove add_special_tokens=False to match collaborator
                # Remove .to("cuda") - let DataLoader handle device movement
            )
            
            self.data_js.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.float32),
                'dataset': dataset
            })

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        return self.data_js[idx]


def custom_collate_fn(batch):
    """
    Collate for train: pad sequences, stack labels and datasets list.
    """
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b['label'] for b in batch])
    datasets = [b['dataset'] for b in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'dataset': datasets
    }


def meta_collate_fn(batch):
    """
    Collate for meta: stack per-step input_ids/attention_masks, and labels.
    """
    labels = torch.stack([b['labels'] for b in batch])
    datasets = [b['dataset'] for b in batch]
    collated = {'labels': labels, 'dataset': datasets}

    # find step keys (assume all samples have same number of steps)
    step_keys = sorted([k for k in batch[0].keys() if k.isdigit()], key=lambda x: int(x))
    for k in step_keys:
        collated[k] = {
            'input_ids':      torch.stack([b[k]['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b[k]['attention_mask'] for b in batch]),
            'dataset':        [b['dataset'] for b in batch]
        }
    return collated


def build_dataloader(
    processor_path,
    train_json_file,
    meta_json_file,
    train_batch_size,
    meta_batch_size
):
    tokenizer = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=True)
    
    # Load train data
    train_records = read_json(train_json_file)
    train_dataset = MyDataset_QwenMath(train_records, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    # Load meta data if provided
    if meta_json_file:
        meta_records = read_json(meta_json_file)
        meta_dataset = MyMetaDataset_QwenMath(meta_records, tokenizer)
        meta_loader = DataLoader(
            meta_dataset,
            batch_size=meta_batch_size,
            shuffle=True,
            collate_fn=meta_collate_fn
        )
    else:
        meta_loader = None

    print(f"Train samples: {len(train_dataset)}")
    print(f"Meta samples: {len(meta_dataset) if meta_loader else 0}")
    return train_loader, meta_loader

