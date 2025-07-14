import json
from PIL import Image
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info


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

class MyDataset_QwenVL(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']
        prompt = prompt + "\n\n" + add
        label = self.data_js[idx]['accuracy']
        dset = self.data_js[idx]['dataset']

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = [resize_image_if_needed(image_inputs[0])]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_grid_thw': inputs['image_grid_thw'].squeeze(),
            'label': label,
            'dataset': dset
        }


class MyDataset_Llava(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        i = self.data_js[idx]['id']
        prompt = self.data_js[idx]['input']
        add = self.data_js[idx]['add']
        image_path = self.data_js[idx]['image_path']
        prompt = prompt + "\n\n" + add
        label = self.data_js[idx]['accuracy']
        dset = self.data_js[idx]['dataset']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        raw_image = Image.open(image_path)
        inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(0, torch.bfloat16)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image_sizes': inputs['image_sizes'].squeeze(),
            'label': label,
            'dataset': dset
        }


class MyDataset_QwenMath(Dataset):
    """
    Single-step PRM training examples for QwenMath.
    Each sample provides one <extra_0> reasoning step.
    """
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        prompt = item['input']
        add = item['add']
        label = float(item.get('score', item.get('accuracy', 0)))
        dataset = item.get('dataset', 'prm800k')

        # Build the PRM-formatted message
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user",   "content": prompt},
            {"role": "assistant", "content": add + "<extra_0>"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.tokenizer(
            text=[text],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        return {
            'input_ids':     inputs['input_ids'].squeeze(0),
            'attention_mask':inputs['attention_mask'].squeeze(0),
            'label':         torch.tensor(label, dtype=torch.float),
            'dataset':       dataset
        }
    

class MyMetaDataset_QwenVL(Dataset):
    def __init__(self, data_js, processor):
        self.data_js = data_js
        self.processor = processor

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        input = self.data_js[idx]['input']
        image_path = self.data_js[idx]['image_path']
        label = self.data_js[idx]['true_false']

        r_dict = {}
        step_num = find_max_step(input)
        for index in range(step_num):
            step = split_step(index+1, input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": step},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs = [resize_image_if_needed(image_inputs[0])]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'image_grid_thw': inputs['image_grid_thw'].squeeze(),
            }
        r_dict["labels"] = torch.tensor(label).to(dtype=torch.float)
        return r_dict


class MyMetaDataset_Llava(Dataset):
    def __init__(self, data_js, processor, step_num = 5):
        self.data_js = data_js
        self.processor = processor
        self.step_num = step_num

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        # find question and prompt
        input = self.data_js[idx]['input']
        image_path = self.data_js[idx]['image_path']
        label = self.data_js[idx]['true_false']
        r_dict = {}
        for index in range(self.step_num):
            step = split_step(index+1, input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": step},
                        {"type": "image"},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            raw_image = Image.open(image_path)
            inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(0, torch.bfloat16)
            r_dict[f"{index+1}"] = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'image_sizes': inputs['image_sizes'].squeeze(),
            }
        r_dict["labels"] = torch.tensor(label).to(dtype=torch.float)
        return r_dict


class MyMetaDataset_QwenMath(Dataset):
    """
    Multi-step meta-learning examples. The `input` field contains all steps
    joined by <extra_0>. We split and re-wrap each step individually.
    """
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
        self.sep_token = "<extra_0>"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        full_input = item['input']
        label = float(item['true_false'])
        dataset = item.get('dataset', 'aime')

        # Split into steps by separator
        raw_steps = full_input.split(self.sep_token)
        steps = [s.strip() for s in raw_steps if s.strip()]

        r_dict = {}
        for i, step in enumerate(steps, start=1):
            # Wrap each step as a PRM message
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user",   "content": item.get('question', '')},
                {"role": "assistant", "content": step + self.sep_token}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.tokenizer(
                text=[text],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
            r_dict[str(i)] = {
                'input_ids':      inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            }
        r_dict['labels'] = torch.tensor(label, dtype=torch.float)
        r_dict['dataset'] = dataset
        return r_dict


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
            'attention_mask': torch.stack([b[k]['attention_mask'] for b in batch])
        }
    return collated


def build_dataloader(
    processor_path,
    train_json_file,
    meta_json_file,
    train_batch_size,
    meta_batch_size,
    dataset_type="qwen_math"
):
    # Only for QwenMath
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
        # ensure dataset tag
        for rec in meta_records:
            rec.setdefault('dataset', 'aime')
        meta_dataset = MyMetaDataset_QwenMath(meta_records, tokenizer)
        meta_loader = DataLoader(
            meta_dataset,
            batch_size=meta_batch_size,
            shuffle=True,
            collate_fn=meta_collate_fn
        )
    else:
        meta_loader = None

    return train_loader, meta_loader

