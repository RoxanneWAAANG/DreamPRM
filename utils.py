from model import QwenMath_RM
import random
import numpy as np
import torch
from time import sleep
from qwen_vl_utils import process_vision_info
import json

def set_cudnn(device="cuda"):
    torch.backends.cudnn.enabled = device == "cuda"
    torch.backends.cudnn.benchmark = device == "cuda"


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stop_epoch(time=3):
    try:
        print("can break now")
        for i in range(time):
            sleep(1)
        print("wait for next epoch")
        return False
    except KeyboardInterrupt:
        return True


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)


def load_QwenVL_RM(model_id = "reweighting/weights/base_model",
                      LN_id = "reweighting/weights/LN_weights.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = QwenVL_RM(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_id)
    best_model.LN.load_state_dict(torch.load(LN_id))
    best_model.to(device)
    return best_model


def load_QwenMath_RM(model_id = "reweighting/weights/qwen_math_prm"):
    """Load QwenMath PRM model - no LN weights needed since PRM handles scoring internally"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = QwenMath_RM(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_id)
    best_model.to(device)
    return best_model


def generate_reward_model_input(input, response_step, image_path, processor):
    prompt = input + "\n\n" + response_step
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs


def generate_reward_model_input_math(input, response_step, processor):
    """Generate input for math reward model (text-only, no image) with PRM format"""
    # Add step separator for PRM
    prompt = input + "\n\n" + response_step + "<extra_0>"
    
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": input},
        {"role": "assistant", "content": response_step + "<extra_0>"}
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    inputs = inputs.to("cuda")
    return inputs


def create_dataset_mapping(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_datasets = set()
    for item in data:
        if "dataset" in item:
            unique_datasets.add(item["dataset"])

    sorted_datasets = sorted(list(unique_datasets))
    mapping = {dataset: idx for idx, dataset in enumerate(sorted_datasets)}

    return mapping

