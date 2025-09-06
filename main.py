# All code is original unless otherwise noted.
'''
pip uninstall torch safetensors transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python main.py \
  --train_json_file data/train_prm800k.json \
  --meta_json_file data/meta_aime.json \
  --weights_path outputs/qwen_math_prm_v2 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --lr 1e-5 \
  --meta_lr 0.0005 \
  --iteration_num 1000 \
  --save_every_iterations 200 \
  --scheduler_step_size 1000 \
  --unroll_steps 5 \
  --precision bf16 \
  --scheduler_gamma 0.9 \
  --weight_decay 1e-4 \
  --max_epoch 10 \
  --reward_model Qwen/Qwen2.5-Math-1.5B \
  --device cuda
'''
import torch
import os

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:False,backend:native'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import wandb

from transformers import get_cosine_schedule_with_warmup

# from peft import LoraConfig, get_peft_model, TaskType
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

from model import *
from data import *
from utils import *

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description="DreamPRM")
parser.add_argument('--train_json_file', type=str)
parser.add_argument('--meta_json_file', type=str)
parser.add_argument('--weights_path', type=str)
parser.add_argument("--iteration_num", type=int, default=10000)
parser.add_argument("--save_every_iterations", type=int, default=1000)
parser.add_argument("--unroll_steps", type=int, default=5)
parser.add_argument("--gradient_accumulation", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-7)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--scheduler_step_size", type=int, default=5000)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--meta_lr", type=float, default=0.01)
parser.add_argument("--meta_weight_decay", type=float, default=0.0)
parser.add_argument("--reward_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument("--num_meta", type=int, default=1000)
parser.add_argument("--imbalanced_factor", type=int, default=None)
parser.add_argument("--corruption_type", type=str, default=None)
parser.add_argument("--corruption_ratio", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=120)
parser.add_argument("--meta_interval", type=int, default=1)
parser.add_argument("--paint_interval", type=int, default=20)
parser.add_argument("--peft_rank", type=int, default=-1, help="Rank for PEFT, -1 for no PEFT")
parser.add_argument("--lora_alpha", type=float, default=32.0)
parser.add_argument("--lora_dropout", type=float, default=0.05)

args = parser.parse_args()
set_seed(args.seed)

# === Data, Domains, Device ===
instance_list = create_dataset_mapping(args.train_json_file)
device = torch.device(args.device)

# Dataloaders
train_dataloader, meta_dataloader = build_dataloader(
    processor_path=args.reward_model,
    train_json_file=args.train_json_file,
    meta_json_file=args.meta_json_file,
    train_batch_size=args.batch_size,
    meta_batch_size=args.batch_size
)

# Loss
criterion = nn.MSELoss()
criterion_meta = nn.MSELoss()

# Tracking
lower_weighted_loss, lower_loss, upper_loss = [], [], []
best_loss = float('inf')

# wandb init
wandb.init(
    project="DreamPRM-v2",
    name=f"meta-bs{args.batch_size}-lr{args.lr}",
    config=vars(args)
)


# ---------------------------------
# Upper Problem (Instance Reweighting)
# ---------------------------------
class Upper(ImplicitProblem):
    def forward(self, instance_strings, x):
        return self.module(instance_strings, x)

    # def parameters(self):
    #     """
    #     Yield only the trainable (i.e., LoRA) parameters.
    #     """
    #     # return self.module.parameters()
    #     return (p for p in self.module.parameters() if p.requires_grad)

    def training_step(self, batch):
        # torch.cuda.empty_cache()

        # steps = [batch['1'], batch['2'], batch['3'], batch['4'], batch['5'],]
        numeric_keys = [k for k in batch.keys() if k.isdigit()]
        sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
        steps = [batch[key] for key in sorted_keys]
        labels = batch['labels'].to(device)
        labels = labels.squeeze()

        # print(f"Processing {len(steps)} steps for dataset: {batch['dataset'][0]}")
        # print(f"Batch keys: {batch.keys()}")

        max_steps = 5 
        # limit steps to max_steps
        if len(steps) > max_steps:
            # print(f"Limiting to {max_steps} steps from {len(steps)}")
            # select key steps: first, last, and evenly spaced middle steps
            indices = [0, len(steps)//4, 2*len(steps)//4, 3*len(steps)//4, len(steps)-1]
            selected_steps = [steps[i] for i in indices]
        else:
            selected_steps = steps

        mean_score = 0
        for i in selected_steps:
            score = self.lower(
                i['input_ids'].to(device),
                i['attention_mask'].to(device),
            )
            # print(f"Step {i['input_ids']}: Score: {score}")
            # logits = torch.logit(score, eps=1e-6)  # logit transformation
            # print(f"Logits: {logits}")
            # mean_score += logits.squeeze()  # [B,1] → [B]

            score = score.squeeze()  # [B,1] → [B]
            mean_score += torch.log(score + 1e-6) - torch.log(1 - score + 1e-6)  # logit transformation
            # mean_score += torch.log(score / (1 - score))
            # 0.5 is the initial value for sigmoid, so we can use logit transformation

        # Aggregate function: sigmoid of mean logits
        outputs = torch.sigmoid(mean_score / len(selected_steps))
        # print(f"Outputs: {outputs}")

        # compute loss
        # print("labels shape: ", labels.shape)
        # print("outputs shape: ", outputs.shape)
        loss = criterion_meta(outputs, labels)
        upper_loss.append(loss.item())

        # clean memory
        del steps, labels, outputs, mean_score, selected_steps
        torch.cuda.empty_cache()

        # logging
        wandb.log({
            "upper_loss": np.mean(upper_loss)
        })

        # clear upper_loss every 100 steps
        if len(upper_loss) >= 100:
            upper_loss.clear()

        return {"loss": loss}

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        return InstanceTable(instance_list)

    def configure_optimizer(self):
        return AdamW(
            self.module.parameters(),
            lr=args.meta_lr,
            weight_decay=args.meta_weight_decay
        )
    
    def configure_scheduler(self):
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * args.iteration_num),  # 10% warmup for meta
            num_training_steps=args.iteration_num
        )


# ---------------------------------
# Lower Problem: PRM Fine-tuning + LoRA
# ---------------------------------
class Lower(ImplicitProblem):
    def forward(self, input_ids, attention_mask):
        return self.module(input_ids, attention_mask)  #, pixel_values, image_grid_thw)

    # def parameters(self):
    #     return (p for p in self.module.parameters() if p.requires_grad)

    def training_step(self, batch):
        torch.cuda.empty_cache()

        # ---- 1) forward pass ----
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)
        instance_strings = batch['dataset']

        # print(f"Processing dataset: {instance_strings[0]} with {len(ids)} steps")

        # ---- 2) single forward pass ----
        outputs = self.forward(
            input_ids=ids, 
            attention_mask=mask
        )
        # calculate loss
        loss = criterion(outputs, labels)

        # ---- 3) meta-reweight via upper ----
        # shape it for InstanceTable: [1,1,B] or similar
        # weighted_loss = self.upper.forward(batch['dataset'], lt).squeeze()
        weighted_loss = self.upper( # [B,1] → squeeze → [B]
            instance_strings,
            loss    # [B,1]
        )

        # ---- 4) logging ----
        lower_loss.append(loss.mean().item())
        lower_weighted_loss.append(weighted_loss.mean().item())

        wandb.log({
            "lower_loss": np.mean(lower_loss),
            "lower_weighted_loss": np.mean(lower_weighted_loss)
        })

        del ids, mask, labels, outputs, loss

        return weighted_loss.mean()

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        # return QwenMath_RM(device, args).base_model.train()
        return QwenMath_RM(device, args)

    def configure_optimizer(self):
        return optim.AdamW(
            self.module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def configure_scheduler(self):
        # return optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size = args.scheduler_step_size,
        #     gamma=args.scheduler_gamma
        # )
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.05 * args.iteration_num),  # 5% warmup
            num_training_steps=args.iteration_num
        )


upper_config = Config(
    type="darts", 
    precision=args.precision, 
    retain_graph=True, 
    gradient_accumulation=args.gradient_accumulation,
)
lower_config = Config(
    type="darts", 
    precision=args.precision, 
    unroll_steps=args.unroll_steps, 
    gradient_accumulation=args.gradient_accumulation,
)

engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.save_every_iterations,
    # strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)

# Instantiate problems
upper = Upper(name="upper", config=upper_config)
lower = Lower(name="lower", config=lower_config)

# Define bidirectional dependencies
problems = [upper, lower]
dependencies = {
    "l2u": {lower: [upper]},    # Lower depends on Upper (since Lower calls self.upper)
    "u2l": {upper: [lower]}     # Upper depends on Lower (bilevel structure)
}

# Run
# engine = ReweightingEngine(
engine = Engine(
    config=engine_config,
    problems=problems,
    dependencies=dependencies
)

print("Starting training...")
# engine.run()
try:
    engine.run()
except Exception as e:
    # Handle any exceptions during training
    # save the models when an error occurs
    print(f"Training interrupted: {e}")
    save_path = f"{args.weights_path}/emergency_save"
    os.makedirs(save_path, exist_ok=True)
    torch.save(lower.state_dict(), f"{save_path}/lower_state.pt")
    torch.save(upper.state_dict(), f"{save_path}/upper_state.pt")
    print(f"Models saved to {save_path}")
    exit(1)


print("Saving models...")
# Save the lower problem's model and tokenizer
save_path = f"{args.weights_path}/qwen_math_prm"
os.makedirs(save_path, exist_ok=True)
torch.save(lower.state_dict(), f"{save_path}/lower_state.pt")
torch.save(upper.state_dict(), f"{save_path}/upper_state.pt")

# Save the upper problem's state dict
torch.save(
    upper.state_dict(),
    f"{args.weights_path}/domain_weights.pt"
)

wandb.finish()

# Clear memory
torch.cuda.empty_cache()
