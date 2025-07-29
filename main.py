# All code is original unless otherwise noted.
'''
python3 main.py \
  --train_json_file data/train_prm800k.json \
  --meta_json_file data/meta_aime.json \
  --weights_path outputs/qwen_math_prm_v0 \
  --batch_size 1 \
  --gradient_accumulation 16 \
  --lr 1e-5 \
  --iteration_num 1000 \
  --save_every_iterations 200 \
  --scheduler_step_size 2000 \
  --unroll_steps 1 \
  --precision bf16 \
  --scheduler_gamma 0.9 \
  --weight_decay 1e-4 \
  --max_epoch 10 \
  --reward_model Qwen/Qwen2.5-Math-1.5B \
  --device cuda
'''

import argparse
import os
import gc
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import AdamW
from torch.optim import AdamW
import wandb

# from peft import LoraConfig, get_peft_model, TaskType

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

from model import *
from data import *
from utils import *


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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Prepare data
domain_list = create_dataset_mapping(args.train_json_file)
domain_to_idx = {domain: idx for idx, domain in enumerate(domain_list)}
# print("Domain to index mapping:", domain_to_idx)

epoch_losses = []
current_epoch = 0
steps_per_epoch = 0

# Device
device = torch.device(args.device)

# Loss
criterion = nn.MSELoss()
criterion_meta = nn.MSELoss()

# Globals
step_count = 0
best_loss = float('inf')

# Track losses for validation
lower_weighted_loss = []
lower_loss = []
upper_loss = []

(
    train_dataloader,
    meta_dataloader,
) = build_dataloader(
    processor_path = args.reward_model,
    train_json_file = args.train_json_file,
    meta_json_file = args.meta_json_file,
    train_batch_size= args.batch_size,
    meta_batch_size= args.batch_size
)

# wandb init
wandb.init(
    project="DreamPRM-v0",
    name=f"meta-bs{args.batch_size}-lr{args.lr}",
    config=vars(args)
)


# ---------------------------------
# Upper Problem (Instance Reweighting)
# ---------------------------------
class Upper(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        global step_count
        labels = batch['labels'].to(device)
        loss_tensor = torch.zeros_like(labels)
        
        for k in sorted([k for k in batch if k.isdigit()], key=lambda x: int(x)):
            step_scores = self.lower.module(
                batch[k]['input_ids'].to(device),
                batch[k]['attention_mask'].to(device)
            )
            step_scores = step_scores.clamp(1e-8, 1-1e-8)
            loss_tensor += torch.log(step_scores / (1 - step_scores))

        # Compute mean score and apply sigmoid
        outputs = torch.sigmoid(loss_tensor / len([k for k in batch if k.isdigit()]))
        
        # Simple meta-learning loss (NO domain reweighting here)
        loss = criterion_meta(outputs, labels)
        wandb.log({'upper/loss': loss.item()})
        
        return loss

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        return InstanceTable(domain_to_idx)

    def configure_optimizer(self):
        return AdamW(
            self.module.parameters(),
            lr=1e-2,
            weight_decay=args.meta_weight_decay
        )


# ---------------------------------
# Lower Problem (PRM Fine-tuning)
# ---------------------------------
class Lower(ImplicitProblem):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.epoch_losses = []
        self.current_epoch = 0
        self.steps_per_epoch = len(train_dataloader)  # total steps in one epoch

    def forward(self, input_ids, attention_mask):   #, pixel_values, image_grid_thw):
        # torch.cuda.empty_cache()
        return self.module(input_ids, attention_mask)  #, pixel_values, image_grid_thw)

    def training_step(self, batch):
        global step_count
        step_count += 1

        # Inputs
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)
        domains = batch.get('dataset', None)

        # Forward pass - now returns step-level scores
        step_scores = self.forward(ids, mask)  # [B] - mean score per sequence

        # Handle labels properly - extract overall correctness
        if labels.dim() > 1:
            # If labels is [B, T], extract the overall correctness
            label_mask = (labels != -100).float()
            overall_labels = []
            for i in range(labels.size(0)):
                valid_positions = label_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    # Take the last valid label (overall problem correctness)
                    last_pos = valid_positions[-1]
                    overall_labels.append(labels[i, last_pos])
                else:
                    overall_labels.append(0.5)  # Default neutral score
            labels = torch.stack(overall_labels)  # [B]
        
        # Compute loss
        loss = criterion(step_scores, labels)
        
        # Track losses for logging
        self.epoch_losses.append(loss.item())
        lower_loss.append(loss.item())

        # Epoch tracking
        if step_count % self.steps_per_epoch == 0:
            avg_loss = np.mean(self.epoch_losses)
            self.current_epoch += 1
            wandb.log({
                'epoch/avg_loss': avg_loss,
                'epoch/number': self.current_epoch
            }, step=step_count)
            print(f"\n[Epoch {self.current_epoch}] Average Loss: {avg_loss:.6f}")

            # ==== Save model each epoch ==============================================
            # 1) Save the fine-tuned PRM model
            save_path = f"{args.weights_path}/qwen_math_prm/epoch_{self.current_epoch}"
            os.makedirs(save_path, exist_ok=True)
            self.module.base_model.save_pretrained(save_path)
            self.module.tokenizer.save_pretrained(save_path)

            # 2) Save the upper problem's state dict
            torch.save(
                upper_problem.state_dict(),
                os.path.join(args.weights_path, "instance_weights.pt")
            )
            # =========================================================================

            self.epoch_losses.clear()

        # Get current learning rate
        lr = self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else args.lr

        # Meta-learning: domain reweighting
        # This is where domain reweighting happens - domains come from PRM800K data
        if domains is not None:
            # Ensure loss tensor has correct shape for InstanceTable
            loss_tensor = loss.unsqueeze(0) if loss.dim() == 0 else loss.unsqueeze(1)
            # Apply domain reweighting through the upper problem
            weighted_loss = self.upper(domains, loss_tensor).squeeze()
        else:
            # Fallback if no domain info
            weighted_loss = loss
        
        # Track the weighted loss for validation
        lower_weighted_loss.append(weighted_loss.item())
        
        # Log metrics
        wandb.log({
            'train/weighted_loss': weighted_loss.item(), 
            'train/unweighted_loss': loss.item(),
            'train/lr': lr
        }, step=step_count)
        
        # Periodic logging
        if step_count % 500 == 0:
            print(f"Meta Step {step_count}: weighted_loss={weighted_loss.item():.4f}, unweighted_loss={loss.item():.4f}, lr={lr:.2e}")
        
        # Step scheduler
        self.scheduler.step()
        
        # Log epoch-level metrics when epoch completes
        if len(lower_loss) == len(train_dataloader):
            mean_inner_loss = np.mean(lower_loss)
            mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log({
                "inner_loss": mean_inner_loss,
                "inner_weighted_loss": mean_inner_weighted_loss,
            })
            lower_loss.clear()
            lower_weighted_loss.clear()
        
        return weighted_loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        # return QwenVL_RM(device)
        # return QwenMath_RM(device, args.reward_model).train()
        return QwenMath_RM(device, args).train()

    def configure_optimizer(self):
        return torch.optim.Adam(
            self.module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
        return scheduler


class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        global best_loss, step_count
        print(f"[Validation at step {step_count}] Saving model...")
        torch.cuda.empty_cache()

        upper_problem = self.problems[0]
        lower_problem = self.problems[1]
        
        # Save the fine-tuned PRM model.
        lower_problem.module.base_model.save_pretrained(f"{args.weights_path}/qwen_math_prm")
        lower_problem.module.tokenizer.save_pretrained(f"{args.weights_path}/qwen_math_prm")

        # save the upper problem's state dict.
        torch.save(
            upper_problem.state_dict(),
            f"{args.weights_path}/domain_weights.pt",
        )

        val_loss = float('inf')
        if lower_weighted_loss:
            val_loss = np.mean(lower_weighted_loss)
        # update best
        if val_loss < best_loss:
            best_loss = val_loss
        # clear for next round
        lower_weighted_loss.clear()

        # now log both current and best
        wandb.log({
            "validation/loss": val_loss,
            "validation/best_loss": best_loss,
        }, step=step_count)

        return {"loss": val_loss}
    

upper_config = Config(type="darts", precision=args.precision, retain_graph=True)
lower_config = Config(
    type="darts", 
    precision=args.precision, 
    unroll_steps=args.unroll_steps,
    gradient_accumulation=args.gradient_accumulation
)

engine_config = EngineConfig(
    train_iters=args.iteration_num * args.max_epoch,
    valid_step=args.save_every_iterations,
    # strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)

# Instantiate problems
upper_problem = Upper(name='upper', config=upper_config)
lower_problem = Lower(name='lower', config=lower_config)
problems = [upper_problem, lower_problem]

# Define bidirectional dependencies properly
dependencies = {
    'l2u': {lower_problem: [upper_problem]},  # Lower depends on Upper (since Lower calls self.upper)
    'u2l': {upper_problem: [lower_problem]}   # Upper depends on Lower (bilevel structure)
}

# Run
# engine = ReweightingEngine(
engine = Engine(
    config=engine_config,
    problems=problems,
    dependencies=dependencies
)
engine.run()

print("Training completed, saving models...")

# Save the models after training
upper_problem = problems[0]
lower_problem = problems[1]

# Save the lower problem's model and tokenizer
save_path = f"{args.weights_path}/qwen_math_prm"
os.makedirs(save_path, exist_ok=True)
lower_problem.module.base_model.save_pretrained(save_path)
lower_problem.module.tokenizer.save_pretrained(save_path)

# Save the upper problem's state dict
torch.save(
    upper_problem.state_dict(),
    f"{args.weights_path}/domain_weights.pt"
)

wandb.finish()

# Clear memory
torch.cuda.empty_cache()
