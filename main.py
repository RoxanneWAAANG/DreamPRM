# All code is original unless otherwise noted.
'''
PYTHONPATH=/workspace/DreamPRM \
python3 main.py \
  --train_json_file data/train_prm800k.json \
  --weights_path    outputs/qwen_math_prm_v2 \
  --batch_size 2 \
  --gradient_accumulation 16 \
  --lr              1e-5 \
  --iteration_num   174250 \
  --save_every_iterations 5000 \
  --scheduler_step_size 25000 \
  --scheduler_gamma 0.8 \
  --weight_decay    1e-4 \
  --dataset_type    qwen_math \
  --reward_model    Qwen/Qwen2.5-Math-PRM-7B \
  --baseline \
  --device          cuda
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import AdamW
from torch.optim import AdamW
import wandb

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
parser.add_argument("--baseline", action="store_true")
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
parser.add_argument("--dataset_type", type=str, default="qwen_math")

args = parser.parse_args()
set_seed(args.seed)

# Prepare data
domain_list = create_dataset_mapping(args.train_json_file)
domain_to_idx = {domain: idx for idx, domain in enumerate(domain_list)}
print("Domain to index mapping:", domain_to_idx)

sampler = None
resume_idxes = None
resume_labels = None

(
    train_dataloader,
    meta_dataloader,
) = build_dataloader(
    processor_path = args.reward_model,
    train_json_file = args.train_json_file,
    meta_json_file=(None if args.baseline else args.meta_json_file),
    train_batch_size= args.batch_size,
    meta_batch_size= args.batch_size,
    dataset_type=args.dataset_type
)

# wandb init
# os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="DreamPRM-v0",
    name=f"{'baseline' if args.baseline else 'meta'}-bs{args.batch_size}-lr{args.lr}",
    config=vars(args)
)

# Device
device = torch.device(args.device)

# Loss
if args.dataset_type == "qwen_math":
    # Use BCE for binary classification
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    criterion_meta = nn.BCELoss()
else:
    # Use MSE for regression tasks
    criterion = nn.MSELoss()
    criterion_meta = nn.MSELoss()

# Globals
step_count = 0
best_val_loss = float('inf')

lower_weighted_loss = []
lower_loss = []
upper_loss = []
best_loss = float('inf')
step_count = 0
current_lr = args.lr

# ---------------------------------
# Upper Problem (Instance Reweighting)
# ---------------------------------
class Upper(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        # steps = [batch['1'], batch['2'], batch['3'], batch['4'], batch['5'],]
        labels = batch['labels'].to(device)
        loss_tensor = torch.zeros_like(labels)
        domains = []

        for k in sorted([k for k in batch if k.isdigit()], key=lambda x: int(x)):
            # Text-only input for QwenMath
            out = self.inner(
                batch[k]['input_ids'].to(device),
                batch[k]['attention_mask'].to(device),
                # batch[k]['pixel_values'].to(device),
                # batch[k]['image_grid_thw'].to(device)
            )
            out = out.clamp(1e-8, 1-1e-8)
            loss_tensor += torch.log(out / (1 - out))
            domains.append(batch[k]['dataset'])

        weighted = torch.sigmoid(loss_tensor / len(domains))
        loss = criterion(weighted, labels)
        return {'loss': loss}

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
        # pixel_values = batch['pixel_values'].to(device)
        # image_grid_thw = batch['image_grid_thw'].to(device)

        # Forward pass
        logits = self.forward(ids, mask)
        loss   = criterion(logits, labels)

        lr = self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else args.lr

        if args.baseline:
            # Baseline: simple fine-tuning
            wandb.log({'train/loss': loss.item(), 'train/lr': lr}, step=step_count)
            if args.debug and step_count % 500 == 0:
                print(f"Baseline Step {step_count}: loss={loss.item():.4f}, lr={lr:.2e}")
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            return loss
        
        else:
            # Meta-learning: instance reweighting
            # Ensure losses tensor shape (batch_size, 1)
            loss_tensor = loss.unsqueeze(1) if loss.dim()==1 else loss
            weighted_loss = self.upper(domains, loss_tensor).squeeze()
            wandb.log({'train/weighted_loss': weighted_loss.item(), 'train/lr': lr}, step=step_count)
            if args.debug and step_count % 500 == 0:
                print(f"Meta Step {step_count}: weighted_loss={weighted_loss.item():.4f}, lr={lr:.2e}")
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            return weighted_loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        # return QwenVL_RM(device)
        return QwenMath_RM(device, args.reward_model).train()

    def configure_optimizer(self):
        # optimizer = AdamW(
        #     self.module.parameters(),
        #     lr=args.lr,
        # )
        # return optimizer
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
        # Save the fine-tuned PRM model
        # torch.save(
        #     self.inner.module.LN.state_dict(), f"{args.weights_path}/LN_weights.pt"
        # )
        # self.inner.module.base_model.save_pretrained(f"{args.weights_path}/base_model")

        if args.baseline:
            # Baseline mode: only fine-tune the lower problem.
            lower_problem = self.problems[0]
            val_loss = np.mean(lower_loss) if lower_loss else best_loss
            wandb.log({
                "validation/loss": val_loss,
                "validation/best_loss": best_loss,
                "validation/step": step_count,
                "validation/learning_rate": current_lr,
            })

            # save the fine-tuned PRM model.
            lower_problem.module.base_model.save_pretrained(f"{args.weights_path}/qwen_math_prm")
            lower_problem.module.tokenizer.save_pretrained(f"{args.weights_path}/qwen_math_prm")

            print(f"Model saved. Val Loss: {val_loss:.4f}, Best: {best_loss:.4f}")
        
        else:
            # Non-baseline mode: save both upper and lower problems.
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

            wandb.log({
                "validation/loss": best_loss,
                "validation/step": step_count,
            })
            
        # return {"loss": 1}
        return {"loss": best_loss}
    

# upper_config = Config(type="darts", precision=args.precision, retain_graph=True)
# lower_config = Config(type="darts", precision=args.precision, unroll_steps=args.unroll_steps,
#                     gradient_accumulation=args.gradient_accumulation)
upper_config = Config(type="darts", retain_graph=True)
lower_config = Config(
    type='darts',
    unroll_steps=args.unroll_steps,
    gradient_accumulation=args.gradient_accumulation
)
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.save_every_iterations,
    # strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)

# Instantiate problems
problems = [Lower(name='lower', config=lower_config)]
dependencies = {}

if not args.baseline:
    problems.insert(0, Upper(name='upper', config=upper_config))
    dependencies = {problems[0]: [problems[1]]}
    print("Running META-LEARNING mode")
else:
    print("Running BASELINE mode")

# Run
engine = Engine(
    config=engine_config,
    problems=problems,
    dependencies={'l2u': {}, 'u2l': dependencies}
)
engine.run()

