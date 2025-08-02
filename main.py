# All code is original unless otherwise noted.
'''
python3 main.py \
  --train_json_file data/train_prm800k.json \
  --meta_json_file data/meta_aime.json \
  --weights_path outputs/qwen_math_prm_v0 \
  --batch_size 1 \
  --gradient_accumulation 8 \
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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import wandb

from peft import LoraConfig, get_peft_model, TaskType
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
criterion = nn.MSELoss(reduction="none")
criterion_meta = nn.MSELoss()

# Tracking
lower_weighted_loss, lower_loss, upper_loss = [], [], []
best_loss = float('inf')

# wandb init
wandb.init(
    project="DreamPRM-v0",
    name=f"meta-bs{args.batch_size}-lr{args.lr}",
    config=vars(args)
)

# === Globals ===
step_count = 0

# ---------------------------------
# Upper Problem (Instance Reweighting)
# ---------------------------------
class Upper(ImplicitProblem):
    def forward(self, instance_strings, x):
        # torch.cuda.empty_cache()
        return self.module(instance_strings, x)

    def parameters(self):
        """
        Yield only the trainable (i.e., LoRA) parameters.
        """
        # return self.module.parameters()
        return (p for p in self.module.parameters() if p.requires_grad)

    def training_step(self, batch):
        # steps = [batch['1'], batch['2'], batch['3'], batch['4'], batch['5'],]
        numeric_keys = [k for k in batch.keys() if k.isdigit()]
        sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
        steps = [batch[key] for key in sorted_keys]
        labels = batch['labels'].to(device)

        mean_score = 0
        for i in steps:
            score = self.lower(
                i['input_ids'].to(device),
                i['attention_mask'].to(device),
            )
            # mean_score += torch.log(score / (1 - score))
            mean_score += score
            
        outputs = torch.sigmoid(mean_score / len(steps))
        loss = criterion_meta(outputs, labels)
        upper_loss.append(loss.item())

        # torch.cuda.empty_cache()
        if len(upper_loss) == len(meta_dataloader):
            mean_outer_loss = np.mean(upper_loss)
            wandb.log({"outer_loss": mean_outer_loss})
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


# ---------------------------------
# Lower Problem: PRM Fine-tuning + LoRA
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

    def parameters(self):
        return (p for p in self.module.parameters() if p.requires_grad)

    def training_step(self, batch):
        global step_count
        step_count += 1

        # ---- 1) forward pass ----
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)

        step_scores = self.forward(
            input_ids=ids, 
            attention_mask=mask
        )

        loss = criterion(step_scores, labels)

        # ---- 2) meta-reweight via upper ----
        # shape it for InstanceTable: [1,1,B] or similar
        # weighted_loss = self.upper.forward(batch['dataset'], lt).squeeze()
        weighted_loss = self.upper( # [B,1] → squeeze→[B]
            batch['dataset'],
            loss    # [B,1]
        )
        lower_loss.append(loss.item())
        lower_weighted_loss.append(weighted_loss.item())

        # 4) logging
        if len(lower_loss) == 100:
            mean_inner_loss = np.mean(lower_loss)
            mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log(
                {
                    "inner_loss": np.mean(lower_loss),
                    "inner_weighted_loss": np.mean(lower_weighted_loss)
                }
            )
            lower_loss.clear()
            lower_weighted_loss.clear()
        # torch.cuda.empty_cache()

        return weighted_loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        # return QwenMath_RM(device, args).base_model.train()
        return QwenMath_RM(device, args)

    def configure_optimizer(self):
        optimizer = optim.AdamW(
            self.module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
        return scheduler


upper_config = Config(
    type="darts", 
    precision=args.precision, 
    # retain_graph=True, 
    retain_graph=False,
    gradient_clipping=1.0, 
    gradient_accumulation=args.gradient_accumulation,
)
lower_config = Config(
    type="darts", 
    precision=args.precision, 
    unroll_steps=args.unroll_steps, 
    gradient_accumulation=args.gradient_accumulation,
    gradient_clipping=1.0
)

engine_config = EngineConfig(
    train_iters=args.iteration_num * args.max_epoch,
    valid_step=args.save_every_iterations,
    # strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)

# Instantiate problems
upper = Upper(name="upper", config=upper_config)
lower = Lower(name="lower", config=lower_config)
problems = [upper, lower]

# Define bidirectional dependencies
dependencies = {
    "l2u": {lower: [upper]},    # Lower depends on Upper (since Lower calls self.upper)
    "u2l": {upper: [lower]} # Upper depends on Lower (bilevel structure)
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
