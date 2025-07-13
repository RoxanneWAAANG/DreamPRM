# All code is original unless otherwise noted.
'''
PYTHONPATH=/workspace/DreamPRM \
python3 main.py \
  --train_json_file data/train_prm800k.json \
  --weights_path    outputs/qwen_math_prm_v2 \
  --batch_size      8 \
  --lr              5e-7 \
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
import torch.optim as optim
from model import *
from data import *
from utils import *
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
import wandb
# from transformers import AdamW
from torch.optim import AdamW, Adam
import numpy as np
import os


parser = argparse.ArgumentParser(description="DreamPRM")
parser.add_argument('--train_json_file', type=str)
parser.add_argument('--meta_json_file', type=str)
parser.add_argument('--weights_path', type=str)
parser.add_argument("--iteration_num", type=int, default=10000)
parser.add_argument("--save_every_iterations", type=int, default=1000)
parser.add_argument("--unroll_steps", type=int, default=5)
parser.add_argument("--gradiant_accumulation", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-7)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--scheduler_step_size", type=int, default=5000)
parser.add_argument("--scheduler_gamma", type=float, default=0.8)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--meta_lr", type=float, default=0.005)
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

# Handle string "None" passed as argument
if args.meta_json_file == "None":
    args.meta_json_file = None

# Validate arguments
if not args.baseline and args.meta_json_file is None:
    raise ValueError("--meta_json_file is required when not using --baseline")

print("Training Configuration:")
print("="*50)
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("="*50)

set_seed(args.seed)
domain_list = create_dataset_mapping(args.train_json_file)
print(domain_list)
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
    meta_json_file=args.meta_json_file if not args.baseline else None,
    train_batch_size= args.batch_size,
    meta_batch_size= args.batch_size,
    dataset_type=args.dataset_type
)
os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="DreamPRM-v0",
    name=f"baseline-bs{args.batch_size}-lr{args.lr}-steps{args.iteration_num}",
    config=vars(args),
    tags=["baseline", "qwen-math", "prm-training"]
)

device = torch.device(args.device)
if args.dataset_type == "qwen_math":
    # Use BCE for binary classification
    criterion = nn.BCELoss()
    criterion_meta = nn.BCELoss()
else:
    # Use MSE for regression tasks
    criterion = nn.MSELoss()
    criterion_meta = nn.MSELoss()

lower_weighted_loss = []
lower_loss = []
upper_loss = []
best_loss = float('inf')
step_count = 0
current_lr = args.lr


class Upper(ImplicitProblem):
    def forward(self, domain_strings, x):
        # torch.cuda.empty_cache()
        return self.module(domain_strings, x)

    def training_step(self, batch):
        # steps = [batch['1'], batch['2'], batch['3'], batch['4'], batch['5'],]
        global step_count
        numeric_keys = [k for k in batch.keys() if k.isdigit()]
        sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
        steps = [batch[key] for key in sorted_keys]
        labels = batch['labels'].to(device)

        mean_score = 0
        for i in steps:
            # Text-only input for QwenMath
            score = self.inner(
                i['input_ids'].to(device),
                i['attention_mask'].to(device),
                # i['pixel_values'].to(device),
                # i['image_grid_thw'].to(device)
            )
            mean_score += torch.log(score / (1 - score))
            # mean_score += torch.log(score / (1 - score + 1e-8))

        outputs = torch.sigmoid(mean_score / len(steps))
        loss = criterion_meta(outputs, labels)
        upper_loss.append(loss.item())
        # print(f"Pred: {outputs.item():.3f}, Label: {labels.item():.3f}, Loss: {loss.item():.3f}")
        if step_count % 50 == 0:  # Log every 50 steps
            print(f"[Meta Step {step_count}] Pred: {outputs.item():.3f}, Label: {labels.item():.3f}, Loss: {loss.item():.3f}")

        # torch.cuda.empty_cache()
        if len(upper_loss) == 10:
            mean_outer_loss = np.mean(upper_loss)
            wandb.log({
                "outer_loss": mean_outer_loss,
                "meta_step": step_count,
                "meta_pred_avg": outputs.item(),
                "meta_label_avg": labels.item(),
            })
            upper_loss.clear()

        return {"loss": loss}

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        domain_to_idx = {domain: idx for idx, domain in enumerate(domain_list)}
        meta_net = InstanceTable(domain_to_idx)
        # meta_net = DomainTable(domain_list)
        return meta_net

    def configure_optimizer(self):
        meta_optimizer = AdamW(
            self.module.parameters(),
            lr=args.meta_lr,
            weight_decay=args.weight_decay
        )
        return meta_optimizer


class Lower(ImplicitProblem):
    def forward(self, input_ids, attention_mask):   #, pixel_values, image_grid_thw):
        # torch.cuda.empty_cache()
        return self.module(input_ids, attention_mask)  #, pixel_values, image_grid_thw)

    def training_step(self, batch):
        global step_count, best_loss, current_lr
        step_count += 1

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # pixel_values = batch['pixel_values'].to(device)
        # image_grid_thw = batch['image_grid_thw'].to(device)
        labels = batch['label'].to(dtype=torch.float32).to(device)
        domain_strings = batch['dataset']
        labels = torch.clamp(labels, 0.0, 1.0)

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask )
                                #, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        outputs = outputs.to(dtype=torch.float32)

        if args.baseline or args.retrain:
            return criterion(outputs, labels)

        loss = criterion(outputs, labels)
        # weighted_loss = self.upper(domain_strings, loss)
        # weighted_loss = self.upper(domain_strings, loss.unsqueeze(0)).squeeze()
        weighted_loss = self.upper([domain_strings], loss.unsqueeze(0)).squeeze()
        print(f"Domain: {domain_strings}, Loss: {loss.item()}, Weighted Loss: {weighted_loss.item()}")

        lower_loss.append(loss.item())
        lower_weighted_loss.append(weighted_loss.item())

        if len(lower_loss) == 100:
            mean_inner_loss = np.mean(lower_loss)
            mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log({
                "inner_loss": mean_inner_loss,
                "inner_weighted_loss": mean_inner_weighted_loss, 
            })
            lower_loss.clear()
            lower_weighted_loss.clear()
        # torch.cuda.empty_cache()

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
#                     gradient_accumulation=args.gradiant_accumulation)
upper_config = Config(type="darts", retain_graph=True)
lower_config = Config(type="darts", unroll_steps=args.unroll_steps) 
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.save_every_iterations,
    # strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)

upper = Upper(name="upper", config=upper_config)
lower = Lower(name="lower", config=lower_config)

if args.baseline or args.retrain:
    problems = [lower]
    u2l, l2u = {}, {}
    print("Running in BASELINE mode - single model training")
else:
    problems = [upper, lower]
    u2l = {upper: [lower]}
    l2u = {lower: [upper]}
    print("Running in META-LEARNING mode - bilevel optimization")

dependencies = {"l2u": l2u, "u2l": u2l}

print(f"Starting training for {args.iteration_num:,} iterations...")
print(f"Will save every {args.save_every_iterations} iterations")
print(f"Estimated training time: {args.iteration_num * 0.5 / 60:.1f} minutes")

engine = ReweightingEngine(
    config=engine_config,
    problems=problems,
    dependencies=dependencies
)
engine.run()

