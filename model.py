# from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn

if not hasattr(nn, "RMSNorm"):
    # from transformers.models.qwen2.modeling_qwen2 import RMSNorm 
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm
    nn.RMSNorm = RMSNorm

class QwenMath_RM(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.args = args
        self.device = device

        # self.base_model = AutoModelForCausalLM.from_pretrained(
        self.base_model = AutoModel.from_pretrained(
            args.reward_model, 
            # device_map=device, 
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True,
            # low_cpu_mem_usage=True,
        )

        # ----- LoRA: Only wrap if peft_rank > 0 -----
        if args.peft_rank > 0:
            lora_conf = LoraConfig(
                r=args.peft_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["q_proj", "v_proj"]
            )
            self.base_model = get_peft_model(self.base_model, lora_conf)
            # self.base_model.train()  # Ensure training mode
            # for name, param in self.base_model.named_parameters():
            #     if "lora_" in name:
            #         param.requires_grad = True
        #-----------------------------------#
        
        # Add classification head - map from hidden_size to 2 classes
        self.LN = nn.Linear(
            self.base_model.config.hidden_size, 
            1, 
            dtype=torch.bfloat16,
        )

        self.softmax = nn.Softmax(dim=-1)
        # Move to device
        self.to(device)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        Args:
          input_ids:      [batch_size, seq_len]
          attention_mask: [batch_size, seq_len]
        Returns:
          final_scores:   [batch_size]  (mean positive class probability over all steps)
        """
        # Get hidden states from the causal LM
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True,
            # use_cache=False
        )
        
        # Apply classification head
        logits = self.LN(outputs.last_hidden_state[:, -1, :])  # [B, 3]

        return logits.squeeze(-1).to(dtype=torch.float32)
    

class InstanceTable(nn.Module):
    def __init__(self, instance_to_idx, eps=1e-8):
        """
        Args:
            instance_to_idx (dict):
                字符串 -> 整数索引 的映射，例如 {"domain_a": 0, "domain_b": 1}。
        """
        super().__init__()
        self.instance_to_idx = instance_to_idx
        self.num_instance = len(instance_to_idx)

        # self.raw_weights = nn.Parameter(
        #     torch.zeros(self.num_instance)
        # )  # 初始为1
        self.raw_weights = nn.Parameter(
            torch.ones(self.num_instance, dtype=torch.float32)
        )

        # self.relu = torch.nn.ReLU()
        # self.eps = eps  # Small value to avoid division by zero

    def forward(self, instance_strings, x):
        """
        Args:
            instance_strings (list[str] or tuple[str]):
                每个样本对应的 domain 名称，长度与 x 的 batch_size 相同。
            x (torch.Tensor):
                形状为 (batch_size, 1)，表示每个样本一个数值。

        Returns:
            torch.Tensor:
                同形状 (batch_size, 1) 的张量，每个元素等于原输入乘以对应的 domain 权重。
        """
        # positive_weights = self.raw_weights
        # Apply softplus to ensure weights are positive
        positive_weights = torch.nn.functional.softplus(self.raw_weights)

        # Normalize weights by their mean to maintain scale
        normalized_weights = positive_weights / positive_weights.mean() + 1e-8

        idxes = [self.instance_to_idx[d] for d in instance_strings]
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        instance_weights = normalized_weights[idxes]

        instance_weights = instance_weights.view(-1, 1)

        out = x * instance_weights
        return out


class DomainTable(nn.Module):
    def __init__(self, domain_to_idx):
        """
        Args:
            domain_to_idx (dict):
                Mapping from domain strings to integer indices, e.g., {"domain_a": 0, "domain_b": 1}.
        """
        super(DomainTable, self).__init__()
        self.domain_to_idx = domain_to_idx  # Maps domain names (like "AI2D", "M3CoT") to indices.
        self.num_domains = len(domain_to_idx)   # Number of unique domains.

        # Creates learnable parameters for domain weights,
        # (initialized to zero, will be optimized during bi-level optimization).
        self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                Domain names for each sample in the batch. Length should match x's batch_size.
            x (torch.Tensor):
                Input tensor of shape (batch_size, 1), containing a single value per sample.

        Returns:
            torch.Tensor:
                Output tensor of same shape (batch_size, 1), where each element is the original input
                multiplied by its corresponding domain weight.
        """
        # Apply softplus activation to ensure weights are positive.
        # Softplus(x) = log(1 + exp(x)) ensures output > 0.
        positive_weights = torch.nn.functional.softplus(self.raw_weights)

        # Normalize weights by their mean to maintain scale.
        # Ensures the average weight remains around 1.0.
        # This stabilizes training and makes weights interpretable.
        mean_weights = positive_weights.mean()
        normalized_weights = positive_weights / mean_weights

        # Convert domain strings to indices matching batch order.
        # Maps domain names to their corresponding indices.
        idxes = [self.domain_to_idx[d] for d in domain_strings]
        # Creates tensor of indices for batch lookup.
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        # Retrieve domain weights for each sample in the batch [batch_size].
        # Uses advanced indexing to get weight for each sample's domain.
        domain_weights = normalized_weights[idxes]

        # Reshape weights to match input tensor dimensions [batch_size, 1].
        domain_weights = domain_weights.view(-1, 1)

        # Element-wise multiplication: each input value multiplied by its domain weight.
        # Implements the domain reweighting in the lower-level optimization.
        out = x * domain_weights
        return out
