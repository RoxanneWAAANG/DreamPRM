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

        torch.cuda.empty_cache()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            args.reward_model, 
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            device_map=device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # # ----- LoRA: Only wrap if peft_rank > 0 -----
        # if args.peft_rank > 0:
        #     lora_conf = LoraConfig(
        #         r=args.peft_rank,
        #         lora_alpha=args.lora_alpha,
        #         lora_dropout=args.lora_dropout,
        #         bias='none',
        #         task_type=TaskType.FEATURE_EXTRACTION,
        #         target_modules=["q_proj", "v_proj"]
        #     )
        #     self.base_model = get_peft_model(self.base_model, lora_conf)
        #     # self.base_model.train()  # Ensure training mode
        #     # for name, param in self.base_model.named_parameters():
        #     #     if "lora_" in name:
        #     #         param.requires_grad = True
        # #-----------------------------------#
        
        # Add classification head - map from hidden_size to 2 classes
        self.LN = nn.Linear(
            self.base_model.config.vocab_size, 
            1, 
            dtype=self.base_model.dtype,
        )

        self.sigmoid = nn.Sigmoid()
        
        # # Move to device
        # self.to(device)

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
        )   # [batch_size, seq_len, hidden_size]

        # Apply classification head
        logits = self.LN(outputs.logits[:, -1, :])  # [batch_size, 1]
        # Apply sigmoid to get probabilities
        logits = self.sigmoid(logits)  # [batch_size, 1]

        return logits.squeeze(-1)   # [batch_size]
    

class InstanceTable(nn.Module):
    def __init__(self, instance_to_idx):
        """
        Args:
            instance_to_idx (dict):
                字符串 -> 整数索引 的映射，例如 {"domain_a": 0, "domain_b": 1}。
        """
        super(InstanceTable, self).__init__()
        self.instance_to_idx = instance_to_idx
        self.num_instance = len(instance_to_idx)

        self.raw_weights = nn.Parameter(
            torch.zeros(self.num_instance)
        )  # 初始为1

        # self.relu = torch.nn.ReLU()

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
        positive_weights = self.raw_weights
        
        # Convert instance strings to indices matching batch order
        idxes = [self.instance_to_idx[d] for d in instance_strings]
        # Create tensor of indices for batch lookup
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        # Retrieve instance weights for each sample in the batch [batch_size].
        instance_weights = positive_weights[idxes]

        # Reshape weights to match input tensor dimensions [batch_size, 1].
        instance_weights = instance_weights.view(-1, 1)

        # Element-wise multiplication: each input value multiplied by its instance weight.
        out = x * instance_weights
        return out

