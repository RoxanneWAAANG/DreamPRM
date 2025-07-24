# from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=8,             # Rank for dimensionality reduction (higher = better performance but more compute)
#     lora_alpha=16,   # Scaling factor for LoRA weights
#     target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to (GPT example)
#     lora_dropout=0.1,  # Dropout probability for LoRA layers
#     bias="none"      # Whether to apply LoRA to biases ("none", "all", or "lora_only")
# )

class QwenVL_RM(nn.Module):
    def __init__(self, device, model_path="Qwen/Qwen2-VL-2B-Instruct"):
        super(QwenVL_RM, self).__init__()
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        # self.lora_model = get_peft_model(base_model, lora_config)
        # Linear layer mapping from vocabulary size to single scalar reward.
        self.LN = nn.Linear(self.base_model.config.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # Passes multimodal inputs through the base Qwen2-VL model to get logits.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values = pixel_values, image_grid_thw = image_grid_thw)
        # Passes multimodal inputs through the base Qwen2-VL model to get logits.
        # [:, -1, :]: Takes logits of the final token position.
        outputs = outputs.logits[:, -1, :].to(dtype=torch.float)
        # print(outputs)
        # Maps logits to scalar reward using linear layer.
        value_outputs = self.LN(outputs)
        # Applies sigmoid to get probability in [0,1] range.
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        # Removes dimension to return shape [batch_size] instead of [batch_size, 1].
        return value_outputs.squeeze(dim=1)


class Llava_RM(nn.Module):
    def __init__(self, device):
        super(Llava_RM, self).__init__()
        self.base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True
        ).to(0)
        # self.lora_model = get_peft_model(base_model, lora_config)
        self.LN = nn.Linear(self.base_model.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values = pixel_values, image_sizes = image_sizes)
        outputs = outputs.logits[:, -1, :].to(dtype=torch.float)
        # print(outputs)
        value_outputs = self.LN(outputs)
        value_outputs = self.sigmoid(value_outputs)
        # print(value_outputs)
        return value_outputs.squeeze(dim=1)


class QwenMath_RM(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.args = args
        self.device = device
        model_path = args.reward_model

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<pad>',
            pad_token_id=0,
            trust_remote_code=True
        )
        
        # Add classification head - map from hidden_size to 2 classes
        self.LN = nn.Linear(
            self.base_model.config.hidden_size, 
            3, 
            device=device, 
            dtype=torch.bfloat16
        )

        self.sigmoid = nn.Sigmoid()
        
        # Apply PEFT
        if hasattr(args, 'peft_rank') and args.peft_rank > 0:
            from peft import LoraConfig, get_peft_model, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.peft_rank,
                lora_alpha=getattr(args, 'lora_alpha', 32),
                lora_dropout=getattr(args, 'lora_dropout', 0.1),
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
                inference_mode=False,
            )
            self.base_model = get_peft_model(self.base_model, peft_config)

            # Enable gradients for LoRA layers
            self.base_model.enable_input_require_grads()

            # Make sure LoRA parameters require gradients
            for name, param in self.base_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            
            print("Using LoRA with Qwen2.5-Math-1.5B:")
            self.base_model.print_trainable_parameters()
        
        if getattr(args, 'freeze_base_model', False):
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Base model frozen, only training classification head")

        # Move to device
        self.to(device)

        # Ensure classification head always requires gradients
        for param in self.LN.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        Args:
          input_ids:      [batch_size, seq_len]
          attention_mask: [batch_size, seq_len]
        Returns:
          final_scores:   [batch_size]  (mean positive class probability over all steps)
        """
        batch_size = input_ids.size(0)
        
        # Get hidden states from the causal LM
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
        
        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]  # [B, T, hidden_size]
        del outputs # Free memory from outputs
        # print("Hidden states shape:", hidden_states.shape)
        
        # Apply classification head - keep in bfloat16 for efficiency
        sequence_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        last_hidden_states = hidden_states[batch_indices, sequence_lengths]  # [B, hidden_size]
        
        # Apply classification head
        logits = self.LN(last_hidden_states)  # [B, num_classes]

        # Method 1: Compute expected score
        class_scores = torch.tensor([0.0, 0.5, 1.0], device=logits.device, dtype=logits.dtype)
        # probs = torch.softmax(logits, dim=-1)  # [B, 3]
        # output = torch.sum(probs * class_scores, dim=-1)  # [B]
        output = torch.sum(logits * class_scores, dim=-1)  # [batch_size]
        return output  # [batch_size]
        
        # Method 2: Use max class probability
        # max_class = torch.argmax(logits, dim=-1)  # [batch_size]
        # return class_scores[max_class]
    

class InstanceTable(nn.Module):
    def __init__(self, domain_to_idx, eps=1e-8):
        """
        Args:
            domain_to_idx (dict):
                字符串 -> 整数索引 的映射，例如 {"domain_a": 0, "domain_b": 1}。
        """
        super(InstanceTable, self).__init__()
        self.domain_to_idx = domain_to_idx
        self.num_domains = len(domain_to_idx)

        # start from balanced importance
        init_val = 1.0
        self.raw_weights = nn.Parameter(torch.ones(self.num_domains) * init_val)  # 初始为1
        # self.relu = torch.nn.ReLU()
        self.eps = eps

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                每个样本对应的 domain 名称，长度与 x 的 batch_size 相同。
            x (torch.Tensor):
                形状为 (batch_size, 1)，表示每个样本一个数值。

        Returns:
            torch.Tensor:
                同形状 (batch_size, 1) 的张量，每个元素等于原输入乘以对应的 domain 权重。
        """
        # positive_weights = self.raw_weights
        w = torch.relu(self.raw_weights) + self.eps # (D,)
        # normalize so sum(w)=D or sum(w)=1
        w = w / w.sum() * self.num_domains  # (D,)

        # map domains → weights
        # idxes = [self.domain_to_idx[d] for d in domain_strings]
        # idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]
        idxs = torch.tensor(
            [self.domain_to_idx[d] for d in domain_strings],
            device=x.device,
            dtype=torch.long
        )   # (B,)

        domain_weights = w[idxs].unsqueeze(1)  # (B,1)
        # domain_weights = domain_weights.view(-1, 1)

        out = x * domain_weights
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
