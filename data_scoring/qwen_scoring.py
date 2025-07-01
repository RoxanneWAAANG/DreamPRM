#!/usr/bin/env python3
"""
Dual Labeling Approach: Use PRM800k reasoning steps with both human hard labels 
and Qwen PRM soft labels for controlled comparison
"""

import json
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenPRMScorer:
    """Qwen PRM scorer following the exact Hugging Face example"""
    
    def __init__(self, model_path="Qwen/Qwen2.5-Math-PRM-7B", device="auto"):
        self.device = device
        logger.info(f"Loading Qwen PRM model: {model_path}")
        
        # EXACTLY following the original example
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        logger.info("Qwen PRM model loaded successfully")
    
    def make_step_rewards(self, logits, token_masks):
        """EXACTLY the same function from Hugging Face example"""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def score_step_sequence(self, problem: str, step_list: List[str]) -> List[float]:
        """
        Score a sequence of reasoning steps - EXACTLY following Hugging Face example
        
        Args:
            problem: Math problem statement
            step_list: List of reasoning steps to evaluate
            
        Returns:
            List of scores (0-1) for each step
        """
        # EXACTLY follow the original example format
        data = {
            "system": "Please reason step by step, and put your final answer within \\boxed{}.",
            "query": problem,
            "response": step_list
        }
        
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ]
        
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        input_ids = self.tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            token_masks = (input_ids == self.step_sep_id)
            step_rewards = self.make_step_rewards(outputs[0], token_masks)
        
        # Return the step rewards (should match the number of steps)
        return step_rewards[0] if step_rewards else []
    
    def score_single_step(self, problem: str, step_text: str) -> float:
        """
        Score a single step by treating it as a 1-step sequence
        """
        scores = self.score_step_sequence(problem, [step_text])
        return scores[0] if scores else 0.0

def process_prm800k_dual_labeling(input_file: str, output_file: str, 
                                 batch_size: int = 16, max_samples: int = None) -> Dict[str, Any]:
    """
    Process PRM800k with dual labeling: human hard + Qwen soft labels
    
    Args:
        input_file: Path to PRM800k jsonl file
        output_file: Path to save dual-labeled data
        batch_size: Batch size for Qwen PRM inference
        max_samples: Limit samples for testing (None for all)
    
    Returns:
        Dictionary with processing statistics and analysis
    """
    logger.info("Starting dual labeling process")
    
    # Load Qwen PRM scorer
    scorer = QwenPRMScorer()
    
    dual_labeled_data = []
    stats = {
        "total_samples": 0,
        "human_positive": 0,
        "human_negative": 0,
        "qwen_scores": [],
        "human_scores": [],
        "agreements": [],
        "disagreements": []
    }
    
    with open(input_file) as fin:
        for line_idx, line in enumerate(tqdm(fin, desc="Processing PRM800k")):
            if max_samples and line_idx >= max_samples:
                break
                
            ex = json.loads(line)
            problem = ex["question"]["problem"]
            ground_truth = ex["question"].get("ground_truth_answer", "")
            
            for sid, step_obj in enumerate(ex["label"]["steps"], start=1):
                comps = step_obj.get("completions", [])
                if not comps:
                    continue
                
                # Get human-chosen completion
                idx = step_obj.get("chosen_completion")
                if idx is None or not (0 <= idx < len(comps)):
                    # Pick highest-rated completion
                    numeric_ratings = [
                        c.get("rating") if isinstance(c.get("rating"), (int, float)) else 0
                        for c in comps
                    ]
                    idx = numeric_ratings.index(max(numeric_ratings))
                
                comp = comps[idx]
                step_text = comp.get("text", "").strip()
                human_rating = comp.get("rating")
                human_rating = human_rating if isinstance(human_rating, (int, float)) else 0
                
                # Human hard label (binary)
                human_hard_label = 1.0 if human_rating > 0 else 0.0
                
                # Qwen soft label (continuous)
                try:
                    qwen_soft_label = scorer.score_single_step(problem, step_text)
                except Exception as e:
                    logger.warning(f"Error scoring step {sid}: {e}")
                    qwen_soft_label = 0.0
                
                # Create dual-labeled sample
                sample = {
                    "id": len(dual_labeled_data) + 1,
                    "sid": sid,
                    "input": problem,
                    "add": step_text,
                    "ground_truth": ground_truth,
                    "dataset": "prm800k_dual",
                    
                    # Dual labels
                    "human_hard_label": human_hard_label,
                    "qwen_soft_label": qwen_soft_label,
                    "human_rating": human_rating,
                    
                    # For compatibility with existing training code
                    "accuracy": human_hard_label,  # Default to human
                    "score": human_rating,
                    "times": 1,
                    "image_path": "",
                    "source": "prm800k_dual_labeled"
                }
                
                dual_labeled_data.append(sample)
                
                # Collect statistics
                stats["total_samples"] += 1
                stats["human_scores"].append(human_hard_label)
                stats["qwen_scores"].append(qwen_soft_label)
                
                if human_hard_label == 1:
                    stats["human_positive"] += 1
                else:
                    stats["human_negative"] += 1
                
                # Analyze agreement (using 0.5 threshold for Qwen)
                qwen_binary = 1.0 if qwen_soft_label > 0.5 else 0.0
                if human_hard_label == qwen_binary:
                    stats["agreements"].append((human_hard_label, qwen_soft_label))
                else:
                    stats["disagreements"].append((human_hard_label, qwen_soft_label))
    
    # Save dual-labeled data
    with open(output_file, "w") as fout:
        json.dump(dual_labeled_data, fout, indent=2)
    
    # Calculate final statistics
    stats["agreement_rate"] = len(stats["agreements"]) / stats["total_samples"]
    stats["human_positive_rate"] = stats["human_positive"] / stats["total_samples"]
    stats["qwen_mean_score"] = np.mean(stats["qwen_scores"])
    stats["qwen_std_score"] = np.std(stats["qwen_scores"])
    
    logger.info(f"Dual labeling completed: {stats['total_samples']} samples")
    logger.info(f"Agreement rate: {stats['agreement_rate']:.3f}")
    logger.info(f"Human positive rate: {stats['human_positive_rate']:.3f}")
    logger.info(f"Qwen mean score: {stats['qwen_mean_score']:.3f} ± {stats['qwen_std_score']:.3f}")
    
    return stats

def analyze_label_differences(dual_labeled_data: List[Dict], stats: Dict) -> None:
    """Analyze differences between human and Qwen labels"""
    
    human_scores = stats["human_scores"]
    qwen_scores = stats["qwen_scores"]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Score distribution
    axes[0, 0].hist(qwen_scores, bins=50, alpha=0.7, label='Qwen Soft Labels')
    axes[0, 0].hist([h for h in human_scores], bins=2, alpha=0.7, label='Human Hard Labels')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distributions')
    axes[0, 0].legend()
    
    # Scatter plot
    axes[0, 1].scatter(human_scores, qwen_scores, alpha=0.5)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
    axes[0, 1].set_xlabel('Human Hard Labels')
    axes[0, 1].set_ylabel('Qwen Soft Labels')
    axes[0, 1].set_title('Human vs Qwen Scoring')
    axes[0, 1].legend()
    
    # Agreement analysis
    agreement_data = []
    for human, qwen in zip(human_scores, qwen_scores):
        qwen_binary = 1 if qwen > 0.5 else 0
        agreement_data.append(int(human == qwen_binary))
    
    axes[1, 0].bar(['Disagree', 'Agree'], [
        agreement_data.count(0), agreement_data.count(1)
    ])
    axes[1, 0].set_title('Agreement/Disagreement Counts')
    axes[1, 0].set_ylabel('Count')
    
    # Qwen scores for human positive vs negative
    pos_qwen = [q for h, q in zip(human_scores, qwen_scores) if h == 1]
    neg_qwen = [q for h, q in zip(human_scores, qwen_scores) if h == 0]
    
    axes[1, 1].boxplot([pos_qwen, neg_qwen], labels=['Human Positive', 'Human Negative'])
    axes[1, 1].set_ylabel('Qwen Soft Labels')
    axes[1, 1].set_title('Qwen Scores by Human Labels')
    
    plt.tight_layout()
    plt.savefig('dual_labeling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("\n=== DETAILED ANALYSIS ===")
    print(f"Total samples: {len(human_scores)}")
    print(f"Human positive: {sum(human_scores)} ({sum(human_scores)/len(human_scores)*100:.1f}%)")
    print(f"Qwen mean score: {np.mean(qwen_scores):.3f}")
    print(f"Agreement rate: {np.mean(agreement_data):.3f}")
    
    # Analyze disagreements
    disagreements = [(h, q) for h, q in zip(human_scores, qwen_scores) 
                    if h != (1 if q > 0.5 else 0)]
    
    human_pos_qwen_neg = [(h, q) for h, q in disagreements if h == 1 and q <= 0.5]
    human_neg_qwen_pos = [(h, q) for h, q in disagreements if h == 0 and q > 0.5]
    
    print(f"\nDisagreement Analysis:")
    print(f"Human positive, Qwen negative: {len(human_pos_qwen_neg)}")
    print(f"Human negative, Qwen positive: {len(human_neg_qwen_pos)}")
    
    if pos_qwen and neg_qwen:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(pos_qwen, neg_qwen)
        print(f"\nT-test between human pos/neg groups:")
        print(f"Qwen scores - Human positive: {np.mean(pos_qwen):.3f} ± {np.std(pos_qwen):.3f}")
        print(f"Qwen scores - Human negative: {np.mean(neg_qwen):.3f} ± {np.std(neg_qwen):.3f}")
        print(f"P-value: {p_value:.6f}")

def create_training_datasets(dual_labeled_data: List[Dict], output_dir: str) -> None:
    """Create separate training datasets for hard vs soft labeling experiments"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset 1: Human hard labels
    hard_label_data = []
    for item in dual_labeled_data:
        hard_item = item.copy()
        hard_item["accuracy"] = item["human_hard_label"]
        hard_item["score"] = item["human_hard_label"]
        hard_item["dataset"] = "prm800k_human_hard"
        hard_label_data.append(hard_item)
    
    # Dataset 2: Qwen soft labels
    soft_label_data = []
    for item in dual_labeled_data:
        soft_item = item.copy()
        soft_item["accuracy"] = item["qwen_soft_label"]
        soft_item["score"] = item["qwen_soft_label"]
        soft_item["dataset"] = "prm800k_qwen_soft"
        soft_label_data.append(soft_item)
    
    # Dataset 3: Mixed (for comparison)
    mixed_data = dual_labeled_data.copy()
    
    # Save all three datasets
    datasets = {
        "human_hard_labels": hard_label_data,
        "qwen_soft_labels": soft_label_data, 
        "dual_labels_mixed": mixed_data
    }
    
    for name, data in datasets.items():
        output_file = os.path.join(output_dir, f"prm800k_{name}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} samples to {output_file}")

def main():
    """Main execution function"""
    
    # Configuration
    config = {
        "input_file": "/workspace/data/dreamprm/prm800k/phase2_test.jsonl",
        "output_file": "/workspace/DreamPRM/data/prm800k_dual_labeled.json",
        "output_dir": "/workspace/DreamPRM/data/dual_labeling_experiments",
        "max_samples": 100,  # Set to None for full dataset
        "batch_size": 16
    }
    
    # Step 1: Process with dual labeling
    logger.info("Step 1: Processing PRM800k with dual labeling")
    stats = process_prm800k_dual_labeling(
        config["input_file"], 
        config["output_file"],
        max_samples=config["max_samples"]
    )
    
    # Step 2: Load and analyze
    logger.info("Step 2: Analyzing label differences")
    with open(config["output_file"], 'r') as f:
        dual_labeled_data = json.load(f)
    
    analyze_label_differences(dual_labeled_data, stats)
    
    # Step 3: Create training datasets
    logger.info("Step 3: Creating separate training datasets")
    create_training_datasets(dual_labeled_data, config["output_dir"])
    
    # Step 4: Print summary
    print("\n=== EXPERIMENT SETUP COMPLETE ===")
    print("You now have three datasets to experiment with:")
    print("1. prm800k_human_hard_labels.json - Original human binary labels")
    print("2. prm800k_qwen_soft_labels.json - Qwen continuous labels")  
    print("3. prm800k_dual_labels_mixed.json - Both labels for analysis")

if __name__ == "__main__":
    main()
