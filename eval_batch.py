#!/usr/bin/env python3
"""
Batch evaluation script for DreamPRM model
"""

import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from model import QwenMath_RM
from data import MyDataset_QwenMath, custom_collate_fn
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DreamPRM model")
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model directory')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test JSON file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    return parser.parse_args()

def load_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    model = QwenMath_RM(device, model_path)
    model.eval()
    return model

def load_test_data(test_file, model_path, batch_size):
    """Load test data and create dataloader"""
    print(f"Loading test data from {test_file}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create dataset
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_dataset = MyDataset_QwenMath(test_data, tokenizer)
    
    # Custom collate function for batching
    def collate_fn(batch):
        from torch.nn.utils.rnn import pad_sequence
        
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['label'] for item in batch]
        datasets = [item['dataset'] for item in batch]
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels,
            'dataset': datasets
        }
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return test_dataloader, test_data

def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate the model and return predictions and metrics"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_datasets = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            datasets = batch['dataset']
            
            # Get model predictions
            outputs = model(input_ids, attention_mask)
            probabilities = outputs.cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())
            all_datasets.extend(datasets)
    
    return all_predictions, all_probabilities, all_labels, all_datasets

def calculate_metrics(predictions, probabilities, labels):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float('nan')  # In case all labels are the same class
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def analyze_by_dataset(predictions, probabilities, labels, datasets):
    """Analyze results by dataset type"""
    unique_datasets = list(set(datasets))
    dataset_results = {}
    
    for dataset in unique_datasets:
        # Get indices for this dataset
        indices = [i for i, d in enumerate(datasets) if d == dataset]
        
        if len(indices) == 0:
            continue
            
        dataset_preds = [predictions[i] for i in indices]
        dataset_probs = [probabilities[i] for i in indices]
        dataset_labels = [labels[i] for i in indices]
        
        metrics = calculate_metrics(dataset_preds, dataset_probs, dataset_labels)
        metrics['count'] = len(indices)
        dataset_results[dataset] = metrics
    
    return dataset_results

def save_detailed_results(test_data, predictions, probabilities, labels, output_dir):
    """Save detailed results for analysis"""
    results = []
    for i, item in enumerate(test_data):
        result = {
            'id': item['id'],
            'input': item['input'][:100] + '...' if len(item['input']) > 100 else item['input'],
            'add': item['add'][:100] + '...' if len(item['add']) > 100 else item['add'],
            'true_label': labels[i],
            'predicted_label': predictions[i],
            'probability': probabilities[i],
            'correct': predictions[i] == labels[i],
            'dataset': item.get('dataset', 'unknown')
        }
        results.append(result)
    
    # Save as CSV
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    return df

def print_results(overall_metrics, dataset_metrics):
    """Print evaluation results"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall:    {overall_metrics['recall']:.4f}")
    print(f"F1 Score:  {overall_metrics['f1']:.4f}")
    print(f"AUC:       {overall_metrics['auc']:.4f}")
    
    print(f"\nResults by Dataset:")
    print("-" * 50)
    for dataset, metrics in dataset_metrics.items():
        print(f"{dataset} (n={metrics['count']}):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    
    # Load test data
    test_dataloader, test_data = load_test_data(args.test_file, args.model_path, args.batch_size)
    
    # Run evaluation
    predictions, probabilities, labels, datasets = evaluate_model(
        model, test_dataloader, device, args.threshold
    )
    
    # Calculate metrics
    overall_metrics = calculate_metrics(predictions, probabilities, labels)
    dataset_metrics = analyze_by_dataset(predictions, probabilities, labels, datasets)
    
    # Print results
    print_results(overall_metrics, dataset_metrics)
    
    # Save detailed results
    detailed_df = save_detailed_results(test_data, predictions, probabilities, labels, args.output_dir)
    
    # Save summary metrics
    summary = {
        'overall_metrics': overall_metrics,
        'dataset_metrics': dataset_metrics,
        'total_samples': len(predictions),
        'threshold': args.threshold
    }
    
    summary_file = os.path.join(args.output_dir, 'summary_metrics.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary metrics saved to {summary_file}")
    
    # Show some example predictions
    print(f"\nExample Predictions:")
    print("-" * 50)
    for i in range(min(5, len(test_data))):
        correct_mark = "✓" if predictions[i] == labels[i] else "✗"
        print(f"{correct_mark} ID {test_data[i]['id']}: True={labels[i]}, Pred={predictions[i]}, Prob={probabilities[i]:.3f}")
        print(f"   Text: {test_data[i]['add'][:80]}...")
        print()

if __name__ == "__main__":
    main()