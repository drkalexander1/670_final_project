#!/usr/bin/env python3
"""
Quick test of feature engineering with small sample.
Runs fast to get comparison results, then can run full version overnight.
"""
import pandas as pd
import numpy as np
from train import train_and_evaluate_cv
from evaluate import (
    plot_cv_results, 
    plot_model_comparison, 
    generate_summary_report
)
from data_loader import load_all_years
import os

print("="*60)
print("Quick Feature Engineering Test (Small Sample)")
print("Includes: Features + Count Prediction (Multi-task Learning)")
print("="*60)

# Step 1: Load transitions
print("\nStep 1: Loading transition pairs...")
transitions_file = 'transitions.parquet'
if not os.path.exists(transitions_file):
    raise FileNotFoundError(f"Transitions file not found: {transitions_file}")

transitions = pd.read_parquet(transitions_file)
print(f"Loaded {len(transitions):,} transition pairs")

# Use a sample for quick testing (first 10% of transitions)
print("\nUsing sample for quick testing (10% of transitions)...")
sample_size = max(1000, len(transitions) // 10)
transitions_sample = transitions.sample(n=min(sample_size, len(transitions)), 
                                        random_state=42).copy()
print(f"Testing with {len(transitions_sample):,} transition pairs")

# Step 2: Load minimal raw data for feature extraction
print("\nStep 2: Loading minimal raw data (1 file per year) for feature extraction...")
raw_data = load_all_years(max_files_per_year=1)  # Just 1 file per year for speed
print(f"Loaded raw data for {len(raw_data)} years")

# Step 3: Define model parameters
print("\nStep 3: Setting up model parameters...")
model_params = {
    'collaborative': {
        'alpha': 0.1,  # Smoothing parameter for co-occurrence scores
        'min_cooccurrence': 2  # Minimum co-occurrences to consider
    },
    'neural': {
        'embedding_dim': 32,  # Reduced for speed
        'hidden_dims': [64, 32],  # Smaller network
        'dropout': 0.3,
        'predict_count': True  # Enable count prediction (multi-task learning)
    }
}

# Step 4: Train WITHOUT features (baseline)
print("\n" + "="*60)
print("Training models WITHOUT additional features")
print("="*60)
results_no_features = train_and_evaluate_cv(
    transitions_sample,
    model_types=['baseline', 'collaborative', 'neural'],
    model_params=model_params,
    raw_data=None,  # No raw data needed when not using features
    use_features=False
)

# Step 5: Train WITH features
print("\n" + "="*60)
print("Training models WITH additional features")
print("="*60)
results_with_features = train_and_evaluate_cv(
    transitions_sample,
    model_types=['collaborative', 'neural'],  # Skip baseline (doesn't use features)
    model_params=model_params,
    raw_data=raw_data,  # Need raw data for feature extraction
    use_features=True
)

# Add baseline to with_features results (baseline doesn't use features, so use same results)
# This ensures baseline appears in graphs and reports for comparison
results_with_features['baseline'] = results_no_features['baseline']

# Step 6: Compare results
print("\n" + "="*60)
print("QUICK TEST RESULTS: With vs Without Features")
print("="*60)

comparison_results = {}

for model_name in ['collaborative', 'neural']:
    if model_name in results_no_features and model_name in results_with_features:
        no_feat = results_no_features[model_name]
        with_feat = results_with_features[model_name]
        
        comparison_results[model_name] = {
            'without': {
                'precision': no_feat['mean_precision@k'],
                'recall': no_feat['mean_recall@k'],
                'map': no_feat['mean_map@k']
            },
            'with': {
                'precision': with_feat['mean_precision@k'],
                'recall': with_feat['mean_recall@k'],
                'map': with_feat['mean_map@k']
            }
        }
        
        print(f"\n{model_name.upper()} Model:")
        print(f"  Without features:")
        print(f"    Precision@10: {no_feat['mean_precision@k']:.4f}")
        print(f"    Recall@10: {no_feat['mean_recall@k']:.4f}")
        print(f"    MAP@10: {no_feat['mean_map@k']:.4f}")
        print(f"  With features:")
        print(f"    Precision@10: {with_feat['mean_precision@k']:.4f}")
        print(f"    Recall@10: {with_feat['mean_recall@k']:.4f}")
        print(f"    MAP@10: {with_feat['mean_map@k']:.4f}")
        
        # Calculate improvement
        p_improvement = ((with_feat['mean_precision@k'] - no_feat['mean_precision@k']) / 
                        max(no_feat['mean_precision@k'], 0.001) * 100)
        r_improvement = ((with_feat['mean_recall@k'] - no_feat['mean_recall@k']) / 
                        max(no_feat['mean_recall@k'], 0.001) * 100)
        m_improvement = ((with_feat['mean_map@k'] - no_feat['mean_map@k']) / 
                        max(no_feat['mean_map@k'], 0.001) * 100)
        
        print(f"  Improvement:")
        print(f"    Precision: {p_improvement:+.2f}%")
        print(f"    Recall: {r_improvement:+.2f}%")
        print(f"    MAP: {m_improvement:+.2f}%")

# Step 7: Save comparison
print("\nStep 7: Saving comparison results...")
import json

comparison_file = 'results/quick_feature_comparison.json'
with open(comparison_file, 'w') as f:
    json.dump(comparison_results, f, indent=2)
print(f"Saved: {comparison_file}")

# Generate summary report
report_lines = []
report_lines.append("="*70)
report_lines.append("Quick Feature Engineering Test Results")
report_lines.append("="*70)
report_lines.append("")
report_lines.append("NOTE: This is a QUICK TEST with sample data.")
report_lines.append("Run full training overnight for complete results.")
report_lines.append("")
report_lines.append("Comparison (With vs Without Features):")
report_lines.append("")

for model_name, comp in comparison_results.items():
    report_lines.append(f"{model_name.upper()} Model:")
    report_lines.append("-"*70)
    report_lines.append(f"Without Features:")
    report_lines.append(f"  Precision@10: {comp['without']['precision']:.4f}")
    report_lines.append(f"  Recall@10: {comp['without']['recall']:.4f}")
    report_lines.append(f"  MAP@10: {comp['without']['map']:.4f}")
    report_lines.append(f"With Features:")
    report_lines.append(f"  Precision@10: {comp['with']['precision']:.4f}")
    report_lines.append(f"  Recall@10: {comp['with']['recall']:.4f}")
    report_lines.append(f"  MAP@10: {comp['with']['map']:.4f}")
    
    p_imp = ((comp['with']['precision'] - comp['without']['precision']) / 
             max(comp['without']['precision'], 0.001) * 100)
    report_lines.append(f"Improvement: {p_imp:+.2f}% precision")
    report_lines.append("")

report_lines.append("="*70)

report_text = '\n'.join(report_lines)
with open('results/quick_feature_comparison.txt', 'w') as f:
    f.write(report_text)
print(f"Saved: results/quick_feature_comparison.txt")

print("\n" + report_text)

print("\n" + "="*60)
print("Quick Test Complete!")
print("="*60)
print("\nNext step: Run full training overnight:")
print("  sbatch batch_process_years.sh")
print("\nOr manually:")
print("  python3 run_training_with_features.py")

