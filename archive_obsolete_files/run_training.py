#!/usr/bin/env python3
"""
Run the complete training pipeline for bird species prediction.
This script loads transitions, prepares CV splits, trains models, and evaluates them.
"""
import pandas as pd
import numpy as np
from train import train_and_evaluate_cv
from evaluate import (
    plot_cv_results, 
    plot_model_comparison,
    plot_count_prediction_results,
    plot_count_prediction_comparison,
    generate_summary_report
)
import os

print("="*60)
print("Bird Species Prediction - Training Pipeline")
print("Includes: Count Prediction (Multi-task Learning)")
print("="*60)

# Step 1: Load transitions
print("\nStep 1: Loading transition pairs...")
transitions_file = 'transitions.parquet'
if not os.path.exists(transitions_file):
    raise FileNotFoundError(f"Transitions file not found: {transitions_file}")

transitions = pd.read_parquet(transitions_file)
print(f"Loaded {len(transitions):,} transition pairs")
print(f"Year transitions: {sorted(transitions[['year_from', 'year_to']].drop_duplicates().values.tolist())}")

# Step 2: Define model parameters
print("\nStep 2: Setting up model parameters...")
model_params = {
    'collaborative': {
        'alpha': 0.1,  # Smoothing parameter for co-occurrence scores
        'min_cooccurrence': 2  # Minimum co-occurrences to consider
    },
    'neural': {
        'embedding_dim': 64,
        'hidden_dims': [128, 64],
        'dropout': 0.3,
        'predict_count': True  # Enable count prediction (multi-task learning)
    }
}

# Step 3: Train and evaluate models
print("\nStep 3: Training models with time-series cross-validation...")
print("This may take a while depending on data size...")
print("Note: Training WITHOUT additional features (use run_training_with_features.py for features)")
print("Note: Neural network includes count prediction (multi-task learning)\n")

results = train_and_evaluate_cv(
    transitions,
    model_types=['baseline', 'collaborative', 'neural'],
    model_params=model_params,
    use_features=False  # Set to True and provide raw_data to use features
)

# Step 4: Generate visualizations
print("\nStep 4: Generating visualizations...")
try:
    plot_cv_results(results, output_file='cv_results.png')
    print("  Saved: cv_results.png")
except Exception as e:
    print(f"  Warning: Could not create cv_results.png: {e}")

try:
    plot_model_comparison(results, output_file='model_comparison.png')
    print("  Saved: model_comparison.png")
except Exception as e:
    print(f"  Warning: Could not create model_comparison.png: {e}")

try:
    plot_count_prediction_results(results, output_file='count_prediction_results.png')
    print("  Saved: count_prediction_results.png")
except Exception as e:
    print(f"  Warning: Could not create count_prediction_results.png: {e}")

try:
    plot_count_prediction_comparison(results, output_file='count_prediction_comparison.png')
    print("  Saved: count_prediction_comparison.png")
except Exception as e:
    print(f"  Warning: Could not create count_prediction_comparison.png: {e}")

# Step 5: Generate summary report
print("\nStep 5: Generating summary report...")
report = generate_summary_report(results, output_file='evaluation_report.txt')
print("\n" + report)

# Step 6: Save results
print("\nStep 6: Saving results...")
import json

# Convert numpy types to native Python types for JSON
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

results_serializable = convert_to_serializable(results)
with open('training_results.json', 'w') as f:
    json.dump(results_serializable, f, indent=2)
print("  Saved: training_results.json")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print("\nGenerated files:")
print("  - cv_results.png (CV metrics across folds)")
print("  - model_comparison.png (Model comparison chart)")
print("  - count_prediction_results.png (Count prediction metrics)")
print("  - count_prediction_comparison.png (Count prediction comparison)")
print("  - evaluation_report.txt (Text summary)")
print("  - training_results.json (Detailed results)")

