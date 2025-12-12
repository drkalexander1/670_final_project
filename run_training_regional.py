#!/usr/bin/env python3
"""
Run training with feature engineering enabled on regional (Northeast) data.
Compares models with and without additional features for regional subset.
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
import sys

print("="*60)
print("Bird Species Prediction - Regional Training (Northeast)")
print("Training WITH Features on Northeast Regional Data")
print("Includes: Features + Count Prediction (Multi-task Learning)")
print("="*60)
sys.stdout.flush()

# Step 1: Load regional transitions
print("\nStep 1: Loading regional transition pairs...")
sys.stdout.flush()
transitions_file = 'transitions_northeast.parquet'
if not os.path.exists(transitions_file):
    raise FileNotFoundError(
        f"Regional transitions file not found: {transitions_file}\n"
        f"Please run extract_regional.py first to create regional data."
    )

transitions = pd.read_parquet(transitions_file)
print(f"Loaded {len(transitions):,} regional transition pairs")
sys.stdout.flush()

# Step 2: We'll load birder_species data lazily per CV fold to save memory
# Using pre-aggregated birder_species_northeast parquet files for fast feature extraction
print("\nStep 2: birder_species_northeast data will be loaded lazily per CV fold")
print("        Using pre-aggregated files instead of raw data for faster feature extraction")
sys.stdout.flush()
raw_data = None  # Will be loaded on-demand per fold

# Step 3: Define model parameters
print("\nStep 3: Setting up model parameters...")
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

# Step 4: Train WITHOUT features (baseline)
print("\n" + "="*60)
print("Training models WITHOUT additional features (Regional)")
print("="*60)
sys.stdout.flush()
results_no_features = train_and_evaluate_cv(
    transitions,
    model_types=['baseline', 'collaborative', 'neural'],
    model_params=model_params,
    use_features=False
)

# Step 5: Train WITH features (using lazy loading)
print("\n" + "="*60)
print("Training models WITH additional features (Regional)")
print("="*60)
print("Using lazy birder_species data loading - will load only years needed per CV fold")
sys.stdout.flush()

# Create a data loader function for lazy loading birder_species_northeast files
def load_birder_species_data(year: int):
    """Load birder_species_northeast file for a single year - called on-demand per fold"""
    print(f"    Loading birder_species_{year}_northeast.parquet...")
    birder_species_file = f"birder_species_{year}_northeast.parquet"
    if not os.path.exists(birder_species_file):
        raise FileNotFoundError(
            f"birder_species file not found: {birder_species_file}\n"
            f"Please run extract_regional.py first to create regional birder_species files."
        )
    df = pd.read_parquet(birder_species_file)
    print(f"    Loaded {len(df):,} birder-year pairs for year {year}")
    return df

results_with_features = train_and_evaluate_cv(
    transitions,
    model_types=['collaborative', 'neural'],  # Skip baseline (doesn't use features)
    model_params=model_params,
    raw_data=None,  # Don't pre-load all data
    use_features=True,
    data_loader_func=load_birder_species_data  # Use birder_species loading
)

# Add baseline to with_features results (baseline doesn't use features, so use same results)
# This ensures baseline appears in graphs and reports for comparison
results_with_features['baseline'] = results_no_features['baseline']

# Step 6: Compare results
print("\n" + "="*60)
print("Comparison: With vs Without Features (Regional)")
print("="*60)

for model_name in ['collaborative', 'neural']:
    if model_name in results_no_features and model_name in results_with_features:
        print(f"\n{model_name.upper()} Model:")
        print(f"  Without features:")
        print(f"    Precision@10: {results_no_features[model_name]['mean_precision@k']:.4f}")
        print(f"    Recall@10: {results_no_features[model_name]['mean_recall@k']:.4f}")
        print(f"    MAP@10: {results_no_features[model_name]['mean_map@k']:.4f}")
        print(f"  With features:")
        print(f"    Precision@10: {results_with_features[model_name]['mean_precision@k']:.4f}")
        print(f"    Recall@10: {results_with_features[model_name]['mean_recall@k']:.4f}")
        print(f"    MAP@10: {results_with_features[model_name]['mean_map@k']:.4f}")
        
        # Calculate improvement
        p_improvement = ((results_with_features[model_name]['mean_precision@k'] - 
                         results_no_features[model_name]['mean_precision@k']) / 
                        results_no_features[model_name]['mean_precision@k'] * 100)
        print(f"  Improvement: {p_improvement:+.2f}% precision")

# Step 7: Generate visualizations
print("\nStep 7: Generating visualizations...")
try:
    plot_cv_results(results_with_features, output_file='results/cv_results_regional.png')
    print("  Saved: results/cv_results_regional.png")
except Exception as e:
    print(f"  Warning: Could not create cv_results_regional.png: {e}")

try:
    plot_model_comparison(results_with_features, output_file='results/model_comparison_regional.png')
    print("  Saved: results/model_comparison_regional.png")
except Exception as e:
    print(f"  Warning: Could not create model_comparison_regional.png: {e}")

try:
    plot_count_prediction_results(results_with_features, output_file='results/count_prediction_regional.png')
    print("  Saved: results/count_prediction_regional.png")
except Exception as e:
    print(f"  Warning: Could not create count_prediction_regional.png: {e}")

try:
    plot_count_prediction_comparison(results_with_features, output_file='results/count_prediction_comparison_regional.png')
    print("  Saved: results/count_prediction_comparison_regional.png")
except Exception as e:
    print(f"  Warning: Could not create count_prediction_comparison_regional.png: {e}")

# Step 8: Generate summary report
print("\nStep 8: Generating summary report...")
report = generate_summary_report(results_with_features, output_file='results/evaluation_report_regional.txt')
print("\n" + report)

print("\n" + "="*60)
print("Regional Training Complete!")
print("="*60)
print("\nResults saved:")
print("  - results/evaluation_report_regional.txt")
print("  - results/cv_results_regional.png")
print("  - results/model_comparison_regional.png")
print("  - results/count_prediction_regional.png")
print("  - results/count_prediction_comparison_regional.png")



