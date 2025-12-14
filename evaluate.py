"""
Evaluation and visualization utilities for model predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from train import evaluate_model, precision_at_k, recall_at_k, mean_average_precision, coverage


def plot_cv_results(results: Dict, output_file: str = None):
    """
    Plot cross-validation results across folds.
    
    Args:
        results: Dictionary with model results from train_and_evaluate_cv
        output_file: Optional path to save figure
    """
    model_names = list(results.keys())
    metrics = ['precision@k', 'recall@k', 'map@k', 'coverage']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Extract fold results for each model
        fold_data = []
        for model_name in model_names:
            fold_results = results[model_name]['fold_results']
            values = [r[metric] for r in fold_results]
            fold_data.append(values)
        
        # Create box plot
        bp = ax.boxplot(fold_data, labels=model_names, patch_artist=True)
        ax.set_title(metric.replace('@', '@').title())
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Color boxes (extend colors if more than 3 models)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_model_comparison(results: Dict, output_file: str = None):
    """
    Create bar chart comparing mean metrics across models.
    
    Args:
        results: Dictionary with model results
        output_file: Optional path to save figure
    """
    model_names = list(results.keys())
    metrics = ['mean_precision@k', 'mean_recall@k', 'mean_map@k', 'mean_coverage']
    metric_labels = ['Precision@10', 'Recall@10', 'MAP@10', 'Coverage']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[model][metric] for model in model_names]
        offset = (i - len(metrics)/2) * width + width/2
        ax.bar(x + offset, values, width, label=label)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Mean Metrics Across CV Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_cv_results_full(results: Dict, output_file: str = None):
    """
    Plot cross-validation results across folds for full evaluation metrics.
    
    Args:
        results: Dictionary with model results from train_and_evaluate_cv
        output_file: Optional path to save figure
    """
    model_names = list(results.keys())
    metrics = ['precision_full', 'recall_full', 'map_full']
    metric_labels = ['Precision (Full)', 'Recall (Full)', 'MAP (Full)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Extract fold results for each model
        fold_data = []
        for model_name in model_names:
            fold_results = results[model_name]['fold_results']
            values = [r.get(metric, 0) for r in fold_results if metric in r]
            if values:  # Only add if metric exists
                fold_data.append(values)
            else:
                fold_data.append([])
        
        # Filter out empty lists and corresponding model names
        valid_data = [(data, name) for data, name in zip(fold_data, model_names) if len(data) > 0]
        if valid_data:
            valid_fold_data, valid_model_names = zip(*valid_data)
            # Create box plot
            bp = ax.boxplot(valid_fold_data, labels=valid_model_names, patch_artist=True)
            ax.set_title(label)
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            
            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_model_comparison_full(results: Dict, output_file: str = None):
    """
    Create bar chart comparing mean full evaluation metrics across models.
    
    Args:
        results: Dictionary with model results
        output_file: Optional path to save figure
    """
    model_names = list(results.keys())
    metrics = ['mean_precision_full', 'mean_recall_full', 'mean_map_full']
    metric_labels = ['Precision (Full)', 'Recall (Full)', 'MAP (Full)']
    
    # Filter to only models that have full metrics
    model_names = [m for m in model_names if 'mean_precision_full' in results[m]]
    
    if len(model_names) == 0:
        print("No full evaluation metrics found in results.")
        return
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[model].get(metric, 0) for model in model_names]
        offset = (i - len(metrics)/2) * width + width/2
        bars = ax.bar(x + offset, values, width, label=label)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Full Evaluation Metrics (All Species People See)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_count_prediction_results(results: Dict, output_file: str = None):
    """
    Plot regression metrics for count prediction (MAE, RMSE, Correlation).
    
    Args:
        results: Dictionary with model results from train_and_evaluate_cv
        output_file: Optional path to save figure
    """
    # Filter models that have count prediction metrics
    models_with_count = {}
    for model_name, model_results in results.items():
        if 'mean_count_mae' in model_results:
            models_with_count[model_name] = model_results
    
    if len(models_with_count) == 0:
        print("No count prediction metrics found in results.")
        return
    
    model_names = list(models_with_count.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. Box plot for MAE across folds
    ax = axes[0]
    fold_data_mae = []
    for model_name in model_names:
        fold_results = models_with_count[model_name]['fold_results']
        values = [r['count_mae'] for r in fold_results if 'count_mae' in r]
        fold_data_mae.append(values)
    
    if fold_data_mae:
        bp = ax.boxplot(fold_data_mae, labels=model_names, patch_artist=True)
        ax.set_title('Count Prediction MAE (Mean Absolute Error)')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    # 2. Box plot for RMSE across folds
    ax = axes[1]
    fold_data_rmse = []
    for model_name in model_names:
        fold_results = models_with_count[model_name]['fold_results']
        values = [r['count_rmse'] for r in fold_results if 'count_rmse' in r]
        fold_data_rmse.append(values)
    
    if fold_data_rmse:
        bp = ax.boxplot(fold_data_rmse, labels=model_names, patch_artist=True)
        ax.set_title('Count Prediction RMSE (Root Mean Squared Error)')
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    # 3. Box plot for Correlation across folds
    ax = axes[2]
    fold_data_corr = []
    for model_name in model_names:
        fold_results = models_with_count[model_name]['fold_results']
        values = [r['count_correlation'] for r in fold_results if 'count_correlation' in r]
        fold_data_corr.append(values)
    
    if fold_data_corr:
        bp = ax.boxplot(fold_data_corr, labels=model_names, patch_artist=True)
        ax.set_title('Count Prediction Correlation')
        ax.set_ylabel('Correlation')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    # 4. Bar chart comparing mean metrics
    ax = axes[3]
    metrics = ['mean_count_mae', 'mean_count_rmse', 'mean_count_correlation']
    metric_labels = ['MAE', 'RMSE', 'Correlation']
    
    x = np.arange(len(model_names))
    width = 0.25
    
    # Normalize metrics for comparison (MAE and RMSE should be lower, correlation should be higher)
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [models_with_count[model][metric] for model in model_names]
        # For correlation, we want higher = better, so don't normalize
        # For MAE/RMSE, normalize to 0-1 scale for comparison
        if metric == 'mean_count_correlation':
            # Correlation is already in -1 to 1 range
            normalized_values = values
        else:
            # Normalize MAE and RMSE (assuming they're positive)
            max_val = max(values) if max(values) > 0 else 1
            normalized_values = [v / max_val for v in values]
        
        offset = (i - len(metrics)/2) * width + width/2
        ax.bar(x + offset, normalized_values, width, label=label)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Count Prediction Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved count prediction plot to {output_file}")
    else:
        plt.show()


def plot_count_prediction_comparison(results: Dict, output_file: str = None):
    """
    Create bar chart comparing count prediction metrics across models.
    
    Args:
        results: Dictionary with model results
        output_file: Optional path to save figure
    """
    # Filter models that have count prediction metrics
    models_with_count = {}
    for model_name, model_results in results.items():
        if 'mean_count_mae' in model_results:
            models_with_count[model_name] = model_results
    
    if len(models_with_count) == 0:
        print("No count prediction metrics found in results.")
        return
    
    model_names = list(models_with_count.keys())
    metrics = ['mean_count_mae', 'mean_count_rmse', 'mean_count_correlation']
    metric_labels = ['MAE', 'RMSE', 'Correlation']
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        values = [models_with_count[model][metric] for model in model_names]
        
        bars = ax.bar(x, values, width, label=label)
        ax.set_xlabel('Model')
        ax.set_ylabel(label)
        ax.set_title(f'Count Prediction - {label}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}' if metric != 'mean_count_correlation' else f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved count prediction comparison to {output_file}")
    else:
        plt.show()


def analyze_predictions_by_activity(model, fold_data: Dict, 
                                   birder_activity: np.ndarray = None,
                                   top_k: int = 10) -> pd.DataFrame:
    """
    Analyze model performance by birder activity level.
    
    Args:
        model: Trained model
        fold_data: Dictionary with test data
        birder_activity: Array of activity levels (number of species viewed)
        top_k: Number of top predictions
    
    Returns:
        DataFrame with performance by activity level
    """
    # Get predictions
    y_pred_binary = model.predict(fold_data['test_X'], top_k=top_k)
    y_true = fold_data['test_y']
    
    # Calculate activity levels if not provided
    if birder_activity is None:
        birder_activity = y_true.sum(axis=1)
    
    # Bin activity levels
    activity_bins = pd.cut(birder_activity, bins=[0, 5, 10, 20, 50, np.inf], 
                          labels=['1-5', '6-10', '11-20', '21-50', '50+'])
    
    # Calculate metrics for each bin
    results = []
    for bin_label in activity_bins.cat.categories:
        mask = activity_bins == bin_label
        if mask.sum() == 0:
            continue
        
        y_true_bin = y_true[mask]
        y_pred_bin = y_pred_binary[mask]
        
        results.append({
            'activity_level': bin_label,
            'n_birders': mask.sum(),
            'precision@k': precision_at_k(y_true_bin, y_pred_bin, k=top_k),
            'recall@k': recall_at_k(y_true_bin, y_pred_bin, k=top_k),
            'map@k': mean_average_precision(y_true_bin, y_pred_bin, k=top_k)
        })
    
    return pd.DataFrame(results)


def plot_activity_analysis(activity_df: pd.DataFrame, output_file: str = None):
    """
    Plot performance by activity level.
    
    Args:
        activity_df: DataFrame from analyze_predictions_by_activity
        output_file: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['precision@k', 'recall@k', 'map@k']
    metric_labels = ['Precision@10', 'Recall@10', 'MAP@10']
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        ax.bar(activity_df['activity_level'], activity_df[metric])
        ax.set_title(label)
        ax.set_xlabel('Species Viewed (Previous Year)')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def generate_summary_report(results: Dict, output_file: str = None) -> str:
    """
    Generate a text summary report of results.
    
    Args:
        results: Dictionary with model results
        output_file: Optional path to save report
    
    Returns:
        Report string
    """
    report = []
    report.append("="*60)
    report.append("Bird Species Prediction - Model Evaluation Report")
    report.append("="*60)
    report.append("")
    
    for model_name, model_results in results.items():
        report.append(f"Model: {model_name.upper()}")
        report.append("-"*60)
        report.append(f"Mean Precision@10: {model_results['mean_precision@k']:.4f}")
        report.append(f"Mean Recall@10: {model_results['mean_recall@k']:.4f}")
        report.append(f"Mean MAP@10: {model_results['mean_map@k']:.4f}")
        report.append(f"Full Evaluation (over all species people see):")
        report.append(f"  Mean Precision (full): {model_results.get('mean_precision_full', 0):.4f}")
        report.append(f"  Mean Recall (full): {model_results.get('mean_recall_full', 0):.4f}")
        report.append(f"  Mean MAP (full): {model_results.get('mean_map_full', 0):.4f}")
        report.append(f"Mean Coverage: {model_results['mean_coverage']:.4f}")
        
        # Add count prediction metrics if available
        if 'mean_count_mae' in model_results:
            report.append("")
            report.append("Count Prediction Metrics:")
            report.append(f"  Mean Test MAE: {model_results['mean_count_mae']:.2f}")
            report.append(f"  Mean Test RMSE: {model_results['mean_count_rmse']:.2f}")
            report.append(f"  Mean Test Correlation: {model_results['mean_count_correlation']:.4f}")
            report.append(f"  Mean Actual Count: {model_results['mean_actual_count']:.2f}")
            report.append(f"  Mean Predicted Count: {model_results['mean_predicted_count']:.2f}")
        
        report.append("")
        
        report.append("Fold-by-fold results:")
        for fold_result in model_results['fold_results']:
            fold_line = f"  Fold {fold_result['fold']}: "
            fold_line += f"P@10={fold_result['precision@k']:.4f}, "
            fold_line += f"R@10={fold_result['recall@k']:.4f}, "
            fold_line += f"MAP@10={fold_result['map@k']:.4f}"
            if 'precision_full' in fold_result:
                fold_line += f" | P_full={fold_result['precision_full']:.4f}, "
                fold_line += f"R_full={fold_result['recall_full']:.4f}, "
                fold_line += f"MAP_full={fold_result['map_full']:.4f}"
            
            # Add count metrics if available
            if 'count_mae' in fold_result:
                fold_line += f", Count_MAE@{fold_result['count_mae']:.2f}"
                fold_line += f", Count_Corr@{fold_result['count_correlation']:.4f}"
            
            report.append(fold_line)
        report.append("")
    
    # Best model
    best_model = max(results.keys(), 
                    key=lambda k: results[k]['mean_map@k'])
    report.append(f"Best Model (by MAP@10): {best_model}")
    report.append("="*60)
    
    report_str = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_str)
        print(f"Saved report to {output_file}")
    
    return report_str


if __name__ == "__main__":
    print("Evaluation module loaded successfully")

