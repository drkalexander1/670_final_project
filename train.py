"""
Training and evaluation script for bird species prediction models.
Implements time-series cross-validation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from models import create_model, NeuralNetworkModel, SpeciesCooccurrenceModel
from prepare_training_data import prepare_all_cv_data
import json
import os
import sys


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Calculate Precision@K.
    
    Args:
        y_true: True binary matrix (n_birders, n_species)
        y_pred: Predicted binary matrix (n_birders, n_species)
        k: Number of top predictions to consider
    
    Returns:
        Precision@K score
    """
    n_birders = y_true.shape[0]
    precisions = []
    
    for i in range(n_birders):
        true_species = set(np.where(y_true[i] > 0)[0])
        if len(true_species) == 0:
            continue
        
        # Get top K predicted species
        pred_scores = y_pred[i]
        top_k_indices = np.argsort(pred_scores)[::-1][:k]
        pred_species = set(top_k_indices)
        
        # Calculate precision
        if len(pred_species) > 0:
            precision = len(true_species & pred_species) / min(len(pred_species), k)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Calculate Recall@K.
    
    Args:
        y_true: True binary matrix (n_birders, n_species)
        y_pred: Predicted binary matrix (n_birders, n_species)
        k: Number of top predictions to consider
    
    Returns:
        Recall@K score
    """
    n_birders = y_true.shape[0]
    recalls = []
    
    for i in range(n_birders):
        true_species = set(np.where(y_true[i] > 0)[0])
        if len(true_species) == 0:
            continue
        
        # Get top K predicted species
        pred_scores = y_pred[i]
        top_k_indices = np.argsort(pred_scores)[::-1][:k]
        pred_species = set(top_k_indices)
        
        # Calculate recall
        if len(true_species) > 0:
            recall = len(true_species & pred_species) / len(true_species)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def mean_average_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Calculate Mean Average Precision (MAP@K).
    
    Args:
        y_true: True binary matrix (n_birders, n_species)
        y_pred: Predicted binary matrix (n_birders, n_species)
        k: Number of top predictions to consider
    
    Returns:
        MAP@K score
    """
    n_birders = y_true.shape[0]
    aps = []
    
    for i in range(n_birders):
        true_species = set(np.where(y_true[i] > 0)[0])
        if len(true_species) == 0:
            continue
        
        # Get top K predicted species
        pred_scores = y_pred[i]
        top_k_indices = np.argsort(pred_scores)[::-1][:k]
        
        # Calculate average precision
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, species_idx in enumerate(top_k_indices, 1):
            if species_idx in true_species:
                relevant_count += 1
                precision_sum += relevant_count / rank
        
        if relevant_count > 0:
            ap = precision_sum / min(len(true_species), k)
            aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def coverage(y_pred: np.ndarray) -> float:
    """
    Calculate coverage: fraction of species that are predicted for at least one birder.
    
    Args:
        y_pred: Predicted binary matrix (n_birders, n_species)
    
    Returns:
        Coverage score
    """
    n_species = y_pred.shape[1]
    predicted_species = np.sum(y_pred, axis=0) > 0
    return np.sum(predicted_species) / n_species


def evaluate_count_predictions(y_true: np.ndarray, y_pred_counts: np.ndarray, 
                               actual_counts: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate count predictions (regression metrics).
    
    Args:
        y_true: True binary matrix (n_birders, n_species) - actual species viewed
        y_pred_counts: Predicted counts (n_birders,) - predicted number
        actual_counts: Optional actual counts (if None, uses species count from y_true)
    
    Returns:
        Dictionary with count prediction metrics
    """
    # Calculate actual counts
    if actual_counts is not None:
        actual_counts = actual_counts.astype(np.float32)
    else:
        actual_counts = np.sum(y_true, axis=1).astype(np.float32)
    
    # Ensure predictions are non-negative
    y_pred_counts = np.maximum(y_pred_counts, 0)
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_counts - y_pred_counts))
    rmse = np.sqrt(np.mean((actual_counts - y_pred_counts) ** 2))
    
    # Calculate correlation (handle edge cases)
    if np.std(actual_counts) > 0 and np.std(y_pred_counts) > 0:
        correlation = np.corrcoef(actual_counts, y_pred_counts)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Mean absolute percentage error (avoid division by zero)
    mape = np.mean(np.abs((actual_counts - y_pred_counts) / (actual_counts + 1e-8))) * 100
    
    return {
        'count_mae': mae,
        'count_rmse': rmse,
        'count_correlation': correlation,
        'count_mape': mape,
        'mean_actual_count': np.mean(actual_counts),
        'mean_predicted_count': np.mean(y_pred_counts)
    }


def evaluate_model(model, fold_data: Dict, top_k: int = 10) -> Dict:
    """
    Evaluate a model on a CV fold.
    
    Args:
        model: Trained model
        fold_data: Dictionary with test data
        top_k: Number of top predictions
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions (with features if available)
    if isinstance(model, NeuralNetworkModel) and fold_data.get('test_features') is not None:
        y_pred_binary = model.predict(fold_data['test_X'], top_k=top_k, 
                                     X_features=fold_data['test_features'])
    elif isinstance(model, SpeciesCooccurrenceModel):
        idx_to_species = fold_data.get('idx_to_species')
        y_pred_binary = model.predict(fold_data['test_X'], top_k=top_k,
                                     idx_to_species=idx_to_species)
    else:
        y_pred_binary = model.predict(fold_data['test_X'], top_k=top_k)
    y_true = fold_data['test_y']
    
    # Calculate metrics
    metrics = {
        'precision@k': precision_at_k(y_true, y_pred_binary, k=top_k),
        'recall@k': recall_at_k(y_true, y_pred_binary, k=top_k),
        'map@k': mean_average_precision(y_true, y_pred_binary, k=top_k),
        'coverage': coverage(y_pred_binary)
    }
    
    # Evaluate count predictions if model supports it
    if isinstance(model, NeuralNetworkModel) and model.predict_count:
        # Get count predictions
        test_input = fold_data['test_X']
        if fold_data.get('test_features') is not None:
            test_input = [fold_data['test_X'], fold_data['test_features']]
        
        model_output = model.model.predict(test_input, verbose=0)
        
        # Extract count predictions
        if isinstance(model_output, dict):
            predicted_counts = model_output['species_count'].flatten()
        elif isinstance(model_output, list):
            predicted_counts = model_output[1].flatten()
        else:
            predicted_counts = None
        
        if predicted_counts is not None:
            # Get actual total bird counts (not species counts)
            actual_bird_counts = None
            if fold_data.get('raw_data') is not None:
                test_transition = fold_data.get('test_transition')
                if test_transition:
                    year_to = test_transition[1]
                    test_transitions_df = fold_data.get('test_transitions_df')
                    if test_transitions_df is not None:
                        actual_bird_counts = extract_bird_counts(
                            test_transitions_df,
                            fold_data['raw_data'],
                            year_to
                        )
            
            # Fallback to species count if bird count not available
            if actual_bird_counts is None:
                actual_bird_counts = None  # Will use default in evaluate_count_predictions
            
            count_metrics = evaluate_count_predictions(y_true, predicted_counts, actual_counts=actual_bird_counts)
            metrics.update(count_metrics)
    
    return metrics


def extract_bird_counts(transitions_df: pd.DataFrame, 
                       raw_data: Dict[int, pd.DataFrame],
                       year_to: int) -> np.ndarray:
    """
    Extract total bird counts (not species counts) for each birder in target year.
    
    Args:
        transitions_df: DataFrame with transition pairs
        raw_data: Dictionary mapping year to raw DataFrame
        year_to: Target year to extract counts from
    
    Returns:
        Array of total bird counts per birder, or None if data not available
    """
    bird_counts = []
    
    if year_to not in raw_data:
        return None
    
    df_year = raw_data[year_to]
    
    for _, row in transitions_df.iterrows():
        observer_id = row['observer_id']
        birder_data = df_year[df_year['observer_id'] == observer_id]
        
        if len(birder_data) > 0 and 'observation_count' in birder_data.columns:
            total_birds = birder_data['observation_count'].sum()
            bird_counts.append(total_birds)
        else:
            bird_counts.append(0.0)
    
    return np.array(bird_counts, dtype=np.float32)


def train_and_evaluate_cv(transitions_df: pd.DataFrame, 
                          model_types: List[str] = ['baseline', 'collaborative', 'neural'],
                          model_params: Dict = None,
                          raw_data: Optional[Dict[int, pd.DataFrame]] = None,
                          use_features: bool = False,
                          data_loader_func=None) -> Dict:
    """
    Train and evaluate models using time-series cross-validation.
    
    Args:
        transitions_df: DataFrame with transition pairs
        model_types: List of model types to train
        model_params: Dictionary mapping model type to parameters
        raw_data: Optional dictionary mapping year to raw DataFrame (for feature extraction)
        use_features: Whether to extract and use temporal features
        data_loader_func: Optional function(year: int) -> pd.DataFrame to load data on-demand per fold
    
    Returns:
        Dictionary with results for each model and fold
    """
    if model_params is None:
        model_params = {}
    
    # Prepare CV data
    print("Preparing cross-validation data...")
    sys.stdout.flush()
    cv_folds = prepare_all_cv_data(transitions_df, raw_data=raw_data, 
                                   extract_features=use_features,
                                   data_loader_func=data_loader_func)
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type} model")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        fold_results = []
        
        for fold_idx, fold_data in enumerate(cv_folds, 1):
            print(f"\nFold {fold_idx}/{len(cv_folds)}")
            print(f"  Train shape: {fold_data['train_X'].shape}")
            print(f"  Test shape: {fold_data['test_X'].shape}")
            sys.stdout.flush()
            
            # Create model
            params = model_params.get(model_type, {}).copy()
            if model_type == 'neural':
                params['n_species'] = fold_data['n_species']
                # Add feature dimension if features available
                if fold_data.get('train_features') is not None:
                    params['n_additional_features'] = fold_data['train_features'].shape[1]
            elif model_type == 'collaborative':
                # Add species features if available
                if 'species_features' in fold_data:
                    params['species_features'] = fold_data['species_features']
            
            model = create_model(model_type, **params)
            
            # Train model
            print("  Training...")
            sys.stdout.flush()
            training_history = None
            if model_type == 'neural':
                # Use a subset for validation
                n_val = min(1000, fold_data['train_X'].shape[0] // 10)
                val_indices = np.random.choice(fold_data['train_X'].shape[0], n_val, replace=False)
                train_indices = np.setdiff1d(np.arange(fold_data['train_X'].shape[0]), val_indices)
                
                X_train = fold_data['train_X'][train_indices]
                y_train = fold_data['train_y'][train_indices]
                X_val = fold_data['train_X'][val_indices]
                y_val = fold_data['train_y'][val_indices]
                
                # Extract total bird counts (not species counts)
                train_bird_counts = None
                val_bird_counts = None
                
                # Use raw_data from fold_data (may be loaded lazily per fold)
                fold_raw_data = fold_data.get('raw_data')
                if fold_raw_data is not None and model_params.get(model_type, {}).get('predict_count', False):
                    train_transitions_df = fold_data.get('train_transitions_df')
                    train_transitions_list = fold_data.get('train_transitions', [])
                    if train_transitions_df is not None and train_transitions_list:
                        # Extract bird counts for each row in train_transitions_df
                        # Match the order of rows in train_X (which is stacked from all transitions)
                        all_train_counts = []
                        for _, row in train_transitions_df.iterrows():
                            year_to_train = int(row['year_to'])
                            # Extract count for this specific birder-year pair
                            counts = extract_bird_counts(
                                pd.DataFrame([row]), 
                                fold_raw_data, 
                                year_to_train
                            )
                            if counts is not None and len(counts) > 0:
                                all_train_counts.append(counts[0])
                            else:
                                all_train_counts.append(0.0)
                        
                        if len(all_train_counts) == fold_data['train_X'].shape[0]:
                            all_train_counts = np.array(all_train_counts, dtype=np.float32)
                            train_bird_counts = all_train_counts[train_indices]
                            val_bird_counts = all_train_counts[val_indices]
                
                # Include features if available
                X_train_features = None
                X_val_features = None
                if fold_data.get('train_features') is not None:
                    X_train_features = fold_data['train_features'][train_indices]
                    X_val_features = fold_data['train_features'][val_indices]
                
                if X_train_features is not None:
                    if train_bird_counts is not None:
                        training_history = model.fit(X_train, y_train, X_features=X_train_features,
                                                     y_bird_count=train_bird_counts,
                                                     validation_data=(X_val, X_val_features, y_val, val_bird_counts),
                                                     epochs=20, batch_size=256, verbose=0)
                    else:
                        training_history = model.fit(X_train, y_train, X_features=X_train_features,
                                                     validation_data=(X_val, X_val_features, y_val),
                                                     epochs=20, batch_size=256, verbose=0)
                else:
                    if train_bird_counts is not None:
                        training_history = model.fit(X_train, y_train, y_bird_count=train_bird_counts,
                                                     validation_data=(X_val, y_val, val_bird_counts),
                                                     epochs=20, batch_size=256, verbose=0)
                    else:
                        training_history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                                     epochs=20, batch_size=256, verbose=0)
            else:
                model.fit(fold_data['train_X'], fold_data['train_y'])
            
            # Evaluate
            print("  Evaluating...")
            sys.stdout.flush()
            metrics = evaluate_model(model, fold_data, top_k=10)
            metrics['fold'] = fold_idx
            
            # Add training history metrics if available
            if training_history is not None:
                history = training_history.history
                # Extract final epoch metrics
                if 'species_count_mae' in history:
                    metrics['train_count_mae'] = history['species_count_mae'][-1]
                    metrics['val_count_mae'] = history.get('val_species_count_mae', [0])[-1]
                if 'loss' in history:
                    metrics['train_loss'] = history['loss'][-1]
                    metrics['val_loss'] = history.get('val_loss', [0])[-1]
            
            fold_results.append(metrics)
            
            print(f"  Precision@10: {metrics['precision@k']:.4f}")
            print(f"  Recall@10: {metrics['recall@k']:.4f}")
            print(f"  MAP@10: {metrics['map@k']:.4f}")
            print(f"  Coverage: {metrics['coverage']:.4f}")
            
            # Print count prediction metrics if available
            if 'count_mae' in metrics:
                print(f"  Count Prediction:")
                print(f"    Test MAE: {metrics['count_mae']:.2f}")
                print(f"    Test RMSE: {metrics['count_rmse']:.2f}")
                print(f"    Test Correlation: {metrics['count_correlation']:.4f}")
                print(f"    Mean Actual: {metrics['mean_actual_count']:.2f}")
                print(f"    Mean Predicted: {metrics['mean_predicted_count']:.2f}")
                if 'train_count_mae' in metrics:
                    print(f"    Train MAE: {metrics['train_count_mae']:.2f}")
                    print(f"    Val MAE: {metrics['val_count_mae']:.2f}")
        
        # Aggregate results
        results[model_type] = {
            'fold_results': fold_results,
            'mean_precision@k': np.mean([r['precision@k'] for r in fold_results]),
            'mean_recall@k': np.mean([r['recall@k'] for r in fold_results]),
            'mean_map@k': np.mean([r['map@k'] for r in fold_results]),
            'mean_coverage': np.mean([r['coverage'] for r in fold_results])
        }
        
        # Aggregate count prediction metrics if available
        if 'count_mae' in fold_results[0]:
            results[model_type]['mean_count_mae'] = np.mean([r['count_mae'] for r in fold_results])
            results[model_type]['mean_count_rmse'] = np.mean([r['count_rmse'] for r in fold_results])
            results[model_type]['mean_count_correlation'] = np.mean([r['count_correlation'] for r in fold_results])
            results[model_type]['mean_actual_count'] = np.mean([r['mean_actual_count'] for r in fold_results])
            results[model_type]['mean_predicted_count'] = np.mean([r['mean_predicted_count'] for r in fold_results])
        
        print(f"\n{model_type} Summary:")
        print(f"  Mean Precision@10: {results[model_type]['mean_precision@k']:.4f}")
        print(f"  Mean Recall@10: {results[model_type]['mean_recall@k']:.4f}")
        print(f"  Mean MAP@10: {results[model_type]['mean_map@k']:.4f}")
        print(f"  Mean Coverage: {results[model_type]['mean_coverage']:.4f}")
        
        # Print count prediction summary if available
        if 'mean_count_mae' in results[model_type]:
            print(f"  Count Prediction:")
            print(f"    Mean Test MAE: {results[model_type]['mean_count_mae']:.2f}")
            print(f"    Mean Test RMSE: {results[model_type]['mean_count_rmse']:.2f}")
            print(f"    Mean Test Correlation: {results[model_type]['mean_count_correlation']:.4f}")
            print(f"    Mean Actual Count: {results[model_type]['mean_actual_count']:.2f}")
            print(f"    Mean Predicted Count: {results[model_type]['mean_predicted_count']:.2f}")
    
    return results


if __name__ == "__main__":
    print("Training module loaded successfully")
    print("Import and use train_and_evaluate_cv() with transitions DataFrame")

