"""
Prepare training data with time-series cross-validation.
Creates train/test splits for year-over-year transitions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from sklearn.preprocessing import RobustScaler
from feature_engineering import (
    create_birder_species_matrix, 
    get_all_species,
    extract_temporal_features,
    extract_geographic_features,
    extract_enhanced_geographic_features,
    extract_comprehensive_temporal_features,
    extract_temporal_features_from_birder_species,
    extract_geographic_features_from_birder_species,
    create_species_features
)
try:
    from feature_engineering_simple import extract_features_from_summaries
except ImportError:
    extract_features_from_summaries = None


def create_time_series_cv_splits(transitions_df: pd.DataFrame) -> List[Dict]:
    """
    Create time-series cross-validation splits.
    Each fold trains on all previous years and tests on the next year.
    
    CV fold 1: Train (2018→2019), Test (2019→2020)
    CV fold 2: Train (2018→2019, 2019→2020), Test (2020→2021)
    CV fold 3: Train (2018→2019, 2019→2020, 2020→2021), Test (2021→2022)
    CV fold 4: Train (2018→2019, 2019→2020, 2020→2021, 2021→2022), Test (2022→2023)
    
    Args:
        transitions_df: DataFrame with transition pairs
    
    Returns:
        List of dictionaries, each containing 'train' and 'test' DataFrames
    """
    # Get all unique year transitions
    year_transitions = sorted(transitions_df[['year_from', 'year_to']].drop_duplicates().values.tolist())
    
    cv_splits = []
    
    # Create 4 folds
    for fold_idx in range(1, 5):
        # Test set is always the (fold_idx)th transition
        test_transition = year_transitions[fold_idx]
        
        # Train set is all transitions before the test transition
        train_transitions = year_transitions[:fold_idx]
        
        # Split data
        train_mask = transitions_df.apply(
            lambda row: [int(row['year_from']), int(row['year_to'])] in train_transitions,
            axis=1
        )
        test_mask = transitions_df.apply(
            lambda row: [int(row['year_from']), int(row['year_to'])] == test_transition,
            axis=1
        )
        
        train_df = transitions_df[train_mask].copy()
        test_df = transitions_df[test_mask].copy()
        
        cv_splits.append({
            'fold': fold_idx,
            'train': train_df,
            'test': test_df,
            'train_transitions': train_transitions,
            'test_transition': test_transition
        })
        
        print(f"Fold {fold_idx}: Train={len(train_df):,} rows ({train_transitions}), "
              f"Test={len(test_df):,} rows ({test_transition})")
    
    return cv_splits


def prepare_cv_fold_data(cv_split: Dict, all_species: Set[str], 
                         raw_data: Optional[Dict[int, pd.DataFrame]] = None,
                         extract_features: bool = True) -> Dict:
    """
    Prepare data matrices for a single CV fold.
    
    Args:
        cv_split: Dictionary with 'train' and 'test' DataFrames
        all_species: Set of all unique species
        raw_data: Optional dictionary mapping year to raw DataFrame for feature extraction
        extract_features: Whether to extract temporal features
    
    Returns:
        Dictionary with train/test matrices, features, and metadata
    """
    # Create matrices for training transitions
    train_matrices, species_to_idx, idx_to_species = create_birder_species_matrix(
        cv_split['train'], all_species
    )
    
    # Create matrices for test transition
    test_matrices, _, _ = create_birder_species_matrix(
        cv_split['test'], all_species
    )
    
    # Combine training data from all training transitions
    train_X_list = []
    train_y_list = []
    train_birder_ids_list = []
    
    for transition_key, matrix_dict in train_matrices.items():
        train_X_list.append(matrix_dict['X'])
        train_y_list.append(matrix_dict['y'])
        train_birder_ids_list.append(matrix_dict['birder_ids'])
    
    # Stack training matrices
    train_X = np.vstack(train_X_list) if train_X_list else np.array([])
    train_y = np.vstack(train_y_list) if train_y_list else np.array([])
    train_birder_ids = np.concatenate(train_birder_ids_list) if train_birder_ids_list else np.array([])
    
    # Get test data (should be single transition)
    test_transition_key = list(test_matrices.keys())[0] if test_matrices else None
    if test_transition_key:
        test_X = test_matrices[test_transition_key]['X']
        test_y = test_matrices[test_transition_key]['y']
        test_birder_ids = test_matrices[test_transition_key]['birder_ids']
    else:
        test_X = np.array([])
        test_y = np.array([])
        test_birder_ids = np.array([])
    
    # Extract temporal features if requested
    train_features = None
    test_features = None
    if extract_features:
        try:
            # Try full temporal features if raw_data available
            if raw_data is not None:
                # Check if raw_data contains birder_species DataFrames (has 'scientific_name' as set)
                # vs raw observation DataFrames
                sample_year = list(raw_data.keys())[0] if raw_data else None
                is_birder_species_format = False
                
                if sample_year is not None:
                    sample_df = raw_data[sample_year]
                    # Check if this looks like birder_species (has 'scientific_name' column that might be a set)
                    if 'scientific_name' in sample_df.columns and len(sample_df) > 0:
                        # Try to detect if it's birder_species format
                        first_row = sample_df.iloc[0]
                        if isinstance(first_row.get('scientific_name', None), set) or \
                           isinstance(first_row.get('effort_hours', None), set):
                            is_birder_species_format = True
                
                if is_birder_species_format:
                    # This is birder_species data - use new extraction functions
                    train_temporal = extract_temporal_features_from_birder_species(cv_split['train'], raw_data)
                    test_temporal = extract_temporal_features_from_birder_species(cv_split['test'], raw_data)
                    
                    train_geo, geo_metadata = extract_geographic_features_from_birder_species(cv_split['train'], raw_data)
                    test_geo, _ = extract_geographic_features_from_birder_species(cv_split['test'], raw_data)
                    
                    # Select feature columns (skip day_of_year, hours_of_day, months - not available)
                    feature_cols = [
                        'num_checklists', 'total_observations', 'avg_observations_per_checklist',
                        'avg_effort_hours', 'effort_total', 'effort_max', 'effort_std',
                        'avg_duration', 'duration_total', 'duration_max', 'duration_std',
                        'checklists_per_month', 'num_locations', 'num_states'
                    ]
                    
                    # Filter to only columns that exist
                    available_cols = [col for col in feature_cols if col in train_temporal.columns]
                else:
                    # This is raw data - use original extraction functions
                    train_temporal = extract_comprehensive_temporal_features(cv_split['train'], raw_data)
                    test_temporal = extract_comprehensive_temporal_features(cv_split['test'], raw_data)
                    
                    train_geo, geo_metadata = extract_enhanced_geographic_features(cv_split['train'], raw_data)
                    test_geo, _ = extract_enhanced_geographic_features(cv_split['test'], raw_data)
                    
                    # Select temporal feature columns to use
                    # Remove 'num_species' as it's redundant (already in birder-species matrix)
                    # Use comprehensive temporal features
                    base_temporal_cols = ['num_checklists', 'total_observations', 'avg_observations_per_checklist',
                                         'avg_day_of_year', 'doy_std', 'doy_range',
                                         'avg_hours_of_day', 'hod_std',
                                         'avg_effort_hours', 'effort_total', 'effort_max', 'effort_std',
                                         'avg_duration', 'duration_total', 'duration_max', 'duration_std',
                                         'checklists_per_month', 'num_locations', 'num_states']
                    
                    # Add month frequency features (12 months)
                    month_cols = [f'month_{m}_freq' for m in range(1, 13)]
                    
                    # Combine all temporal feature columns
                    feature_cols = base_temporal_cols + month_cols
                    
                    # Filter to only columns that exist in the temporal DataFrame
                    available_cols = [col for col in feature_cols if col in train_temporal.columns]
                
                if len(train_temporal) > 0:
                    n_temporal_features = len(available_cols)
                    n_geo_features = train_geo.shape[1]
                    
                    train_features = np.zeros((train_X.shape[0], n_temporal_features + n_geo_features), dtype=np.float32)
                    birder_year_map = {}
                    for idx, (_, row) in enumerate(cv_split['train'].iterrows()):
                        # Convert to Python native types to ensure hashability
                        obs_id = int(row['observer_id']) if pd.notna(row['observer_id']) else None
                        year = int(row['year_from']) if pd.notna(row['year_from']) else None
                        if obs_id is not None and year is not None:
                            birder_year_map[(obs_id, year)] = idx
                    
                    # Fill temporal features
                    for _, feat_row in train_temporal.iterrows():
                        # Convert to Python native types
                        obs_id = int(feat_row['observer_id']) if pd.notna(feat_row['observer_id']) else None
                        year = int(feat_row['year_from']) if pd.notna(feat_row['year_from']) else None
                        if obs_id is not None and year is not None:
                            key = (obs_id, year)
                            if key in birder_year_map:
                                feat_idx = birder_year_map[key]
                                for col_idx, col in enumerate(available_cols):
                                    val = feat_row[col]
                                    train_features[feat_idx, col_idx] = val if not pd.isna(val) else 0.0
                    
                    # Fill geographic features
                    train_features[:, n_temporal_features:] = train_geo
                
                if len(test_temporal) > 0:
                    n_temporal_features = len(available_cols)
                    n_geo_features = test_geo.shape[1]
                    
                    test_features = np.zeros((test_X.shape[0], n_temporal_features + n_geo_features), dtype=np.float32)
                    birder_year_map = {}
                    for idx, (_, row) in enumerate(cv_split['test'].iterrows()):
                        # Convert to Python native types to ensure hashability
                        obs_id = int(row['observer_id']) if pd.notna(row['observer_id']) else None
                        year = int(row['year_from']) if pd.notna(row['year_from']) else None
                        if obs_id is not None and year is not None:
                            birder_year_map[(obs_id, year)] = idx
                    
                    # Fill temporal features
                    for _, feat_row in test_temporal.iterrows():
                        # Convert to Python native types
                        obs_id = int(feat_row['observer_id']) if pd.notna(feat_row['observer_id']) else None
                        year = int(feat_row['year_from']) if pd.notna(feat_row['year_from']) else None
                        if obs_id is not None and year is not None:
                            key = (obs_id, year)
                            if key in birder_year_map:
                                feat_idx = birder_year_map[key]
                                for col_idx, col in enumerate(available_cols):
                                    val = feat_row[col]
                                    test_features[feat_idx, col_idx] = val if not pd.isna(val) else 0.0
                    
                    # Fill geographic features
                    test_features[:, n_temporal_features:] = test_geo
                
                # Normalize features: fit scaler on train, transform both train and test
                # Note: Don't normalize dummy variables (state/county one-hot) but normalize geographic metrics
                if train_features is not None and len(train_features) > 0:
                    scaler = RobustScaler()  # More robust to outliers
                    n_temporal_features = len(available_cols)
                    
                    # Determine which geographic features are dummy variables vs metrics
                    # Dummy variables: state and county one-hot encodings
                    # Metrics: state_diversity, num_states, num_counties, num_localities, geographic_spread
                    n_state_dummy = geo_metadata['n_state_features']
                    n_county_dummy = geo_metadata['n_county_features']
                    n_geo_dummy = n_state_dummy + n_county_dummy
                    n_geo_metrics = geo_metadata['n_geo_metrics']
                    
                    # Normalize temporal features
                    train_temporal_part = train_features[:, :n_temporal_features]
                    train_temporal_scaled = scaler.fit_transform(train_temporal_part).astype(np.float32)
                    
                    # Normalize geographic metrics (but not dummy variables)
                    geo_metrics_start = n_temporal_features + n_geo_dummy
                    geo_metrics_end = geo_metrics_start + n_geo_metrics
                    
                    if n_geo_metrics > 0:
                        geo_scaler = RobustScaler()  # More robust to outliers
                        train_geo_metrics = train_features[:, geo_metrics_start:geo_metrics_end]
                        train_geo_metrics_scaled = geo_scaler.fit_transform(train_geo_metrics).astype(np.float32)
                        
                        # Combine: normalized temporal + geo dummy + normalized geo metrics
                        train_features = np.hstack([
                            train_temporal_scaled,
                            train_features[:, n_temporal_features:geo_metrics_start],  # Dummy variables
                            train_geo_metrics_scaled  # Normalized metrics
                        ])
                    else:
                        # No geo metrics, just temporal + dummy
                        train_features = np.hstack([
                            train_temporal_scaled,
                            train_features[:, n_temporal_features:]  # All geo features (dummy only)
                        ])
                    
                    if test_features is not None and len(test_features) > 0:
                        test_temporal_part = test_features[:, :n_temporal_features]
                        test_temporal_scaled = scaler.transform(test_temporal_part).astype(np.float32)
                        
                        if n_geo_metrics > 0:
                            test_geo_metrics = test_features[:, geo_metrics_start:geo_metrics_end]
                            test_geo_metrics_scaled = geo_scaler.transform(test_geo_metrics).astype(np.float32)
                            
                            test_features = np.hstack([
                                test_temporal_scaled,
                                test_features[:, n_temporal_features:geo_metrics_start],
                                test_geo_metrics_scaled
                            ])
                        else:
                            test_features = np.hstack([
                                test_temporal_scaled,
                                test_features[:, n_temporal_features:]
                            ])
            # Otherwise try simplified features from summaries
            elif extract_features_from_summaries is not None:
                # This would require birder_species_df - simplified version
                # For now, create basic features from transitions
                feature_cols = ['num_species', 'year']
                train_features = np.zeros((train_X.shape[0], len(feature_cols)), dtype=np.float32)
                for idx, (_, row) in enumerate(cv_split['train'].iterrows()):
                    train_features[idx, 0] = len(row['species_from'])
                    train_features[idx, 1] = row['year_from'] / 2023.0  # Normalize year
                
                test_features = np.zeros((test_X.shape[0], len(feature_cols)), dtype=np.float32)
                for idx, (_, row) in enumerate(cv_split['test'].iterrows()):
                    test_features[idx, 0] = len(row['species_from'])
                    test_features[idx, 1] = row['year_from'] / 2023.0
                
                # Normalize simplified features: fit on train, transform both
                if train_features is not None and len(train_features) > 0:
                    scaler = RobustScaler()  # More robust to outliers
                    train_features = scaler.fit_transform(train_features).astype(np.float32)
                    if test_features is not None and len(test_features) > 0:
                        test_features = scaler.transform(test_features).astype(np.float32)
        except Exception as e:
            print(f"  Warning: Could not extract features: {e}")
            train_features = None
            test_features = None
    
    return {
        'train_X': train_X,
        'train_y': train_y,
        'train_birder_ids': train_birder_ids,
        'train_features': train_features,  # New: temporal features
        'test_X': test_X,
        'test_y': test_y,
        'test_birder_ids': test_birder_ids,
        'test_features': test_features,  # New: temporal features
        'species_to_idx': species_to_idx,
        'idx_to_species': idx_to_species,
        'n_species': len(all_species),
        'fold': cv_split['fold'],
        'test_transition': cv_split.get('test_transition'),
        'train_transitions': cv_split.get('train_transitions'),
        'raw_data': raw_data,
        'test_transitions_df': cv_split.get('test'),
        'train_transitions_df': cv_split.get('train')
    }


def get_years_needed_for_fold(cv_split: Dict) -> List[int]:
    """
    Determine which years are needed for a CV fold.
    
    Args:
        cv_split: Dictionary with train/test transitions
    
    Returns:
        List of years needed for this fold
    """
    years_needed = set()
    
    # Add years from train transitions
    for transition in cv_split['train_transitions']:
        years_needed.add(transition[0])  # year_from
        years_needed.add(transition[1])  # year_to
    
    # Add years from test transition
    test_transition = cv_split['test_transition']
    if test_transition:
        years_needed.add(test_transition[0])  # year_from
        years_needed.add(test_transition[1])  # year_to
    
    return sorted(list(years_needed))


def prepare_all_cv_data(transitions_df: pd.DataFrame, 
                       raw_data: Optional[Dict[int, pd.DataFrame]] = None,
                       extract_features: bool = False,
                       data_loader_func=None) -> List[Dict]:
    """
    Prepare all CV folds with matrices.
    
    Args:
        transitions_df: DataFrame with transition pairs
        raw_data: Optional dictionary mapping year to raw DataFrame for feature extraction
        extract_features: Whether to extract temporal features (requires raw_data or data_loader_func)
        data_loader_func: Optional function(year: int) -> pd.DataFrame to load data on-demand
    
    Returns:
        List of prepared fold data dictionaries
    """
    # Get all species
    all_species = get_all_species(transitions_df)
    print(f"Total unique species: {len(all_species)}")
    
    # Create species features (for use in models)
    species_features = create_species_features(transitions_df)
    print(f"Created features for {len(species_features)} species")
    
    # Create CV splits
    cv_splits = create_time_series_cv_splits(transitions_df)
    
    # Prepare data for each fold
    prepared_folds = []
    for cv_split in cv_splits:
        print(f"\nPreparing fold {cv_split['fold']} data...")
        
        # Load data for this fold if using lazy loading
        fold_raw_data = raw_data
        if extract_features and data_loader_func is not None:
            years_needed = get_years_needed_for_fold(cv_split)
            print(f"  Loading data for years: {years_needed}")
            fold_raw_data = {}
            for year in years_needed:
                fold_raw_data[year] = data_loader_func(year)
            print(f"  Loaded {len(fold_raw_data)} years for this fold")
        
        fold_data = prepare_cv_fold_data(cv_split, all_species, fold_raw_data, extract_features)
        fold_data['species_features'] = species_features  # Add species features
        prepared_folds.append(fold_data)
        
        print(f"  Train: {fold_data['train_X'].shape}")
        if fold_data['train_features'] is not None:
            print(f"  Train features: {fold_data['train_features'].shape}")
        print(f"  Test: {fold_data['test_X'].shape}")
        if fold_data['test_features'] is not None:
            print(f"  Test features: {fold_data['test_features'].shape}")
        
        # Free memory for this fold's raw data if we loaded it lazily
        if extract_features and data_loader_func is not None:
            del fold_raw_data
            import gc
            gc.collect()
    
    return prepared_folds


if __name__ == "__main__":
    # Test with sample data
    print("Training data preparation module loaded successfully")
    print("Run with actual transitions DataFrame to prepare CV folds")

