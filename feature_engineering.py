"""
Feature engineering for birding prediction.
Creates features from birder-species interaction data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
from collections import Counter
from sklearn.preprocessing import LabelEncoder


def create_birder_species_matrix(transitions_df: pd.DataFrame, 
                                 all_species: Set[str]) -> Tuple[Dict, Dict[str, int], Dict[int, str]]:
    """
    Create birder-species interaction matrices for each year transition.
    
    Args:
        transitions_df: DataFrame with transition pairs (from data_loader)
        all_species: Set of all unique species
    
    Returns:
        Tuple of:
        - Dictionary mapping (year_from, year_to) to interaction matrix dict
        - Species to index mapping
        - Index to species mapping
    """
    # Create species encoders
    species_list = sorted(list(all_species))
    species_to_idx = {species: idx for idx, species in enumerate(species_list)}
    idx_to_species = {idx: species for species, idx in species_to_idx.items()}
    
    matrices = {}
    
    for (year_from, year_to), group in transitions_df.groupby(['year_from', 'year_to']):
        # Get unique birders for this transition
        unique_birders = group['observer_id'].unique()
        birder_to_idx = {birder: idx for idx, birder in enumerate(unique_birders)}
        
        # Create matrix: rows = birders, cols = species
        n_birders = len(unique_birders)
        n_species = len(species_list)
        
        matrix_from = np.zeros((n_birders, n_species), dtype=np.float32)
        matrix_to = np.zeros((n_birders, n_species), dtype=np.float32)
        
        for _, row in group.iterrows():
            birder_idx = birder_to_idx[row['observer_id']]
            
            # Fill matrix_from (species viewed in year_from)
            for species in row['species_from']:
                if species in species_to_idx:
                    species_idx = species_to_idx[species]
                    matrix_from[birder_idx, species_idx] = 1.0
            
            # Fill matrix_to (species viewed in year_to)
            for species in row['species_to']:
                if species in species_to_idx:
                    species_idx = species_to_idx[species]
                    matrix_to[birder_idx, species_idx] = 1.0
        
        matrices[(year_from, year_to)] = {
            'X': matrix_from,
            'y': matrix_to,
            'birder_ids': unique_birders,
            'birder_to_idx': birder_to_idx
        }
    
    return matrices, species_to_idx, idx_to_species


def extract_temporal_features(transitions_df: pd.DataFrame, 
                            raw_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract temporal features for each birder-year pair.
    (Backward compatible wrapper - uses comprehensive version internally)
    
    Args:
        transitions_df: DataFrame with transition pairs
        raw_data: Dictionary mapping year to raw DataFrame
    
    Returns:
        DataFrame with temporal features
    """
    # Use comprehensive temporal features
    return extract_comprehensive_temporal_features(transitions_df, raw_data)


def extract_comprehensive_temporal_features(transitions_df: pd.DataFrame,
                                           raw_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract comprehensive temporal features for each birder-year pair.
    Includes detailed temporal patterns, effort metrics, and observation statistics.
    
    Args:
        transitions_df: DataFrame with transition pairs
        raw_data: Dictionary mapping year to raw DataFrame
    
    Returns:
        DataFrame with comprehensive temporal features
    """
    features = []
    
    for _, row in transitions_df.iterrows():
        observer_id = row['observer_id']
        year_from = int(row['year_from'])
        year_to = int(row['year_to'])
        
        # Get raw data for year_from
        if year_from in raw_data:
            df_year = raw_data[year_from]
            birder_data = df_year[df_year['observer_id'] == observer_id]
            
            if len(birder_data) > 0:
                feat_dict = {
                    'observer_id': observer_id,
                    'year_from': year_from,
                    'year_to': year_to
                }
                
                # Count features
                # Use locality_id as proxy for checklists (consistent with birder_species version)
                if 'locality_id' in birder_data.columns:
                    num_checklists = birder_data['locality_id'].nunique()
                else:
                    # Fallback: if locality_id not available, use 0
                    num_checklists = 0
                num_species = birder_data['scientific_name'].nunique()
                feat_dict['num_checklists'] = num_checklists
                feat_dict['num_species'] = num_species
                
                # Observation features
                if 'observation_count' in birder_data.columns:
                    obs_counts = birder_data['observation_count'].dropna()
                    if len(obs_counts) > 0:
                        feat_dict['total_observations'] = obs_counts.sum()
                        feat_dict['avg_observations_per_checklist'] = obs_counts.sum() / num_checklists if num_checklists > 0 else 0
                        feat_dict['max_observations_per_checklist'] = obs_counts.max()
                    else:
                        feat_dict['total_observations'] = 0
                        feat_dict['avg_observations_per_checklist'] = 0
                        feat_dict['max_observations_per_checklist'] = 0
                else:
                    feat_dict['total_observations'] = 0
                    feat_dict['avg_observations_per_checklist'] = 0
                    feat_dict['max_observations_per_checklist'] = 0
                
                # Day of year features - REQUIRED
                if 'day_of_year' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'day_of_year' in raw data for year {year_from}")
                doy = birder_data['day_of_year'].dropna()
                if len(doy) > 0:
                    feat_dict['doy_mean'] = doy.mean()
                    feat_dict['doy_std'] = doy.std() if len(doy) > 1 else 0
                    feat_dict['doy_min'] = doy.min()
                    feat_dict['doy_max'] = doy.max()
                    feat_dict['doy_range'] = doy.max() - doy.min()
                else:
                    feat_dict['doy_mean'] = np.nan
                    feat_dict['doy_std'] = 0
                    feat_dict['doy_min'] = np.nan
                    feat_dict['doy_max'] = np.nan
                    feat_dict['doy_range'] = 0
                
                # Hours of day features - REQUIRED
                if 'hours_of_day' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'hours_of_day' in raw data for year {year_from}")
                hod = birder_data['hours_of_day'].dropna()
                if len(hod) > 0:
                    feat_dict['hod_mean'] = hod.mean()
                    feat_dict['hod_std'] = hod.std() if len(hod) > 1 else 0
                    feat_dict['hod_min'] = hod.min()
                    feat_dict['hod_max'] = hod.max()
                else:
                    feat_dict['hod_mean'] = np.nan
                    feat_dict['hod_std'] = 0
                    feat_dict['hod_min'] = np.nan
                    feat_dict['hod_max'] = np.nan
                
                # Month distribution (if observation_date available)
                if 'observation_date' in birder_data.columns:
                    try:
                        birder_data_copy = birder_data.copy()
                        birder_data_copy['month'] = pd.to_datetime(birder_data_copy['observation_date']).dt.month
                        month_dist = birder_data_copy['month'].value_counts(normalize=True)
                        for month in range(1, 13):
                            feat_dict[f'month_{month}_freq'] = month_dist.get(month, 0.0)
                    except:
                        # If date parsing fails, set all months to 0
                        for month in range(1, 13):
                            feat_dict[f'month_{month}_freq'] = 0.0
                else:
                    for month in range(1, 13):
                        feat_dict[f'month_{month}_freq'] = 0.0
                
                # Effort features - REQUIRED
                if 'effort_hours' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'effort_hours' in raw data for year {year_from}")
                effort = birder_data['effort_hours'].dropna()
                if len(effort) > 0:
                    feat_dict['effort_mean'] = effort.mean()
                    feat_dict['effort_total'] = effort.sum()
                    feat_dict['effort_max'] = effort.max()
                    feat_dict['effort_std'] = effort.std() if len(effort) > 1 else 0
                else:
                    feat_dict['effort_mean'] = np.nan
                    feat_dict['effort_total'] = 0
                    feat_dict['effort_max'] = 0
                    feat_dict['effort_std'] = 0
                
                # Duration features - REQUIRED
                if 'duration_minutes' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'duration_minutes' in raw data for year {year_from}")
                duration = birder_data['duration_minutes'].dropna()
                if len(duration) > 0:
                    feat_dict['duration_mean'] = duration.mean()
                    feat_dict['duration_total'] = duration.sum()
                    feat_dict['duration_max'] = duration.max()
                    feat_dict['duration_std'] = duration.std() if len(duration) > 1 else 0
                else:
                    feat_dict['duration_mean'] = np.nan
                    feat_dict['duration_total'] = 0
                    feat_dict['duration_max'] = 0
                    feat_dict['duration_std'] = 0
                
                # Checklist frequency features
                feat_dict['checklists_per_month'] = num_checklists / 12.0 if num_checklists > 0 else 0
                
                # Location diversity - REQUIRED
                if 'locality_id' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'locality_id' in raw data for year {year_from}")
                if 'state' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'state' in raw data for year {year_from}")
                num_locations = birder_data['locality_id'].nunique()
                num_states = birder_data['state'].nunique()
                feat_dict['num_locations'] = num_locations
                feat_dict['num_states'] = num_states
                
                # Backward compatibility: keep old column names
                feat_dict['avg_day_of_year'] = feat_dict.get('doy_mean', np.nan)
                feat_dict['avg_hours_of_day'] = feat_dict.get('hod_mean', np.nan)
                feat_dict['avg_effort_hours'] = feat_dict.get('effort_mean', np.nan)
                feat_dict['avg_duration'] = feat_dict.get('duration_mean', np.nan)
                
                features.append(feat_dict)
    
    return pd.DataFrame(features)


def extract_geographic_features(transitions_df: pd.DataFrame,
                               raw_data: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
    """
    Extract geographic dummy variables (state) for each birder-year pair.
    (Backward compatible wrapper - uses enhanced version internally)
    
    Args:
        transitions_df: DataFrame with transition pairs
        raw_data: Dictionary mapping year to raw DataFrame
    
    Returns:
        Tuple of:
        - Geographic feature matrix (n_samples, n_states) with dummy variables
        - Dictionary mapping state to column index
    """
    # Use enhanced geographic features but return only state dummy variables for backward compatibility
    geo_features, metadata = extract_enhanced_geographic_features(transitions_df, raw_data)
    
    # Extract only state dummy variables (first n_state_features columns)
    n_state_features = metadata['n_state_features']
    state_features = geo_features[:, :n_state_features]
    
    return state_features, {'state_to_idx': metadata['state_to_idx'], 'state_list': metadata['state_list']}


def extract_enhanced_geographic_features(transitions_df: pd.DataFrame,
                                       raw_data: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
    """
    Extract enhanced geographic features with state-level aggregation.
    Includes state dummy variables, county dummy variables (if available),
    and geographic diversity metrics.
    
    Args:
        transitions_df: DataFrame with transition pairs
        raw_data: Dictionary mapping year to raw DataFrame
    
    Returns:
        Tuple of:
        - Feature matrix (n_samples, n_features) with all geographic features
        - Feature metadata dictionary with encoders and feature names
    """
    # Collect all states and counties from raw data
    all_states = set()
    all_counties = set()
    
    for year, df in raw_data.items():
        if 'state' in df.columns:
            all_states.update(df['state'].dropna().unique())
        if 'county' in df.columns:
            all_counties.update(df['county'].dropna().unique())
    
    # Create encoders
    state_list = sorted([s for s in all_states if pd.notna(s)])
    state_to_idx = {state: idx for idx, state in enumerate(state_list)}
    
    county_list = sorted([c for c in all_counties if pd.notna(c)]) if all_counties else []
    county_to_idx = {county: idx for idx, county in enumerate(county_list)} if county_list else {}
    
    # Feature dimensions:
    # - State dummy variables: len(state_list)
    # - County dummy variables (if available): len(county_list)
    # - Geographic metrics: 5 (state_diversity, num_states, num_counties, num_localities, geographic_spread)
    n_state_features = len(state_list)
    n_county_features = len(county_list) if county_list else 0
    n_geo_metrics = 5
    
    n_samples = len(transitions_df)
    n_total_features = n_state_features + n_county_features + n_geo_metrics
    
    geo_features = np.zeros((n_samples, n_total_features), dtype=np.float32)
    
    for idx, row in transitions_df.iterrows():
        observer_id = row['observer_id']
        year_from = int(row['year_from'])
        
        if year_from in raw_data:
            df_year = raw_data[year_from]
            birder_data = df_year[df_year['observer_id'] == observer_id]
            
            if len(birder_data) > 0:
                # 1. State dummy variables (one-hot) - REQUIRED
                if 'state' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'state' in raw data for year {year_from}")
                states = birder_data['state'].dropna()
                if len(states) > 0:
                    state_counts = states.value_counts()
                    most_common_state = state_counts.index[0]
                    
                    if most_common_state in state_to_idx:
                        geo_features[idx, state_to_idx[most_common_state]] = 1.0
                    
                    # State diversity (entropy)
                    state_probs = state_counts / len(states)
                    state_entropy = -np.sum(state_probs * np.log(state_probs + 1e-10))
                    geo_features[idx, n_state_features + n_county_features] = state_entropy
                    
                    # Number of unique states
                    geo_features[idx, n_state_features + n_county_features + 1] = states.nunique()
                else:
                    # No states found
                    geo_features[idx, n_state_features + n_county_features] = 0.0
                    geo_features[idx, n_state_features + n_county_features + 1] = 0
                
                # 2. County dummy variables - REQUIRED
                if 'county' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'county' in raw data for year {year_from}")
                counties = birder_data['county'].dropna()
                if len(counties) > 0:
                    county_counts = counties.value_counts()
                    most_common_county = county_counts.index[0]
                    
                    if county_list and most_common_county in county_to_idx:
                        geo_features[idx, n_state_features + county_to_idx[most_common_county]] = 1.0
                    
                    # Number of unique counties
                    geo_features[idx, n_state_features + n_county_features + 2] = counties.nunique()
                else:
                    geo_features[idx, n_state_features + n_county_features + 2] = 0
                
                # 3. Location diversity metrics - REQUIRED
                if 'locality_id' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'locality_id' in raw data for year {year_from}")
                num_localities = birder_data['locality_id'].nunique()
                geo_features[idx, n_state_features + n_county_features + 3] = num_localities
                
                # 4. Geographic spread - REQUIRED
                if 'latitude' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'latitude' in raw data for year {year_from}")
                if 'longitude' not in birder_data.columns:
                    raise ValueError(f"Missing required column 'longitude' in raw data for year {year_from}")
                lats = birder_data['latitude'].dropna()
                lons = birder_data['longitude'].dropna()
                
                if len(lats) > 0 and len(lons) > 0:
                    # Geographic spread: std of lat + std of lon
                    lat_std = lats.std() if len(lats) > 1 else 0
                    lon_std = lons.std() if len(lons) > 1 else 0
                    geo_features[idx, n_state_features + n_county_features + 4] = lat_std + lon_std
                else:
                    geo_features[idx, n_state_features + n_county_features + 4] = 0
    
    # Create feature metadata
    feature_names = (
        [f'state_{s}' for s in state_list] +
        ([f'county_{c}' for c in county_list] if county_list else []) +
        ['state_diversity', 'num_states', 'num_counties', 'num_localities', 'geographic_spread']
    )
    
    feature_metadata = {
        'state_list': state_list,
        'state_to_idx': state_to_idx,
        'county_list': county_list,
        'county_to_idx': county_to_idx,
        'n_state_features': n_state_features,
        'n_county_features': n_county_features,
        'n_geo_metrics': n_geo_metrics,
        'feature_names': feature_names
    }
    
    return geo_features, feature_metadata


def create_species_features(transitions_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Create features for each species (popularity, rarity, etc.)
    
    Args:
        transitions_df: DataFrame with transition pairs
    
    Returns:
        Dictionary mapping species to features
    """
    # Count how many birders viewed each species in each year
    species_stats = {}
    
    for _, row in transitions_df.iterrows():
        year_from = row['year_from']
        year_to = row['year_to']
        
        # Count species in year_from
        for species in row['species_from']:
            if species not in species_stats:
                species_stats[species] = {'viewers_by_year': Counter(), 'total_viewers': 0}
            species_stats[species]['viewers_by_year'][year_from] += 1
            species_stats[species]['total_viewers'] += 1
        
        # Count species in year_to
        for species in row['species_to']:
            if species not in species_stats:
                species_stats[species] = {'viewers_by_year': Counter(), 'total_viewers': 0}
            species_stats[species]['viewers_by_year'][year_to] += 1
            species_stats[species]['total_viewers'] += 1
    
    # Calculate popularity metrics
    all_viewer_counts = [stats['total_viewers'] for stats in species_stats.values()]
    if all_viewer_counts:
        median_viewers = np.median(all_viewer_counts)
        mean_viewers = np.mean(all_viewer_counts)
    else:
        median_viewers = 0
        mean_viewers = 0
    
    # Add popularity labels
    for species, stats in species_stats.items():
        stats['popularity'] = 'common' if stats['total_viewers'] >= median_viewers else 'rare'
        stats['popularity_score'] = stats['total_viewers'] / max(mean_viewers, 1)
    
    return species_stats


def get_all_species(transitions_df: pd.DataFrame) -> Set[str]:
    """
    Get set of all unique species across all transitions.
    
    Args:
        transitions_df: DataFrame with transition pairs
    
    Returns:
        Set of all unique species
    """
    all_species = set()
    
    for _, row in transitions_df.iterrows():
        all_species.update(row['species_from'])
        all_species.update(row['species_to'])
    
    return all_species


def inspect_available_columns(raw_data: Dict[int, pd.DataFrame], 
                            sample_size: int = 1000) -> Dict[int, List[str]]:
    """
    Inspect what columns are available in raw data.
    Useful for determining what features can be extracted.
    
    Args:
        raw_data: Dictionary mapping year to raw DataFrame
        sample_size: Number of rows to sample for inspection
    
    Returns:
        Dictionary mapping year to list of available columns
    """
    available_columns = {}
    
    for year, df in raw_data.items():
        # Sample a small portion to check columns
        sample_df = df.head(min(sample_size, len(df)))
        available_columns[year] = list(sample_df.columns)
        
        print(f"\nYear {year} - Available columns ({len(sample_df.columns)}):")
        print(f"  {sample_df.columns.tolist()}")
        
        # Check for geographic columns
        geo_cols = [c for c in sample_df.columns if any(term in c.lower() 
                    for term in ['state', 'county', 'lat', 'lon', 'location', 'region', 'country', 'locality'])]
        if geo_cols:
            print(f"  Geographic columns: {geo_cols}")
        
        # Check for temporal columns
        temporal_cols = [c for c in sample_df.columns if any(term in c.lower() 
                        for term in ['date', 'time', 'day', 'hour', 'month', 'year'])]
        if temporal_cols:
            print(f"  Temporal columns: {temporal_cols}")
        
        # Check for effort columns
        effort_cols = [c for c in sample_df.columns if any(term in c.lower() 
                      for term in ['effort', 'duration', 'distance', 'protocol'])]
        if effort_cols:
            print(f"  Effort columns: {effort_cols}")
    
    return available_columns


def extract_temporal_features_from_birder_species(transitions_df: pd.DataFrame,
                                                  birder_species_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract temporal features from birder_species summaries (sets).
    Uses available data: effort_hours, duration_minutes sets, and total_observations.
    
    Args:
        transitions_df: DataFrame with transition pairs
        birder_species_data: Dictionary mapping year to birder_species DataFrame
    
    Returns:
        DataFrame with temporal features
    """
    features = []
    
    for _, row in transitions_df.iterrows():
        observer_id = row['observer_id']
        year_from = int(row['year_from'])
        year_to = int(row['year_to'])
        
        # Get birder_species data for year_from
        if year_from in birder_species_data:
            df_year = birder_species_data[year_from]
            birder_row = df_year[df_year['observer_id'] == observer_id]
            
            if len(birder_row) > 0:
                birder_row = birder_row.iloc[0]
                feat_dict = {
                    'observer_id': observer_id,
                    'year_from': year_from,
                    'year_to': year_to
                }
                
                # Count features from sets
                species_set = birder_row.get('scientific_name', set())
                if isinstance(species_set, set):
                    feat_dict['num_species'] = len(species_set)
                else:
                    feat_dict['num_species'] = 0
                
                # Use number of unique localities as proxy for checklists
                locality_set = birder_row.get('locality_id', set())
                if isinstance(locality_set, set):
                    num_checklists = len(locality_set)  # Proxy: unique locations
                    feat_dict['num_checklists'] = num_checklists
                else:
                    num_checklists = 0
                    feat_dict['num_checklists'] = 0
                
                # Observation features
                total_obs = birder_row.get('total_observations', 0)
                feat_dict['total_observations'] = total_obs if pd.notna(total_obs) else 0
                feat_dict['avg_observations_per_checklist'] = total_obs / num_checklists if num_checklists > 0 else 0
                
                # Effort features from set
                effort_set = birder_row.get('effort_hours', set())
                if isinstance(effort_set, set) and len(effort_set) > 0:
                    effort_list = list(effort_set)
                    feat_dict['effort_mean'] = np.mean(effort_list)
                    feat_dict['effort_total'] = np.sum(effort_list) if len(effort_list) > 0 else 0
                    feat_dict['effort_max'] = np.max(effort_list)
                    feat_dict['effort_std'] = np.std(effort_list) if len(effort_list) > 1 else 0
                    feat_dict['avg_effort_hours'] = feat_dict['effort_mean']
                else:
                    feat_dict['effort_mean'] = 0.0
                    feat_dict['effort_total'] = 0.0
                    feat_dict['effort_max'] = 0.0
                    feat_dict['effort_std'] = 0.0
                    feat_dict['avg_effort_hours'] = 0.0
                
                # Duration features from set
                duration_set = birder_row.get('duration_minutes', set())
                if isinstance(duration_set, set) and len(duration_set) > 0:
                    duration_list = list(duration_set)
                    feat_dict['duration_mean'] = np.mean(duration_list)
                    feat_dict['duration_total'] = np.sum(duration_list) if len(duration_list) > 0 else 0
                    feat_dict['duration_max'] = np.max(duration_list)
                    feat_dict['duration_std'] = np.std(duration_list) if len(duration_list) > 1 else 0
                    feat_dict['avg_duration'] = feat_dict['duration_mean']
                else:
                    feat_dict['duration_mean'] = 0.0
                    feat_dict['duration_total'] = 0.0
                    feat_dict['duration_max'] = 0.0
                    feat_dict['duration_std'] = 0.0
                    feat_dict['avg_duration'] = 0.0
                
                # Checklist frequency (proxy)
                feat_dict['checklists_per_month'] = num_checklists / 12.0 if num_checklists > 0 else 0
                
                # Location diversity
                locality_set = birder_row.get('locality_id', set())
                state_set = birder_row.get('state', set())
                if isinstance(locality_set, set):
                    feat_dict['num_locations'] = len(locality_set)
                else:
                    feat_dict['num_locations'] = 0
                
                if isinstance(state_set, set):
                    feat_dict['num_states'] = len(state_set)
                else:
                    feat_dict['num_states'] = 0
                
                # Set missing temporal features to 0 (day_of_year, hours_of_day, months not available)
                feat_dict['avg_day_of_year'] = 0.0
                feat_dict['doy_std'] = 0.0
                feat_dict['doy_range'] = 0.0
                feat_dict['avg_hours_of_day'] = 0.0
                feat_dict['hod_std'] = 0.0
                for month in range(1, 13):
                    feat_dict[f'month_{month}_freq'] = 0.0
                
                features.append(feat_dict)
    
    return pd.DataFrame(features)


def extract_geographic_features_from_birder_species(transitions_df: pd.DataFrame,
                                                    birder_species_data: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
    """
    Extract geographic features from birder_species summaries (sets).
    
    Args:
        transitions_df: DataFrame with transition pairs
        birder_species_data: Dictionary mapping year to birder_species DataFrame
    
    Returns:
        Tuple of:
        - Feature matrix (n_samples, n_features) with all geographic features
        - Feature metadata dictionary
    """
    # Collect all states and counties from birder_species data
    all_states = set()
    all_counties = set()
    
    for year, df in birder_species_data.items():
        if 'state' in df.columns:
            for state_set in df['state'].dropna():
                if isinstance(state_set, set):
                    all_states.update(state_set)
        if 'county' in df.columns:
            for county_set in df['county'].dropna():
                if isinstance(county_set, set):
                    all_counties.update(county_set)
    
    # Create encoders
    state_list = sorted([s for s in all_states if pd.notna(s)])
    state_to_idx = {state: idx for idx, state in enumerate(state_list)}
    
    county_list = sorted([c for c in all_counties if pd.notna(c)]) if all_counties else []
    county_to_idx = {county: idx for idx, county in enumerate(county_list)} if county_list else {}
    
    # Feature dimensions
    n_state_features = len(state_list)
    n_county_features = len(county_list) if county_list else 0
    n_geo_metrics = 5  # state_diversity, num_states, num_counties, num_localities, geographic_spread
    
    n_samples = len(transitions_df)
    n_total_features = n_state_features + n_county_features + n_geo_metrics
    
    geo_features = np.zeros((n_samples, n_total_features), dtype=np.float32)
    
    for idx, row in transitions_df.iterrows():
        observer_id = row['observer_id']
        year_from = int(row['year_from'])
        
        if year_from in birder_species_data:
            df_year = birder_species_data[year_from]
            birder_row = df_year[df_year['observer_id'] == observer_id]
            
            if len(birder_row) > 0:
                birder_row = birder_row.iloc[0]
                
                # 1. State dummy variables
                state_set = birder_row.get('state', set())
                if isinstance(state_set, set) and len(state_set) > 0:
                    state_list_birder = list(state_set)
                    # Use first state (sets don't preserve frequency)
                    if state_list_birder:
                        most_common_state = state_list_birder[0]
                        if most_common_state in state_to_idx:
                            geo_features[idx, state_to_idx[most_common_state]] = 1.0
                    
                    # State diversity (entropy approximation - uniform distribution)
                    n_states = len(state_list_birder)
                    if n_states > 1:
                        state_entropy = -np.sum((1.0 / n_states) * np.log(1.0 / n_states + 1e-10))
                    else:
                        state_entropy = 0.0
                    geo_features[idx, n_state_features + n_county_features] = state_entropy
                    geo_features[idx, n_state_features + n_county_features + 1] = n_states
                else:
                    geo_features[idx, n_state_features + n_county_features] = 0.0
                    geo_features[idx, n_state_features + n_county_features + 1] = 0
                
                # 2. County dummy variables
                county_set = birder_row.get('county', set())
                if isinstance(county_set, set) and len(county_set) > 0:
                    county_list_birder = list(county_set)
                    if county_list_birder:
                        most_common_county = county_list_birder[0]
                        if county_list and most_common_county in county_to_idx:
                            geo_features[idx, n_state_features + county_to_idx[most_common_county]] = 1.0
                    geo_features[idx, n_state_features + n_county_features + 2] = len(county_list_birder)
                else:
                    geo_features[idx, n_state_features + n_county_features + 2] = 0
                
                # 3. Location diversity metrics
                locality_set = birder_row.get('locality_id', set())
                if isinstance(locality_set, set):
                    num_localities = len(locality_set)
                else:
                    num_localities = 0
                geo_features[idx, n_state_features + n_county_features + 3] = num_localities
                
                # 4. Geographic spread (using locality count as proxy - lat/lon not available)
                geo_features[idx, n_state_features + n_county_features + 4] = num_localities
    
    metadata = {
        'n_state_features': n_state_features,
        'n_county_features': n_county_features,
        'n_geo_metrics': n_geo_metrics,
        'state_to_idx': state_to_idx,
        'county_to_idx': county_to_idx,
        'state_list': state_list,
        'county_list': county_list
    }
    
    return geo_features, metadata


if __name__ == "__main__":
    # This will be tested with actual data
    print("Feature engineering module loaded successfully")
