"""
Simplified feature engineering that works with birder_species summaries.
Extracts features that don't require raw data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Set
from collections import Counter


def extract_features_from_summaries(birder_species_df: pd.DataFrame, 
                                    transitions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from birder_species summaries (no raw data needed).
    
    Args:
        birder_species_df: DataFrame with birder-species by year
        transitions_df: DataFrame with transition pairs
    
    Returns:
        DataFrame with features for each transition
    """
    features = []
    
    # Create birder activity by year
    birder_activity = {}
    for _, row in birder_species_df.iterrows():
        key = (row['observer_id'], row['year'])
        birder_activity[key] = len(row['scientific_name'])
    
    # Get all species for popularity
    all_species_counts = Counter()
    for _, row in birder_species_df.iterrows():
        all_species_counts.update(row['scientific_name'])
    
    for _, trans_row in transitions_df.iterrows():
        observer_id = trans_row['observer_id']
        year_from = trans_row['year_from']
        year_to = trans_row['year_to']
        
        # Activity features
        activity_key = (observer_id, year_from)
        num_species_viewed = birder_activity.get(activity_key, 0)
        
        # Species diversity (unique species in year_from)
        species_from = trans_row['species_from']
        num_unique_species = len(species_from)
        
        # Popularity of viewed species (average)
        if len(species_from) > 0:
            avg_popularity = np.mean([all_species_counts.get(s, 0) for s in species_from])
        else:
            avg_popularity = 0
        
        # Year features
        year_numeric = year_from
        
        features.append({
            'observer_id': observer_id,
            'year_from': year_from,
            'year_to': year_to,
            'num_species': num_species_viewed,
            'num_unique_species': num_unique_species,
            'avg_species_popularity': avg_popularity,
            'year': year_numeric
        })
    
    return pd.DataFrame(features)


def create_species_popularity_features(transitions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Create species popularity scores from transitions.
    
    Args:
        transitions_df: DataFrame with transition pairs
    
    Returns:
        Dictionary mapping species to popularity score
    """
    species_counts = Counter()
    
    for _, row in transitions_df.iterrows():
        species_counts.update(row['species_from'])
        species_counts.update(row['species_to'])
    
    # Normalize to 0-1 scale
    if len(species_counts) > 0:
        max_count = max(species_counts.values())
        min_count = min(species_counts.values())
        range_count = max_count - min_count if max_count > min_count else 1
        
        species_scores = {}
        for species, count in species_counts.items():
            species_scores[species] = (count - min_count) / range_count
    else:
        species_scores = {}
    
    return species_scores


