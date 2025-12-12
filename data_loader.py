"""
Data loading utilities for birding checklist data.
Memory-efficient loading of parquet files from 2018-2023.
Processes files incrementally to avoid memory issues.
"""
import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np


def load_year_data_incremental(year: int, 
                               max_files: Optional[int] = None,
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load parquet files incrementally, processing in chunks to save memory.
    Only loads specified columns to reduce memory usage.
    
    Args:
        year: Year to load (2018-2023)
        max_files: Maximum number of files to load (for testing). None = load all.
        columns: List of columns to load. If None, loads only essential columns.
    
    Returns:
        Combined DataFrame for the year
    """
    if columns is None:
        # Load all columns needed for feature extraction
        # REQUIRED columns - will fail if missing
        columns = [
            'observer_id', 
            'scientific_name', 
            'year', 
            'species_observed', 
            'checklist_id', 
            'observation_date', 
            'observation_count', 
            'state',
            'day_of_year',      # Required for temporal features
            'hours_of_day',     # Required for temporal features
            'effort_hours',     # Required for effort features
            'effort_distance_km',  # Required for effort distance features
            'duration_minutes', # Required for duration features
            'number_observers',  # Required for social preference features
            'locality_id',      # Required for location diversity
            'county',           # Required for geographic features
            'latitude',         # Required for geographic spread
            'longitude'         # Required for geographic spread
        ]
    
    year_dir = Path(str(year))
    if not year_dir.exists():
        raise ValueError(f"Directory {year_dir} does not exist")
    
    # Find all parquet files for this year
    parquet_files = sorted(list(year_dir.glob("*.parquet")))
    
    if max_files:
        parquet_files = parquet_files[:max_files]
    
    print(f"Loading {len(parquet_files)} files for year {year} (columns: {columns})...")
    
    # Process files incrementally and aggregate
    all_data = []
    total_rows = 0
    
    for i, file_path in enumerate(parquet_files):
        try:
            # Read only specified columns - will fail if columns don't exist
            df = pd.read_parquet(file_path, columns=columns)
            
            # Validate all required columns are present
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in {file_path}: {missing_cols}\n"
                    f"Available columns: {df.columns.tolist()}"
                )
            
            # Filter to only observed species immediately to reduce memory
            df = df[df['species_observed'] == True].copy()
            
            # Drop species_observed column since we've filtered
            df = df.drop(columns=['species_observed'], errors='ignore')
            
            all_data.append(df)
            total_rows += len(df)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(parquet_files)} files ({total_rows:,} rows so far)...")
                
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No data loaded for year {year}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df):,} rows for year {year}")
    
    return combined_df


def load_year_data(year: int, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Wrapper for backward compatibility. Uses incremental loading.
    """
    return load_year_data_incremental(year, max_files=max_files)


def load_all_years(years: List[int] = [2018, 2019, 2020, 2021, 2022, 2023],
                   max_files_per_year: Optional[int] = None) -> Dict[int, pd.DataFrame]:
    """
    Load data for multiple years.
    
    Args:
        years: List of years to load
        max_files_per_year: Maximum files per year (for testing)
    
    Returns:
        Dictionary mapping year to DataFrame
    """
    data = {}
    for year in years:
        try:
            data[year] = load_year_data(year, max_files=max_files_per_year)
        except Exception as e:
            print(f"Error loading year {year}: {e}")
            continue
    
    return data


def get_birder_species_by_year(data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract birder-species interactions by year.
    Creates a summary: for each birder-year pair, which species were observed.
    Memory-efficient: processes each year separately.
    
    Args:
        data: Dictionary mapping year to DataFrame (should already be filtered to observed species)
    
    Returns:
        DataFrame with columns: observer_id, year, scientific_name (set of species)
    """
    birder_species = []
    
    for year, df in data.items():
        print(f"Processing year {year} ({len(df):,} rows)...")
        
        # Group by observer_id and get unique species
        # Use a more memory-efficient approach
        grouped = df.groupby('observer_id')['scientific_name'].apply(
            lambda x: set(x.unique())
        ).reset_index()
        grouped.columns = ['observer_id', 'scientific_name']
        grouped['year'] = year
        
        birder_species.append(grouped)
        
        # Clear memory
        del df
    
    result = pd.concat(birder_species, ignore_index=True)
    return result


def combine_year_summaries(years: List[int], 
                           summary_dir: str = ".") -> pd.DataFrame:
    """
    Combine pre-processed year summaries from parquet files.
    Useful after batch processing individual years.
    
    Args:
        years: List of years to combine
        summary_dir: Directory containing birder_species_YYYY.parquet files
    
    Returns:
        Combined DataFrame with all birder-species summaries
    """
    summaries = []
    
    for year in years:
        summary_file = Path(summary_dir) / f"birder_species_{year}.parquet"
        if summary_file.exists():
            print(f"Loading summary for year {year}...")
            df = pd.read_parquet(summary_file)
            summaries.append(df)
        else:
            print(f"Warning: Summary file not found for year {year}: {summary_file}")
    
    if not summaries:
        raise ValueError("No summary files found")
    
    result = pd.concat(summaries, ignore_index=True)
    print(f"Combined {len(summaries)} year summaries: {len(result):,} birder-year pairs")
    return result


def create_transition_pairs(birder_species_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create year-over-year transition pairs.
    For each birder with data in consecutive years, create pairs:
    (species_viewed_year_N, species_viewed_year_N+1)
    
    Args:
        birder_species_df: DataFrame from get_birder_species_by_year
    
    Returns:
        DataFrame with columns: observer_id, year_from, year_to, 
                               species_from (set), species_to (set)
    """
    transitions = []
    
    # Sort by observer_id and year
    df_sorted = birder_species_df.sort_values(['observer_id', 'year'])
    
    # Group by observer_id
    for observer_id, group in df_sorted.groupby('observer_id'):
        years = sorted(group['year'].unique())
        
        # Create pairs for consecutive years
        for i in range(len(years) - 1):
            year_from = years[i]
            year_to = years[i + 1]
            
            species_from = group[group['year'] == year_from]['scientific_name'].iloc[0]
            species_to = group[group['year'] == year_to]['scientific_name'].iloc[0]
            
            transitions.append({
                'observer_id': observer_id,
                'year_from': year_from,
                'year_to': year_to,
                'species_from': species_from,
                'species_to': species_to
            })
    
    return pd.DataFrame(transitions)


def process_year_incremental(year: int, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Process a single year incrementally and optionally save intermediate results.
    Useful for batch processing on servers.
    
    Args:
        year: Year to process
        output_file: Optional path to save birder-species summary
    
    Returns:
        DataFrame with birder-species summary for the year
    """
    print(f"\n{'='*60}")
    print(f"Processing year {year}")
    print(f"{'='*60}")
    
    # Load year data (only essential columns)
    df = load_year_data_incremental(year)
    
    # Extract birder-species
    print(f"Extracting birder-species interactions for year {year}...")
    birder_species = df.groupby('observer_id')['scientific_name'].apply(
        lambda x: set(x.unique())
    ).reset_index()
    birder_species.columns = ['observer_id', 'scientific_name']
    birder_species['year'] = year
    
    if output_file:
        birder_species.to_parquet(output_file, compression='snappy')
        print(f"Saved to {output_file}")
    
    return birder_species


if __name__ == "__main__":
    import sys
    
    # Check if running in batch mode (process single year)
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        output_file = f"birder_species_{year}.parquet"
        process_year_incremental(year, output_file)
    else:
        # Test loading with small sample
        print("Testing data loader with small sample...")
        print("For full processing, run: python data_loader.py <year>")
        
        # Load a small sample first
        print("\nLoading sample data (first 2 files per year)...")
        data = load_all_years(max_files_per_year=2)
        
        print(f"\nLoaded {len(data)} years")
        for year, df in data.items():
            print(f"  Year {year}: {df.shape[0]:,} rows, {df['observer_id'].nunique():,} unique birders")
        
        # Extract birder-species by year
        print("\nExtracting birder-species interactions...")
        birder_species = get_birder_species_by_year(data)
        print(f"Birder-species summary shape: {birder_species.shape}")
        
        # Create transition pairs
        print("\nCreating transition pairs...")
        transitions = create_transition_pairs(birder_species)
        print(f"Transition pairs shape: {transitions.shape}")
        print("\nSample transitions:")
        print(transitions.head())

