#!/usr/bin/env python3
"""
Extract a subset of data from birder_species parquet files (2018-2023)
filtered to birders who visited counties within the Northeast region of the United States.

This script reads the pre-processed birder_species_YYYY.parquet files,
filters for birders who visited Northeast states, and creates regional summaries.
"""
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Optional, Set
import sys
from data_loader import create_transition_pairs

# Northeast region states (US Census Bureau definition)
NORTHEAST_STATES = {
    'Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 
    'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania'
}

NORTHEAST_STATE_ABBR = {
    'ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'
}


def is_northeast_birder(state_set) -> bool:
    """
    Check if a birder visited any Northeast states.
    
    Args:
        state_set: Set, list, or array of state names visited by the birder
                   (parquet files may store sets as lists/arrays)
    
    Returns:
        True if birder visited any Northeast state
    """
    # Handle None, NaN, or empty values
    if state_set is None:
        return False
    
    # Convert to list if it's a set, array, or other iterable
    try:
        if isinstance(state_set, (list, tuple)):
            states = state_set
        elif isinstance(state_set, set):
            states = list(state_set)
        else:
            # Try to convert (handles numpy arrays, pandas Series, etc.)
            states = list(state_set)
    except (TypeError, ValueError):
        return False
    
    # Check if empty
    if len(states) == 0:
        return False
    
    # Check if any state in the set is a Northeast state
    for state in states:
        if isinstance(state, str):
            state_clean = state.strip()
            if state_clean in NORTHEAST_STATES:
                return True
    
    return False


def filter_birder_species_northeast(year: int) -> pd.DataFrame:
    """
    Load birder_species file for a year and filter for Northeast birders.
    
    Args:
        year: Year to process (2018-2023)
    
    Returns:
        Filtered DataFrame with Northeast birders only (includes all columns from birder_species file)
    """
    birder_species_file = Path(f"birder_species_{year}.parquet")
    
    if not birder_species_file.exists():
        raise FileNotFoundError(f"birder_species file not found: {birder_species_file}")
    
    print(f"\n{'='*60}")
    print(f"Processing year {year}")
    print(f"{'='*60}")
    print(f"Loading {birder_species_file}...")
    
    # Load birder_species file
    df = pd.read_parquet(birder_species_file)
    total_birders = len(df)
    
    print(f"  Loaded {total_birders:,} birder-year pairs")
    
    # Filter for birders who visited Northeast states
    # The 'state' column contains a set of states visited
    df_northeast = df[df['state'].apply(is_northeast_birder)].copy()
    filtered_birders = len(df_northeast)
    
    print(f"\nYear {year} summary:")
    print(f"  Total birders: {total_birders:,}")
    print(f"  Northeast birders: {filtered_birders:,} ({filtered_birders/total_birders*100:.2f}%)")
    
    if filtered_birders > 0:
        # Show sample of states found
        sample_states = set()
        for state_set in df_northeast['state'].head(100):
            if isinstance(state_set, set):
                sample_states.update(state_set)
        northeast_states_found = [s for s in sample_states if s in NORTHEAST_STATES]
        print(f"  Sample Northeast states found: {sorted(northeast_states_found)[:5]}")
    
    return df_northeast


def extract_northeast_data(years: List[int] = [2018, 2019, 2020, 2021, 2022, 2023],
                          save_summaries: bool = True) -> None:
    """
    Extract Northeast regional data for multiple years from birder_species files.
    
    Args:
        years: List of years to process
        save_summaries: Whether to save birder_species summaries
    """
    print("="*60)
    print("Extracting Northeast Regional Data")
    print("="*60)
    print(f"Years: {years}")
    print(f"Northeast states: {', '.join(sorted(NORTHEAST_STATES))}")
    print("="*60)
    print("\nNote: This script reads from birder_species_YYYY.parquet files")
    print("      and filters for birders who visited Northeast states.")
    print("="*60)
    
    birder_species_summaries = []
    
    for year in years:
        try:
            # Filter birder_species data for Northeast
            df_northeast = filter_birder_species_northeast(year)
            
            if len(df_northeast) == 0:
                print(f"  No Northeast birders found for year {year}, skipping...")
                continue
            
            # Save individual year summary (includes all features from birder_species file)
            if save_summaries:
                output_file = f"birder_species_{year}_northeast.parquet"
                df_northeast.to_parquet(output_file, compression='snappy')
                print(f"\n  ✓ Saved: {output_file} ({len(df_northeast):,} rows)")
                print(f"    Columns: {df_northeast.columns.tolist()}")
                
                birder_species_summaries.append(df_northeast)
            
            # Free memory
            del df_northeast
            
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print(f"  Make sure birder_species_{year}.parquet exists in the current directory.")
            continue
        except Exception as e:
            print(f"\nERROR processing year {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all year summaries
    if save_summaries and birder_species_summaries:
        print(f"\n{'='*60}")
        print("Combining all year summaries")
        print(f"{'='*60}")
        
        combined_summary = pd.concat(birder_species_summaries, ignore_index=True)
        output_file = "birder_species_2018_2023_northeast.parquet"
        combined_summary.to_parquet(output_file, compression='snappy')
        print(f"✓ Saved combined summary: {output_file}")
        print(f"Total birder-year pairs: {len(combined_summary):,}")
        print(f"Unique observers: {combined_summary['observer_id'].nunique():,}")
        print(f"Years covered: {sorted(combined_summary['year'].unique())}")
        
        # Create transition pairs from regional data
        print(f"\n{'='*60}")
        print("Creating transition pairs for regional data")
        print(f"{'='*60}")
        try:
            transitions = create_transition_pairs(combined_summary)
            transitions_file = "transitions_northeast.parquet"
            transitions.to_parquet(transitions_file, compression='snappy')
            print(f"✓ Saved transitions: {transitions_file} ({len(transitions):,} rows)")
            print(f"\nYear transitions:")
            print(transitions[['year_from', 'year_to']].drop_duplicates().sort_values('year_from'))
        except Exception as e:
            print(f"Warning: Could not create transitions: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Extraction Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Northeast regional data from birder species datasets"
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2018, 2019, 2020, 2021, 2022, 2023],
        help='Years to process (default: 2018-2023)'
    )
    parser.add_argument(
        '--no-summaries',
        action='store_true',
        help='Skip saving birder_species summaries'
    )
    
    args = parser.parse_args()
    
    extract_northeast_data(
        years=args.years,
        save_summaries=not args.no_summaries
    )
