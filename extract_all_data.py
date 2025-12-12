#!/usr/bin/env python3
"""
Extract all required data columns for feature extraction.
This script will FAIL loudly if any required columns are missing.

Test mode: Run with --test flag to process only 1 file per year for quick verification.
Example: python3 extract_all_data.py --test
"""
import sys
import pandas as pd
import argparse
from data_loader import load_year_data_incremental, create_transition_pairs

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract birder-species data with features')
parser.add_argument('--test', action='store_true', 
                    help='Test mode: process only 1 file per year for quick verification')
args = parser.parse_args()

test_mode = args.test
if test_mode:
    print("="*60)
    print("TEST MODE: Processing only 1 file per year")
    print("="*60)
    print()

print("="*60)
print("Data Extraction for Feature Engineering")
print("="*60)
print("This script will:")
print("  1. Load raw data with ALL required columns")
print("  2. Validate all columns exist (will FAIL loudly if missing)")
print("  3. Process and save data summaries with features:")
print("     - Species observed (set)")
print("     - States visited (set)")
print("     - Counties visited (set)")
print("     - Locations visited (set)")
print("     - Effort hours (set)")
print("     - Effort distance km (set)")
print("     - Duration minutes (set)")
print("     - Number of observers (set)")
print("     - Total observations (sum of bird counts)")
print("="*60)
print()

# Step 1: Inspect columns in sample files
print("="*60)
print("Step 1: Inspecting available columns in raw data")
print("="*60)

from pathlib import Path

for year in [2018, 2019, 2020, 2021, 2022, 2023]:
    print(f"\nChecking year {year}...")
    year_dir = Path(str(year))
    parquet_files = sorted(list(year_dir.glob("*.parquet")))
    
    if not parquet_files:
        print(f"  ERROR: No parquet files found in {year_dir}")
        sys.exit(1)
    
    # Read first file to check columns (read full file then take first row)
    df_sample = pd.read_parquet(parquet_files[0]).head(1)
    print(f"  Sample file: {parquet_files[0].name}")
    print(f"  Available columns ({len(df_sample.columns)}): {df_sample.columns.tolist()}")
    
    # Check required columns
    required_cols = [
        'observer_id', 'scientific_name', 'year', 'species_observed',
        'checklist_id', 'observation_date', 'observation_count', 'state',
        'day_of_year', 'hours_of_day', 'effort_hours', 'duration_minutes',
        'locality_id', 'county', 'latitude', 'longitude'
    ]
    missing = set(required_cols) - set(df_sample.columns)
    if missing:
        print(f"  ERROR: Missing required columns: {missing}")
        sys.exit(1)
    else:
        print(f"  ✓ All required columns present")

# Step 2: Process each year individually to save memory
print("\n" + "="*60)
print("Step 2: Processing years individually (memory-efficient)")
print("="*60)
print("This will FAIL loudly if any required columns are missing...")
print()

years = [2018, 2019, 2020, 2021, 2022, 2023]

# Process each year individually: load, process, save, then free memory
for year in years:
    print(f"\n{'='*60}")
    print(f"Processing year {year}")
    print(f"{'='*60}")
    
    try:
        # Load year data
        print(f"Loading year {year}...")
        max_files = 1 if test_mode else None
        if test_mode:
            print(f"  TEST MODE: Processing only 1 file per year")
        df = load_year_data_incremental(year, max_files=max_files)
        print(f"  ✓ Loaded {len(df):,} rows, {df['observer_id'].nunique():,} unique birders")
        print(f"    Columns: {df.columns.tolist()}")
        
        # Extract birder-species for this year with regional data and features
        print(f"Extracting birder-species interactions for year {year}...")
        print("  Including: species (set), states (set), counties (set), locations (set),")
        print("             effort_hours (set), effort_distance_km (set), duration_minutes (set),")
        print("             number_observers (set), total observations...")
        
        # Group by observer_id and aggregate all features
        birder_species_year = df.groupby('observer_id').agg({
            'scientific_name': lambda x: set(x.unique()),  # Set of all species observed
            'state': lambda x: set(x.dropna().unique()),  # Set of all states visited
            'county': lambda x: set(x.dropna().unique()),  # Set of all counties visited
            'locality_id': lambda x: set(x.dropna().unique()),  # Set of all locations visited
            'effort_hours': lambda x: set(x.dropna().unique()),  # Set of unique effort hours
            'effort_distance_km': lambda x: set(x.dropna().unique()),  # Set of unique effort distances
            'duration_minutes': lambda x: set(x.dropna().unique()),  # Set of unique durations
            'number_observers': lambda x: set(x.dropna().unique()),  # Set of unique observer counts (social preference)
            'observation_count': 'sum'  # Total number of birds observed (raw count)
        }).reset_index()
        
        birder_species_year.columns = ['observer_id', 'scientific_name', 'state', 'county', 'locality_id', 
                                        'effort_hours', 'effort_distance_km', 'duration_minutes', 
                                        'number_observers', 'total_observations']
        birder_species_year['year'] = year
        
        # Save individual year summary
        output_file = f'birder_species_{year}.parquet'
        birder_species_year.to_parquet(output_file, compression='snappy')
        print(f"  ✓ Saved: {output_file} ({len(birder_species_year):,} rows)")
        print(f"    Columns: {birder_species_year.columns.tolist()}")
        
        # Free memory
        del df
        del birder_species_year
        
    except Exception as e:
        print(f"\nERROR processing year {year}: {e}")
        print("\nThis error means required columns are missing from the parquet files.")
        print("Please check the parquet file schema and ensure all required columns exist.")
        sys.exit(1)

# Step 3: Combine all year summaries
print("\n" + "="*60)
print("Step 3: Combining all year summaries")
print("="*60)

try:
    birder_species_list = []
    for year in years:
        year_file = f'birder_species_{year}.parquet'
        print(f"Loading {year_file}...")
        df_year = pd.read_parquet(year_file)
        birder_species_list.append(df_year)
    
    birder_species = pd.concat(birder_species_list, ignore_index=True)
    print(f"✓ Combined birder-species summaries: {birder_species.shape}")
    
    # Save combined summary
    output_file = 'all_birder_species.parquet'
    birder_species.to_parquet(output_file, compression='snappy')
    print(f"✓ Saved combined summary: {output_file} ({len(birder_species):,} rows)")
    
except Exception as e:
    print(f"\nERROR combining summaries: {e}")
    sys.exit(1)

# Step 4: Create transition pairs
print("\n" + "="*60)
print("Step 4: Creating transition pairs")
print("="*60)

try:
    transitions = create_transition_pairs(birder_species)
    print(f"✓ Created transition pairs: {transitions.shape}")
    
    # Save transitions
    transitions_file = 'transitions.parquet'
    transitions.to_parquet(transitions_file, compression='snappy')
    print(f"✓ Saved transitions: {transitions_file} ({len(transitions):,} rows)")
    
    print(f"\nYear transitions:")
    print(transitions[['year_from', 'year_to']].drop_duplicates().sort_values('year_from'))
    
except Exception as e:
    print(f"\nERROR creating transition pairs: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Data Extraction Complete!")
print("="*60)
print("\nAll required columns validated and data extracted successfully.")
print("\nNext step: Run training with features")
print("  sbatch batch_process_years.sh")


