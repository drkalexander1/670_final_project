#!/usr/bin/env python3
"""
Combine birder_species summaries and create transition pairs for training.
Run this after processing all years.
"""
import pandas as pd
from data_loader import combine_year_summaries, create_transition_pairs

print("="*60)
print("Step 1: Combining year summaries")
print("="*60)

# Combine all year summaries
years = [2018, 2019, 2020, 2021, 2022, 2023]
birder_species = combine_year_summaries(years)

print(f"\nCombined summary shape: {birder_species.shape}")
print(f"Sample data:")
print(birder_species.head())

# Save combined summary
output_file = 'all_birder_species.parquet'
birder_species.to_parquet(output_file, compression='snappy')
print(f"\nSaved combined summary to: {output_file}")

print("\n" + "="*60)
print("Step 2: Creating transition pairs")
print("="*60)

# Create transition pairs
transitions = create_transition_pairs(birder_species)

print(f"\nTransition pairs shape: {transitions.shape}")
print(f"\nYear transitions available:")
print(transitions[['year_from', 'year_to']].drop_duplicates().sort_values('year_from'))

print(f"\nSample transitions:")
print(transitions.head(10))

# Save transitions
transitions_file = 'transitions.parquet'
transitions.to_parquet(transitions_file, compression='snappy')
print(f"\nSaved transitions to: {transitions_file}")

# Print statistics
print("\n" + "="*60)
print("Statistics")
print("="*60)
print(f"Total birder-year pairs: {len(birder_species):,}")
print(f"Total transition pairs: {len(transitions):,}")
print(f"Unique birders: {birder_species['observer_id'].nunique():,}")

# Count species per birder-year
species_counts = []
for _, row in birder_species.iterrows():
    species_counts.append(len(row['scientific_name']))

print(f"\nSpecies per birder-year:")
print(f"  Mean: {pd.Series(species_counts).mean():.2f}")
print(f"  Median: {pd.Series(species_counts).median():.2f}")
print(f"  Min: {pd.Series(species_counts).min()}")
print(f"  Max: {pd.Series(species_counts).max()}")

print("\n" + "="*60)
print("Done! Ready for training.")
print("="*60)
print("\nNext steps:")
print("1. Open 670_final_project.ipynb")
print("2. Load transitions: transitions = pd.read_parquet('transitions.parquet')")
print("3. Run the training pipeline")

