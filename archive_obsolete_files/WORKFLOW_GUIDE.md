# Workflow Guide for Server Batch Submission

## Overview

This guide explains the correct order of operations for running the bird species prediction project on a server using batch submission.

## Prerequisites

Before running training, ensure you have:
1. ✅ Processed individual years → `birder_species_YYYY.parquet` files (already done)
2. ✅ Combined year summaries → `all_birder_species.parquet` (already done)
3. ✅ Created transition pairs → `transitions.parquet` (already done)

## Execution Order

### Step 1: Data Preparation (Already Complete)
If you need to re-run data preparation:

```bash
# Process individual years (can run in parallel)
sbatch --array=0-5 batch_process_years.sh  # Process years 2018-2023

# After all years are processed, combine and create transitions:
python3 combine_and_prepare.py
```

**Status**: ✅ Already complete - you have `transitions.parquet`

### Step 2: Training with Enhanced Features

The main training script (`run_training_with_features.py`) does the following:
1. Loads `transitions.parquet`
2. Loads raw data for feature extraction (geographic, temporal features)
3. Trains models with and without features
4. Generates visualizations and reports

**Important**: The script currently loads only 2 files per year for feature extraction (`max_files_per_year=2`). For full feature extraction, you may want to load more data, but this requires more memory.

### Step 3: Batch Submission

Submit the batch job:

```bash
cd /home/drkalex/670_final_project
sbatch batch_process_years.sh
```

## Memory Considerations

- **Current batch script**: Allocates 64GB RAM per CPU
- **Feature extraction**: Currently samples 2 files per year (for speed)
- **Full feature extraction**: Would require loading more raw data files

## Options for Feature Extraction

### Option A: Sample Data (Current - Fast)
- Loads 2 files per year
- Faster, uses less memory
- Good for testing enhanced features
- **Current setting in `run_training_with_features.py`**

### Option B: More Data (Better Features)
- Load more files per year (e.g., 10-20 files)
- Better geographic/temporal feature coverage
- Requires more memory
- **Modify**: Change `max_files_per_year=2` to higher value

### Option C: Full Data (Best Features)
- Load all files per year
- Complete feature extraction
- Requires significant memory (may need to process in chunks)
- **Not recommended** unless you have 100+ GB RAM

## Expected Output Files

After running `run_training_with_features.py`, you'll get:

```
results/
├── evaluation_report_with_features.txt
├── cv_results_with_features.png
├── model_comparison_with_features.png
├── count_prediction_results.png
└── count_prediction_comparison.png
```

## Troubleshooting

### If transitions.parquet is missing:
```bash
python3 combine_and_prepare.py
```

### If you need to process individual years:
```bash
# Process a single year
python3 data_loader.py 2018

# Or use batch array job
sbatch --array=0-5 batch_process_years.sh
```

### To inspect available columns for feature extraction:
```python
from data_loader import load_all_years
from feature_engineering import inspect_available_columns

# Load sample
raw_data = load_all_years(max_files_per_year=2)
inspect_available_columns(raw_data)
```

## Current Status

✅ Data preparation complete
✅ Ready to run training
→ Submit batch job: `sbatch batch_process_years.sh`




