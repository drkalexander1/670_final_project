# Execution Order for Server Batch Submission

## Quick Start

Since your data files already exist, you can **skip directly to Step 3**:

```bash
cd /home/drkalex/670_final_project
sbatch batch_process_years.sh
```

## Complete Workflow (For Reference)

### Step 1: Data Preparation ✅ (Already Complete)

**Status**: Your files already exist:
- ✅ `birder_species_2018.parquet` through `birder_species_2023.parquet`
- ✅ `all_birder_species.parquet`
- ✅ `transitions.parquet`

**If you need to re-run** (not needed now):
```bash
# Process individual years (parallel)
for year in 2018 2019 2020 2021 2022 2023; do
    python3 data_loader.py $year
done

# Combine and create transitions
python3 combine_and_prepare.py
```

### Step 2: Verify Prerequisites

Check that required files exist:
```bash
cd /home/drkalex/670_final_project
ls -lh transitions.parquet all_birder_species.parquet birder_species_*.parquet
```

### Step 3: Run Training with Enhanced Features 🚀

**This is what you need to run now:**

```bash
cd /home/drkalex/670_final_project
sbatch batch_process_years.sh
```

**What this does:**
1. Loads `transitions.parquet` (training data)
2. Loads raw data for feature extraction (samples 2 files per year)
3. Extracts enhanced geographic features (states, counties, metrics)
4. Extracts comprehensive temporal features (day/year, hours, months, effort)
5. Trains models with and without features
6. Generates visualizations and reports

**Expected runtime**: ~2-6 hours depending on data size

**Output location**: `results/` directory

### Step 4: Check Results

After job completes, check:
```bash
# Check job status
squeue -u $USER

# View output
tail -f results/bird_training_*.out

# Check results
ls -lh results/*.png results/*.txt
```

## Feature Extraction Details

The training script (`run_training_with_features.py`) currently:
- **Samples**: 2 files per year for feature extraction (for speed)
- **Extracts**: 
  - Geographic: State dummy variables, county (if available), diversity metrics
  - Temporal: Day of year, hours, months, effort, duration, observations
  - ~30+ temporal features + variable geographic features

**To use more data** (better features, more memory):
Edit `run_training_with_features.py` line 42:
```python
raw_data = load_all_years(max_files_per_year=10)  # Increase from 2
```

## Summary

**Current Status**: ✅ Ready to run training

**Action Required**: 
```bash
sbatch batch_process_years.sh
```

**Expected Output**:
- `results/evaluation_report_with_features.txt`
- `results/cv_results_with_features.png`
- `results/model_comparison_with_features.png`
- `results/count_prediction_results.png`
- `results/count_prediction_comparison.png`




