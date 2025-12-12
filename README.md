# Bird Species Prediction - Final Project

## Project Overview

This project predicts which bird species a birder will view next year based on their viewing patterns from the current year. The dataset contains birding checklists from 2018-2023 with approximately 13.6 billion observations.

## Project Structure

```
670_final_project/
├── data_loader.py              # Memory-efficient data loading utilities
├── feature_engineering.py       # Feature extraction and engineering
├── prepare_training_data.py    # Time-series CV split preparation
├── models.py                   # ML model implementations
├── train.py                    # Training and evaluation functions
├── evaluate.py                 # Visualization and analysis tools
├── 670_final_project.ipynb    # Main project notebook
├── batch_process_years.sh      # SLURM batch script for server processing
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── README_DATA_PROCESSING.md   # Data processing guide
└── [year directories]/         # Parquet files organized by year
```

## Quick Start

### 1. Set Up Environment

```bash
cd /home/drkalex/670_final_project
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Process Data (Choose One)

**Option A: Batch Processing (Recommended for Full Dataset)**
```bash
sbatch batch_process_years.sh
# Then combine summaries:
python3 -c "from data_loader import combine_year_summaries; \
           df = combine_year_summaries([2018,2019,2020,2021,2022,2023]); \
           df.to_parquet('all_birder_species.parquet')"
```

**Option B: Sample Data (For Testing)**
```python
from data_loader import load_all_years, get_birder_species_by_year, create_transition_pairs

# Load sample (first 2 files per year)
data = load_all_years(max_files_per_year=2)
birder_species = get_birder_species_by_year(data)
transitions = create_transition_pairs(birder_species)
```

### 3. Run Main Notebook

```bash
jupyter notebook 670_final_project.ipynb
```

Or run individual components:

```python
from train import train_and_evaluate_cv
from evaluate import plot_cv_results, generate_summary_report

# Train models
results = train_and_evaluate_cv(transitions)

# Visualize results
plot_cv_results(results)
generate_summary_report(results)
```

## Models Implemented

1. **Baseline Popularity Model**: Predicts most popular species overall
2. **Collaborative Filtering (NMF)**: Matrix factorization to learn birder and species embeddings
3. **Neural Network**: Deep learning model with embeddings and hidden layers

## Evaluation Metrics

- **Precision@K**: Fraction of predicted species that are correct
- **Recall@K**: Fraction of actual species that were predicted
- **MAP@K**: Mean Average Precision at K
- **Coverage**: Diversity of predicted species

## Time-Series Cross-Validation

Uses 4-fold time-series CV to handle temporal dependencies and potential 2020 outlier:

- Fold 1: Train (2018→2019), Test (2019→2020)
- Fold 2: Train (2018→2019, 2019→2020), Test (2020→2021)
- Fold 3: Train (2018→2019, 2019→2020, 2020→2021), Test (2021→2022)
- Fold 4: Train (2018→2019, 2019→2020, 2020→2021, 2021→2022), Test (2022→2023)

## Memory Considerations

The full dataset is very large (~13.6B rows). The data loader is optimized for memory efficiency:

- Processes files incrementally
- Only loads essential columns
- Filters to observed species immediately
- Supports batch processing for server environments

See `README_DATA_PROCESSING.md` for detailed memory management guidance.

## Output Files

After running the notebook, you'll get:

- `cv_results.png`: Box plots of metrics across CV folds
- `model_comparison.png`: Bar chart comparing models
- `activity_analysis.png`: Performance by birder activity level
- `evaluation_report.txt`: Text summary of results

## Key Features

- Memory-efficient data processing
- Time-series cross-validation
- Multiple model architectures
- Comprehensive evaluation metrics
- Activity-level analysis
- Visualization tools

## Notes

- For development/testing, use `max_files_per_year=2` to limit data size
- Full dataset processing requires significant memory (8-16GB per year)
- Batch submission recommended for server environments
- Results may vary based on data sample size

