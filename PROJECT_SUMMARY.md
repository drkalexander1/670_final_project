# Bird Species Prediction Project - Complete Summary

## Project Overview

This project predicts which bird species a birder will view **next year** based on their viewing patterns from the **current year**. We use machine learning to learn from historical birding checklists (2018-2023) and predict future viewing behavior.

**Key Question**: Given what bird species a birder viewed this year, what will they view next year?

## Dataset

- **Source**: Birding checklists from 2018-2023
- **Total Observations**: ~13.6 billion rows
- **Format**: Parquet files organized by year, quarter, and groups
- **Key Fields**: 
  - `observer_id`: Unique birder identifier
  - `scientific_name`: Bird species name
  - `year`: Year of observation
  - `species_observed`: Boolean indicating if species was seen

## Project Architecture

### Data Pipeline

1. **Data Loading** (`data_loader.py`)
   - Memory-efficient loading of parquet files
   - Processes files incrementally to handle large datasets
   - Extracts birder-species interactions by year
   - Creates year-over-year transition pairs

2. **Feature Engineering** (`feature_engineering.py`)
   - Creates birder-species interaction matrices
   - Extracts temporal features (seasonality, viewing frequency)
   - Calculates species popularity metrics

3. **Training Data Preparation** (`prepare_training_data.py`)
   - Creates time-series cross-validation splits
   - Handles sequential year transitions for testing
   - Allows non-sequential transitions for training

### Machine Learning Models

We implement and compare **three different approaches**:

#### 1. Baseline Popularity Model
- **Approach**: Simple recommendation based on overall species popularity
- **Method**: Predicts the most commonly viewed species that the birder hasn't seen yet
- **Use Case**: Baseline for comparison

#### 2. Collaborative Filtering (NMF)
- **Approach**: Matrix factorization to learn latent patterns
- **Method**: Non-negative Matrix Factorization (NMF) learns:
  - Birder embeddings (latent preferences)
  - Species embeddings (latent characteristics)
- **Advantage**: Captures similarity between birders and species

#### 3. Neural Network
- **Approach**: Deep learning with embeddings
- **Architecture**:
  - Input: Birder-species interaction vector (binary)
  - Embedding layer: Learns birder representation
  - Hidden layers: 128 → 64 neurons with dropout
  - Output: Probability distribution over all species
- **Advantage**: Can learn complex non-linear patterns

### Evaluation Framework

**Time-Series Cross-Validation** (4 folds):
- **Fold 1**: Train on transitions ending before 2020, Test on 2019→2020
- **Fold 2**: Train on transitions ending before 2021, Test on 2020→2021
- **Fold 3**: Train on transitions ending before 2022, Test on 2021→2022
- **Fold 4**: Train on transitions ending before 2023, Test on 2022→2023

**Why Time-Series CV?**
- Respects temporal order (can't use future to predict past)
- Handles potential 2020 outlier (COVID-19 impact)
- More realistic evaluation scenario

**Evaluation Metrics**:
- **Precision@K**: Of the top K predicted species, how many are correct?
- **Recall@K**: Of the actual species viewed, how many were in top K predictions?
- **MAP@K**: Mean Average Precision - considers ranking quality
- **Coverage**: How diverse are the predictions? (fraction of species predicted)

## Key Files and Their Purpose

### Core Modules

- **`data_loader.py`**: Loads and processes parquet files, creates transition pairs
- **`feature_engineering.py`**: Creates features and interaction matrices
- **`prepare_training_data.py`**: Sets up cross-validation splits
- **`models.py`**: Implements all three ML models
- **`train.py`**: Training loop and evaluation functions
- **`evaluate.py`**: Visualization and analysis tools

### Scripts

- **`combine_and_prepare.py`**: Combines year summaries and creates transitions
- **`run_training.py`**: Complete training pipeline (load → train → evaluate)
- **`batch_process_years.sh`**: SLURM script for processing years on server

### Data Files

- **`birder_species_YYYY.parquet`**: Processed summaries for each year
- **`all_birder_species.parquet`**: Combined summary for all years
- **`transitions.parquet`**: Year-over-year transition pairs (ready for training)

## How to Use This Project

### Step 1: Set Up Environment

```bash
cd /home/drkalex/670_final_project
source venv/bin/activate  # Virtual environment already set up
```

### Step 2: Data is Already Processed!

The following files are ready to use:
- `birder_species_2018.parquet` through `birder_species_2023.parquet`
- `all_birder_species.parquet` (combined)
- `transitions.parquet` (ready for training)

### Step 3: Run Training

**Option A: Complete Pipeline Script**
```bash
python3 run_training.py
```

**Option B: Interactive Python**
```python
from train import train_and_evaluate_cv
import pandas as pd

# Load transitions
transitions = pd.read_parquet('transitions.parquet')

# Train and evaluate
results = train_and_evaluate_cv(transitions)

# Results dictionary contains:
# - 'baseline': metrics for baseline model
# - 'collaborative': metrics for NMF model  
# - 'neural': metrics for neural network model
```

**Option C: Jupyter Notebook**
```bash
jupyter notebook 670_final_project.ipynb
```

### Step 4: View Results

Results are already available in the `results/` directory:
- **`results/cv_results.png`**: Box plots showing metric distributions across CV folds
- **`results/model_comparison.png`**: Bar chart comparing all three models
- **`results/evaluation_report.txt`**: Text summary with all metrics (see above for actual results)
- **`results/training_results.json`**: Detailed results in JSON format

You can also generate new results by running the training pipeline again.

## Understanding the Results

### What the Models Predict

Each model outputs **top K species** (default K=10) that a birder is likely to view next year.

**Example**:
- Birder viewed in 2021: [American Robin, Blue Jay, Cardinal]
- Model predicts for 2022: [Cardinal, Blue Jay, Sparrow, Crow, ...]

### Interpreting Metrics

- **High Precision@10**: Model is accurate - most predictions are correct
- **High Recall@10**: Model is comprehensive - catches most actual species
- **High MAP@10**: Model ranks well - correct species appear early in predictions
- **High Coverage**: Model suggests diverse species (not just popular ones)

### Actual Results

Based on training with the full dataset:

**Baseline Model**:
- Precision@10: **0.319** (32% of predictions are correct)
- Recall@10: **0.121** (12% of actual species predicted)
- MAP@10: **0.264** (good ranking quality)
- Coverage: **0.092** (predicts ~9% of all species)

**Collaborative Filtering (NMF)**:
- Precision@10: **0.031** (3% of predictions correct)
- Recall@10: **0.011** (1% of actual species predicted)
- MAP@10: **0.031** (poor performance)
- Coverage: **0.016** (very limited diversity)

**Neural Network** (Best Model):
- Precision@10: **0.357** (36% of predictions are correct)
- Recall@10: **0.138** (14% of actual species predicted)
- MAP@10: **0.299** (best ranking quality)
- Coverage: **0.193** (predicts ~19% of all species - most diverse)

**Key Finding**: The Neural Network performs best overall, achieving the highest precision, recall, MAP, and coverage. The baseline model is surprisingly competitive, while collaborative filtering underperformed (possibly due to data sparsity).

## Project Statistics

- **Total Transition Pairs**: 441,106 birder-year transitions
- **Unique Species**: 1,095 bird species
- **Unique Birders**: Varies by year (hundreds of thousands)
- **Time Period**: 2018-2023 (6 years of data)
- **Training Data Size**: ~65,000-540,000 birder pairs per CV fold
- **Test Data Size**: ~1,200-6,500 birder pairs per CV fold

## Key Design Decisions

1. **Memory Efficiency**: Processes data incrementally to handle 13.6B rows
2. **Time-Series CV**: Respects temporal order and handles 2020 outlier
3. **Multiple Models**: Compares simple vs. complex approaches
4. **Top-K Prediction**: Practical recommendation format (not binary classification)

## Extending the Project

### Adding New Models

Add to `models.py`:
```python
class YourNewModel:
    def fit(self, X, y):
        # Training logic
        pass
    
    def predict(self, X, top_k=10):
        # Prediction logic
        return predictions
```

### Modifying Evaluation

Edit `train.py` to add new metrics or change K values.

### Processing More Data

Use `batch_process_years.sh` to process additional years as they become available.

## Troubleshooting

### Memory Issues
- Use `max_files_per_year` parameter to limit data size
- Process years individually using batch script
- Increase available RAM for training

### Import Errors
- Ensure virtual environment is activated: `source venv/bin/activate`
- Check all dependencies: `pip install -r requirements.txt`

### Slow Training
- Neural network is slowest (~15-60 min depending on data)
- Consider reducing `n_components` for collaborative filtering
- Reduce `hidden_dims` for neural network

## Contact and Collaboration

This project is shared with the class. Feel free to:
- Use the code for your own analysis
- Modify models and experiment
- Share improvements and insights

## Citation

If using this code, please cite appropriately and acknowledge the data source (birding checklists).

---

**Last Updated**: December 2024
**Project Status**: Complete and ready for use

