# Bird Species Prediction - Final Project

## Research Question

Can we predict which bird species an eBird observer will report in the coming year, 
based on their historical activity? This project shifts focus from the observed birds 
— their abundance, migration patterns, habitat usage — to the observers themselves, 
who generate the data underlying ecological research.

Understanding observer behavior addresses a persistent challenge in citizen science: 
observation bias. eBird data reflects not just where birds are, but where birders go. 
Modeling observer patterns helps organizations like the Cornell Lab of Ornithology 
interpret systematic gaps in coverage, identify emerging hotspots, and improve 
population abundance estimates that depend on standardizing for observer effort.

The problem also has an economic dimension: predicting where observers concentrate 
highlights habitat value, informs wildlife management and conservation resource 
allocation, and connects ecological health to local tourism and policy.

## Project Overview

This project predicts which bird species a birder will view **next year** based on their viewing patterns from the **current year**. The dataset contains birding checklists from 2018–2023 with approximately 13.6 billion observations. The project implements both **species prediction** (classification) and **count prediction** (regression) tasks.

**Key Question**: Given what bird species a birder viewed this year, what will they view next year?

## Project Structure

```
670_final_project/
├── data_loader.py                 # Memory-efficient data loading utilities
├── feature_engineering.py         # Feature extraction and engineering
├── prepare_training_data.py       # Time-series CV split preparation
├── models.py                      # ML model implementations
├── train.py                       # Training and evaluation functions
├── evaluate.py                    # Visualization and analysis tools
├── extract_all_data.py            # Full-dataset feature extraction
├── extract_regional.py            # Northeast regional data extraction
├── run_training_regional.py       # Regional (Northeast) training script
├── run_training_with_features.py  # Full training with engineered features
├── batch_process_years.sh         # SLURM batch script for full-feature training
├── batch_training_regional.sh     # SLURM batch script for regional training
├── batch_extract_data.sh          # SLURM batch script for data extraction
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── README_DATA_PROCESSING.md      # Memory management and data processing guide
```

## Dataset

- **Source**: eBird birding checklists, 2018–2023
- **Total Observations**: ~13.6 billion rows
- **Format**: Parquet files organized by year, quarter, and groups
- **Key Fields**:
  - `observer_id`: Unique birder identifier
  - `scientific_name`: Bird species name
  - `year`: Year of observation
  - `species_observed`: Boolean indicating if species was seen

## Quick Start

### 1. Set Up Environment

```bash
cd 670_final_project
python3.12 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Process Data

**Option A: Full Dataset via SLURM (HPC cluster)**
```bash
# Extract processed summaries for each year
sbatch batch_extract_data.sh

# Then train with engineered features
sbatch batch_process_years.sh
```

**Option B: Regional Subset (Northeast, faster)**
```bash
# Extract Northeast regional data
python extract_regional.py

# Train on regional data
python run_training_regional.py
```

**Option C: Sample Data (for local development/testing)**
```python
from data_loader import load_all_years, get_birder_species_by_year, create_transition_pairs

# Load a small sample (first 2 files per year)
data = load_all_years(max_files_per_year=2)
birder_species = get_birder_species_by_year(data)
transitions = create_transition_pairs(birder_species)
```

### 3. Run Training

**Regional training (recommended starting point):**
```bash
python run_training_regional.py
```

**Full training with engineered features:**
```bash
python run_training_with_features.py
```

**Or use the API directly:**
```python
from train import train_and_evaluate_cv
from evaluate import plot_cv_results, generate_summary_report

results = train_and_evaluate_cv(transitions, model_types=['baseline', 'collaborative', 'neural'])
plot_cv_results(results)
generate_summary_report(results)
```

### 4. View Results

After training, results are saved to the `results/` directory:

| File | Description |
|---|---|
| `cv_results.png` | Box plots of metrics across CV folds |
| `model_comparison.png` | Bar chart comparing all models |
| `count_prediction_results.png` | Count prediction metrics |
| `evaluation_report.txt` | Text summary of all metrics |
| `training_results.json` | Full results in JSON format |

## Models Implemented

### 1. Baseline Popularity Model

A naive baseline that predicts the most popular species overall.

- **Species Prediction**: Ranks species by overall popularity (number of birders who viewed each) and recommends the top K unseen species
- **Count Prediction**: Always predicts the mean number of species seen across all birders
- **Naive Baseline Behavior**: Learns from fold 1 only and uses fixed parameters for all subsequent folds — this ensures a fair comparison that doesn't benefit from seeing more data

### 2. Collaborative Filtering Model (Species Co-occurrence)

Learns which species tend to co-occur across birders and uses those patterns to make recommendations.

- **Method**: Builds a co-occurrence matrix where `cooccurrence[i, j]` = number of birders who saw both species i and j, then normalizes to conditional probabilities P(j | i)
- **Prediction**: For each birder, aggregates co-occurrence scores from all species they've viewed and recommends the top K
- **Parameters**: `alpha` (smoothing, default 0.1), `min_cooccurrence` (minimum co-occurrences, default 2)

### 3. Neural Network Model

A deep learning model with embeddings and optional multi-task learning.

**Architecture**:
- Input: Birder–species interaction vector
- Birder embedding layer (default dim: 64, ReLU, dropout 0.3)
- Optional feature integration for temporal/geographic features
- Hidden layers (default: 128 → 64, ReLU, dropout 0.3)
- Output: sigmoid over all species for classification; optional ReLU head for count regression

**Multi-Task Learning**: When `predict_count=True`, the model jointly predicts which species a birder will see and how many, with a combined loss (binary cross-entropy + 0.1 × MSE).

**Training**: Adam (lr=0.001), early stopping (patience=5), learning rate reduction on plateau, 20 max epochs.

## Evaluation Metrics

### Species Prediction (Classification)

| Metric | Description |
|---|---|
| **Precision@K** | `\|predicted ∩ actual\| / K` — accuracy of predictions |
| **Recall@K** | `\|predicted ∩ actual\| / \|actual\|` — coverage of actual species |
| **MAP@K** | Mean Average Precision — rewards correct species ranked higher |
| **Coverage** | Fraction of all species predicted for at least one birder |

### Count Prediction (Regression)

| Metric | Description |
|---|---|
| **MAE** | Mean absolute error in number of species |
| **RMSE** | Root mean squared error — penalizes large misses more |
| **Correlation** | Pearson correlation between predicted and actual counts |
| **MAPE** | Mean absolute percentage error |

## Time-Series Cross-Validation

4-fold time-series CV that respects temporal ordering and handles the potential 2020 COVID-19 outlier:

| Fold | Training Data | Test Data |
|---|---|---|
| 1 | 2018→2019 | 2019→2020 |
| 2 | 2018→2019, 2019→2020 | 2020→2021 |
| 3 | 2018→2019 … 2020→2021 | 2021→2022 |
| 4 | 2018→2019 … 2021→2022 | 2022→2023 |

Each fold adds one more year of training data, allowing evaluation of how models improve with more history.

## Actual Results

Results from training on the full dataset (441,106 transition pairs, 1,095 unique species):

### Species Prediction (Precision/Recall/MAP/Coverage @K=10)

| Model | Precision@10 | Recall@10 | MAP@10 | Coverage |
|---|---|---|---|---|
| Baseline Popularity | 0.319 | 0.121 | 0.264 | 0.092 |
| Collaborative Filtering | 0.031 | 0.011 | 0.031 | 0.016 |
| **Neural Network** | **0.357** | **0.138** | **0.299** | **0.193** |

**Key Finding**: The Neural Network achieves the best performance across all metrics and has the most diverse predictions (19% species coverage vs. 9% for the baseline). The baseline is surprisingly competitive; collaborative filtering underperformed, likely due to data sparsity.

## Memory Considerations

The full dataset is very large (~13.6B rows). See `README_DATA_PROCESSING.md` for detailed memory management guidance. Key points:

- Use `max_files_per_year=2` for local development/testing
- Full dataset processing requires ~8–16 GB RAM per year
- SLURM batch scripts are configured for 64 GB per CPU on the cluster
- The data loader processes files incrementally and filters to observed species immediately

## Notes

- Regional analysis focuses on the Northeast region as a faster alternative to the full dataset
- Baseline models are intentionally naive (fixed parameters from fold 1) to provide fair comparison
- Results may vary based on data sample size
- Neural network training takes ~15–60 minutes depending on data size; reduce `hidden_dims` to speed it up
