# Bird Species Prediction - Final Project

## Project Overview

This project predicts which bird species a birder will view next year based on their viewing patterns from the current year. The dataset contains birding checklists from 2018-2023 with approximately 13.6 billion observations. The project implements both **species prediction** (classification) and **count prediction** (regression) tasks.

## Project Structure

```
670_final_project/
├── data_loader.py              # Memory-efficient data loading utilities
├── feature_engineering.py       # Feature extraction and engineering
├── prepare_training_data.py    # Time-series CV split preparation
├── models.py                   # ML model implementations
├── train.py                    # Training and evaluation functions
├── evaluate.py                 # Visualization and analysis tools
├── run_training_regional.py    # Regional training script
├── run_training_with_features.py  # Training with features script
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

### 3. Run Training

**Regional Training (Northeast Region):**
```bash
python run_training_regional.py
```

**Training with Features:**
```bash
python run_training_with_features.py
```

Or run individual components:

```python
from train import train_and_evaluate_cv
from evaluate import plot_cv_results, generate_summary_report

# Train models
results = train_and_evaluate_cv(transitions, model_types=['baseline', 'collaborative', 'neural'])

# Visualize results
plot_cv_results(results)
generate_summary_report(results)
```

## Models Implemented

### 1. Baseline Popularity Model

A naive baseline model that serves as a simple comparison point for both classification and regression tasks.

**Species Prediction (Classification):**
- **Method**: Predicts the most popular species overall
- **Training**: Learns species popularity ranking from training data (number of birders who viewed each species)
- **Prediction**: For each birder, predicts the top K most popular species they haven't already viewed
- **Naive Baseline Behavior**: 
  - Learns popular species from **fold 1 only**
  - Uses the same fixed ranking for all subsequent folds (does not update)
  - This ensures a true naive baseline that doesn't adapt to new data

**Count Prediction (Regression):**
- **Method**: Always predicts the mean number of species seen
- **Training**: Calculates the mean number of species per birder from training data
- **Prediction**: Returns the same mean count for all birders
- **Naive Baseline Behavior**:
  - Learns mean count from **fold 1 only**
  - Uses the same fixed mean for all subsequent folds (does not update)

**Use Case**: Provides a simple baseline to compare against more sophisticated models. The naive behavior (no updating) ensures fair comparison across time-series folds.

### 2. Collaborative Filtering Model (Species Co-occurrence)

A collaborative filtering approach based on species co-occurrence patterns.

**Method**: 
- Learns which species tend to co-occur together across birders
- Builds a co-occurrence matrix: `cooccurrence_matrix[i, j]` = number of birders who saw both species i and j
- Normalizes to conditional probabilities: `P(species_j | species_i)`

**Training Process**:
1. Combines training input (X) and targets (y) to learn co-occurrence patterns
2. For each birder, computes outer product of their species vector
3. Aggregates across all birders to build global co-occurrence matrix
4. Normalizes by species frequency with additive smoothing (alpha parameter)
5. Computes species popularity as fallback for birders with no history

**Prediction**:
- For each birder, aggregates co-occurrence scores from all species they've viewed
- Averages scores across viewed species
- Optionally applies species popularity weighting if features available
- Selects top K species with highest scores

**Parameters**:
- `alpha`: Smoothing parameter for co-occurrence scores (default: 0.1)
- `min_cooccurrence`: Minimum co-occurrences to consider (default: 2)
- `species_features`: Optional dictionary with species-level features

**Use Case**: Captures collaborative patterns - if birder A and birder B both saw species X, and birder A saw species Y, then birder B might also see species Y.

### 3. Neural Network Model

A deep learning model with embeddings and multi-task learning capabilities.

**Architecture**:
- **Input Layer**: Birder-species interaction vector (binary or count matrix)
- **Birder Embedding**: Dense layer that learns birder representations from species interactions
  - Dimension: `embedding_dim` (default: 64)
  - Activation: ReLU
  - Dropout: 0.3
- **Feature Integration** (optional): 
  - Additional features (temporal, regional, etc.) processed through separate embedding layers
  - Concatenated with birder embedding
- **Hidden Layers**: 
  - Configurable dimensions (default: [128, 64])
  - ReLU activation with dropout (default: 0.3)
- **Output Layers**:
  - **Species Prediction**: Dense layer with sigmoid activation (n_species outputs)
  - **Count Prediction** (optional): Dense layer with ReLU activation (1 output)

**Multi-Task Learning**:
- When `predict_count=True`, the model simultaneously predicts:
  1. Which species a birder will see (classification)
  2. How many species they will see (regression)
- Loss function combines:
  - Binary cross-entropy for species prediction (weight: 1.0)
  - Mean squared error for count prediction (weight: 0.1)
- Both tasks share the same hidden representation, allowing the model to learn related patterns

**Training**:
- Optimizer: Adam (learning rate: 0.001)
- Callbacks: Early stopping (patience: 5) and learning rate reduction on plateau
- Validation split: 10% of training data (min 1000 samples)
- Epochs: 20 (with early stopping)

**Parameters**:
- `n_species`: Number of unique species
- `embedding_dim`: Dimension of birder/species embeddings (default: 64)
- `hidden_dims`: List of hidden layer dimensions (default: [128, 64])
- `dropout`: Dropout rate (default: 0.3)
- `n_additional_features`: Number of additional features (default: 0)
- `predict_count`: Whether to predict count (default: False)

**Use Case**: Most sophisticated model that can learn complex non-linear patterns and leverage additional features. Multi-task learning allows count prediction to inform species prediction and vice versa.

## Evaluation Metrics

### Species Prediction Metrics (Classification)

- **Precision@K**: Fraction of predicted species that are correct
  - Measures accuracy of predictions
  - Formula: `|predicted ∩ actual| / K`
  
- **Recall@K**: Fraction of actual species that were predicted
  - Measures coverage of actual species
  - Formula: `|predicted ∩ actual| / |actual|`
  
- **MAP@K** (Mean Average Precision): Average precision across all birders
  - Considers ranking quality, not just presence
  - Higher weight to correctly predicted species ranked higher
  
- **Coverage**: Diversity of predicted species
  - Fraction of all species that are predicted for at least one birder
  - Measures model diversity vs. always predicting same popular species

### Count Prediction Metrics (Regression)

- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual counts
  - Formula: `mean(|predicted - actual|)`
  - Units: Same as count (number of species)
  
- **RMSE** (Root Mean Squared Error): Square root of mean squared differences
  - Formula: `sqrt(mean((predicted - actual)^2))`
  - Penalizes large errors more than MAE
  
- **Correlation**: Pearson correlation coefficient between predicted and actual counts
  - Range: [-1, 1]
  - Measures linear relationship strength
  - 1.0 = perfect positive correlation, 0 = no correlation
  
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
  - Formula: `mean(|predicted - actual| / actual) * 100`
  - Useful for understanding relative error magnitude

## Time-Series Cross-Validation

Uses 4-fold time-series CV to handle temporal dependencies and potential 2020 outlier:

- **Fold 1**: Train (2018→2019), Test (2019→2020)
- **Fold 2**: Train (2018→2019, 2019→2020), Test (2020→2021)
- **Fold 3**: Train (2018→2019, 2019→2020, 2020→2021), Test (2021→2022)
- **Fold 4**: Train (2018→2019, 2019→2020, 2020→2021, 2021→2022), Test (2022→2023)

**Key Features**:
- Respects temporal ordering (no future data in training)
- Each fold adds one more year of training data
- Allows evaluation of model performance over time
- Handles potential distribution shifts across years

**Naive Baseline Behavior**:
- Baseline models learn from **fold 1 only** and use fixed parameters for all folds
- This ensures fair comparison - baseline doesn't benefit from seeing more data

## Memory Considerations

The full dataset is very large (~13.6B rows). The data loader is optimized for memory efficiency:

- Processes files incrementally
- Only loads essential columns
- Filters to observed species immediately
- Supports batch processing for server environments
- Lazy loading per CV fold (for regional data)

See `README_DATA_PROCESSING.md` for detailed memory management guidance.

## Output Files

After running training, you'll get:

**Visualizations**:
- `cv_results.png`: Box plots of metrics across CV folds
- `cv_results_regional.png`: Regional CV results
- `model_comparison.png`: Bar chart comparing models
- `model_comparison_regional.png`: Regional model comparison
- `count_prediction_results.png`: Count prediction metrics visualization
- `count_prediction_regional.png`: Regional count prediction results
- `count_prediction_comparison.png`: Comparison of count prediction across models

**Reports**:
- `evaluation_report.txt`: Text summary of results
- `evaluation_report_regional.txt`: Regional evaluation report

**Data**:
- `training_results.json`: JSON file with all metrics and results

## Key Features

- **Memory-efficient data processing**: Handles large datasets without loading everything into memory
- **Time-series cross-validation**: Proper temporal evaluation methodology
- **Multiple model architectures**: Baseline, collaborative filtering, and neural networks
- **Multi-task learning**: Simultaneous species and count prediction
- **Comprehensive evaluation metrics**: Both classification and regression metrics
- **Naive baselines**: Fixed baselines that don't update across folds for fair comparison
- **Visualization tools**: Comprehensive plotting and analysis capabilities
- **Regional analysis**: Support for regional subsets (e.g., Northeast)

## Model Comparison Strategy

1. **Baseline Model**: Simple popularity-based predictions (both species and count)
   - Provides a lower bound on performance
   - Naive (doesn't update across folds)

2. **Collaborative Filtering**: Co-occurrence based recommendations
   - Captures collaborative patterns
   - Updates with each fold (learns from all training data)

3. **Neural Network**: Deep learning with embeddings
   - Most flexible, can learn complex patterns
   - Supports multi-task learning (species + count)
   - Updates with each fold

## Notes

- For development/testing, use `max_files_per_year=2` to limit data size
- Full dataset processing requires significant memory (8-16GB per year)
- Batch submission recommended for server environments
- Results may vary based on data sample size
- Regional analysis focuses on Northeast region for faster processing
- Baseline models are intentionally naive (fixed parameters) to provide fair comparison

## Running Regional Analysis

The regional analysis focuses on the Northeast region for faster processing:

```bash
python run_training_regional.py
```

This script:
- Loads pre-processed regional data
- Trains all three models
- Evaluates both with and without additional features
- Generates comprehensive visualizations and reports
