#!/bin/bash
# Batch submission script for training models with enhanced features
# Uses birder_species parquet files for fast feature extraction (no raw data loading)
# Usage: sbatch batch_process_years.sh
#
# Prerequisites: 
#   - birder_species_YYYY.parquet files must exist (created by extract_all_data.py)
#   - transitions.parquet must exist

#SBATCH --job-name=bird_training_features
#SBATCH --account=si650f25s101_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/drkalex/670_final_project/results/bird_training_%j.out
#SBATCH --error=/home/drkalex/670_final_project/results/bird_training_%j.err
#SBATCH --time=06:00:00

# Activate virtual environment
source /home/drkalex/670_final_project/venv/bin/activate

# Change to project directory
cd /home/drkalex/670_final_project

# Use -u flag for unbuffered output so we can see progress in real-time
# This script now uses birder_species files instead of loading raw data,
# which should be much faster and avoid timeout issues
python3.12 -u run_training_with_features.py

