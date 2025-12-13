#!/bin/bash
# Batch submission script for training models on regional (Northeast) data
# Usage: sbatch batch_training_regional.sh
#
# Prerequisites: Run extract_regional.py first to create transitions_northeast.parquet

#SBATCH --job-name=bird_training_regional
#SBATCH --account=si650f25s101_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/drkalex/670_final_project/results/bird_training_regional_%j.out
#SBATCH --error=/home/drkalex/670_final_project/results/bird_training_regional_%j.err
#SBATCH --time=08:00:00

# Activate virtual environment
source /home/drkalex/670_final_project/venv/bin/activate

# Change to project directory
cd /home/drkalex/670_final_project

# Use -u flag for unbuffered output so we can see progress in real-time
# python3.12 -u extract_regional.py
python3.12 -u run_training_regional.py


