#!/bin/bash
# Batch script to extract all required data columns for feature extraction
# Usage: sbatch batch_extract_data.sh

#SBATCH --job-name=extract_data_features
#SBATCH --account=si650f25s101_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/drkalex/670_final_project/results/extract_data_%j.out
#SBATCH --error=/home/drkalex/670_final_project/results/extract_data_%j.err
#SBATCH --time=4:00:00

# Activate virtual environment
source /home/drkalex/670_final_project/venv/bin/activate

# Change to project directory
cd /home/drkalex/670_final_project

# Use -u flag for unbuffered output
python3.12 -u extract_all_data.py


