# Data Processing Guide

## Data Size
- **Total data**: ~13.6 billion rows across 6 years
- **Per year**: ~2.3 billion rows, 120 parquet files
- **Disk space**: ~7GB total (compressed parquet)

## Memory Considerations

The full dataset is too large to load into memory at once. Use one of these approaches:

### Option 1: Incremental Processing (Recommended)
Process years one at a time and save intermediate results:

```bash
# Process individual years (can be run in parallel)
python3 data_loader.py 2018
python3 data_loader.py 2019
# ... etc

# Then combine summaries
python3 -c "from data_loader import combine_year_summaries; \
           df = combine_year_summaries([2018,2019,2020,2021,2022,2023]); \
           df.to_parquet('all_birder_species.parquet')"
```

### Option 2: Batch Submission (For HPC/Server)
Use the provided SLURM script:

```bash
sbatch batch_process_years.sh
```

### Option 3: Sample for Development
For testing/development, use a small sample:

```python
from data_loader import load_all_years

# Load only first 2 files per year (for testing)
data = load_all_years(max_files_per_year=2)
```

## Memory Requirements

- **Per year processing**: ~8-16GB RAM recommended
- **Full dataset in memory**: Not recommended (would need 100+ GB)
- **Processed summaries**: Much smaller (~100MB-1GB per year)

## Processing Steps

1. **Extract birder-species by year** (saves to `birder_species_YYYY.parquet`)
   - This reduces data from billions of rows to millions of birder-year pairs
   - Each file is ~100-500MB

2. **Combine year summaries** (creates `all_birder_species.parquet`)
   - Combines all year summaries into one file
   - Still manageable size (~1-5GB)

3. **Create transition pairs** (creates training data)
   - Much smaller: only birder pairs with consecutive years
   - Final training data: ~millions of rows

## Tips

- Process years in parallel if you have multiple nodes
- Save intermediate results frequently
- Use `max_files_per_year` parameter for testing
- Monitor memory usage with `htop` or `free -h`

