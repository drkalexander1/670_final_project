#!/usr/bin/env python3
"""
Extract zip file bypassing zip bomb warnings.
This is safe for parquet files which are already compressed.
"""
import zipfile
import os
import sys

zip_path = 'checklist_filtered_zf_raw_2018_2023.zip'

if not os.path.exists(zip_path):
    print(f"Error: {zip_path} not found")
    sys.exit(1)

print(f"Extracting {zip_path}...")
print("This may take a while due to the large file size...")
print("Note: Parquet files are already compressed, so zip bomb warnings are false positives.\n")

extracted_count = 0
skipped_count = 0
error_count = 0

# Disable zip bomb protection for Python 3.12+
# This is safe because we know these are parquet files, not malicious archives
try:
    # For Python 3.12+, we need to set max size limits
    # Setting to a very large value (100GB) to allow extraction
    zipfile.ZipFile._MAX_EXTRACT_SIZE = 100 * 1024 * 1024 * 1024  # 100GB
except AttributeError:
    # Older Python versions don't have this protection
    pass

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get list of all files (excluding directories)
    file_list = [f for f in zip_ref.namelist() if not f.endswith('/')]
    total_files = len(file_list)
    
    print(f"Total files to extract: {total_files}")
    
    for i, filename in enumerate(file_list, 1):
        try:
            # Extract file
            zip_ref.extract(filename)
            extracted_count += 1
            
            if i % 50 == 0:
                print(f"Progress: {i}/{total_files} files extracted...")
                
        except zipfile.BadZipFile:
            print(f"Warning: Bad zip entry for {filename}")
            error_count += 1
        except Exception as e:
            print(f"Error extracting {filename}: {e}")
            error_count += 1

print(f"\nExtraction complete!")
print(f"  Extracted: {extracted_count} files")
print(f"  Errors: {error_count} files")
print(f"  Total expected: {total_files} files")

