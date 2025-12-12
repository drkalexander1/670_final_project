# Sharing Instructions for 670_final_project

## Current Setup
The folder has been configured with group permissions so all members of `si670f25_class_root` can access it.

## Access Methods

### Method 1: Direct Path Access (Recommended)
Your classmate can access the folder directly using the full path:
```bash
cd /home/drkalex/670_final_project
```

Or access specific files:
```bash
cat /home/drkalex/670_final_project/README.md
```

### Method 2: Create a Symlink
Your classmate can create a symlink in their home directory:
```bash
ln -s /home/drkalex/670_final_project ~/670_final_project_shared
cd ~/670_final_project_shared
```

### Method 3: Copy to Shared Location
If there's a shared class directory, copy the folder there:
```bash
# Example (adjust path as needed):
cp -r /home/drkalex/670_final_project /shared/si670f25/670_final_project
```

## Current Permissions
- Folder owner: drkalex
- Folder group: si670f25_class_root
- Permissions: drwxr-xr-x (readable/executable by group and others)
- Accessible by: si670f25_class_root group members AND si650f25s001_class_root/si650f25s101_class_root group members (via world-readable permissions)

## Troubleshooting

If your classmate still can't access:

1. **Check group membership:**
   ```bash
   groups
   # Should include: si670f25_class_root
   ```

2. **Verify permissions:**
   ```bash
   ls -ld /home/drkalex/670_final_project
   # Should show: drwxr-xr-x ... si670f25_class_root
   ```

3. **Try direct path:**
   ```bash
   ls /home/drkalex/670_final_project
   ```

4. **If still having issues, contact the instructor to:**
   - Add them to the si670f25_class_root group
   - Or create a shared directory for the class

## Files Included
- All Python modules (data_loader.py, models.py, etc.)
- Jupyter notebook (670_final_project.ipynb)
- Batch processing script (batch_process_years.sh)
- Documentation (README.md, README_DATA_PROCESSING.md)
- Data directories (2018-2023)
- Processed summaries (birder_species_*.parquet)

