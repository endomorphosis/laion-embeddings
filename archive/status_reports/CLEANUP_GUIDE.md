# LAION Embeddings Directory Cleanup Guide

## Overview

This guide provides instructions for cleaning up and organizing the LAION Embeddings project directory structure. The goal is to improve maintainability, make the codebase more navigable, and enforce better organization practices.

## Before You Begin

1. **Backup Your Work**: The cleanup script will create a backup automatically, but it's always good to have your own backup too.
2. **Review the Plan**: Take a look at the `DIRECTORY_CLEANUP_PLAN.md` document to understand the changes that will be made.

## Running the Cleanup Script

### Option 1: Run from the Root Directory

```bash
bash scripts/cleanup_directory.sh
```

### Option 2: Make Executable and Run

```bash
chmod +x scripts/cleanup_directory.sh
./scripts/cleanup_directory.sh
```

## Post-Cleanup Tasks

After running the cleanup script, you should:

1. **Verify File Organization**
   - Check that files have been moved to their proper locations
   - Ensure no critical files were missed

2. **Test the Code**
   - Run the test suite to make sure everything works correctly:
   ```bash
   python -m pytest
   ```

3. **Update Import Statements** (if needed)
   - The script attempts to update imports, but may miss some cases
   - Check for any import errors and fix them manually

4. **Remove Duplicate Files**
   - The script copies `mock_ipfs.py` instead of moving it (for safety)
   - After verifying, manually remove the root copy:
   ```bash
   rm mock_ipfs.py
   ```

5. **Update CI/CD Configuration**
   - If you use CI/CD tools, update their configuration to reflect the new directory structure

6. **Update Documentation**
   - Update any documentation that refers to files by their old locations

## Directory Structure

After cleanup, your project should have the following structure:

```
laion-embeddings-1/
├── __init__.py
├── auth.py
├── main.py
├── monitoring.py
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── README.md
├── LICENSE
├── Dockerfile
├── run.sh -> scripts/run.sh
├── install_depends.sh -> scripts/install_depends.sh
├── search.sh -> scripts/search.sh
├── docs/
│   ├── README.md
│   ├── DOCUMENTATION_UPDATES.md
│   ├── api/
│   ├── examples/
│   ├── ipfs/
│   ├── planning/
│   └── archive/
├── scripts/
│   ├── cleanup_directory.sh
│   ├── run.sh
│   └── [other scripts]
├── test/
│   ├── __init__.py
│   ├── unit/
│   ├── ipfs/
│   └── performance/
├── test_results/
├── src/
│   └── archive/
└── [module directories]
```

## Reverting Changes

If something goes wrong, you can restore from the backup:

```bash
# Find the backup file
ls -la ../*.tar.gz

# Restore from backup (replace with your backup filename)
cd ..
rm -rf laion-embeddings-1
mkdir laion-embeddings-1
tar -xzf laion-embeddings-backup-20250531.tar.gz -C laion-embeddings-1
```

## Common Issues

### Missing Files
The script uses `2>/dev/null || true` to prevent errors if files don't exist. If you find that files weren't moved, check if they existed at the script's runtime.

### Import Errors
If you encounter import errors, update the import statements in your code to reflect the new file locations.

### Broken Symlinks
If symlinks are broken, recreate them manually:
```bash
ln -sf scripts/run.sh run.sh
```

### Test Failures
If tests fail after reorganization, check the import paths and make sure `__init__.py` files exist in the proper directories.

## Contact

If you encounter any issues during cleanup that you cannot resolve, please open an issue on the project's GitHub repository or contact the project maintainers.
