# LAION Embeddings Project - Directory Cleanup Plan

## Objectives
- Reduce clutter in the root directory
- Organize files into logical categories
- Maintain a clean, maintainable project structure
- Preserve important files and history

## Current Directory Analysis

The root directory currently contains 72 files and directories, many of which could be better organized. The main issues include:

1. Multiple README versions
2. Many documentation files in the root directory
3. Test files mixed with application code
4. Numerous shell scripts in the root directory
5. Test result text files in the root directory

## Action Plan

### 1. README Files Consolidation

| File | Action | Details |
|------|--------|---------|
| README.md | Keep | Update with latest content from README_UPDATED.md |
| README_ENHANCED.md | Archive | Move to docs/archive/README_ENHANCED.md |
| README_UPDATED.md | Apply & Remove | Apply changes to README.md then remove |

```bash
# Apply README_UPDATED.md content to README.md
cp README_UPDATED.md README.md

# Move enhanced readme to archive
mkdir -p docs/archive
mv README_ENHANCED.md docs/archive/
```

### 2. Documentation Organization

| File | Action | Details |
|------|--------|---------|
| DEPRECATION_COMPLETION.md | Move | docs/planning/DEPRECATION_COMPLETION.md |
| DEPRECATION_PLAN.md | Move | docs/planning/DEPRECATION_PLAN.md |
| DOCUMENTATION_UPDATES.md | Move | docs/DOCUMENTATION_UPDATES.md |
| IMMEDIATE_ACTION_PLAN.md | Move | docs/planning/IMMEDIATE_ACTION_PLAN.md |
| IMPLEMENTATION_PLAN.md | Move | docs/planning/IMPLEMENTATION_PLAN.md |
| IPFS_FIXES_SUMMARY.md | Move | docs/ipfs/IPFS_FIXES_SUMMARY.md |

```bash
# Create directories
mkdir -p docs/planning
mkdir -p docs/ipfs

# Move documentation files
mv DEPRECATION_COMPLETION.md docs/planning/
mv DEPRECATION_PLAN.md docs/planning/
mv DOCUMENTATION_UPDATES.md docs/
mv IMMEDIATE_ACTION_PLAN.md docs/planning/
mv IMPLEMENTATION_PLAN.md docs/planning/
mv IPFS_FIXES_SUMMARY.md docs/ipfs/
```

### 3. Test Files Organization

| File | Action | Details |
|------|--------|---------|
| test_imports.py | Move | test/unit/test_imports.py |
| test_ipfs.py | Move | test/ipfs/test_ipfs.py |
| test_ipfs_debug.py | Move | test/ipfs/test_ipfs_debug.py |
| test_ipfs_fixed.py | Move | test/ipfs/test_ipfs_fixed.py |
| test_main_new_verification.py | Move | test/unit/test_main_new_verification.py |
| test_max_batch_size_timeout.py | Move | test/performance/test_max_batch_size_timeout.py |
| test_patches.py | Move | test/ipfs/test_patches.py |
| test_timeout_comprehensive.py | Move | test/performance/test_timeout_comprehensive.py |
| test_timeout_implementation.py | Move | test/performance/test_timeout_implementation.py |

```bash
# Create test directories
mkdir -p test/unit
mkdir -p test/ipfs
mkdir -p test/performance

# Move test files
mv test_imports.py test/unit/
mv test_ipfs.py test/ipfs/
mv test_ipfs_debug.py test/ipfs/
mv test_ipfs_fixed.py test/ipfs/
mv test_main_new_verification.py test/unit/
mv test_max_batch_size_timeout.py test/performance/
mv test_patches.py test/ipfs/
mv test_timeout_comprehensive.py test/performance/
mv test_timeout_implementation.py test/performance/
```

### 4. Test Results Files

| File | Action | Details |
|------|--------|---------|
| distributed_test_results.txt | Move | test_results/distributed_test_results.txt |
| full_ipfs_results.txt | Move | test_results/full_ipfs_results.txt |
| ipfs_test_results.txt | Move | test_results/ipfs_test_results.txt |
| test_output.txt | Move | test_results/test_output.txt |

```bash
# Ensure test_results directory exists
mkdir -p test_results

# Move test result files
mv distributed_test_results.txt test_results/
mv full_ipfs_results.txt test_results/
mv ipfs_test_results.txt test_results/
mv test_output.txt test_results/
```

### 5. Scripts Organization

| Location | Action | Details |
|----------|--------|---------|
| Root directory | Create scripts directory | mkdir -p scripts |
| *.sh files | Move to scripts directory | Move all shell scripts to scripts/ |
| Run scripts | Create symlinks in root | Create symlinks for frequently used scripts |

```bash
# Create scripts directory
mkdir -p scripts

# Move shell scripts
mv *.sh scripts/
mv run_patched_tests.py scripts/
mv run_tests.py scripts/
mv run_tests_with_env.sh scripts/

# Create symlinks for commonly used scripts
ln -s scripts/run.sh run.sh
ln -s scripts/install_depends.sh install_depends.sh
ln -s scripts/search.sh search.sh
```

### 6. Main Application Files

| File | Action | Details |
|------|--------|---------|
| __init__.py | Keep | Keep in root for package structure |
| auth.py | Keep | Core application file |
| main.py | Keep | Main application entry point |
| main_old.py | Archive | Move to archive directory |
| mock_ipfs.py | Move | Move to ipfs_embeddings_py/mock_ipfs.py |
| monitoring.py | Keep | Core monitoring functionality |

```bash
# Create archive directory if needed
mkdir -p src/archive

# Move old main file
mv main_old.py src/archive/

# Move mock_ipfs to proper location
mv mock_ipfs.py ipfs_embeddings_py/
```

### 7. Update Import References

After moving files, update import references in all affected files:

```bash
# Find all Python files and update imports
find . -name "*.py" -type f -exec sed -i 's/from test_ipfs/from test.ipfs.test_ipfs/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/import test_ipfs/import test.ipfs.test_ipfs/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/from mock_ipfs/from ipfs_embeddings_py.mock_ipfs/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/import mock_ipfs/import ipfs_embeddings_py.mock_ipfs/g' {} \;
```

### 8. Update pytest.ini

Update pytest.ini to reflect the new test structure:

```ini
[pytest]
python_files = test_*.py
testpaths = test
python_functions = test_*
```

### 9. Update Documentation References

Update documentation to reflect the new directory structure:

```bash
# Find all Markdown files and update references
find . -name "*.md" -type f -exec sed -i 's/test_ipfs.py/test\/ipfs\/test_ipfs.py/g' {} \;
```

## Final Directory Structure

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
│   │   └── IPFS_FIXES_SUMMARY.md
│   ├── planning/
│   │   ├── DEPRECATION_COMPLETION.md
│   │   ├── DEPRECATION_PLAN.md
│   │   ├── IMMEDIATE_ACTION_PLAN.md
│   │   └── IMPLEMENTATION_PLAN.md
│   └── archive/
│       └── README_ENHANCED.md
├── scripts/
│   ├── autofaiss.sh
│   ├── create.sh
│   ├── create_sparse.sh
│   ├── index_cluster.sh
│   ├── install_depends.sh
│   ├── launch_tei.sh
│   ├── load.sh
│   ├── load2.sh
│   ├── load3.sh
│   ├── run.sh
│   ├── run_full_ipfs_tests.sh
│   ├── run_ipfs_test_debug.sh
│   ├── run_ipfs_test_summary.sh
│   ├── run_ipfs_tests.sh
│   ├── run_patched_tests.py
│   ├── run_tests.py
│   ├── run_tests_with_env.sh
│   ├── search.sh
│   ├── search2.sh
│   ├── shard_cluster.sh
│   └── storacha.sh
├── test/
│   ├── unit/
│   │   ├── test_imports.py
│   │   └── test_main_new_verification.py
│   ├── ipfs/
│   │   ├── test_ipfs.py
│   │   ├── test_ipfs_debug.py
│   │   ├── test_ipfs_fixed.py
│   │   └── test_patches.py
│   └── performance/
│       ├── test_max_batch_size_timeout.py
│       ├── test_timeout_comprehensive.py
│       └── test_timeout_implementation.py
├── test_results/
│   ├── distributed_test_results.txt
│   ├── full_ipfs_results.txt
│   ├── ipfs_test_results.txt
│   └── test_output.txt
├── ipfs_embeddings_py/
│   ├── mock_ipfs.py
│   └── ...
├── create_embeddings/
├── search_embeddings/
├── sparse_embeddings/
├── shard_embeddings/
├── ipfs_cluster_index/
├── storacha_clusters/
└── services/
```

## Implementation Steps

1. **Backup**: Create a backup of the entire project before reorganizing
2. **Documentation**: Move and organize documentation files first
3. **Tests**: Reorganize test files and update import references
4. **Scripts**: Move scripts to dedicated directory and create symlinks
5. **Application**: Move application files to appropriate locations
6. **References**: Update import references and documentation
7. **Validation**: Run tests to ensure everything still works properly

## Cleanup Script

A shell script will be created in `scripts/cleanup_directory.sh` to automate this process:

```bash
#!/bin/bash
# cleanup_directory.sh - Reorganizes the project directory structure

echo "Creating backup of current project..."
tar -czf ../laion-embeddings-backup-$(date +%Y%m%d).tar.gz .

# Create directories
echo "Creating directory structure..."
mkdir -p docs/planning docs/ipfs docs/archive
mkdir -p test/unit test/ipfs test/performance
mkdir -p scripts
mkdir -p src/archive

# Move documentation files
echo "Organizing documentation files..."
mv DEPRECATION_COMPLETION.md docs/planning/ 2>/dev/null || true
mv DEPRECATION_PLAN.md docs/planning/ 2>/dev/null || true
mv DOCUMENTATION_UPDATES.md docs/ 2>/dev/null || true
mv IMMEDIATE_ACTION_PLAN.md docs/planning/ 2>/dev/null || true
mv IMPLEMENTATION_PLAN.md docs/planning/ 2>/dev/null || true
mv IPFS_FIXES_SUMMARY.md docs/ipfs/ 2>/dev/null || true

# Handle README files
echo "Handling README files..."
cp README_UPDATED.md README.md 2>/dev/null || true
mv README_ENHANCED.md docs/archive/ 2>/dev/null || true

# Move test files
echo "Organizing test files..."
mv test_imports.py test/unit/ 2>/dev/null || true
mv test_ipfs.py test/ipfs/ 2>/dev/null || true
mv test_ipfs_debug.py test/ipfs/ 2>/dev/null || true
mv test_ipfs_fixed.py test/ipfs/ 2>/dev/null || true
mv test_main_new_verification.py test/unit/ 2>/dev/null || true
mv test_max_batch_size_timeout.py test/performance/ 2>/dev/null || true
mv test_patches.py test/ipfs/ 2>/dev/null || true
mv test_timeout_comprehensive.py test/performance/ 2>/dev/null || true
mv test_timeout_implementation.py test/performance/ 2>/dev/null || true

# Move test result files
echo "Organizing test result files..."
mv distributed_test_results.txt test_results/ 2>/dev/null || true
mv full_ipfs_results.txt test_results/ 2>/dev/null || true
mv ipfs_test_results.txt test_results/ 2>/dev/null || true
mv test_output.txt test_results/ 2>/dev/null || true

# Move shell scripts
echo "Organizing script files..."
for script in $(find . -maxdepth 1 -name "*.sh"); do
  mv $script scripts/ 2>/dev/null || true
done

mv run_patched_tests.py scripts/ 2>/dev/null || true
mv run_tests.py scripts/ 2>/dev/null || true
mv run_tests_with_env.sh scripts/ 2>/dev/null || true

# Create symlinks for common scripts
echo "Creating symlinks for common scripts..."
ln -sf scripts/run.sh run.sh 2>/dev/null || true
ln -sf scripts/install_depends.sh install_depends.sh 2>/dev/null || true
ln -sf scripts/search.sh search.sh 2>/dev/null || true

# Move application files
echo "Organizing application files..."
mv main_old.py src/archive/ 2>/dev/null || true
mv mock_ipfs.py ipfs_embeddings_py/ 2>/dev/null || true

# Update import references
echo "Updating import references..."
find . -name "*.py" -type f -exec sed -i 's/from test_ipfs/from test.ipfs.test_ipfs/g' {} \; 2>/dev/null || true
find . -name "*.py" -type f -exec sed -i 's/import test_ipfs/import test.ipfs.test_ipfs/g' {} \; 2>/dev/null || true
find . -name "*.py" -type f -exec sed -i 's/from mock_ipfs/from ipfs_embeddings_py.mock_ipfs/g' {} \; 2>/dev/null || true
find . -name "*.py" -type f -exec sed -i 's/import mock_ipfs/import ipfs_embeddings_py.mock_ipfs/g' {} \; 2>/dev/null || true

# Update pytest.ini
echo "Updating pytest configuration..."
cat > pytest.ini << EOF
[pytest]
python_files = test_*.py
testpaths = test
python_functions = test_*
EOF

echo "Directory cleanup complete!"
```
