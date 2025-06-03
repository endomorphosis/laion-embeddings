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
rm README_UPDATED.md 2>/dev/null || true

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
mkdir -p test_results
mv distributed_test_results.txt test_results/ 2>/dev/null || true
mv full_ipfs_results.txt test_results/ 2>/dev/null || true
mv ipfs_test_results.txt test_results/ 2>/dev/null || true
mv test_output.txt test_results/ 2>/dev/null || true

# Move shell scripts (except this one)
echo "Organizing script files..."
for script in $(find . -maxdepth 1 -name "*.sh"); do
  # Skip this script if we're running from the root directory
  if [ "$(basename $script)" != "$(basename $0)" ]; then
    mv $script scripts/ 2>/dev/null || true
  fi
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
cp mock_ipfs.py ipfs_embeddings_py/ 2>/dev/null || true  # Copy instead of move, for safety
echo "NOTE: mock_ipfs.py has been copied to ipfs_embeddings_py/ - please remove the root copy manually after verifying"

# Create __init__.py files for proper imports
echo "Creating __init__.py files for package structure..."
touch test/__init__.py
touch test/unit/__init__.py
touch test/ipfs/__init__.py
touch test/performance/__init__.py
touch src/__init__.py
touch src/archive/__init__.py

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

# Generate a report of changes
echo "Generating cleanup report..."
cat > CLEANUP_REPORT.md << EOF
# Directory Cleanup Report

## Changes Made
- Created organized directory structure
- Moved documentation files to docs/ directory
- Consolidated README files
- Organized test files into test/unit/, test/ipfs/, test/performance/
- Moved test result files to test_results/
- Moved shell scripts to scripts/
- Created symlinks for commonly used scripts
- Updated import references
- Updated pytest configuration

## Manual Actions Required
1. Remove the original mock_ipfs.py from the root directory after verifying the copy in ipfs_embeddings_py/ works correctly
2. Update any project-specific imports that the script may have missed
3. Run the test suite to verify everything works correctly
4. Update any CI/CD configurations to reflect the new directory structure

## New Directory Structure
\`\`\`
$(find . -type f -o -type d | grep -v "__pycache__" | grep -v ".git" | sort)
\`\`\`
EOF

echo "Directory cleanup complete!"
echo "A report has been generated in CLEANUP_REPORT.md"
echo "Please review the report and perform any required manual actions."
