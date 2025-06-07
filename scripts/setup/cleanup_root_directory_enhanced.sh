#!/bin/bash

# Enhanced Root Directory Cleanup Script
# This script organizes files that have accumulated in the root directory
# after the previous cleanup, maintaining the established structure

echo "=========================================="
echo "ENHANCED ROOT DIRECTORY CLEANUP"
echo "Date: $(date)"
echo "=========================================="

# Create necessary directories if they don't exist
echo "Creating directory structure..."
mkdir -p docs/project-status
mkdir -p docs/reports
mkdir -p test/validation
mkdir -p test/mcp
mkdir -p test/integration
mkdir -p scripts/validation
mkdir -p scripts/testing
mkdir -p tools/testing
mkdir -p tmp/results
mkdir -p tmp/logs

echo ""
echo "Moving files to appropriate directories..."

# Move documentation files to docs/project-status/
echo "Organizing documentation files..."
mv CLEANUP_IMPLEMENTATION_PLAN.md docs/project-status/ 2>/dev/null
mv CLEANUP_SUMMARY.md docs/project-status/ 2>/dev/null
mv FINAL_DOCUMENTATION_UPDATE.md docs/project-status/ 2>/dev/null
mv FINAL_DOCUMENTATION_UPDATE_COMPLETE.md docs/project-status/ 2>/dev/null
mv FINAL_SYSTEM_VALIDATION_COMPLETE.md docs/project-status/ 2>/dev/null

# Move report files to docs/reports/
echo "Organizing report files..."
mv MCP_BUG_FIX_AND_VALIDATION_REPORT.md docs/reports/ 2>/dev/null
mv MCP_SERVER_FINAL_COMPLETION_REPORT.md docs/reports/ 2>/dev/null

# Move test files to appropriate test directories
echo "Organizing test files..."
mv test_final_integration.py test/integration/ 2>/dev/null
mv test_imports_only.py test/validation/ 2>/dev/null
mv test_mcp_components_final.py test/mcp/ 2>/dev/null
mv test_mcp_fix.py test/mcp/ 2>/dev/null
mv test_mcp_fix_validation.py test/mcp/ 2>/dev/null
mv test_simple_mcp.py test/mcp/ 2>/dev/null
mv test_simple_validation.py test/validation/ 2>/dev/null

# Move validation scripts to scripts/validation/
echo "Organizing validation scripts..."
mv final_mcp_validation.py scripts/validation/ 2>/dev/null
mv mcp_comprehensive_validation.py scripts/validation/ 2>/dev/null
mv final_validation_check.sh scripts/validation/ 2>/dev/null
mv run_validation.sh scripts/validation/ 2>/dev/null

# Move testing scripts to scripts/testing/
echo "Organizing testing scripts..."
mv test_terminal.sh scripts/testing/ 2>/dev/null

# Move status/summary scripts to tools/testing/
echo "Organizing utility scripts..."
mv final_status_summary.py tools/testing/ 2>/dev/null

# Move result files to tmp/results/
echo "Organizing result files..."
mv mcp_test_results.txt tmp/results/ 2>/dev/null
mv mcp_validation_results.txt tmp/results/ 2>/dev/null
mv tool_test_result.txt tmp/results/ 2>/dev/null

# Clean up Python cache and temporary files
echo "Cleaning up cache files..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo ""
echo "File organization summary:"
echo "=========================="

# Count files in each organized directory
echo "Documentation (docs/project-status/): $(find docs/project-status/ -type f 2>/dev/null | wc -l) files"
echo "Reports (docs/reports/): $(find docs/reports/ -type f 2>/dev/null | wc -l) files"
echo "Integration tests (test/integration/): $(find test/integration/ -type f -name "*.py" 2>/dev/null | wc -l) files"
echo "MCP tests (test/mcp/): $(find test/mcp/ -type f -name "*.py" 2>/dev/null | wc -l) files"
echo "Validation tests (test/validation/): $(find test/validation/ -type f -name "*.py" 2>/dev/null | wc -l) files"
echo "Validation scripts (scripts/validation/): $(find scripts/validation/ -type f 2>/dev/null | wc -l) files"
echo "Testing scripts (scripts/testing/): $(find scripts/testing/ -type f 2>/dev/null | wc -l) files"
echo "Testing tools (tools/testing/): $(find tools/testing/ -type f 2>/dev/null | wc -l) files"
echo "Result files (tmp/results/): $(find tmp/results/ -type f 2>/dev/null | wc -l) files"

echo ""
echo "Remaining files in root directory:"
echo "================================="
ls -la *.* 2>/dev/null | grep -v "^d" | wc -l || echo "0"

echo ""
echo "Root directory cleanup completed successfully!"
echo "All files have been organized into appropriate subdirectories."
echo ""
echo "Core files preserved in root:"
echo "- main.py (FastAPI application)"
echo "- mcp_server_enhanced.py (MCP server)"
echo "- README.md (main documentation)"
echo "- requirements.txt (dependencies)"
echo "- LICENSE (license file)"
echo "- pyproject.toml (project configuration)"
echo "- pytest.ini (test configuration)"
echo "- conftest.py (pytest configuration)"
echo "- Dockerfile (container configuration)"
echo "- __init__.py (package marker)"
echo ""
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
