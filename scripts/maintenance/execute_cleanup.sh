#!/bin/bash

# Comprehensive Root Directory Cleanup Script
# This script implements the cleanup plan in phases

set -e

ROOT_DIR="/home/barberb/laion-embeddings-1"
cd "$ROOT_DIR"

echo "🧹 Starting Comprehensive Cleanup of Root Directory"
echo "=================================================="

# Phase 1: Create backup
echo "Phase 1: Creating backup..."
if [ ! -d "backup_before_cleanup" ]; then
    mkdir backup_before_cleanup
    # Only backup the files we're about to move, not entire directories
    echo "Creating selective backup of files to be moved..."
else
    echo "Backup already exists, skipping..."
fi

# Phase 2: Move Status Reports and Documentation to Archive
echo ""
echo "Phase 2: Moving status reports and documentation to archive..."

# Status reports
status_files=(
    "CLEANUP_COMPLETION.md"
    "CLEANUP_GUIDE.md" 
    "CLEANUP_REPORT.md"
    "DEPRECATION_COMPLETION.md"
    "DEPRECATION_PLAN.md"
    "DIRECTORY_CLEANUP_PLAN.md"
    "DOCUMENTATION_UPDATES.md"
    "DOCUMENTATION_UPDATE_COMPLETION.md"
    "FINAL_COMPLETION_STATUS.md"
    "FINAL_DOCUMENTATION_UPDATE.md"
    "FINAL_DOCUMENTATION_UPDATE_COMPLETE.md"
    "FINAL_PROJECT_STATUS.md"
    "FINAL_PROJECT_STATUS_COMPLETE.md"
    "IMMEDIATE_ACTION_PLAN.md"
    "IMPLEMENTATION_PLAN.md"
    "IPFS_FIXES_SUMMARY.md"
    "MCP_COMPREHENSIVE_AUDIT_FINAL.md"
    "MCP_CONFIGURATION_COMPLETE.md"
    "MCP_FASTAPI_AUDIT_REPORT.md"
    "MCP_FEATURE_COVERAGE_ANALYSIS.md"
    "MCP_SERVER_COMPLETE.md"
    "MCP_SERVER_FINAL_REPORT.md"
    "MCP_SERVICE_INTEGRATION_PLAN.md"
    "MCP_TOOL_COVERAGE_ANALYSIS.md"
    "MCP_TOOL_COVERAGE_IMPLEMENTATION_COMPLETE.md"
    "MCP_TOOL_IMPLEMENTATION_COMPLETE.md"
    "PROJECT_COMPLETION_SUMMARY.md"
    "PYTEST_FINAL_STATUS.md"
    "PYTEST_FIX_COMPLETION.md"
    "PYTEST_STATUS_FINAL.md"
    "TASK_COMPLETION_SUMMARY.md"
    "VECTOR_DEPENDENCIES.md"
    "VECTOR_DOCUMENTATION_COMPLETE.md"
    "VECTOR_INTEGRATION_COMPLETE.md"
    "VECTOR_INTEGRATION_STATUS.md"
    "VECTOR_STORE_README.md"
    "VECTOR_TEST_SUITE_FINAL_REPORT.md"
    "audit_final_output.txt"
    "audit_output.log"
    "debug_output.txt"
    "laion_embeddings.log"
    "mcp_test_output.log"
    "mcp_test_results.txt"
    "test_output.log"
    "test_output.txt"
    "test_output_new.log"
    "test_results.log"
)

for file in "${status_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to archive/status_reports/"
        mv "$file" archive/status_reports/
    fi
done

# Documentation versions
doc_files=(
    "README_ENHANCED.md"
    "README_UPDATED.md"
    "QUICK_START.md"
)

for file in "${doc_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to archive/documentation/"
        mv "$file" archive/documentation/
    fi
done

# Phase 3: Move Development and Debug Files
echo ""
echo "Phase 3: Moving development and debug files to archive..."

dev_files=(
    "debug_mcp.py"
    "debug_sklearn.py"
    "debug_test_import.py"
    "debug_tool_registration.py"
    "demo_vector_architecture.py"
    "example_usage.py"
    "mock_ipfs.py"
    "monitoring.py"
    "auth.py"
    "pytest_plugins.py"
    "conftest.py.bak"
    "conftest.py.disabled"
)

for file in "${dev_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to archive/development/"
        mv "$file" archive/development/
    fi
done

# Phase 4: Move MCP Experiments
echo ""
echo "Phase 4: Moving MCP experiments to archive..."

mcp_files=(
    "mcp_server_enhanced.py"
    "mcp_server_minimal.py"
    "mcp_server_stdio.py"
    "start_mcp_server.py"
    "mcp_status_check.py"
    "simple_mcp_test.py"
    "simple_status_check.py"
    "quick_status_check.py"
)

for file in "${mcp_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to archive/mcp_experiments/"
        mv "$file" archive/mcp_experiments/
    fi
done

# Phase 5: Move Test Experiments
echo ""
echo "Phase 5: Moving test experiments to archive..."

test_files=(
    "test_all_vectors.py"
    "test_basic.py"
    "test_basic_imports.py"
    "test_clustering_debug.py"
    "test_deprecation.py"
    "test_diagnostic.py"
    "test_final_integration.py"
    "test_imports.py"
    "test_imports_only.py"
    "test_integration_final.py"
    "test_integration_standalone.py"
    "test_ipfs.py"
    "test_ipfs_debug.py"
    "test_ipfs_fixed.py"
    "test_master_suite.py"
    "test_mcp_complete.py"
    "test_mcp_components.py"
    "test_mcp_components_final.py"
    "test_mcp_coverage.py"
    "test_mcp_imports.py"
    "test_mcp_integration.py"
    "test_mcp_server.py"
    "test_mcp_server_startup.py"
    "test_mcp_simple.py"
    "test_mcp_startup_debug.py"
    "test_mcp_subprocess.py"
    "test_mcp_write_results.py"
    "test_minimal_imports.py"
    "test_new_tools.py"
    "test_new_tools_direct.py"
    "test_patches.py"
    "test_python_basic.py"
    "test_quick_integration.py"
    "test_simple_debug.py"
    "test_simple_mcp.py"
    "test_simple_validation.py"
    "test_summary.py"
    "test_terminal.sh"
    "test_tool_fix.py"
    "test_vector_advanced.py"
    "test_vector_architecture.py"
    "test_vector_benchmarks.py"
    "test_vector_integration.py"
    "test_vector_integrity.py"
    "test_vector_security.py"
    "test_vector_stores.py"
    "test_vector_validation.py"
    "test_vector_validation_quick.py"
    "simple_test.py"
    "quick_test.py"
)

for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to archive/test_experiments/"
        mv "$file" archive/test_experiments/
    fi
done

# Phase 6: Move Tools and Utilities
echo ""
echo "Phase 6: Moving tools and utilities..."

# Audit tools
audit_files=(
    "comprehensive_audit.py"
    "final_comprehensive_audit.py"
    "mcp_final_audit_report.py"
    "run_audit.py"
)

for file in "${audit_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to tools/audit/"
        mv "$file" tools/audit/
    fi
done

# Testing tools
testing_files=(
    "run_comprehensive_tests.py"
    "run_vector_tests_standalone.py"
    "run_patched_tests.py"
    "run_tests.py"
    "run_tests_with_env.sh"
    "run_ipfs_tests.sh"
    "run_full_ipfs_tests.sh"
    "run_ipfs_test_debug.sh"
    "run_ipfs_test_summary.sh"
)

for file in "${testing_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to tools/testing/"
        mv "$file" tools/testing/
    fi
done

# Validation tools
validation_files=(
    "validate_mcp_server.py"
    "validate_tools.py"
    "final_mcp_validation.py"
    "final_mcp_status_check.py"
)

for file in "${validation_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to tools/validation/"
        mv "$file" tools/validation/
    fi
done

# Phase 7: Move Scripts
echo ""
echo "Phase 7: Moving utility scripts..."

script_files=(
    "install_depends.sh"
    "run.sh"
    "search.sh"
    "setup_project.sh"
    "project_summary.sh"
    "run_validation.sh"
)

for file in "${script_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to scripts/"
        mv "$file" scripts/
    fi
done

# Phase 8: Move Configuration
echo ""
echo "Phase 8: Moving configuration files..."

if [ -f "pytest.ini" ]; then
    echo "Moving pytest.ini to config/"
    mv pytest.ini config/
fi

if [ -f "conftest.py" ]; then
    echo "Moving conftest.py to config/"
    mv conftest.py config/
fi

if [ -d ".vscode" ]; then
    echo "Moving .vscode to config/"
    mv .vscode config/
fi

# Phase 9: Move remaining Python files that are utilities
echo ""
echo "Phase 9: Moving remaining utility Python files..."

utility_python_files=(
    "MCP_SERVER_UPDATE_METHODS.py"
)

for file in "${utility_python_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $file to archive/development/"
        mv "$file" archive/development/
    fi
done

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📊 Root directory contents after cleanup:"
echo "========================================"
ls -la | grep -v "^d" | wc -l | xargs echo "Files in root:"
echo ""
echo "📁 Directory structure:"
echo "======================"
tree -d -L 2

echo ""
echo "📋 Next steps:"
echo "============="
echo "1. Update .vscode/mcp.json path (now in config/.vscode/)"
echo "2. Update import paths in moved files"
echo "3. Update documentation references"
echo "4. Test functionality"
echo ""
echo "✨ Cleanup completed successfully!"
