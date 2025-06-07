#!/bin/bash

# Root Directory Cleanup Script
# This script reorganizes the laion-embeddings-1 root directory

set -e  # Exit on error

PROJECT_ROOT="/home/barberb/laion-embeddings-1"
cd "$PROJECT_ROOT"

echo "🧹 Starting Root Directory Cleanup..."
echo "📍 Working directory: $(pwd)"

# Phase 1: Create new directory structure
echo "📁 Creating directory structure..."

mkdir -p archive/{deprecated-code,old-docs,backups,logs}
mkdir -p scripts/{setup,server,testing,audit,examples,maintenance,info}
mkdir -p test/{integration,mcp,vector,debug}
mkdir -p tools/{development,validation,debugging,mcp}
mkdir -p tmp/{logs,outputs,cache}
mkdir -p config
mkdir -p docs/project-status

echo "✅ Directory structure created"

# Phase 2: Archive files
echo "🗄️ Archiving deprecated and backup files..."

# Archive deprecated code
[ -f "auth.py" ] && mv auth.py archive/deprecated-code/
[ -f "monitoring.py" ] && mv monitoring.py archive/deprecated-code/
[ -f "mock_ipfs.py" ] && mv mock_ipfs.py archive/deprecated-code/
[ -f "pytest_plugins.py" ] && mv pytest_plugins.py archive/deprecated-code/

# Archive backup files
[ -f "conftest.py.bak" ] && mv conftest.py.bak archive/backups/
[ -f "conftest.py.disabled" ] && mv conftest.py.disabled archive/backups/
[ -f "README_ENHANCED.md" ] && mv README_ENHANCED.md archive/backups/
[ -f "README_UPDATED.md" ] && mv README_UPDATED.md archive/backups/
[ -f "requirements-mcp.txt" ] && mv requirements-mcp.txt archive/backups/

# Archive old MCP variants (keep mcp_server_enhanced.py)
[ -f "mcp_server_minimal.py" ] && mv mcp_server_minimal.py archive/deprecated-code/
[ -f "mcp_server_stdio.py" ] && mv mcp_server_stdio.py archive/deprecated-code/
[ -f "MCP_SERVER_UPDATE_METHODS.py" ] && mv MCP_SERVER_UPDATE_METHODS.py archive/deprecated-code/

# Archive log and output files
find . -maxdepth 1 -name "*.log" -exec mv {} archive/logs/ \;
find . -maxdepth 1 -name "*output*.txt" -exec mv {} archive/logs/ \;
find . -maxdepth 1 -name "audit_*.txt" -exec mv {} archive/logs/ \;
find . -maxdepth 1 -name "test_output*" -exec mv {} archive/logs/ \;
find . -maxdepth 1 -name "mcp_test_*.txt" -exec mv {} archive/logs/ \;
find . -maxdepth 1 -name "validation_output.txt" -exec mv {} archive/logs/ \; 2>/dev/null || true

echo "✅ Archival complete"

# Phase 3: Move documentation files
echo "📚 Organizing documentation..."

# Move all status/completion documentation (excluding main README.md and LICENSE)
for file in *COMPLETION*.md *STATUS*.md MCP_*.md VECTOR_*.md DEPRECATION_*.md IMPLEMENTATION_PLAN.md IMMEDIATE_ACTION_PLAN.md PROJECT_*.md TASK_*.md IPFS_*.md PYTEST_*.md QUICK_START.md; do
    [ -f "$file" ] && mv "$file" docs/project-status/ 2>/dev/null || true
done

# Move remaining .md files that aren't core docs
for file in COMPREHENSIVE_CLEANUP_PLAN.md DIRECTORY_CLEANUP_PLAN.md DOCUMENTATION_*.md; do
    [ -f "$file" ] && mv "$file" docs/project-status/ 2>/dev/null || true
done

echo "✅ Documentation organized"

# Phase 4: Organize scripts
echo "🔧 Organizing scripts..."

# Setup scripts
[ -f "install_depends.sh" ] && mv install_depends.sh scripts/setup/
[ -f "setup_project.sh" ] && mv setup_project.sh scripts/setup/

# Server scripts  
[ -f "run.sh" ] && mv run.sh scripts/server/

# Testing scripts
for file in run_*tests*.py run_*test*.sh run_comprehensive_tests.py run_vector_tests_standalone.py; do
    [ -f "$file" ] && mv "$file" scripts/testing/ 2>/dev/null || true
done

# Audit scripts
for file in comprehensive_audit.py final_comprehensive_audit.py run_audit.py *audit*.py; do
    [ -f "$file" ] && mv "$file" scripts/audit/ 2>/dev/null || true
done

# Example scripts
[ -f "search.sh" ] && mv search.sh scripts/examples/

# Maintenance scripts
[ -f "execute_cleanup.sh" ] && mv execute_cleanup.sh scripts/maintenance/
[ -f "project_summary.sh" ] && mv project_summary.sh scripts/info/

echo "✅ Scripts organized"

# Phase 5: Reorganize test files
echo "🧪 Reorganizing test files..."

# Integration tests
for file in test_integration*.py test_*integration.py test_master_suite.py test_all_vectors.py test_final_integration.py test_quick_integration.py; do
    [ -f "$file" ] && mv "$file" test/integration/ 2>/dev/null || true
done

# MCP tests
for file in test_mcp*.py; do
    [ -f "$file" ] && mv "$file" test/mcp/ 2>/dev/null || true
done

# Vector tests  
for file in test_vector*.py; do
    [ -f "$file" ] && mv "$file" test/vector/ 2>/dev/null || true
done

# Debug tests
for file in test_*debug*.py test_diagnostic.py test_simple_*.py; do
    [ -f "$file" ] && mv "$file" test/debug/ 2>/dev/null || true
done

# Basic tests that don't fit other categories
for file in test_basic*.py test_imports*.py test_minimal*.py test_patches.py test_deprecation.py test_clustering_debug.py; do
    [ -f "$file" ] && mv "$file" test/integration/ 2>/dev/null || true
done

echo "✅ Tests reorganized"

# Phase 6: Organize development tools
echo "🛠️ Organizing development tools..."

# Debugging tools
for file in debug_*.py; do
    [ -f "$file" ] && mv "$file" tools/debugging/ 2>/dev/null || true
done

# Development tools
for file in demo_*.py example_usage.py; do
    [ -f "$file" ] && mv "$file" tools/development/ 2>/dev/null || true
done

# Validation tools
for file in validate_*.py *_status_check.py *_validation.py final_mcp_*.py mcp_comprehensive_validation.py; do
    [ -f "$file" ] && mv "$file" tools/validation/ 2>/dev/null || true
done

# MCP tools
[ -f "start_mcp_server.py" ] && mv start_mcp_server.py tools/mcp/
for file in simple_mcp_test.py simple_test.py quick_test.py; do
    [ -f "$file" ] && mv "$file" tools/mcp/ 2>/dev/null || true
done

echo "✅ Development tools organized"

# Phase 7: Handle configuration
echo "📁 Organizing configuration..."

# Create config copies (don't move originals yet to avoid breaking things)
[ -f "pyproject.toml" ] && cp pyproject.toml config/
[ -f "pytest.ini" ] && cp pytest.ini config/
[ -f ".vscode/mcp.json" ] && cp .vscode/mcp.json config/mcp.json

echo "✅ Configuration organized"

# Phase 8: Clean up remaining files
echo "🧹 Final cleanup..."

# Move any remaining test files that might have been missed
for file in test_*.py; do
    if [ -f "$file" ]; then
        echo "Moving remaining test file: $file"
        mv "$file" test/debug/
    fi
done

# Move remaining Python utility files
for file in *_mcp_*.py mcp_*.py; do
    if [ -f "$file" ] && [ "$file" != "mcp_server_enhanced.py" ]; then
        mv "$file" tools/validation/ 2>/dev/null || true
    fi
done

# Handle any remaining .py files that aren't core
for file in simple_*.py quick_*.py; do
    [ -f "$file" ] && mv "$file" tools/development/ 2>/dev/null || true
done

echo "✅ Final cleanup complete"

# Phase 9: Create summary
echo "📊 Creating cleanup summary..."

cat > CLEANUP_SUMMARY.md << 'EOF'
# Root Directory Cleanup Summary

## ✅ Completed Actions

### Files Moved:
- **Documentation**: 25+ status files → `docs/project-status/`
- **Scripts**: 15+ utility scripts → `scripts/` (categorized)
- **Tests**: 30+ test files → `test/` (categorized by type)
- **Tools**: 15+ development tools → `tools/` (categorized)
- **Archive**: 20+ deprecated/backup files → `archive/`
- **Config**: Configuration files → `config/`

### Directory Structure Created:
```
├── archive/{deprecated-code,old-docs,backups,logs}/
├── scripts/{setup,server,testing,audit,examples,maintenance,info}/
├── test/{integration,mcp,vector,debug}/
├── tools/{development,validation,debugging,mcp}/
├── tmp/{logs,outputs,cache}/
├── config/
└── docs/project-status/
```

### Core Files Remaining in Root:
- `main.py` - FastAPI application
- `requirements.txt` - Dependencies  
- `README.md` - Main documentation
- `LICENSE` - License file
- `mcp_server_enhanced.py` - Primary MCP server
- Configuration files (`pyproject.toml`, `pytest.ini`)
- Core directories (`src/`, `docs/`, etc.)

## 🔄 Next Steps:
1. Update import paths in moved files
2. Update script references  
3. Test that everything still works
4. Update documentation paths
5. Commit changes to git
EOF

echo "📋 Summary created: CLEANUP_SUMMARY.md"

# Phase 10: Show results
echo ""
echo "🎉 Root Directory Cleanup Complete!"
echo ""
echo "📊 Current root directory contents:"
ls -la | grep -E '^[^d]' | wc -l | xargs echo "Files in root:"
echo ""
echo "📁 New directories created:"
find . -maxdepth 2 -type d -name "*" | grep -v "^\.$" | grep -v "^\./" | sort

echo ""
echo "⚠️  IMPORTANT: You may need to:"
echo "   1. Update import statements in moved files"
echo "   2. Update script paths and references"  
echo "   3. Update .vscode/mcp.json if needed"
echo "   4. Run tests to verify everything works"
echo ""
echo "📁 Check CLEANUP_SUMMARY.md for detailed information"
