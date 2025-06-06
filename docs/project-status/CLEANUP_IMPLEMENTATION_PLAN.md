# Root Directory Cleanup and Reorganization Plan

## 🎯 Objective
Clean up the root directory by organizing 100+ files into logical categories, archiving outdated content, and creating a maintainable structure.

## 📋 Current Analysis
The root directory is cluttered with:
- 25+ status/documentation files
- 30+ test files in root (should be in test/)
- 15+ debug/development scripts
- 10+ MCP server variants
- Multiple backup/duplicate files
- Log files and temporary outputs

## 🗂️ Target Structure

```
/home/barberb/laion-embeddings-1/
├── 📁 archive/                    # Deprecated and historical files
│   ├── deprecated-code/
│   ├── old-docs/
│   ├── backups/
│   └── logs/
├── 📁 config/                     # Configuration files
├── 📁 docs/                       # Documentation (existing + reorganized)
│   └── project-status/            # Status reports and completion docs
├── 📁 scripts/                    # Operational scripts
│   ├── setup/
│   ├── server/
│   ├── testing/
│   ├── audit/
│   └── examples/
├── 📁 src/                        # Source code (existing)
├── 📁 test/                       # Reorganized tests
│   ├── integration/
│   ├── mcp/
│   ├── vector/
│   └── debug/
├── 📁 tools/                      # Development utilities
├── 📁 tmp/                        # Temporary files and logs
├── main.py                        # Core application files
├── requirements.txt
├── README.md
└── LICENSE
```

## 📂 File Actions

### 🗄️ ARCHIVE (Move to archive/)
**Deprecated Code:**
- `auth.py` - superseded by src/auth
- `monitoring.py` - superseded by src/monitoring  
- `mock_ipfs.py` - development artifact
- `pytest_plugins.py` - unused

**Backup Files:**
- `conftest.py.bak`
- `conftest.py.disabled`
- `README_ENHANCED.md`
- `README_UPDATED.md`

**Old MCP Variants:**
- `mcp_server_minimal.py`
- `mcp_server_stdio.py` 
- `MCP_SERVER_UPDATE_METHODS.py`

**Log Files:**
- `*.log`, `*.txt` output files
- `audit_*.txt`, `debug_output.txt`
- `test_output.*`, `validation_output.txt`

### 📚 REORGANIZE Documentation (Move to docs/project-status/)
**Status Reports (25+ files):**
- All `*_COMPLETION*.md` files
- All `*_STATUS*.md` files  
- All `MCP_*.md` files
- All `VECTOR_*.md` files
- `DEPRECATION_*.md`, `IMPLEMENTATION_PLAN.md`, etc.

### 🔧 REORGANIZE Scripts (Move to scripts/)
**Setup Scripts:**
- `install_depends.sh` → `scripts/setup/`
- `setup_project.sh` → `scripts/setup/`

**Server Scripts:**
- `run.sh` → `scripts/server/`

**Testing Scripts:**
- `run_*tests*.py` → `scripts/testing/`
- `run_*.sh` → `scripts/testing/`

**Audit Scripts:**
- `*audit*.py` → `scripts/audit/`

**Example Scripts:**
- `search.sh` → `scripts/examples/`

### 🧪 REORGANIZE Tests (Move within test/)
**Integration Tests:**
- `test_integration_*.py` → `test/integration/`
- `test_*_integration.py` → `test/integration/`
- `test_master_suite.py` → `test/integration/`

**MCP Tests:**
- `test_mcp_*.py` → `test/mcp/`

**Vector Tests:**
- `test_vector_*.py` → `test/vector/`

**Debug Tests:**
- `test_*debug*.py` → `test/debug/`

### 🛠️ REORGANIZE Tools (Move to tools/)
**Development Tools:**
- `debug_*.py` → `tools/debugging/`
- `demo_*.py` → `tools/development/`
- `example_usage.py` → `tools/development/`

**Validation Tools:**
- `validate_*.py` → `tools/validation/`
- `*_status_check.py` → `tools/validation/`
- `*_validation.py` → `tools/validation/`

**MCP Tools:**
- `mcp_server_enhanced.py` → `tools/mcp/` (or keep in root)
- `start_mcp_server.py` → `tools/mcp/`

### 📁 REORGANIZE Config (Move to config/)
- `pyproject.toml` → `config/`
- `pytest.ini` → `config/`
- Create `config/mcp.json` (copy from .vscode/mcp.json)

## 🚀 Implementation Script

Let me create the implementation script:
