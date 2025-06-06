#!/usr/bin/env python3
"""
Minimal import test for MCP server components.
"""
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Starting import tests...")

try:
    print("1. Testing error_handlers import...")
    from mcp_server.error_handlers import ValidationError, MCPError
    print("✓ error_handlers imported successfully")
except Exception as e:
    print(f"✗ error_handlers import failed: {e}")
    raise

try:
    print("2. Testing validators import...")
    from mcp_server.validators import validator
    print("✓ validators imported successfully")
except Exception as e:
    print(f"✗ validators import failed: {e}")
    raise

try:
    print("3. Testing tool_registry import...")
    from mcp_server.tool_registry import ClaudeMCPTool, ToolRegistry
    print("✓ tool_registry imported successfully")
except Exception as e:
    print(f"✗ tool_registry import failed: {e}")
    raise

try:
    print("4. Testing analysis_tools import...")
    from mcp_server.tools.analysis_tools import ClusterAnalysisTool
    print("✓ analysis_tools imported successfully")
except Exception as e:
    print(f"✗ analysis_tools import failed: {e}")
    raise

print("✅ All imports successful!")
