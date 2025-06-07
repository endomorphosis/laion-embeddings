# Development Guide

This guide provides information for developers working on the LAION Embeddings system.

## 🎯 Current Status (v2.2.0)

**Production Ready!** The system has achieved significant stability milestones:
- ✅ All 22 MCP tools fully functional (100% success rate)
- ✅ Comprehensive test suite with systematic validation
- ✅ Professional project organization and structure
- ✅ Complete documentation coverage
- ✅ All critical bugs resolved

## Development Environment Setup

### Prerequisites

1. **Python 3.8+** with virtual environment support
2. **IPFS node** (optional, for IPFS vector store)
3. **Git** for version control

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd laion-embeddings-1
```

2. Configure Python environment:
```bash
# The system will auto-configure the environment
python configure_python_environment.py
```

3. Install dependencies:
```bash
./install_depends.sh
```

4. Run tests to verify setup:
```bash
python test_all_mcp_tools.py
```

Expected output: `All 22 MCP tools working correctly! ✅`

## Project Structure

The project follows a professional, organized structure:

```
laion-embeddings-1/
├── src/                    # Source code
│   ├── mcp_server/        # MCP server implementation
│   │   └── tools/         # 22 MCP tools (all functional)
│   ├── services/          # Core services
│   └── components/        # Reusable components
├── docs/                  # Documentation
│   ├── api/              # API reference
│   ├── mcp/              # MCP-specific docs
│   └── reports/          # Status reports
├── tests/                 # Test suite
├── config/               # Configuration files
├── scripts/              # Utility scripts
└── tools/                # Development tools
```

## MCP Tools Development

### Tool Interface Standard

All MCP tools follow a consistent interface pattern:

```python
from typing import Dict, Any
from ..tool_registry import ClaudeMCPTool

class YourTool(ClaudeMCPTool):
    name: str = "your_tool"
    description: str = "Tool description"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard execute method signature.
        
        Args:
            parameters: Dictionary containing all tool parameters
            
        Returns:
            Dictionary with operation results
        """
        # Extract parameters
        action = parameters.get("action")
        
        # Implement tool logic
        try:
            result = await self._perform_action(action, parameters)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

### Testing MCP Tools

#### Individual Tool Testing
```python
# Test a specific tool
from src.mcp_server.tools.your_tool import YourTool

tool = YourTool()
result = await tool.execute({"action": "test"})
print(f"Tool result: {result}")
```

#### Comprehensive Testing
```python
# Run all tools test
python test_all_mcp_tools.py

# Expected output shows 22/22 tools working
```

### Adding New Tools

1. Create tool file in `src/mcp_server/tools/`:
```python
# src/mcp_server/tools/new_tool.py
from typing import Dict, Any
from ..tool_registry import ClaudeMCPTool

class NewTool(ClaudeMCPTool):
    name: str = "new_tool"
    description: str = "Description of new tool"
    
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action to perform"},
            # Add other parameters
        },
        "required": ["action"]
    }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        action = parameters.get("action")
        
        if action == "test":
            return {"success": True, "message": "Tool working"}
        else:
            return {"success": False, "error": "Unknown action"}
```

2. Register the tool (if using dynamic registration):
```python
# In tool registry or server setup
from .tools.new_tool import NewTool
register_tool(NewTool())
```

3. Add tests:
```python
# Add to test_all_mcp_tools.py or create specific test file
async def test_new_tool():
    tool = NewTool()
    result = await tool.execute({"action": "test"})
    assert result["success"] == True
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes following code standards
4. Add tests for new functionality
5. Run full test suite: `python test_all_mcp_tools.py && pytest`
6. Submit pull request with clear description

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings for public methods
- Maintain test coverage above 80%
- Use consistent error handling patterns

Current version: **v2.2.0** - All MCP tools functional, production ready