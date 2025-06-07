"""
Utility to wrap standalone functions as MCP tools.
"""

import inspect
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from ..tool_registry import ClaudeMCPTool

logger = logging.getLogger(__name__)


class FunctionToolWrapper(ClaudeMCPTool):
    """
    Wrapper to convert a standalone async function into an MCP tool.
    """
    
    def __init__(self, 
                 function: Callable,
                 tool_name: str,
                 category: str = "general",
                 description: Optional[str] = None,
                 tags: Optional[list] = None):
        super().__init__()
        
        self.function = function
        self.name = tool_name
        self.category = category
        self.description = description or function.__doc__ or f"Execute {tool_name}"
        self.tags = tags or []
        
        # Extract input schema from function signature and docstring
        self.input_schema = self._extract_input_schema()
        
    def _extract_input_schema(self) -> Dict[str, Any]:
        """
        Extract input schema from function signature and annotations.
        """
        try:
            sig = inspect.signature(self.function)
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    "type": self._python_type_to_json_type(param.annotation),
                    "description": f"Parameter {param_name}"
                }
                
                # Check if parameter has a default value
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                else:
                    param_info["default"] = param.default
                
                properties[param_name] = param_info
            
            schema = {
                "type": "object",
                "properties": properties
            }
            
            if required:
                schema["required"] = required
                
            return schema
            
        except Exception as e:
            logger.warning(f"Could not extract schema for {self.name}: {e}")
            return {"type": "object", "properties": {}}
    
    def _python_type_to_json_type(self, python_type) -> str:
        """
        Convert Python type annotations to JSON schema types.
        """
        if python_type == inspect.Parameter.empty:
            return "string"  # Default type
        
        type_mapping = {
            str: "string",
            int: "integer", 
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        # Handle Optional types and Union types
        if hasattr(python_type, '__origin__'):
            if python_type.__origin__ is list:
                return "array"
            elif python_type.__origin__ is dict:
                return "object"
            elif python_type.__origin__ is type(None):
                return "null"
            # For Union types (like Optional), return the first non-None type
            elif hasattr(python_type, '__args__'):
                for arg_type in python_type.__args__:
                    if arg_type is not type(None):
                        return self._python_type_to_json_type(arg_type)
        
        return type_mapping.get(python_type, "string")
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the wrapped function with the given parameters.
        """
        try:
            # Call the function with parameters
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**parameters)
            else:
                result = self.function(**parameters)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}
            
            # Add execution metadata
            result.update({
                "tool_name": self.name,
                "executed_at": datetime.utcnow().isoformat(),
                "success": result.get("success", True)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing {self.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name,
                "executed_at": datetime.utcnow().isoformat()
            }


def wrap_function_as_tool(function: Callable, 
                         tool_name: str,
                         category: str = "general",
                         description: Optional[str] = None,
                         tags: Optional[list] = None) -> FunctionToolWrapper:
    """
    Convenience function to wrap a standalone function as an MCP tool.
    
    Args:
        function: The async function to wrap
        tool_name: Name for the tool
        category: Category for the tool
        description: Optional description (uses function docstring if not provided)
        tags: Optional tags for the tool
    
    Returns:
        FunctionToolWrapper instance
    """
    return FunctionToolWrapper(
        function=function,
        tool_name=tool_name,
        category=category,
        description=description,
        tags=tags
    )


def wrap_function_with_metadata(function: Callable, 
                               metadata: Dict[str, Any]) -> FunctionToolWrapper:
    """
    Wrap a function using metadata dictionary (compatible with TOOL_METADATA format).
    
    Args:
        function: The function to wrap
        metadata: Metadata dictionary with tool information
    
    Returns:
        FunctionToolWrapper instance
    """
    tool_info = metadata.get(function.__name__, {})
    
    return FunctionToolWrapper(
        function=function,
        tool_name=tool_info.get("name", function.__name__),
        category=tool_info.get("category", "general"),
        description=tool_info.get("description", function.__doc__),
        tags=tool_info.get("tags", [])
    )
