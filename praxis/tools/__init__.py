import json
import math
from typing import Any, Dict, List, Optional, Union


class Tool:
    """Simple tool wrapper to replace smolagents."""

    def __init__(self, func, name=None, description=None, parameters=None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.inputs = parameters or self._extract_parameters(func)
        self.output_type = "float"  # Default for calc tool

    def _extract_parameters(self, func):
        """Extract parameters from function signature."""
        import inspect

        sig = inspect.signature(func)
        params = {}
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == List[float]:
                    param_type = "array"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == str:
                    param_type = "string"
            params[param_name] = {
                "type": param_type,
                "default": (
                    param.default if param.default != inspect.Parameter.empty else None
                ),
            }
        return params

    def forward(self, **kwargs):
        """Execute the tool with given arguments."""
        return self.func(**kwargs)

    def __call__(self, **kwargs):
        """Allow direct calling of the tool."""
        return self.func(**kwargs)


def tool(func):
    """Decorator to create a tool from a function."""
    return Tool(func)


@tool
def get_tools() -> str:
    """Get the JSON schema of all available tools.

    Returns:
        JSON string containing all available tool schemas
    """
    tools_schema = []
    for tool in AVAILABLE_TOOLS:
        # Skip the get_tools tool itself to avoid recursion
        if isinstance(tool, Tool) and tool.name == "get_tools":
            continue

        # Handle both Tool instances and regular functions
        if isinstance(tool, Tool):
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputs,
            }
        elif callable(tool):
            # Fallback for regular functions
            tool_schema = {
                "name": getattr(tool, "__name__", "unknown"),
                "description": getattr(tool, "__doc__", ""),
                "parameters": {},
            }
        else:
            continue
        tools_schema.append(tool_schema)

    # Also include dynamic tools
    for tool in DYNAMIC_TOOLS.values():
        if isinstance(tool, Tool):
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputs,
            }
            tools_schema.append(tool_schema)

    return json.dumps(tools_schema, indent=2)


@tool
def calc(values: List[float], op: str = "add") -> float:
    """Perform a basic algebraic operation on a list of values.

    Args:
        values: List of numbers to perform the operation on
        op: The operation to perform (choices: ["add", "sub", "mul", "div", "sqrt", "exp"])

    Returns:
        The result of the operation
    """
    if not values:
        raise ValueError("values list cannot be empty")

    if op == "add":
        return sum(values)
    elif op == "sub":
        result = values[0]
        for v in values[1:]:
            result -= v
        return result
    elif op == "mul":
        result = 1.0
        for v in values:
            result *= v
        return result
    elif op == "div":
        result = values[0]
        for v in values[1:]:
            if v == 0:
                raise ValueError("Division by zero")
            result /= v
        return result
    elif op == "sqrt":
        if values[0] < 0:
            raise ValueError("Cannot take square root of negative number")
        return math.sqrt(values[0])
    elif op == "exp":
        if len(values) < 2:
            raise ValueError("exp operation requires at least 2 values")
        return math.pow(values[0], values[1])
    else:
        raise ValueError(f"Unknown operation: {op}")


# Registry of available tools
AVAILABLE_TOOLS = [get_tools, calc]

# Registry for dynamically loaded tools (from JSON)
DYNAMIC_TOOLS: Dict[str, Any] = {}


def get_all_tools() -> List[Any]:
    """Get all available tools (both built-in and dynamic)."""
    return AVAILABLE_TOOLS + list(DYNAMIC_TOOLS.values())


def get_tools() -> List[Any]:
    """Get all available tool instances."""
    return get_all_tools()


def get_tools_json_schema() -> List[Dict[str, Any]]:
    """Get tools in JSON schema format for chat templates."""
    tools_schema = []
    for tool in get_all_tools():
        # Handle both Tool instances and regular functions
        if isinstance(tool, Tool):
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputs,
            }
        elif callable(tool):
            # Fallback for regular functions
            tool_schema = {
                "name": getattr(tool, "__name__", "unknown"),
                "description": getattr(tool, "__doc__", ""),
                "parameters": {},
            }
        else:
            continue
        tools_schema.append(tool_schema)
    return tools_schema


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool by name with arguments (backward compatibility)."""
    tools = get_all_tools()
    for tool in tools:
        tool_name = (
            tool.name if isinstance(tool, Tool) else getattr(tool, "__name__", None)
        )
        if tool_name == name:
            if isinstance(tool, Tool):
                return tool(**arguments)
            elif callable(tool):
                return tool(**arguments)
    raise ValueError(f"Tool '{name}' not found")
