import json
import math
from typing import Any, Dict, List, Optional, Union

from smolagents import tool


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
AVAILABLE_TOOLS = [calc]

# Registry for dynamically loaded tools (from JSON)
DYNAMIC_TOOLS: Dict[str, Any] = {}


def get_all_tools() -> List[Any]:
    """Get all available tools (both built-in and dynamic)."""
    return AVAILABLE_TOOLS + list(DYNAMIC_TOOLS.values())


def get_tools() -> List[Any]:
    """Get all available tool instances for smolagents."""
    return get_all_tools()


def get_tools_json_schema() -> List[Dict[str, Any]]:
    """Get tools in JSON schema format for chat templates."""
    tools_schema = []
    for tool in get_all_tools():
        # Convert smolagents tool to JSON schema format
        tool_schema = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputs  # smolagents tools have inputs attribute
        }
        tools_schema.append(tool_schema)
    return tools_schema


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool by name with arguments (backward compatibility)."""
    tools = get_all_tools()
    for tool in tools:
        if tool.name == name:
            return tool(**arguments)
    raise ValueError(f"Tool '{name}' not found")
