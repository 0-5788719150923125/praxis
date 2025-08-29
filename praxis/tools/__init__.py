import inspect
import json
import math
from typing import Any, Callable, Dict, List

from transformers.utils import get_json_schema


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
AVAILABLE_TOOLS = [
    calc,
]


def get_tools_json_schema() -> List[Dict[str, Any]]:
    """Get JSON schema for all available tools.

    Uses transformers.utils.get_json_schema to automatically generate
    JSON schemas from function signatures and docstrings.
    """
    # The transformers get_json_schema returns schemas in the format
    # expected by chat templates (with type: "function" wrapper)
    return [get_json_schema(tool) for tool in AVAILABLE_TOOLS]


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Call a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to call
        arguments: Dictionary of arguments to pass to the tool

    Returns:
        The result of the tool call
    """
    # Find the tool
    tool_func = None
    for tool in AVAILABLE_TOOLS:
        if tool.__name__ == tool_name:
            tool_func = tool
            break

    if tool_func is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    # Call the tool with arguments
    return tool_func(**arguments)
