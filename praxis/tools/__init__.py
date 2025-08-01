from typing import List, Dict, Any, Callable
import inspect
import json
from transformers.utils import get_json_schema


def get_current_temperature(location: str, unit: str = "celsius") -> float:
    """Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for (e.g., "Paris", "New York")
        unit: The unit to return the temperature in (choices: ["celsius", "fahrenheit"])
    
    Returns:
        The current temperature as a float
    """
    # Simple mock implementation - in real usage, this would call a weather API
    mock_temps = {
        "paris": 15.0,
        "new york": 20.0,
        "london": 12.0,
        "tokyo": 18.0,
    }
    
    temp = mock_temps.get(location.lower(), 22.0)
    
    if unit.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return temp


def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number to add
        b: Second number to add
    
    Returns:
        The sum of a and b
    """
    return a + b


# Registry of available tools
AVAILABLE_TOOLS = [
    get_current_temperature,
    calculate_sum,
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