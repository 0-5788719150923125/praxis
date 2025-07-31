from typing import List, Dict, Any, Callable
import inspect
import json


def get_current_temperature(location: str, unit: str = "celsius") -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for (e.g., "Paris", "New York")
        unit: The unit to return the temperature in ("celsius" or "fahrenheit")
    
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
    """
    Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


# Registry of available tools
AVAILABLE_TOOLS = [
    get_current_temperature,
    calculate_sum,
]


def function_to_json_schema(func: Callable) -> Dict[str, Any]:
    """
    Convert a Python function with type hints and docstring to JSON schema format
    compatible with transformers function calling.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring to extract parameter descriptions
    lines = doc.split('\n')
    description = ""
    args_section = False
    returns_section = False
    param_descriptions = {}
    
    for line in lines:
        line = line.strip()
        if line.lower() == "args:":
            args_section = True
            returns_section = False
            continue
        elif line.lower() == "returns:":
            args_section = False
            returns_section = True
            continue
        elif args_section and ":" in line:
            param_name, param_desc = line.split(":", 1)
            param_descriptions[param_name.strip()] = param_desc.strip()
        elif not args_section and not returns_section and line:
            description = line
            break
    
    # Build parameter schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        param_type = param.annotation
        
        # Convert Python types to JSON schema types
        json_type = "string"  # default
        if param_type == int:
            json_type = "integer"
        elif param_type == float:
            json_type = "number"
        elif param_type == bool:
            json_type = "boolean"
        elif param_type == list or param_type == List:
            json_type = "array"
        elif param_type == dict or param_type == Dict:
            json_type = "object"
        
        prop_schema = {"type": json_type}
        
        # Add description if available
        if param_name in param_descriptions:
            prop_schema["description"] = param_descriptions[param_name]
        
        properties[param_name] = prop_schema
        
        # If no default value, parameter is required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


def get_tools_json_schema() -> List[Dict[str, Any]]:
    """Get JSON schema for all available tools."""
    return [function_to_json_schema(tool) for tool in AVAILABLE_TOOLS]


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