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
def read_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        The contents of the file
    """
    try:
        from pathlib import Path

        path = Path(file_path)

        # Read the file content
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return content
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


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
AVAILABLE_TOOLS = [get_tools, read_file, calc]

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


class ToolValidationError(ValueError):
    """Raised when a tool call fails schema validation.

    Subclasses ValueError so existing ``except ValueError`` callers still
    catch it, but distinct enough to recognize where that matters.
    """


def _find_tool(name: str) -> Optional[Any]:
    for t in get_all_tools():
        tool_name = t.name if isinstance(t, Tool) else getattr(t, "__name__", None)
        if tool_name == name:
            return t
    return None


def _type_matches(value: Any, expected: str) -> bool:
    """Lightweight JSON-shape check. Coarse on purpose - we only know
    the primitive types that ``_extract_parameters`` assigns."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True  # Unknown type specifier - don't reject.


def validate_tool_arguments(name: str, arguments: Any) -> None:
    """Validate ``arguments`` against the named tool's declared schema.

    Raises ``ToolValidationError`` with a short explanation on:
    ``arguments`` not being an object, unknown tool, unknown parameter(s),
    missing required parameter(s), or wrong primitive type. Required
    parameters are those whose schema ``default`` is ``None`` (the
    convention used by ``Tool._extract_parameters``).
    """
    if not isinstance(arguments, dict):
        raise ToolValidationError(
            f"'arguments' must be an object, got {type(arguments).__name__}"
        )

    tool = _find_tool(name)
    if tool is None:
        available = sorted(
            t.name if isinstance(t, Tool) else getattr(t, "__name__", "?")
            for t in get_all_tools()
        )
        raise ToolValidationError(f"Unknown tool '{name}'. Available: {available}")

    if not isinstance(tool, Tool):
        return  # Raw callable - no schema to check against.

    schema = tool.inputs
    unknown = sorted(set(arguments) - set(schema))
    if unknown:
        raise ToolValidationError(
            f"Unknown parameter(s) for '{name}': {unknown}. "
            f"Allowed: {sorted(schema)}"
        )

    missing = sorted(
        pname
        for pname, pspec in schema.items()
        if pspec.get("default") is None and pname not in arguments
    )
    if missing:
        raise ToolValidationError(
            f"Missing required parameter(s) for '{name}': {missing}"
        )

    for pname, value in arguments.items():
        expected = schema[pname].get("type")
        if expected and not _type_matches(value, expected):
            raise ToolValidationError(
                f"Parameter '{pname}' of '{name}' expected type "
                f"'{expected}', got {type(value).__name__}"
            )


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool by name with arguments.

    Raises ``ToolValidationError`` if the call doesn't match the tool's
    declared schema (unknown name, bad args, wrong types). Any exception
    raised by the tool itself propagates unchanged.
    """
    if name is None:
        raise ValueError("Tool name cannot be None")

    validate_tool_arguments(name, arguments)
    tool = _find_tool(name)
    return tool(**arguments)


# Sentinel returned by ``execute_tool_call`` when the caller should
# break out of its loop instead of splicing another result.
STOP_TOOL_LOOP = object()


def execute_tool_call(
    tool_call: Dict[str, Any],
    history: List[tuple],
    call_tool_fn: Any = call_tool,
    log: Any = print,
) -> Any:
    """Run a parsed tool call dict and return what to splice as the result.

    Centralises the malformed / missing-name / duplicate / exception
    branching that both generators were repeating. ``history`` is mutated
    in place to track executed (name, args) signatures across calls.

    Returns:
        - ``str`` payload to splice back as ``[TOOL_RESULT]``, OR
        - ``STOP_TOOL_LOOP`` to signal the caller should break.
    """
    if not isinstance(tool_call, dict):
        log(f"Malformed tool call (not a JSON object): {tool_call!r}")
        return f"Error: tool call must be a JSON object, got {type(tool_call).__name__}."

    if tool_call.get("_malformed"):
        err = tool_call.get("_error", "malformed tool call")
        log(f"Malformed tool call: {err}")
        return f"Error: {err}"

    tool_name = (
        tool_call.get("name")
        or tool_call.get("tool")
        or (tool_call.get("function", {}) or {}).get("name")
    )
    if tool_name is None:
        log(f"Could not extract tool name from: {tool_call}")
        return "Error: tool call is missing the 'name' field."

    tool_args = tool_call.get("arguments", {})
    signature = (tool_name, json.dumps(tool_args, sort_keys=True))
    if signature in history:
        log(f"[TOOL_SAFETY] Duplicate tool call: {tool_name}({tool_args})")
        return STOP_TOOL_LOOP
    history.append(signature)

    try:
        result = call_tool_fn(tool_name, tool_args)
        log(f"Tool {tool_name}({tool_args}) -> {result}")
        return result
    except Exception as exc:
        log(f"Tool {tool_name} failed: {exc}")
        return f"Error: {exc}"


# Export tag utilities for use across the codebase
from praxis.tools.tags import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    build_result_splice_ids,
    find_unprocessed_tool_call_ids,
    format_tool_call_with_result,
    format_tool_input,
    format_tool_output,
    get_tool_input_pattern,
    get_tool_output_pattern,
    get_unprocessed_tool_call,
    has_complete_tool_call,
    has_complete_tool_call_ids,
    has_tool_output,
    has_tool_output_ids,
    parse_tool_call,
    tool_token_ids,
)
