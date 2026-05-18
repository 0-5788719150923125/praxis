"""Tool calling format generation."""

import json
import math
import random
from typing import Dict, List

from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt
from praxis.tools import format_tool_input, format_tool_output


def _log_weighted_int(lo: int, hi: int) -> int:
    """Sample an int in [lo, hi] with weights proportional to 1/n.

    Smaller counts dominate (expected value grows like log(hi/lo)), but
    the full range is still reachable - so the model sees many 2-value
    calls and occasional long ones, without flipping the distribution.
    """
    choices = list(range(lo, hi + 1))
    weights = [1.0 / n for n in choices]
    return random.choices(choices, weights=weights, k=1)[0]


def format_tool_calling(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """
    Format synthetic tool-calling examples for training.
    Generates math problems that require the calc tool.

    Args:
        document: Dictionary containing the document data (unused for synthetic generation)
        keys: List of keys to extract from document (unused for synthetic generation)
        tokenizer: Tokenizer with chat template support

    Returns:
        Dictionary with messages and metadata
    """

    # Choose a random operation
    operation = random.choice(["add", "sub", "mul", "div", "sqrt", "exp"])

    if operation == "add":
        num_values = _log_weighted_int(2, 16)
        values = [random.randint(1, 100_000_000) for _ in range(num_values)]
        result = sum(values)

        if len(values) == 2:
            problem_templates = [
                f"What is {values[0]} + {values[1]}?",
                f"Calculate {values[0]} plus {values[1]}",
                f"Can you add {values[0]} and {values[1]} for me?",
                f"What's the sum of {values[0]} and {values[1]}?",
            ]
            result_phrase = f"The sum of {values[0]} and {values[1]} is {result}."
        else:
            values_str = " + ".join(map(str, values))
            problem_templates = [
                f"What is {values_str}?",
                f"Calculate the sum: {values_str}",
                f"Add these numbers: {', '.join(map(str, values))}",
            ]
            result_phrase = f"The sum of {', '.join(map(str, values))} is {result}."

    elif operation == "sub":
        num_values = _log_weighted_int(2, 12)
        values = [random.randint(1, 100_000_000) for _ in range(num_values)]
        result = values[0]
        for v in values[1:]:
            result -= v

        if len(values) == 2:
            problem_templates = [
                f"What is {values[0]} - {values[1]}?",
                f"Subtract {values[1]} from {values[0]}",
                f"What's {values[0]} minus {values[1]}?",
            ]
            result_phrase = f"{values[0]} minus {values[1]} equals {result}."
        else:
            values_str = " - ".join(map(str, values))
            problem_templates = [
                f"What is {values_str}?",
                f"Calculate: {values_str}",
            ]
            result_phrase = f"The result of {values_str} is {result}."

    elif operation == "mul":
        # Per-value range capped at 1000 because products blow up fast with
        # many factors; 1000^6 is already 18 digits.
        num_values = _log_weighted_int(2, 6)
        values = [random.randint(1, 1000) for _ in range(num_values)]
        result = 1
        for v in values:
            result *= v

        if len(values) == 2:
            problem_templates = [
                f"What is {values[0]} × {values[1]}?",
                f"Multiply {values[0]} by {values[1]}",
                f"What's {values[0]} times {values[1]}?",
            ]
            result_phrase = f"{values[0]} times {values[1]} equals {result}."
        else:
            values_str = " × ".join(map(str, values))
            problem_templates = [
                f"What is {values_str}?",
                f"Calculate the product: {', '.join(map(str, values))}",
            ]
            result_phrase = f"The product of {', '.join(map(str, values))} is {result}."

    elif operation == "div":
        # Generate 2 numbers for division
        b = random.randint(2, 1000)
        a = b * random.randint(1, 10000)
        values = [a, b]
        result = a / b

        problem_templates = [
            f"What is {a} ÷ {b}?",
            f"Divide {a} by {b}",
            f"What's {a} divided by {b}?",
        ]
        result_phrase = f"{a} divided by {b} equals {result}."

    elif operation == "sqrt":
        # Generate a perfect square for nice results
        base = random.randint(1, 1000)
        values = [base * base]
        result = base

        problem_templates = [
            f"What is the square root of {values[0]}?",
            f"Calculate √{values[0]}",
            f"Find the square root of {values[0]}",
        ]
        result_phrase = f"The square root of {values[0]} is {result}."

    else:  # exp
        # Generate base and exponent
        base = random.randint(2, 20)
        exp = random.randint(2, 5)
        values = [base, exp]
        result = math.pow(base, exp)

        problem_templates = [
            f"What is {base}^{exp}?",
            f"Calculate {base} to the power of {exp}",
            f"What's {base} raised to the {exp}th power?",
        ]
        result_phrase = f"{base} to the power of {exp} equals {result:.0f}."

    user_prompt = random.choice(problem_templates)

    # Build the conversation with unified system/developer prompts and tool usage
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": sample_developer_prompt("use_tools")},
        {"role": "user", "content": user_prompt},
    ]

    # Rare get_tools() probe - the schema dump is byte-identical across
    # samples, so a high frequency makes it the most-memorized chunk in
    # this corpus. 1% is enough to teach "you can introspect your tools"
    # without dominating the loss surface.
    if random.random() < 0.01:
        from praxis.tools import get_tools_json_schema

        tools_json = json.dumps(get_tools_json_schema(), indent=2)
        messages.append(
            {
                "role": "assistant",
                "content": format_tool_input(tool_name="get_tools", arguments={}),
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": format_tool_output(tools_json),
            }
        )

    # Calc tool call. Three messages: assistant emits the call, tool
    # emits the result, assistant emits the natural-language phrase.
    # Splitting the tool result into its own role keeps it outside the
    # assistant_mask region so the model isn't trained to predict
    # runtime-injected content.
    messages.append(
        {
            "role": "assistant",
            "content": format_tool_input(
                tool_name="calc",
                arguments={"values": values, "op": operation},
            ),
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": format_tool_output(str(float(result))),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": result_phrase,
        }
    )

    # Return messages and metadata
    return {
        "messages": messages,
        "metadata": {
            "format": "tool_calling",
            "operation": operation,
            "has_tool_call": True,
        },
    }
