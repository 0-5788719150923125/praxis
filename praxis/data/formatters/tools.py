"""Tool calling format generation."""

import json
import math
import random
from typing import Dict, List
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS


def format_tool_calling(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """
    Format synthetic tool-calling examples for training.
    Generates math problems that require the calc tool.
    
    Args:
        document: Dictionary containing the document data (unused for synthetic generation)
        keys: List of keys to extract from document (unused for synthetic generation)
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Formatted text with chat template applied including tool calls
    """

    # Choose a random operation
    operation = random.choice(["add", "sub", "mul", "div", "sqrt", "exp"])

    if operation == "add":
        # Generate 2-4 numbers for addition
        num_values = random.randint(2, 4)
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
        # Generate 2-3 numbers for subtraction
        num_values = random.randint(2, 3)
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
        # Generate 2-3 numbers for multiplication
        num_values = random.randint(2, 3)
        values = [random.randint(1, 10000) for _ in range(num_values)]
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
        {"role": "developer", "content": DEVELOPER_PROMPTS["use_tools"]},
        {"role": "user", "content": user_prompt},
    ]

    # 50% chance to call get_tools() first before using calc
    if random.random() < 0.5:
        from praxis.tools import get_tools_json_schema

        tools_json = json.dumps(get_tools_json_schema(), indent=2)

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"<tool_call>\n{json.dumps({'name': 'get_tools', 'arguments': {}}, indent=2)}\n</tool_call>",
                },
                {"role": "tool", "content": tools_json},
            ]
        )

    # Always call calc tool
    messages.extend(
        [
            {
                "role": "assistant",
                "content": f"<tool_call>\n{json.dumps({'name': 'calc', 'arguments': {'values': values, 'op': operation}}, indent=2)}\n</tool_call>",
            },
            {"role": "tool", "content": str(float(result))},
            {"role": "assistant", "content": result_phrase},
        ]
    )

    # Apply chat template without tools parameter
    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"