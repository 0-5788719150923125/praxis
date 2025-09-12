"""Chain of Thought formatting."""

from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS
from praxis.data.formatters.base import text_formatter


# Chain of Thought tag definitions
COT_TAGS = {
    "wrapper": {
        "thinking": ("<thinking>", "</thinking>"),
        "output": ("<output>", "</output>"),
    },
    "thinking_components": {
        "initial_analysis": ("<initial_analysis>", "</initial_analysis>"),
        "conscious_thought": ("<conscious_thought>", "</conscious_thought>"),
        "step_by_step": ("<step_by_step>", "</step_by_step>"),
        "reflection": ("<reflection>", "</reflection>"),
        "feeling": ("<feeling>", "</feeling>"),
        "self_improvement": ("<self_improvement>", "</self_improvement>"),
        "subcomponent_analysis": (
            "<subcomponent_analysis>",
            "</subcomponent_analysis>",
        ),
    },
    # Tag weights for training
    "tag_weights": {
        "thinking": 1.5,
        "output": 1.0,
        "initial_analysis": 1.3,
        "conscious_thought": 1.3,
        "step_by_step": 1.6,  # Highest weight for step-by-step reasoning
        "reflection": 1.3,
        "feeling": 1.1,
        "self_improvement": 1.2,
        "subcomponent_analysis": 1.4,
    },
    # Reward values for REINFORCE
    "tag_rewards": {
        "thinking": 0.3,
        "output": 0.1,
        "initial_analysis": 0.2,
        "conscious_thought": 0.2,
        "step_by_step": 0.3,
        "reflection": 0.2,
        "feeling": 0.1,
        "self_improvement": 0.1,
        "subcomponent_analysis": 0.2,
    },
}


def format_cot(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Format Chain of Thought dataset for training using chat templates.

    Uses the tokenizer's chat template to properly format the conversation
    with user prompt and assistant response (containing thinking tags).

    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document (at least 2: prompt, response)
        tokenizer: Tokenizer with chat template support

    Returns:
        Dict with text and metadata for token-level reward computation
    """
    assert len(keys) >= 2, "CoT format requires at least 2 keys (prompt, response)"

    prompt = text_formatter(document.get(keys[0], ""))
    response = text_formatter(document.get(keys[1], ""))
    category = document.get(keys[2], "unknown") if len(keys) > 2 else "unknown"
    topic = document.get(keys[3], "unknown") if len(keys) > 3 else "unknown"

    # Use chat template with unified system/developer prompts
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["think_step_by_step"]},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Detect which CoT tags are present in the response
    tags_present = []
    for tag_type in ["wrapper", "thinking_components"]:
        for tag_name, (open_tag, close_tag) in COT_TAGS[tag_type].items():
            if open_tag in response and close_tag in response:
                tags_present.append(tag_name)

    # Return dict format similar to format_rl for metadata pipeline
    return {
        "text": formatted_text,
        "reward": 0.0,  # Sequence-level reward (not used for CoT)
        "cot_metadata": {
            "tags_present": tags_present,
            "category": category,
            "topic": topic,
            "has_cot": len(tags_present) > 0,
        },
    }
