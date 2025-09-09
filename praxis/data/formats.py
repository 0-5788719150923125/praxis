"""Data format definitions and detection logic."""

from enum import Enum


class DataFormat(Enum):
    """Enumeration of supported data formats for different dataset types."""
    
    SIMPLE = "simple"                # Plain text data
    INSTRUCTION = "instruction"        # Instruction-following format
    CONVERSATION = "conversation"      # Multi-turn conversation
    PERSONACHAT = "persona_chat"       # Persona-based chat
    CUSTOM = "custom"                  # Custom format (user-defined)
    MESSAGES = "messages"              # Chat messages format
    SODA = "soda"                      # SODA dialogue format
    WIKI = "wiki"                      # Wikipedia article format
    RL = "rl"                          # Reinforcement learning format
    COT = "cot"                        # Chain-of-thought format
    TOOL_CALLING = "tool_calling"      # Tool calling format


def detect_format(dataset_name: str, dataset_config: dict) -> DataFormat:
    """Detect the appropriate format for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration dictionary for the dataset
        
    Returns:
        The detected DataFormat
    """
    # If format is explicitly specified, use it
    if 'format' in dataset_config:
        return dataset_config['format']
    
    # Otherwise, try to detect based on keys or name
    keys = dataset_config.get('keys', [])
    
    if 'messages' in keys or 'conversations' in keys:
        return DataFormat.MESSAGES
    elif 'instruction' in keys or 'prompt' in keys and 'completion' in keys:
        return DataFormat.INSTRUCTION
    elif 'personality' in keys or 'personas' in dataset_name:
        return DataFormat.PERSONACHAT
    elif 'dialogue' in keys:
        return DataFormat.SODA
    elif 'title' in keys and 'text' in keys:
        return DataFormat.WIKI
    else:
        return DataFormat.SIMPLE