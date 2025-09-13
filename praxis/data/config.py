"""Configuration and constants for Praxis data loading."""

from praxis.data.formats import DataFormat

# Dataset weight constants
DEFAULT_WEIGHT = 1.0
SRC_WEIGHT = 1.0
DIR_WEIGHT = 1.0
TOOLS_WEIGHT = 1.0

# Dataset collections with weights
DATASET_COLLECTIONS = dict(
    base={
        "fineweb-edu-350bt": DEFAULT_WEIGHT,
    },
    phi={
        "fineweb": 0.75,
        "finepdfs": 0.01,
        "textbooks": 0.002,
        "tinystories": 0.01,
        "wikipedia": 0.02,
        "persona-chat": 0.1,
        "soda": 0.1,
        "wildchat": 0.1,
        "natural-instructions": 0.5,
        "cosmopedia-v2": 0.2,
        "smoltalk": 0.1,
        "nextcoder": 0.05,
        "nextcoder-conversational": 0.1,
        "hermes-3-dataset": 0.1,
    },
    pile={
        "minipile-train": DEFAULT_WEIGHT,
    },
    validation={
        "refinedweb": DEFAULT_WEIGHT,
    },
    dev={
        "textbooks": DEFAULT_WEIGHT,
    },
    slimpajama={
        "slimpajama": DEFAULT_WEIGHT,
    },
    rl={
        "intellect-rl": DEFAULT_WEIGHT,
    },
    cot={
        "chain-of-thought": DEFAULT_WEIGHT * 0.1,
    },
)

# System and developer prompts
SYSTEM_PROMPT = "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation."

DEVELOPER_PROMPTS = {
    "continue_text": "Continue or complete the provided text, maintaining style and coherence.",
    "follow_instruction": "Follow the user's instructions precisely and provide a complete response.",
    "engage_conversation": "Engage naturally in this conversation, being helpful and appropriate.",
    "answer_question": "Answer the question accurately based on your knowledge.",
    "think_step_by_step": "Think step-by-step through this problem before providing your answer.",
    "use_tools": "Use the available tools when appropriate to help the user.",
    "write_article": "Write a comprehensive article or explanation on the given topic.",
    "persona_chat": "Engage in conversation while maintaining the specified personas.",
    "soda_dialogue": "Continue this dialogue naturally based on the context provided.",
}


# HuggingFace dataset configurations
HUGGINGFACE_DATASETS = {
    "minipile-train": dict(
        path="JeanKaddour/minipile",
        split="train",
        keys=["text"],
        format=DataFormat.SIMPLE,
        streaming=False,
    ),
    "minipile-validation": dict(
        path="JeanKaddour/minipile",
        split="validation",
        keys=["text"],
        format=DataFormat.SIMPLE,
        streaming=False,
    ),
    "textbooks": dict(
        path="open-phi/textbooks",
        keys=["markdown"],
        format=DataFormat.SIMPLE,
    ),
    "cosmopedia-v2": dict(
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        keys=["prompt", "text"],
        format=DataFormat.INSTRUCTION,
    ),
    "natural-instructions": dict(
        path="Muennighoff/natural-instructions",
        name="default",
        keys=["definition", "inputs", "targets"],
        format=DataFormat.CONVERSATION,
    ),
    "persona-chat": dict(
        path="AlekseyKorshuk/persona-chat",
        keys=["personality", "utterances"],
        format=DataFormat.PERSONACHAT,
    ),
    "smoltalk": dict(
        path="HuggingFaceTB/smoltalk",
        name="all",
        keys=["messages"],
        format=DataFormat.MESSAGES,
    ),
    "nextcoder": dict(
        path="microsoft/NextCoderDataset",
        split="train",
        keys=["prompt", "completion"],
        format=DataFormat.INSTRUCTION,
    ),
    "nextcoder-conversational": dict(
        path="microsoft/NextCoderDataset-Conversational",
        keys=["messages"],
        format=DataFormat.MESSAGES,
    ),
    "soda": dict(
        path="allenai/soda",
        keys=[
            "speakers",
            "narrative",
            "literal",
            "dialogue",
            "head",
            "relation",
            "tail",
        ],
        format=DataFormat.SODA,
    ),
    "hermes-3-dataset": dict(
        path="NousResearch/Hermes-3-Dataset",
        keys=["conversations"],
        format=DataFormat.MESSAGES,
    ),
    "wildchat": dict(
        path="allenai/WildChat",
        keys=["conversation"],
        format=DataFormat.MESSAGES,
    ),
    "github-code": dict(
        path="codeparrot/github-code",
        name="all-all",
        keys=["code"],
        format=DataFormat.SIMPLE,
        trust_remote_code=True,
    ),
    "tinystories": dict(
        path="roneneldan/TinyStories",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "legal": dict(
        path="pile-of-law/pile-of-law",
        name="all",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "wikipedia": dict(
        path="wikimedia/wikipedia",
        name="20231101.en",
        keys=["title", "text"],
        format=DataFormat.WIKI,
    ),
    "c4": dict(
        path="allenai/c4",
        name="en",
        split="validation",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "refinedweb": dict(
        path="tiiuae/falcon-refinedweb", keys=["content"], format=DataFormat.SIMPLE
    ),
    "fineweb-edu-10bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb-edu-100bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb-edu-350bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-350BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb": dict(
        path="HuggingFaceFW/fineweb",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "finepdfs": dict(
        path="HuggingFaceFW/finepdfs",
        name="eng_Latn",
        split="train",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "intellect-rl": dict(
        path="PrimeIntellect/INTELLECT-2-RL-Dataset",
        split="train",
        keys=["prompt", "verification_info", "solve_rate_qwen_r1_distill_7b"],
        format=DataFormat.RL,
        streaming=True,
        mix_simple_math=True,  # Mix in simple problems for untrained models
        trust_remote_code=False,
    ),
    "chain-of-thought": dict(
        path="isaiahbjork/chain-of-thought",
        split="train",
        keys=["prompt", "response", "category", "topic"],
        format=DataFormat.COT,
        streaming=True,
        trust_remote_code=False,
    ),
}
