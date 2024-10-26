"""SPC: Synthetic-Persona-Chat using Huggingface Datasets."""

import json
import os

import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Initialize logger
logger = datasets.logging.get_logger(__name__)

# Step 1: Load the Dataset
print("Loading the Synthetic-Persona-Chat dataset...")
try:
    dataset = load_dataset("google/Synthetic-Persona-Chat")
    print("Dataset loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    exit(1)


# Step 2: Define Updated Formatting Function
def format_conversation(example):
    """
    Formats a single dataset example into a structured text block with clear separators.

    Args:
        example (dict): A single example from the dataset.

    Returns:
        str or None: Formatted text block or None if essential fields are missing.
    """
    # Extract and split personas
    user1_personas = example.get("user 1 personas", "")
    user1_personas = user1_personas.split("\n") if user1_personas else []

    user2_personas = example.get("user 2 personas", "")
    user2_personas = user2_personas.split("\n") if user2_personas else []

    # Extract and split conversation
    raw_conversation = example.get("Best Generated Conversation", "")
    if not raw_conversation:
        logger.warning("Skipping example with empty 'Best Generated Conversation'")
        return None  # Skip this example

    raw_conversation = raw_conversation.split("\n")
    utterances = []

    for u in raw_conversation:
        if ": " in u:
            speaker_label, utterance = u.split(": ", 1)
            speaker_label = speaker_label.strip().upper()
            if speaker_label in ["A", "USER 1", "USER1"]:
                speaker = "User1"
            elif speaker_label in ["B", "USER 2", "USER2"]:
                speaker = "User2"
            else:
                # Unknown speaker label, assign alternately
                speaker = "User1" if len(utterances) % 2 == 0 else "User2"
                # logger.warning(
                #     f"Unknown speaker label '{speaker_label}' in conversation. Assigning to '{speaker}'."
                # )
            utterances.append(f"{speaker}: {utterance.strip()}")
        else:
            # If no prefix, assign alternately
            speaker = "User1" if len(utterances) % 2 == 0 else "User2"
            utterances.append(f"{speaker}: {u.strip()}")

    # Check if conversation has both speakers
    speakers = set()
    for utterance in utterances:
        if ": " in utterance:
            speakers.add(utterance.split(": ", 1)[0])
    if len(speakers) < 2:
        logger.warning("Conversation has less than 2 speakers.")

    # Format personas section
    personas_section = "<BEGIN_PERSONAS>\n"
    personas_section += "User1:\n"
    for persona in user1_personas:
        personas_section += f"- {persona.strip()}\n"
    personas_section += "\nUser2:\n"
    for persona in user2_personas:
        personas_section += f"- {persona.strip()}\n"
    personas_section += "<END_PERSONAS>\n\n"

    # Format conversation section
    conversation_section = "<BEGIN_CONVERSATION>\n"
    for utterance in utterances:
        conversation_section += f"{utterance}\n"
    conversation_section += "<END_CONVERSATION>\n"

    # Combine both sections
    formatted_text = personas_section + conversation_section
    return formatted_text


# Step 3: Format and Save the Dataset
def format_dataset_split(split, split_name):
    """
    Formats an entire dataset split by applying the format_conversation function to each example.

    Args:
        split (Dataset): A Huggingface Dataset split (train, validation, test, synthetic).
        split_name (str): Name of the split.

    Returns:
        list: A list of formatted text blocks.
    """
    formatted_examples = []
    skipped_examples = 0
    total_examples = len(split)
    for idx, example in enumerate(split):
        formatted_text = format_conversation(example)
        if formatted_text:
            formatted_examples.append({"text": formatted_text})
        else:
            skipped_examples += 1
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{total_examples} examples...")
    if skipped_examples > 0:
        logger.info(
            f" - Skipped {skipped_examples} examples in '{split_name}' split due to missing conversations."
        )
    return formatted_examples


formatted_dataset = {}
for split_name in dataset.keys():
    print(f"Formatting the '{split_name}' split...")
    formatted_dataset[split_name] = format_dataset_split(
        dataset[split_name], split_name
    )
    print(f" - Number of formatted examples: {len(formatted_dataset[split_name])}")


# Step 4: Save the formatted data
def save_to_jsonl(formatted_examples, filename):
    """
    Saves a list of formatted examples to a JSONL file.

    Args:
        formatted_examples (list): List of dictionaries with formatted text.
        filename (str): Path to the output JSONL file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for example in formatted_examples:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")


output_dir = "./formatted_synthetic_persona_chat"
os.makedirs(output_dir, exist_ok=True)

for split_name, examples in formatted_dataset.items():
    filename = os.path.join(output_dir, f"{split_name}.jsonl")
    print(f"Saving '{split_name}' split to {filename}...")
    try:
        save_to_jsonl(examples, filename)
        print(f" - Saved {len(examples)} examples.")
    except Exception as e:
        logger.error(f"Failed to save '{split_name}' split: {e}")


# Step 5: Display a Few Formatted Examples
def load_jsonl(filename, num_lines=2):
    """
    Loads and prints a specified number of lines from a JSONL file.

    Args:
        filename (str): Path to the JSONL file.
        num_lines (int): Number of lines to print.
    """
    if not os.path.exists(filename):
        logger.error(f"File '{filename}' does not exist.")
        return
    with open(filename, "r", encoding="utf-8") as f:
        for i in range(num_lines):
            line = f.readline()
            if not line:
                break
            example = json.loads(line)
            # Print the 'text' field directly to interpret newlines
            print(example["text"])


print("\nDisplaying sample formatted examples:")

for split_name in ["train", "validation", "test", "synthetic"]:
    filename = os.path.join(output_dir, f"{split_name}.jsonl")
    print(f"\n--- Sample from '{split_name}' split ---")
    load_jsonl(filename, num_lines=1)
