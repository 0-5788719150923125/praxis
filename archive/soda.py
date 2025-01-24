from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


def format_chatml(example: Dict) -> str:
    """Formats a single example into ChatML format."""
    # Get speaker roles
    unique_speakers = list(
        dict.fromkeys(example["speakers"])
    )  # preserve order, remove duplicates
    speaker_roles = {}

    # Always map first two speakers to user/assistant
    if len(unique_speakers) >= 1:
        speaker_roles[unique_speakers[0]] = "user"
    if len(unique_speakers) >= 2:
        speaker_roles[unique_speakers[1]] = "assistant"

    # Map any additional speakers to "other"
    for speaker in unique_speakers[2:]:
        speaker_roles[speaker] = "other"

    # Start with system message
    chatml = "<|im_start|>system\n"

    # Add role mappings to system context
    chatml += ""
    for speaker, role in speaker_roles.items():
        chatml += f"{role}: {speaker}\n"

    # Add context from literal and narrative
    chatml += f"{example['narrative']}\n"
    chatml += "<|im_end|>\n"

    # Add conversation turns
    for speaker, message in zip(example["speakers"], example["dialogue"]):
        role = speaker_roles[speaker]
        chatml += f"<|im_start|>{role}\n"
        chatml += f"{message}\n"
        chatml += "<|im_end|>\n"

    return chatml


def write_examples_to_file(
    dataset, num_examples: int = 10, output_path: str = "examples.chatml"
):
    """Write multiple examples to a file in ChatML format."""
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_file.open("w", encoding="utf-8") as f:
            # Process specified number of examples
            for i, example in enumerate(dataset["train"]):
                if i >= num_examples:
                    break

                # Add example separator with index
                separator = f"\n{'='*50}\nExample {i+1}\n{'='*50}\n"
                f.write(separator)

                # Format and write example
                chatml_output = format_chatml(example)
                f.write(chatml_output)

                # Print progress
                print(f"Processed example {i+1}/{num_examples}")

        print(f"\nSuccessfully wrote {num_examples} examples to {output_path}")

    except Exception as e:
        print(f"Error writing to file: {str(e)}")
        raise


def main():
    try:
        # Load the dataset in streaming mode
        print("Loading dataset...")
        dataset = load_dataset(
            "allenai/soda", streaming=True, cache_dir="data/datasets"
        )

        # Write multiple examples to file
        output_path = "data/examples/soda_examples.chatml"
        write_examples_to_file(dataset, num_examples=10, output_path=output_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
