from datasets import load_dataset

# Load the dataset in streaming mode
dataset = load_dataset("open-phi/textbooks", streaming=True)

# Access the 'train' split (or another split if available)
train_dataset = dataset["train"]

# Create an iterator for the dataset
dataset_iterator = iter(train_dataset)

# Print a few entries
print("Printing sample entries from the dataset:")
for i in range(3):  # Print 3 samples
    sample = next(dataset_iterator)
    print(f"\n--- Sample {i+1} ---")
    print(f"Text excerpt (first 500 chars): {sample['markdown'][:500]}...")

    # Print other keys if available
    for key in sample:
        if key != "markdown":
            print(f"{key}: {sample[key]}")
