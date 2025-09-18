"""Base dataset classes."""

from typing import Any, Dict, List

from transformers import PreTrainedTokenizer


class PraxisSampler:
    """Base class for all dataset samplers."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.weight = 1.0
        self.tokenizer = tokenizer
        self.sequence_cache = []  # Store raw text sequences

    def fill_sequence_cache(self):
        """Each dataset implementation should override this"""
        raise NotImplementedError

    def get_sequences(self, count: int = 1) -> List[str]:
        """Get raw sequences from this dataset"""
        while len(self.sequence_cache) < count:
            self.fill_sequence_cache()
        return [self.sequence_cache.pop(0) for _ in range(count)]


def load_dataset_smart(dataset_args: Dict) -> Any:
    """
    Load a dataset, handling cases where metadata files interfere.

    This wrapper handles two issues:
    1. Datasets with .METADATA files that break split detection
    2. Datasets like NextCoderDataset-Conversational where state.json gets loaded instead of data

    Args:
        dataset_args: Arguments to pass to load_dataset

    Returns:
        Loaded dataset object
    """
    from datasets import load_dataset

    dataset_path = dataset_args.get("path", "")

    # For NextCoderDataset-Conversational, ALWAYS use the arrow file fix
    # because the standard loading is completely broken
    if "NextCoderDataset-Conversational" in dataset_path:
        print(f"Note: Using arrow file workaround for {dataset_path}")
        fixed_args = dataset_args.copy()

        # Replace path with arrow loader and direct file reference
        fixed_args["path"] = "arrow"
        fixed_args["data_files"] = {
            "train": "hf://datasets/microsoft/NextCoderDataset-Conversational/data-00000-of-00001.arrow"
        }

        return load_dataset(**fixed_args)

    # For all other datasets, try normal loading
    try:
        return load_dataset(**dataset_args)
    except Exception as e:
        if ".METADATA" in str(e) or "splits" in str(e):
            # Try with explicitly specified split
            if "split" not in dataset_args:
                # If no split specified, try train
                dataset_args = dataset_args.copy()
                dataset_args["split"] = "train"
                return load_dataset(**dataset_args)
            else:
                # If split already specified, re-raise original error
                raise
        else:
            raise
