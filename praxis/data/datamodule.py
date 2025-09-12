"""PyTorch Lightning DataModule for Praxis."""

import multiprocessing as mp
from typing import List, Dict, Optional, Any
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from praxis.data.datasets import WeightedIterableDataset
from praxis.data.interruptible import InterruptibleDataLoader, DataLoaderManager


class PraxisDataModule(LightningDataModule):
    """Lightning DataModule for managing train and validation datasets."""

    def __init__(
        self,
        train_datasets: List[Dict],
        val_datasets: List[Dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        block_size: int = 512,
        oversample_chance: float = 0,
        supersample_chance: float = 0,
        hypersample_chance: float = 0,
        rl_type: Optional[str] = None,
    ):
        super().__init__()
        self.rl_type = rl_type
        self.dataloader_manager = DataLoaderManager()  # Track dataloaders for shutdown
        self.train_datasets = self.create_datasets(
            train_datasets,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
            hypersample_chance,
        )
        self.val_datasets = False
        if len(val_datasets) > 0:
            self.val_datasets = self.create_datasets(
                val_datasets, tokenizer, block_size, batch_size, 0, 0, 0
            )

    def create_datasets(
        self,
        datasets,
        tokenizer,
        block_size,
        batch_size,
        oversample_chance=0,
        supersample_chance=0,
        hypersample_chance=0,
    ):
        # Get weights and normalize them while preserving relative magnitudes
        raw_weights = [dataset.weight for dataset in datasets]
        weights = [w / sum(raw_weights) for w in raw_weights]

        # Debug log

        return WeightedIterableDataset(
            datasets,
            weights,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
            hypersample_chance,
            rl_type=self.rl_type,
        )

    def train_dataloader(self):
        # Use 0 workers if spawn method is set (required for MonoForward pipeline)
        num_workers = 0 if mp.get_start_method(allow_none=True) == "spawn" else 1

        dataloader = InterruptibleDataLoader(
            dataset=self.train_datasets,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,  # Ensure clean shutdown
        )

        # Register for shutdown management
        self.dataloader_manager.register(dataloader)
        return dataloader

    def val_dataloader(self):
        if self.val_datasets:
            # Use 0 workers if spawn method is set (required for MonoForward pipeline)
            num_workers = 0 if mp.get_start_method(allow_none=True) == "spawn" else 1

            dataloader = InterruptibleDataLoader(
                dataset=self.val_datasets,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=False,  # Ensure clean shutdown
            )

            # Register for shutdown management
            self.dataloader_manager.register(dataloader)
            return dataloader
        else:
            return []

    def shutdown_dataloaders(self, timeout: float = 5.0):
        """Shutdown all active dataloaders gracefully."""
        self.dataloader_manager.shutdown_all(timeout=timeout)

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict for checkpointing."""
        sampler_states = []
        for sampler in self.train_datasets.data_manager.samplers:
            if hasattr(sampler, "state_dict") and callable(sampler.state_dict):
                sampler_states.append(sampler.state_dict())
            else:
                # If sampler doesn't support state, append None
                sampler_states.append(None)

        return {"sampler_states": sampler_states}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dict from a checkpoint."""
        sampler_states = state_dict.get("sampler_states", [])
        samplers = self.train_datasets.data_manager.samplers
        for sampler, s_state in zip(samplers, sampler_states):
            if (
                s_state is not None
                and hasattr(sampler, "load_state_dict")
                and callable(sampler.load_state_dict)
            ):
                sampler.load_state_dict(s_state)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Lightning hook to save DataModule state to checkpoint."""
        checkpoint["datamodule_state"] = self.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Lightning hook to load DataModule state from checkpoint."""
        if "datamodule_state" in checkpoint:
            self.load_state_dict(checkpoint["datamodule_state"])
