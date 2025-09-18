"""Time-based checkpoint callback for Praxis training."""

import os
import time

from lightning.pytorch.callbacks import ModelCheckpoint


class TimeBasedCheckpoint(ModelCheckpoint):
    """
    Replaces the Pytorch Lightning checkpoint behavior with one that saves on
    a time-based interval (in seconds).
    """

    def __init__(self, save_interval: int, *args, **kwargs):
        # Disable other checkpointing triggers
        kwargs["every_n_train_steps"] = 0
        kwargs["every_n_epochs"] = 0

        super().__init__(*args, **kwargs)
        self.save_interval = save_interval
        self.last_checkpoint_time = time.monotonic()

    def on_train_batch_end(
        self,
        trainer,
        lm,
        outputs,
        batch,
        batch_idx,
    ):
        # Get current time
        current_time = time.monotonic()

        # Check if save_interval has elapsed
        if current_time - self.last_checkpoint_time >= self.save_interval:

            # Get current metrics
            monitor_candidates = self._monitor_candidates(trainer)

            # Save checkpoint
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

            # Update last checkpoint time
            self.last_checkpoint_time = current_time

            # Also save the model in Huggingface format
            # The lm is already the BackpropagationTrainer, which has the model as an attribute
            if hasattr(lm, "model") and hasattr(lm.model, "save_pretrained"):
                # Create a subdirectory for the HF model
                import os

                model_dir = os.path.join(self.dirpath, "hf_model")
                os.makedirs(model_dir, exist_ok=True)
                lm.model.save_pretrained(model_dir, safe_serialization=False)

    def on_train_epoch_end(self, trainer, pl_module):
        # Disable saving checkpoints at the end of every epoch
        pass

    def on_validation_end(self, trainer, pl_module):
        # Disable saving checkpoints at the end of every epoch
        pass
