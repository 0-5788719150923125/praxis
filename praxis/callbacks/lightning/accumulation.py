"""Gradient accumulation scheduling callback for Praxis training."""

from lightning.pytorch.callbacks import GradientAccumulationScheduler


class AccumulationSchedule(GradientAccumulationScheduler):
    """
    Change gradient accumulation factor according to scheduling.
    """

    def __init__(self, batch_size=1, target_batch_size=1):
        # NOTE: must be 1 for Hivemind training; will need adapting once we get there
        self.factor = self._fit_grad_accumulation(batch_size, target_batch_size)
        self.schedule = {1: self.factor}
        super().__init__(self.schedule)

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)
        # TODO: implement cosine oscillation of accumulation size
        trainer.accumulate_grad_batches = self.factor

    def _fit_grad_accumulation(self, batch_size, target_batch_size):
        return (
            1
            if batch_size >= target_batch_size
            else -(-target_batch_size // batch_size)
        )