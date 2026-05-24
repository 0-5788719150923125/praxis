"""Lightning callback that computes BrierLM periodically during validation.

BrierLM (arXiv 2510.27688, section 3.5) is a sample-based proper
scoring rule. Computing it requires drawing paired model samples,
which is more expensive than a log-likelihood, so the callback runs
every Nth validation call instead of every one.

The callback is safe to attach to non-CALM runs; it simply returns a
sample-based proxy against the same model, which is useful as a
discrete-LM sanity signal too (the paper's -0.966 Pearson correlation
between BrierLM and CE loss is the motivation).
"""

from typing import List, Optional, Sequence

import torch
from lightning.pytorch.callbacks import Callback
from transformers import GenerationConfig

from praxis.metrics.brier import compute_brier_lm


class BrierLMCallback(Callback):
    """Compute BrierLM over a small val batch and log ``val_brierlm``.

    Args:
        tokenizer: Tokenizer for the model under training.
        eval_every: Run every Nth validation call (default 1).
        num_prompts: How many prompts to sample from the val dataloader.
        prompt_len: Tokens per prompt.
        continuation_len: Tokens to sample for each continuation.
        temperature: Sampling temperature (passed through to generate).
    """

    def __init__(
        self,
        tokenizer,
        eval_every: int = 1,
        num_prompts: int = 8,
        prompt_len: int = 32,
        continuation_len: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_every = max(1, eval_every)
        self.num_prompts = num_prompts
        self.prompt_len = prompt_len
        self.continuation_len = continuation_len
        self.temperature = temperature
        self._counter = 0

    def on_validation_end(self, trainer, pl_module) -> None:
        # Write directly to trainer.callback_metrics instead of calling
        # pl_module.log(): Lightning forbids log() in on_validation_end,
        # and MetricsLoggerCallback drains callback_metrics in its own
        # on_validation_end. Must be registered before MetricsLoggerCallback
        # so val_brierlm lands in the dict before it is iterated.
        self._counter += 1
        if self._counter % self.eval_every != 0:
            return

        dataloader = self._get_val_dataloader(trainer)
        if dataloader is None:
            return

        samples_a, samples_b, refs = self._build_sample_pairs(dataloader, pl_module)
        if not samples_a:
            return

        score = compute_brier_lm(samples_a, samples_b, refs)
        trainer.callback_metrics["val_brierlm"] = torch.tensor(float(score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_val_dataloader(self, trainer):
        val_dls = getattr(trainer, "val_dataloaders", None)
        if val_dls is None:
            return None
        if isinstance(val_dls, list):
            return val_dls[0] if val_dls else None
        return val_dls

    @torch.no_grad()
    def _build_sample_pairs(self, dataloader, pl_module):
        model = getattr(pl_module, "model", pl_module)
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        samples_a: List[List[int]] = []
        samples_b: List[List[int]] = []
        references: List[List[int]] = []

        prompts_seen = 0
        for batch in dataloader:
            if prompts_seen >= self.num_prompts:
                break
            input_ids = self._extract_input_ids(batch)
            if input_ids is None:
                continue
            B = input_ids.size(0)
            take = min(B, self.num_prompts - prompts_seen)
            input_ids = input_ids[:take].to(device)

            if input_ids.size(1) < self.prompt_len + self.continuation_len:
                continue

            prompts = input_ids[:, : self.prompt_len]
            refs = input_ids[
                :, self.prompt_len : self.prompt_len + self.continuation_len
            ]

            gen_config = GenerationConfig(
                max_new_tokens=self.continuation_len,
                temperature=self.temperature,
                do_sample=True,
            )

            try:
                a = model.generate(prompts, generation_config=gen_config)
                b = model.generate(prompts, generation_config=gen_config)
            except Exception as e:
                print(f"[BrierLMCallback] generate failed: {e}")
                break

            a = a[:, self.prompt_len : self.prompt_len + self.continuation_len]
            b = b[:, self.prompt_len : self.prompt_len + self.continuation_len]

            for i in range(take):
                samples_a.append(a[i].tolist())
                samples_b.append(b[i].tolist())
                references.append(refs[i].tolist())
            prompts_seen += take

        if was_training:
            model.train()
        return samples_a, samples_b, references

    @staticmethod
    def _extract_input_ids(batch) -> Optional[torch.Tensor]:
        if isinstance(batch, dict):
            return batch.get("input_ids")
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if torch.is_tensor(first):
                return first
        if torch.is_tensor(batch):
            return batch
        return None
