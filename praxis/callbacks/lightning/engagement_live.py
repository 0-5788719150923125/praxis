"""Drains live engagement rewards from the web `Print` UI into the training loop.

The web thread submits sparse, asynchronous rewards (a real user answering a
model-led question) to the process-global ``LIVE_ENGAGEMENT`` channel. This
callback - on the training-loop cadence, not the UI's - drains that buffer and
folds each interaction's activation into the EngagementPolicy's homeostatic
energy (its REINFORCE baseline). That is the online-learning seam: the live
signal shifts the operating point of subsequent dense updates as a slow,
integrated return, the way the harmonic-weight callback integrates a delayed EMA.

Metrics (engagement_live_*) are written to ``trainer.callback_metrics`` so
MetricsLogger drains them; order this before MetricsLogger.
"""

import torch
from lightning.pytorch.callbacks import Callback

from praxis.policies.engagement_channel import LIVE_ENGAGEMENT


class EngagementLiveRewardCallback(Callback):
    def __init__(self, period: int = 10):
        super().__init__()
        self.period = int(period)
        self._step = 0
        self._count = 0  # cumulative live interactions consumed
        self._metrics: dict = {}
        self._policy = None

    def _find_policy(self, pl_module):
        if self._policy is not None:
            return self._policy
        model = getattr(pl_module, "model", pl_module)
        model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        policy = getattr(model, "policy", None)
        if policy is not None and policy.__class__.__name__ == "EngagementPolicy":
            self._policy = policy
        return self._policy

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._step += 1
        if self._step % self.period == 0:
            events = LIVE_ENGAGEMENT.drain()
            if events:
                policy = self._find_policy(pl_module)
                for ev in events:
                    if policy is not None:
                        policy.ingest_live(ev["activation"])
                self._count += len(events)
                last = events[-1]
                self._metrics = {
                    "engagement_live_reward": float(last["reward"]),
                    "engagement_live_count": float(self._count),
                    "engagement_live_energy": float(
                        LIVE_ENGAGEMENT.snapshot()["energy"]
                    ),
                }

        # Carry the latest live scalars forward each step (interactions are
        # sparse) so the series stays continuous for MetricsLogger.
        for k, v in self._metrics.items():
            trainer.callback_metrics[k] = torch.tensor(float(v))
