"""Drains live web rewards (Print answers, joke approvals) into the training loop.

The web thread submits sparse, asynchronous rewards to a process-global channel
(``LIVE_ENGAGEMENT`` for Print, ``LIVE_JOKES`` for jokes). This callback - on the
training-loop cadence, not the UI's - drains that buffer and folds each
interaction's activation into the matching forward-path policy's homeostatic
energy (its REINFORCE baseline). That is the online-learning seam: the live
signal shifts the operating point of subsequent dense updates as a slow,
integrated return, the way the harmonic-weight callback integrates a delayed EMA.

One instance per (channel, policy, prefix). Metrics (<prefix>_live_*) are written
to ``trainer.callback_metrics`` so MetricsLogger drains them; order before it.
"""

import torch
from lightning.pytorch.callbacks import Callback

from praxis.policies.engagement_channel import LIVE_ENGAGEMENT


class EngagementLiveRewardCallback(Callback):
    def __init__(
        self,
        period: int = 10,
        channel=LIVE_ENGAGEMENT,
        policy_class_name: str = "EngagementPolicy",
        metric_prefix: str = "engagement",
    ):
        super().__init__()
        self.period = int(period)
        self.channel = channel
        self.policy_class_name = policy_class_name
        self.metric_prefix = metric_prefix
        self._step = 0
        self._count = 0  # cumulative live interactions consumed
        self._metrics: dict = {}
        self._policy = None

    def _find_policy(self, pl_module):
        if self._policy is not None:
            return self._policy
        model = getattr(pl_module, "model", pl_module)
        model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        # Recall-style policies live in a ModuleDict; the legacy single forward
        # policy is model.policy. Match by class name across both.
        candidates = list(getattr(model, "recall_policies", {}).values())
        legacy = getattr(model, "policy", None)
        if legacy is not None:
            candidates.append(legacy)
        for policy in candidates:
            if policy.__class__.__name__ == self.policy_class_name:
                self._policy = policy
                break
        return self._policy

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._step += 1
        if self._step % self.period == 0:
            events = self.channel.drain()
            if events:
                policy = self._find_policy(pl_module)
                for ev in events:
                    if policy is not None:
                        policy.ingest_live(ev["activation"])
                self._count += len(events)
                last = events[-1]
                p = self.metric_prefix
                self._metrics = {
                    f"{p}_live_reward": float(last["reward"]),
                    f"{p}_live_count": float(self._count),
                    f"{p}_live_energy": float(self.channel.snapshot()["energy"]),
                }

        # Carry the latest live scalars forward each step (interactions are
        # sparse) so the series stays continuous for MetricsLogger.
        for k, v in self._metrics.items():
            trainer.callback_metrics[k] = torch.tensor(float(v))
