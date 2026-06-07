"""Forward-path REINFORCE on the engagement-prediction reward (PLAN.md P3).

Dense (simulated-user) path: over the assistant region of each example, the
model's predicted answer tokens A_hat are scored against the ground-truth answer
tokens R (the labels) by recall - "did the model anticipate its own answer". That
recall plus the homeostatic energy is the reward (against a slow reward-EMA
baseline); the policy gradient reweights the LM's own log-probs over the answer
tokens. No extra
parameters and no RL dataset - the reward is computed from labels, not carried in.

This is the teacher-forced dense proxy used to validate that the signal moves a
small model before the live, generation-time channel (P4/P5) exists.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.policies.engagement_reward import HomeostaticEnergy
from praxis.tasks import TaskType

IGNORE_INDEX = -100


class EngagementPolicy(nn.Module):
    # Forward-path policy (built into the model, not a callback) that computes
    # its own reward from labels, so the data pipeline needs no RL collection.
    is_weight_controller = False
    needs_rl_datasets = False
    # Metric namespace; subclasses (e.g. JokePolicy) override to reuse the same
    # recall-over-assistant-region machinery under a different chart family.
    prefix = "engagement"
    # Tasks this policy scores. Coexisting recall policies MUST partition the
    # task space, or they duplicate each other's reward and gradient. None =
    # score everything (single-policy runs).
    task_types: Optional[Tuple[int, ...]] = (TaskType.CONVERSATION,)

    # REINFORCE baseline EMA horizon (~100 steps). Fixed, model-agnostic.
    REWARD_BASELINE_DECAY = 0.99

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rl_weight = getattr(config, "rl_weight", 0.1)
        self.energy = HomeostaticEnergy()
        # Variance-reduction baseline: a slow EMA of the reward itself. The
        # homeostatic energy is a setpoint (~1.0), not a mean-reward baseline -
        # using it as the baseline biases every advantage negative once the
        # model activates, which drives the policy to suppress its own answer
        # tokens (unbounded-negative loss + inference degeneration). The reward
        # EMA keeps advantages zero-mean and balanced.
        self.reward_baseline = 0.0
        self._metrics: dict = {}

    @torch.no_grad()
    def _reward(self, pred_ids, tgt_ids, mask):
        """Per-example (activation, recall) over assistant tokens; rewards are
        constants, so this is detached from the graph."""
        activations, recalls = [], []
        for b in range(pred_ids.shape[0]):
            m = mask[b]
            if not bool(m.any()):
                activations.append(0.0)
                recalls.append(0.0)
                continue
            pred = set(pred_ids[b][m].tolist())
            resp = set(tgt_ids[b][m].tolist())
            overlap = pred & resp
            activations.append(1.0 if overlap else 0.0)
            recalls.append(len(overlap) / len(pred) if pred else 0.0)
        return activations, recalls

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        assistant_mask: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        # Needs full per-token logits and an assistant mask; degrade to a no-op
        # rather than guess if either is missing or misshapen.
        if (
            logits is None
            or logits.dim() != 3
            or assistant_mask is None
            or not self.training
        ):
            return None, {}

        # Next-token alignment: logits_t predict token_{t+1}.
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        mask = assistant_mask[..., 1:].to(shift_logits.device).bool()

        # Byte-latent / patching encoders (e.g. CALM) can return logits, labels,
        # and the mask at slightly different sequence lengths (off-by-one from
        # repadding). Align all three to their common length before combining
        # rather than erroring; if there's nothing to score, no-op.
        seq = min(shift_logits.size(1), shift_labels.size(1), mask.size(1))
        if seq <= 0:
            return None, {}
        shift_logits = shift_logits[:, :seq]
        shift_labels = shift_labels[:, :seq]
        mask = mask[:, :seq]

        mask = mask & (shift_labels != IGNORE_INDEX)

        # Scope to this policy's tasks so coexisting recall policies (e.g.
        # engagement vs joke) train on disjoint examples.
        if self.task_types is not None and task_type_ids is not None:
            shift_task = task_type_ids[..., 1:].to(mask.device)
            if shift_task.size(1) >= seq:
                allowed = torch.zeros_like(mask)
                for t in self.task_types:
                    allowed |= shift_task[:, :seq] == int(t)
                mask = mask & allowed

        if not bool(mask.any()):
            return None, {}

        pred_ids = shift_logits.argmax(dim=-1)
        activations, recalls = self._reward(pred_ids, shift_labels, mask)
        device = shift_logits.device
        # Stats over in-scope rows only - rows another policy owns must not
        # dilute this one's activation/recall averages.
        rows = mask.any(dim=1).tolist()
        scoped = [i for i, r in enumerate(rows) if r] or list(range(len(activations)))
        activation_rate = sum(activations[i] for i in scoped) / len(scoped)
        # Homeostatic energy enters the reward (not the baseline): a live
        # interaction spikes it, the baseline EMA catches up, and the wall-clock
        # decay pulls it back - so engagement yields a transient positive
        # advantage that fades, like eating.
        energy = self.energy.update(activation_rate)
        reward = torch.tensor(recalls, dtype=torch.float32, device=device) + energy

        # Advantage against the reward-EMA baseline (zero-mean, balanced).
        # Update the baseline *after* computing the advantage so it doesn't peek
        # at the current sample.
        advantage = (reward - self.reward_baseline).detach()  # [B]
        scoped_reward = float(sum(reward[i].item() for i in scoped) / len(scoped))
        self.reward_baseline = (
            self.REWARD_BASELINE_DECAY * self.reward_baseline
            + (1.0 - self.REWARD_BASELINE_DECAY) * scoped_reward
        )

        # REINFORCE on the LM's own log-probs over the answer tokens: maximize
        # log_prob(answer) weighted by advantage. log_prob = -CE per token.
        safe_labels = shift_labels.clamp(min=0)
        logprob = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view(
            shift_labels.shape
        )  # [B, T-1]

        adv_tok = advantage.unsqueeze(1).expand_as(logprob)
        weighted = (logprob * adv_tok) * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        policy_loss = -(weighted.sum() / denom)
        loss = self.rl_weight * policy_loss

        p = self.prefix
        self._metrics = {
            f"{p}_energy": energy,
            f"{p}_activation_rate": activation_rate,
            f"{p}_recall": float(sum(recalls[i] for i in scoped) / len(scoped)),
            f"{p}_reward": scoped_reward,
            f"{p}_reward_baseline": self.reward_baseline,
            f"{p}_advantage": float(
                sum(advantage[i].item() for i in scoped) / len(scoped)
            ),
        }
        return loss, self._metrics

    @torch.no_grad()
    def ingest_live(self, activation_rate: float) -> float:
        """Fold a live (real-user) activation into the homeostatic energy. The
        energy is a component of the dense reward, so a live interaction raises
        the reward above its baseline on subsequent updates - a transient
        positive advantage that fades as the energy decays - rather than
        injecting a gradient with no forward context."""
        energy = self.energy.update(activation_rate)
        self._metrics[f"{self.prefix}_energy"] = energy
        return energy

    def get_metrics(self) -> dict:
        return dict(self._metrics)


class JokePolicy(EngagementPolicy):
    """Forward-path reward for the joke task. Structurally identical to
    :class:`EngagementPolicy` - it REINFORCEs the model's reproduction (recall)
    of the assistant-region joke tokens, with the homeostatic energy as the
    baseline. The dense grounding comes from quality-filtered (well-rated) jokes
    in the data mix; the live human-approval channel folds into the energy via
    ``ingest_live``. The model is rewarded for producing jokes a human approves -
    "the model seeks our approval"."""

    prefix = "joke"
    # Loop UI query mode for this policy (see praxis.policies.loop_modes): the
    # model self-predicts its want->need score and the human's correction is
    # the live signal.
    loop_mode = "calibration"
    task_types = (TaskType.JOKE,)
