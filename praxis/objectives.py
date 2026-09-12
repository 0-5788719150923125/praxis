"""The half of a model that turns hidden states into a training loss.

A Praxis model is two things stapled together: a trunk that maps input ids to
hidden states, and everything downstream of that - the criterion, the additive
regularizers, the task weighter, the forward-path RL policies, multi-token
prediction, and the strategy that folds them into one scalar. Only the first
half is Praxis-specific. The second half needs nothing but ``hidden_states``, a
``scorer``, ``input_ids`` and ``labels``, which any ``transformers`` model can
supply, so it lives here and is mixed into both ``PraxisForCausalLM`` and the
foreign-model wrappers in :mod:`praxis.models`.

Two layers, because the HuggingFace suite is more than causal LMs:

* :class:`ObjectiveMixin` - task-agnostic. The objective container, the task
  weighter, the combination strategy, the conflict diagnostic, and the
  per-token loss weights.
* :class:`CausalObjectiveMixin` - next-token specifics: the label shift, the
  recall/RL policies, MTP, and the auxiliary-loss sweep.

A sequence-classification or masked-LM mixin would sit beside the second, on
the first.

``objective_config`` is the seam between the two kinds of host. A Praxis model
returns its own ``PraxisConfig``; a foreign wrapper returns the run's
``PraxisConfig`` while ``self.config`` stays the model's own hub config, so
nothing here can write a Praxis knob into somebody else's ``config.json``.
"""

import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from praxis import registry
from praxis.containers import LossContainer
from praxis.losses import build_objectives
from praxis.losses.conflict import ObjectiveConflict
from praxis.tasks import TASK_NAMES, resolve_task_weighter


def build_rl_policies(config):
    """Construct forward-path RL policies from ``config.rl_type``.

    rl_type is a list of policy/profile keys, so multiple discrete RL tasks
    coexist. Returns ``(policy, policy_type, recall_policies)``:

    - recall-style policies (engagement, joke) share a (logits, labels, mask)
      signature and compute their own reward; any number may coexist, one per
      RL interface with distinct metrics.
    - all others (reinforce/grpo/cot) are mutually exclusive (different
      signatures, may modify hidden states) - the single ``policy``.

    Weight controllers act from a training callback and are never built here.
    """
    from praxis.policies import get_rl_profile, normalize_rl_types

    policy = None
    policy_type = None
    recall = {}
    for rl_name in normalize_rl_types(getattr(config, "rl_type", None)):
        profile = get_rl_profile(rl_name)
        policy_key = profile["policy"] if profile else rl_name
        if not policy_key or policy_key not in registry.namespace("rl_policies"):
            continue
        policy_cls = registry.lookup("rl_policies", policy_key)
        if getattr(policy_cls, "is_weight_controller", False):
            continue
        if getattr(policy_cls, "is_recall", False):
            recall[rl_name] = policy_cls(config)
        else:
            if policy is not None:
                raise ValueError(
                    f"Multiple non-recall forward-path RL policies requested "
                    f"({policy_type!r}, {rl_name!r}); only one is supported."
                )
            policy = policy_cls(config)
            policy_type = rl_name
    return policy, policy_type, recall


class ObjectiveMixin:
    """Everything a model trains against, independent of the task it trains on.

    The host builds its own trunk and output projection, then calls
    :meth:`init_objectives` once both exist. Hosts supply ``objective_config``;
    everything else here reads only that and the tensors it is handed.
    """

    @property
    def objective_config(self):
        """The config carrying the Praxis objective knobs (``loss_func``,
        ``strategy``, ``task_weights``, ``regularizers``, ``rl_type``, ...).

        Defaults to the model's own config, which is right for a Praxis model.
        A foreign wrapper overrides it so the hub config stays untouched.
        """
        return self.config

    def init_objectives(self, config, encoder=None, classifier=None) -> None:
        """Build the criterion, policies, task weighter and fold strategy.

        Call once, after the trunk and output projection exist: the criterion
        collects terms from modules that compute their own (``claim``), and the
        arm-surgery check below reads the classifier.
        """
        # Forward-path RL policies. Weight controllers are built by training
        # callbacks, never here.
        self.policy, self.policy_type, _recall = build_rl_policies(config)
        self.policies = nn.ModuleDict(_recall)
        self._engagement_metrics: Dict[str, float] = {}

        # Every loss this model can add to its objective, in one container:
        # the main criterion, the additive representation-shaping
        # regularizers, and the terms other modules compute (claimed just
        # below). An encoder that owns the loss (CALM) bypasses the main-CE
        # path entirely, so `main` stays unregistered there.
        self.criterion = build_objectives(config, encoder)
        # A module that computes its own term declares it via ``objectives()``
        # and reads it back out of the container; collected once here, so the
        # blueprint is complete before a step runs.
        self.criterion.claim(self)

        # Per-task loss weighting. Identity (no-op) unless --task-weights
        # is set; the assistant mask from the chat template is always
        # applied when present. Learnable variants expose an anchor_loss
        # that gets folded into the combined objective by the strategy below.
        self.tasker = resolve_task_weighter(getattr(config, "task_weights", None))

        # Set by the trainer to the task indices a live dataset produces;
        # get_metrics() uses it to skip charting weights for absent tasks.
        self.active_task_ids = None

        # The strategy for combining multiple losses into a single scalar objective.
        self.strategy = registry.namespace("strategies").get(
            getattr(config, "strategy", None), registry.lookup("strategies", "naive")
        )()
        # Do the model's several objectives agree about the shared trunk? One
        # sampled cosine per loss term; see praxis/losses/conflict.py for why
        # this is the measurement and not the gradient-surgery method itself.
        self._conflict = ObjectiveConflict()
        self._conflict_metrics: Dict[str, float] = {}
        # Drained by the dynamics callback; empty unless the strategy reports.
        self._strategy_metrics: Dict[str, float] = {}
        # Per-arm Jacobian diagnostics, stashed by _collect_aux_losses.
        self._arm_metrics: Dict[str, float] = {}

        # A classifier that owns its arms' objectives takes the HALO geometric term
        # as that arm's Jacobian row, so the criterion must stop also adding it
        # - otherwise it is double-counted AND reaches the trunk uncorrected,
        # bypassing the arbitration it is supposed to be subject to.
        if getattr(classifier, "arm_surgery", False) and hasattr(
            self.criterion.main, "composite_geometry"
        ):
            self.criterion.main.composite_geometry = False

    def objective_metrics(self) -> dict:
        """Scalars the objective half contributes to ``get_metrics()``."""
        metrics: Dict[str, float] = {}
        # Surface dynamic task weights (learnable or difficulty-EMA) so
        # runs can see them drift. Fixed weighters are skipped; tasks with
        # no live dataset are skipped when active_task_ids is set.
        if getattr(self.tasker, "is_dynamic", False):
            eff = self.tasker.effective_weights().cpu().tolist()
            for idx, (name, value) in enumerate(zip(TASK_NAMES, eff)):
                if self.active_task_ids is not None and idx not in self.active_task_ids:
                    continue
                metrics[f"task_weight_{name}"] = float(value)
        # Engagement policy scalars (energy, activation rate, recall, advantage).
        if self._engagement_metrics:
            metrics.update(self._engagement_metrics)
        return metrics

    def _build_loss_weights(
        self,
        labels: torch.Tensor,
        task_type_ids: Optional[torch.Tensor],
        assistant_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compose per-token weights from task IDs and the assistant mask.

        Returns None when neither signal is provided. The returned tensor
        is shifted to align with ``labels`` -- positions in
        ``input_ids[t+1]`` set the weight on the loss for the prediction
        at ``labels[t]``.
        """
        # Honor --no-mask-prompts: drop the assistant_mask before composing
        # so every token contributes to the loss. Task weights still apply.
        if getattr(self.objective_config, "no_mask_prompts", False):
            assistant_mask = None

        if task_type_ids is None and assistant_mask is None:
            return None

        # Both inputs cover input_ids positions [0..T-1]. Labels are usually
        # the trailing T-1 tokens, so the weight aligned to label[t] comes
        # from position t+1. When labels is full-length (e.g. aligned encoder),
        # there's no shift to do.
        target_len = labels.size(-1)

        weights = None
        if task_type_ids is not None:
            shifted_task = task_type_ids[..., -target_len:]
            weights = self.tasker(shifted_task.long())
            # Preference contrast material is never imitated: rejected-tagged
            # tokens are excluded from the main CE regardless of weighter
            # profile - the preference policy's margin is their only training
            # signal (the hh-rlhf card's contract).
            from praxis.tasks import TaskType

            weights = weights * (shifted_task != int(TaskType.PREF_REJECTED)).to(
                weights.dtype
            )

        if assistant_mask is not None:
            shifted_mask = assistant_mask[..., -target_len:].to(
                weights.dtype if weights is not None else torch.float32
            )
            weights = shifted_mask if weights is None else weights * shifted_mask

        return weights

    def _finalize_loss(
        self,
        loss,
        losses: LossContainer,
        labels: Optional[torch.Tensor],
        hidden_states: Optional[torch.Tensor] = None,
    ):
        """Combine all tagged losses via the strategy.

        Auxiliary losses are omitted during validation and inference - except
        for handles_loss encoders (CALM), where the encoder owns the main loss
        and there is nothing else to fall back to (their val_loss would
        otherwise stay at 0).

        Names and the trunk activation ride along because a fold that weights
        per objective needs both: which term is which (the key set is
        conditional, so position is not an identity) and how hard each one
        pulls on the shared representation (which the loss VALUE does not say).
        Strategies that ignore them take the same plain sum they always did.
        """
        handles_loss_encoder = self.encoder is not False and getattr(
            self.encoder, "handles_loss", False
        )
        if labels is None or not (self.training or handles_loss_encoder):
            return loss
        names, loss_values = losses.get_named_losses()
        if len(loss_values) > 1 or (loss == 0 and len(loss_values) > 0):
            # The second case is aux-only (no main), e.g. a handles_loss encoder.
            folded = self.strategy(loss_values, names=names, trunk=hidden_states)
            metrics = getattr(self.strategy, "training_metrics", None)
            if metrics is not None:
                self._strategy_metrics = metrics() or {}
            return folded
        return loss


class CausalObjectiveMixin(ObjectiveMixin):
    """Next-token objectives: the Praxis label shift, forward-path policies,
    multi-token prediction, and the auxiliary-loss sweep.

    The shift convention is Praxis's throughout: ``labels`` arrive already
    shifted (``input_ids[..., 1:]``) and the logits are trimmed here, which is
    why a foreign model must never be handed ``labels`` itself - it would
    shift a second time.
    """

    def init_objectives(self, config, encoder=None, classifier=None) -> None:
        # MTP is a causal-only producer and registers its term on the
        # criterion, so it has to exist before ``claim`` runs.
        self.mtp = None
        if getattr(config, "mtp_type", None) is not None:
            if getattr(config, "bidirectional", False):
                raise ValueError("MTP cannot be combined with --bidirectional")
            from praxis.classifiers.mtp import MultiTokenPrediction

            self.mtp = MultiTokenPrediction(config)
        super().init_objectives(config, encoder=encoder, classifier=classifier)

    def supervise(
        self,
        losses: LossContainer,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        scorer: Optional[nn.Module],
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        *,
        attention_mask: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        assistant_mask: Optional[torch.Tensor] = None,
        pair_ids: Optional[torch.Tensor] = None,
        backward_logits: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        skip_logits: bool = False,
    ):
        """Run the whole objective half and return the folded scalar loss.

        The one entry point a host needs. Hosts that have to interleave
        something between the stages (``PraxisForCausalLM`` re-binds
        ``hidden_states`` from the RL policy) call the stages directly.
        """
        self._apply_recall_policies(
            losses, logits, labels, assistant_mask, task_type_ids, skip_logits, pair_ids
        )
        hidden_states = self._apply_rl_policy(
            losses,
            hidden_states,
            logits,
            labels,
            rewards,
            attention_mask,
            token_weights,
        )
        loss = self._main_loss(
            losses,
            logits,
            labels,
            hidden_states,
            scorer,
            input_ids,
            backward_logits,
            task_type_ids,
            assistant_mask,
        )
        self._collect_aux_losses(
            losses,
            hidden_states,
            logits,
            labels,
            input_ids,
            attention_mask,
            skip_logits,
            assistant_mask,
            scorer,
            patch_embeds,
        )
        return self._finalize_loss(loss, losses, labels, hidden_states)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        scorer: Optional[nn.Module],
        input_ids: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the main loss using the criterion."""
        # cut_cross_entropy needs FULL UNSHIFTED embeddings to avoid materializing shifted tensors
        # Check by class name to avoid hard dependency on integration
        is_cut_ce = type(self.criterion.main).__name__ == "CutCrossEntropyLoss"

        # Check if encoder outputs are already aligned
        if self.encoder and self.encoder.outputs_are_aligned:
            return self.criterion.main(
                logits=logits.contiguous(),
                embeddings=embeddings if is_cut_ce else embeddings,
                scorer=scorer,
                labels=labels,
                input_ids=input_ids,
                loss_weights=loss_weights,
            )
        else:
            return self.criterion.main(
                logits=logits[..., :-1, :].contiguous(),
                embeddings=(
                    embeddings if is_cut_ce else embeddings[..., :-1, :].contiguous()
                ),
                scorer=scorer,
                labels=labels,
                input_ids=input_ids,
                loss_weights=loss_weights,
            )

    def _compute_bidirectional_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        scorer: Optional[nn.Module],
        input_ids: torch.Tensor,
        backward_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute bidirectional loss (forward and backward prediction).

        Note: labels are already shifted right (input_ids[..., 1:]) when passed in.
        """
        # Forward loss: predict next token (standard causal LM)
        forward_loss = self._compute_loss(logits, labels, embeddings, scorer, input_ids)

        # Backward loss: predict previous token
        # For backward prediction, we want logits[1:] to predict input_ids[:-1]
        backward_labels = input_ids[..., :-1].contiguous()

        # Select appropriate logits and scorer for backward prediction
        if backward_logits is not None:
            # Use separate backward classifier
            back_logits = backward_logits[..., 1:, :].contiguous()
            back_scorer = getattr(self.backward_classifier, "scorer", None)
        else:
            # Reuse forward classifier
            back_logits = logits[..., 1:, :].contiguous()
            back_scorer = scorer

        # Compute backward loss
        backward_loss = self.criterion.main(
            logits=back_logits,
            embeddings=embeddings[..., 1:, :].contiguous(),
            scorer=back_scorer,
            labels=backward_labels,
            input_ids=input_ids,
        )

        # Weighted combination based on forward_weight
        forward_weight = self.objective_config.forward_weight
        backward_weight = 1.0 - forward_weight

        return forward_weight * forward_loss + backward_weight * backward_loss

    def _apply_recall_policies(
        self,
        losses: LossContainer,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        assistant_mask: Optional[torch.Tensor],
        task_type_ids: Optional[torch.Tensor],
        skip_logits: bool,
        pair_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Recall-style forward policies (engagement / joke / preference): each
        computes its own reward from the answer labels over the assistant
        region. Any number may coexist; each emits its own namespaced loss and
        metrics. Every channel reaches every policy - a policy that has no use
        for one accepts and ignores it, so adding a channel is one signature
        rather than a dispatch table."""
        if not self.policies or labels is None:
            return
        self._engagement_metrics = {}
        for name, pol in self.policies.items():
            pol_loss, pol_metrics = pol(
                logits=logits if not skip_logits else None,
                labels=labels,
                assistant_mask=assistant_mask,
                task_type_ids=task_type_ids,
                pair_ids=pair_ids,
            )
            if pol_loss is not None:
                losses.add_loss(f"{name}_policy", pol_loss)
            if pol_metrics:
                self._engagement_metrics.update(pol_metrics)

    def _apply_rl_policy(
        self,
        losses: LossContainer,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        rewards: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the single (non-recall) forward-path RL policy.

        Returns ``hidden_states``, which REINFORCE-style policies may modify.
        """
        if self.policy is None:
            return hidden_states
        rl_type = self.policy_type

        if rl_type == "grpo" and rewards is not None and labels is not None:
            # GRPO needs logits and labels for proper loss computation.
            _, rl_losses = self.policy(
                hidden_states,
                logits=logits,
                labels=labels,
                rewards=rewards,
                ref_logits=None,  # TODO: Add reference model support
                mask=attention_mask,
            )
            if rl_losses is not None:
                for key, value in rl_losses.items():
                    losses.add_loss(f"rl_{key}", value)
        elif rl_type == "cot" and labels is not None:
            # Basic CoT uses supervised learning with weighted loss; logits
            # are shifted to match labels.
            _, cot_losses = self.policy(
                hidden_states,
                logits=logits[..., :-1, :].contiguous(),
                labels=labels,
                attention_mask=attention_mask,
                token_weights=token_weights,
            )
            if cot_losses is not None:
                losses.add_loss_container(cot_losses)
        elif rewards is not None and labels is not None:
            # REINFORCE and other methods.
            hidden_states, rl_loss = self.policy(
                hidden_states, rewards=rewards, mask=attention_mask
            )
            if rl_loss is not None:
                losses.add_loss("rl_policy", rl_loss)

        return hidden_states

    def _main_loss(
        self,
        losses: LossContainer,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        scorer: Optional[nn.Module],
        input_ids: torch.Tensor,
        backward_logits: Optional[torch.Tensor],
        task_type_ids: Optional[torch.Tensor],
        assistant_mask: Optional[torch.Tensor],
    ):
        """Register the main objective and return it (0 when none applies:
        no labels, a layer-wise trainer without layer losses, or an encoder
        that owns its loss - those are combined later in _finalize_loss)."""
        if labels is None:
            return 0
        loss_weights = self._build_loss_weights(
            labels=labels,
            task_type_ids=task_type_ids,
            assistant_mask=assistant_mask,
        )
        # Layer-wise trainers (e.g. MonoForward) already trained each layer;
        # just combine their recorded losses.
        if "_layer_wise_complete" in losses.loss_dict:
            layer_losses = [
                v
                for k, v in losses.loss_dict.items()
                if k != "_layer_wise_complete" and k != "main"
            ]
            return self.strategy(layer_losses) if layer_losses else 0
        if self.encoder and self.encoder.handles_loss:
            # Encoder owns its loss bookkeeping (see CALMEncoder); its
            # registered losses are combined in _finalize_loss.
            return 0
        if self.objective_config.bidirectional:
            main_loss = self._compute_bidirectional_loss(
                logits=logits,
                labels=labels,
                embeddings=hidden_states,
                scorer=scorer,
                input_ids=input_ids,
                backward_logits=backward_logits,
            )
            return losses.add_loss("main", main_loss)
        main_loss = self._compute_loss(
            logits=logits,
            labels=labels,
            embeddings=hidden_states,
            scorer=scorer,
            input_ids=input_ids,
            loss_weights=loss_weights,
        )
        return losses.add_loss("main", main_loss)

    def _collect_aux_losses(
        self,
        losses: LossContainer,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        skip_logits: bool,
        assistant_mask: Optional[torch.Tensor] = None,
        scorer: Optional[nn.Module] = None,
        patch_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate training-only auxiliary losses into the container:
        task-weight anchor, classifier aux losses, MTP, and the regularizers."""
        if not self.training or labels is None:
            return

        # Task-weight anchor loss (learnable weighters only).
        anchor = self.tasker.anchor_loss()
        if anchor is not None:
            losses.add_loss("task_weight_anchor", anchor)

        # Classifiers can emit named aux losses (e.g., HarmonicClassifier's
        # forward-shift smoothness loss, CrystalClassifier's centers-RMS
        # regularizer).
        if self.classifier is not None:
            for name, value in self.classifier.aux_losses().items():
                losses.add_loss(name, value)

        # Multi-arm classifiers get the labels here rather than in forward,
        # because this is the first place both the labels and the classifier's
        # own input exist. `hidden_states` is exactly what it classified.
        #   arm_conflict()   - diagnostic, sampled, on every ParallelClassifier
        #   arm_objectives() - per-arm CE + the PCGrad trunk gradient, empty
        #                      unless the profile opts in (prismatic9)
        if self.classifier is not None and hasattr(self.classifier, "arm_objectives"):
            self._arm_metrics = self.classifier.arm_conflict(
                hidden_states, labels, self.criterion
            )
            for name, value in self.classifier.arm_objectives(
                hidden_states, labels, self.criterion
            ).items():
                losses.add_loss(name, value)

        # Router aux losses (e.g. VEAR's parameter-only repulsion), collected once
        # per step here rather than through the per-forward aux return: a
        # parameter-only loss escaping the gradient-checkpointed recurrent forward
        # causes a double-backward.
        if self.decoder is not None and hasattr(self.decoder, "router_aux_losses"):
            for name, value in self.decoder.router_aux_losses().items():
                losses.add_loss(name, value)

        if self.mtp is not None:
            # Byte-level MTP embeds byte IDs through the encoder's byte table
            # (get_input_embeddings() is None in encoder mode); the patch path
            # never touches embed_fn.
            mtp_embed_fn = (
                self.embeds
                if getattr(self.mtp, "byte_level", False)
                else self.get_input_embeddings()
            )
            # MTP consumes UNDETACHED hidden states and the shared classifier, so
            # its loss trains the trunk and the classifier like any other
            # objective. Left
            # unweighted it does that at every position, including the ones
            # assistant_mask zeroes - prompt text would keep shaping the trunk
            # through the auxiliary path no matter what the mask said. Only the
            # prompt mask is passed on: task weights are the main objective's
            # difficulty curriculum and coupling the draft module to the tasker
            # would be a separate decision.
            mtp_weights = (
                None
                if getattr(self.objective_config, "no_mask_prompts", False)
                else assistant_mask
            )
            # MTP classifies its draft states with the SHARED classifier, so it
            # needs that classifier's ordinary gradient path. A surgical
            # classifier (prismatic9)
            # detaches every arm in the blend so the main CE trains only the
            # gate - and detaching an arm's output severs the route back to the
            # classifier's input, which is where MTP's draft states enter. Left alone,
            # MTP's loss reaches nothing: 0 of 9 MTP parameters received a
            # gradient under prismatic9 against 9 of 9 under prismatic8, and
            # every mtp_field_* series sat frozen at its initialization.
            undetach = getattr(self.classifier, "undetached", None)
            with undetach() if undetach else contextlib.nullcontext():
                mtp_inputs = self.mtp.prepare_inputs(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    embed_fn=mtp_embed_fn,
                    classifier=self.classifier,
                    patch_embeds=patch_embeds,
                    loss_weights=mtp_weights,
                )
                losses.add_loss_container(self.mtp(mtp_inputs))

        # Additive representation-shaping regularizers (the ``regularizers`` registry).
        # `scorer` is passed as optional context, not stored: a regularizer
        # holding a reference to it would register it a second time and
        # duplicate its parameters in state_dict and the optimizer.
        for reg in self.criterion.regularizers():
            losses.add_loss(
                reg.name,
                reg(
                    hidden_states,
                    input_ids,
                    scorer=scorer,
                    classifier=self.classifier,
                    # The main term, already in the container - _main_loss runs
                    # before this. A regularizer that balances itself against
                    # what the task will pay needs to see what the task paid.
                    main_loss=losses.loss_dict.get("main"),
                ),
            )

        # Last, so the sampler sees every objective this step actually carries.
        # Stashed rather than returned: the dynamics callback drains it on its
        # own cadence, the same contract the governor and compute profiler use.
        self._conflict_metrics = self._conflict.measure(losses.loss_dict, hidden_states)
