"""Lightning callback to integrate with DynamicsLogger."""

from lightning.pytorch.callbacks import Callback

from praxis.logging.dynamics_logger import DynamicsLogger
from praxis.metrics import extract_layer_dynamics


class DynamicsLoggerCallback(Callback):
    """PyTorch Lightning callback that logs gradient dynamics to DynamicsLogger.

    Logs two categories of dynamics:
      1. **Universal** (always available): per-layer gradient norms, variance,
         and update-to-weight ratios for every decoder layer.
      2. **Expert** (router-dependent): per-expert gradient norms and variance
         from Prismatic/SMEAR routers, when present.

    Args:
        run_dir: Directory for the current run (e.g., "build/runs/83492c812")
        num_experts: Number of experts in router (default: 0, auto-detected)
        log_freq: Log gradients every N steps (default: 10)
    """

    def __init__(
        self,
        run_dir: str,
        num_experts: int = 0,
        log_freq: int = 10,
    ):
        super().__init__()
        self.dynamics_logger = DynamicsLogger(run_dir, num_experts=num_experts)
        self.log_freq = log_freq
        self._success_count = 0
        self._failure_logged = False
        print(
            f"[DynamicsLogger] Initialized: logging every {log_freq} steps to {self.dynamics_logger.filepath}"
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log gradient dynamics before optimizer step.

        Timing is critical: gradients are accumulated and ready, but not yet
        applied or zeroed. Works correctly with gradient accumulation.
        """
        try:
            if trainer.global_step % self.log_freq != 0:
                return

            model = self._unwrap_model(pl_module)
            if model is None:
                return

            dynamics = {}

            # Universal dynamics: per-layer gradient flow (always available)
            dynamics.update(self._extract_layer_dynamics(model, optimizer))

            # Optimizer-state telemetry (lr, update size, momentum/grad cosine,
            # Adam SNR + second moment, schedule-free spread + gate).
            dynamics.update(self._extract_optimizer_dynamics(optimizer))

            # Expert dynamics: per-expert gradients (only when routers exist)
            dynamics.update(self._extract_expert_dynamics(model))

            # Head-specific diagnostics (harmonic field, crystal centers,
            # etc.). Each BaseHead may opt in via training_metrics().
            dynamics.update(self._extract_head_dynamics(model))

            # Titans memory diagnostics (surprise), averaged across layers.
            dynamics.update(self._extract_memory_dynamics(model))

            # Contrastive isotropy diagnostics (loss + repr anisotropy).
            dynamics.update(self._extract_contrastive_dynamics(model))

            # Arc per-depth bias specialization, averaged across Arc modules.
            dynamics.update(self._extract_arc_dynamics(model))

            # Loss-owning encoder diagnostics (e.g. CALM: latent stats, KL β,
            # energy loss). Encoders opt in via training_metrics().
            dynamics.update(self._extract_encoder_dynamics(model))

            # Loss-function diagnostics (e.g. HALO: gamma, shell radius,
            # abstain rate). The criterion opts in via training_metrics().
            dynamics.update(self._extract_loss_dynamics(model))

            # Sequence-length curriculum: per-multiplier sampling mix +
            # learning progress. Empty unless the adaptive curriculum is armed.
            dynamics.update(self._extract_seq_curriculum_dynamics())

            if dynamics:
                self._success_count += 1
                if self._success_count <= 3:
                    keys = sorted(dynamics.keys())[:8]
                    suffix = "..." if len(dynamics) > 8 else ""
                    print(
                        f"[DynamicsLogger] Logged {len(dynamics)} metrics at step {trainer.global_step}: {keys}{suffix}"
                    )
                self.dynamics_logger.log(step=trainer.global_step, dynamics=dynamics)
            elif not self._failure_logged:
                print(
                    f"[DynamicsLogger] WARNING: No dynamics extracted at step {trainer.global_step}. "
                    f"Model type: {type(model).__name__}, "
                    f"has decoder: {hasattr(model, 'decoder')}"
                )
                self._failure_logged = True

        except Exception as e:
            print(f"[DynamicsLogger] Error at step {trainer.global_step}: {e}")
            import traceback

            traceback.print_exc()

    def _unwrap_model(self, pl_module):
        """Return the inner model that holds the actual decoder/locals."""
        return getattr(pl_module, "model", pl_module)

    def _extract_layer_dynamics(self, model, optimizer) -> dict:
        """Extract universal per-layer gradient dynamics from decoder layers.

        Delegates per-layer computation to
        :func:`~praxis.metrics.extract_layer_dynamics` and prefixes the
        returned keys with ``layer_{idx}_``.
        """
        if not hasattr(model, "decoder") or not hasattr(model.decoder, "locals"):
            return {}

        # Get learning rate from optimizer
        lr = 1.0
        actual_opt = optimizer
        # Unwrap pytorch-optimizer wrappers
        while (
            hasattr(actual_opt, "optimizer") and actual_opt.optimizer is not actual_opt
        ):
            actual_opt = actual_opt.optimizer
        if hasattr(actual_opt, "param_groups") and actual_opt.param_groups:
            lr = actual_opt.param_groups[0].get("lr", 1.0)

        dynamics = {}
        for layer_idx, layer in enumerate(model.decoder.locals):
            layer_dyn = extract_layer_dynamics(layer, lr)
            if layer_dyn is not None:
                for key, value in layer_dyn.items():
                    dynamics[f"layer_{layer_idx}_{key}"] = value

        return dynamics

    def _extract_optimizer_dynamics(self, optimizer) -> dict:
        """Optimizer-state telemetry; owns its computation + chart hints in
        :mod:`praxis.metrics.optimizer`. Wrapped so a buggy read can't take
        down dynamics logging."""
        try:
            from praxis.metrics.optimizer import extract_optimizer_dynamics

            return extract_optimizer_dynamics(optimizer)
        except Exception as e:
            print(f"[DynamicsLogger] optimizer dynamics failed: {e}")
            return {}

    def _extract_expert_dynamics(self, model) -> dict:
        """Extract per-expert gradient dynamics from routers (Prismatic/SMEAR).

        Only produces data when decoder layers have routers with
        log_gradient_dynamics() method.
        """
        if not hasattr(model, "decoder") or not hasattr(model.decoder, "locals"):
            return {}

        all_dynamics = {}
        for layer_idx, layer in enumerate(model.decoder.locals):
            if not hasattr(layer, "router"):
                continue

            router = layer.router
            if not hasattr(router, "log_gradient_dynamics"):
                continue

            dynamics = router.log_gradient_dynamics()
            if dynamics:
                for key, value in dynamics.items():
                    all_dynamics[f"layer_{layer_idx}_{key}"] = value

        return all_dynamics

    def _extract_head_dynamics(self, model) -> dict:
        """Delegate to the LM head's own diagnostics.

        Heads opt in by overriding ``BaseHead.training_metrics``; we
        wrap the call in a try/except so a buggy metric in one head
        doesn't kill the whole dynamics log.
        """
        head = getattr(model, "head", None)
        if head is None:
            return {}
        try:
            return head.training_metrics()
        except Exception as e:
            print(f"[DynamicsLogger] head.training_metrics() failed: {e}")
            return {}

    def _extract_contrastive_dynamics(self, model) -> dict:
        """Delegate to the contrastive isotropy loss's own diagnostics.

        The module opts in via ``training_metrics()``; wrapped in try/except
        so a buggy metric doesn't kill the whole dynamics log.
        """
        iso = getattr(model, "contrastive_isotropy", None)
        if iso is None:
            return {}
        try:
            return iso.training_metrics()
        except Exception as e:
            print(f"[DynamicsLogger] contrastive training_metrics() failed: {e}")
            return {}

    def _extract_encoder_dynamics(self, model) -> dict:
        """Delegate to a loss-owning encoder's own diagnostics (e.g. CALM).

        Encoders opt in via ``training_metrics()`` (with chart hints declared
        as a ``metric_descriptions`` class attr). Guard against
        ``model.encoder = False`` (the no-encoder sentinel set in modeling.py).
        """
        encoder = getattr(model, "encoder", None)
        if not encoder or not hasattr(encoder, "training_metrics"):
            return {}
        try:
            return encoder.training_metrics()
        except Exception as e:
            print(f"[DynamicsLogger] encoder.training_metrics() failed: {e}")
            return {}

    def _extract_seq_curriculum_dynamics(self) -> dict:
        """Adaptive sequence-length curriculum telemetry (class-level state,
        not a model module): per-multiplier sampling probability + learning
        progress. Empty when the curriculum is disabled."""
        try:
            from praxis.data.seq_curriculum import SequenceCurriculum

            return SequenceCurriculum.metrics()
        except Exception as e:
            print(f"[DynamicsLogger] seq curriculum metrics failed: {e}")
            return {}

    def _extract_loss_dynamics(self, model) -> dict:
        """Delegate to the loss function's own diagnostics (e.g. HALO).

        The criterion opts in via ``training_metrics()`` (chart hints declared
        as a ``metric_descriptions`` class attr); wrapped so a buggy metric
        doesn't kill the dynamics log.
        """
        criterion = getattr(model, "criterion", None)
        if criterion is None or not hasattr(criterion, "training_metrics"):
            return {}
        try:
            return criterion.training_metrics()
        except Exception as e:
            print(f"[DynamicsLogger] criterion.training_metrics() failed: {e}")
            return {}

    def _extract_arc_dynamics(self, model) -> dict:
        """Collect Arc per-depth specialization, averaged across Arc modules.

        ArcAttention/ArcGLU opt in via ``training_metrics()``; wrapped in
        try/except so one bad metric doesn't kill the dynamics log.
        """
        from praxis.metrics.specialization import collect_arc_metrics

        try:
            return collect_arc_metrics(model)
        except Exception as e:
            print(f"[DynamicsLogger] arc training_metrics failed: {e}")
            return {}

    def _extract_memory_dynamics(self, model) -> dict:
        """Collect Titans memory diagnostics, averaged across memory layers.

        Memory modules opt in via ``MemorySurfacing.training_metrics``; wrapped
        in try/except so one bad metric doesn't kill the dynamics log.
        """
        from praxis.memory.surfacings import MemoryBase

        try:
            return MemoryBase.collect_training_metrics(model)
        except Exception as e:
            print(f"[DynamicsLogger] memory training_metrics failed: {e}")
            return {}

    def on_train_end(self, trainer, pl_module):
        """Close logger on training end."""
        self.dynamics_logger.close()
