"""Lightning callback to integrate with DynamicsLogger."""

from lightning.pytorch.callbacks import Callback

from praxis.logging.dynamics_logger import DynamicsLogger


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

    def __init__(self, run_dir: str, num_experts: int = 0, log_freq: int = 10):
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

            # Expert dynamics: per-expert gradients (only when routers exist)
            dynamics.update(self._extract_expert_dynamics(model))

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
        """Get the actual model from Lightning module, handling torch.compile."""
        model = getattr(pl_module, "model", pl_module)
        # Handle torch.compile wrapper
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _extract_layer_dynamics(self, model, optimizer) -> dict:
        """Extract universal per-layer gradient dynamics from decoder layers.

        Computes for each layer:
          - grad_norm: L2 norm of all parameter gradients
          - grad_var: mean variance of gradient values
          - update_ratio: ||grad|| * lr / ||weight|| (relative update magnitude)
        """
        if not hasattr(model, "decoder") or not hasattr(model.decoder, "locals"):
            return {}

        # Get learning rate from optimizer
        lr = 1.0
        actual_opt = optimizer
        # Unwrap pytorch-optimizer wrappers
        while hasattr(actual_opt, "optimizer") and actual_opt.optimizer is not actual_opt:
            actual_opt = actual_opt.optimizer
        if hasattr(actual_opt, "param_groups") and actual_opt.param_groups:
            lr = actual_opt.param_groups[0].get("lr", 1.0)

        dynamics = {}
        for layer_idx, layer in enumerate(model.decoder.locals):
            grad_norms_sq = []
            grad_vars = []
            weight_norms_sq = []

            for param in layer.parameters():
                if param.grad is None:
                    continue
                grad_norms_sq.append(param.grad.norm().item() ** 2)
                grad_vars.append(param.grad.var().item())
                weight_norms_sq.append(param.norm().item() ** 2)

            if grad_norms_sq:
                grad_norm = sum(grad_norms_sq) ** 0.5
                weight_norm = sum(weight_norms_sq) ** 0.5
                grad_var = sum(grad_vars) / len(grad_vars)
                update_ratio = (grad_norm * lr) / max(weight_norm, 1e-8)

                dynamics[f"layer_{layer_idx}_grad_norm"] = grad_norm
                dynamics[f"layer_{layer_idx}_grad_var"] = grad_var
                dynamics[f"layer_{layer_idx}_update_ratio"] = update_ratio

        return dynamics

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

    def on_train_end(self, trainer, pl_module):
        """Close logger on training end."""
        self.dynamics_logger.close()
