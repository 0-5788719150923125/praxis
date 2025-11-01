"""Lightning callback to integrate with DynamicsLogger."""

from lightning.pytorch.callbacks import Callback

from praxis.logging.dynamics_logger import DynamicsLogger


class DynamicsLoggerCallback(Callback):
    """PyTorch Lightning callback that logs gradient dynamics to DynamicsLogger.

    This callback extracts gradient dynamics from Prismatic routers after backward()
    and logs them for web visualization in the Dynamics tab.

    Args:
        run_dir: Directory for the current run (e.g., "build/runs/83492c812")
        num_experts: Number of experts in Prismatic (default: 2)
        log_freq: Log gradients every N steps (default: 10)
    """

    def __init__(self, run_dir: str, num_experts: int = 2, log_freq: int = 10):
        """Initialize callback.

        Args:
            run_dir: Directory for the current run
            num_experts: Number of experts to track
            log_freq: Log every N steps (gradient logging is expensive)
        """
        super().__init__()
        self.dynamics_logger = DynamicsLogger(run_dir, num_experts=num_experts)
        self.log_freq = log_freq
        print(f"[DynamicsLogger] Initialized: logging every {log_freq} steps to {self.dynamics_logger.filepath}")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log gradient dynamics before optimizer step.

        This is the critical timing: all gradients are accumulated and ready,
        but not yet applied or zeroed. This works correctly with gradient accumulation.
        """
        try:
            # Only log every N steps to reduce overhead
            if trainer.global_step % self.log_freq != 0:
                return

            # Extract gradient dynamics from model
            dynamics = self._extract_gradient_dynamics(pl_module)

            if dynamics:
                # Only print success on first few successful logs
                if not hasattr(self, '_success_count'):
                    self._success_count = 0
                self._success_count += 1

                if self._success_count <= 3:
                    print(f"[DynamicsLogger] ✓ Logged gradient dynamics at step {trainer.global_step}")
                    print(f"  Experts: {list(dynamics.get('expert_gradients', {}).keys())}")

                self.dynamics_logger.log(step=trainer.global_step, dynamics=dynamics)
            else:
                # Debug: log why we're not getting dynamics - but only on first few attempts
                if not getattr(self, '_failure_logged', False):
                    print(f"[DynamicsLogger] WARNING: No dynamics extracted at step {trainer.global_step}")

                    # Get actual model (unwrap from Lightning and torch.compile)
                    model = None
                    if hasattr(pl_module, 'model'):
                        model = pl_module.model
                        print(f"  Lightning module has .model attribute")
                        if hasattr(model, '_orig_mod'):
                            print(f"  Model is torch.compiled, unwrapping to _orig_mod")
                            model = model._orig_mod
                    else:
                        model = pl_module
                        print(f"  Using pl_module directly (no .model attribute)")

                    print(f"  Model type: {type(model).__name__}")
                    print(f"  Model dir: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")
                    print(f"  Has decoder: {hasattr(model, 'decoder')}")
                    if hasattr(model, 'decoder'):
                        print(f"  Has locals: {hasattr(model.decoder, 'locals')}")
                        if hasattr(model.decoder, 'locals'):
                            print(f"  Num layers: {len(model.decoder.locals)}")
                            for i, layer in enumerate(model.decoder.locals):
                                print(f"  Layer {i} has router: {hasattr(layer, 'router')}")
                                if hasattr(layer, 'router'):
                                    router = layer.router
                                    print(f"    Router type: {type(router).__name__}")
                                    print(f"    Has log_gradient_dynamics: {hasattr(router, 'log_gradient_dynamics')}")
                                    if hasattr(router, 'log_gradient_dynamics'):
                                        result = router.log_gradient_dynamics()
                                        print(f"    log_gradient_dynamics() returned: {result is not None}")
                                        if result is None:
                                            print(f"    This likely means no gradients are available yet")
                    self._failure_logged = True

        except Exception as e:
            # LOUD ERROR - Don't let this fail silently!
            print("=" * 80)
            print(f"[DynamicsLogger] ❌ CRITICAL ERROR at step {trainer.global_step}")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 80)
            print("[DynamicsLogger] Callback will continue but gradient logging may be broken")
            print("=" * 80)
            # Don't re-raise - keep training going but make the error VERY visible

    def _extract_gradient_dynamics(self, pl_module) -> dict:
        """Extract gradient dynamics from Prismatic routers in model.

        Searches for Prismatic routers in model.decoder.locals and aggregates
        their gradient dynamics.

        Args:
            pl_module: PyTorch Lightning module (BackpropagationTrainer)

        Returns:
            Aggregated dynamics dict or empty dict if no Prismatic routers found
        """
        try:
            all_dynamics = []

            # Get the actual model from Lightning module
            # Lightning wraps the model in pl_module.model
            model = None
            if hasattr(pl_module, 'model'):
                model = pl_module.model
                # Handle torch.compile wrapper
                if hasattr(model, '_orig_mod'):
                    model = model._orig_mod
            else:
                # Fallback: maybe pl_module IS the model (for other trainers)
                model = pl_module

            # Check for decoder with locals (standard model structure)
            if not hasattr(model, 'decoder'):
                return {}

            if not hasattr(model.decoder, 'locals'):
                return {}

            for layer_idx, layer in enumerate(model.decoder.locals):
                # Check if this layer has a Prismatic router
                if not hasattr(layer, 'router'):
                    continue

                router = layer.router
                if not hasattr(router, 'log_gradient_dynamics'):
                    continue

                # Call log_gradient_dynamics
                dynamics = router.log_gradient_dynamics()

                if dynamics and dynamics.get('expert_gradients'):
                    all_dynamics.append(dynamics)

            if not all_dynamics:
                return {}

            # Aggregate across layers (mean for POC)
            return self._aggregate_dynamics(all_dynamics)

        except Exception as e:
            print(f"[DynamicsLogger] ❌ Error in _extract_gradient_dynamics: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _aggregate_dynamics(self, dynamics_list: list) -> dict:
        """Aggregate gradient dynamics across multiple Prismatic layers.

        For POC: Simple averaging. Could be extended to track per-layer separately.

        Args:
            dynamics_list: List of dynamics dicts from each layer

        Returns:
            Aggregated dynamics dict
        """
        if len(dynamics_list) == 0:
            return {}

        if len(dynamics_list) == 1:
            return dynamics_list[0]

        # Aggregate expert gradients
        expert_gradients = {}
        divergence_scores = {}

        # Collect all expert keys
        all_experts = set()
        for dyn in dynamics_list:
            all_experts.update(dyn.get('expert_gradients', {}).keys())

        # Average gradient norms for each expert
        for expert_key in all_experts:
            tier_sums = {}
            tier_counts = {}

            for dyn in dynamics_list:
                expert_data = dyn.get('expert_gradients', {}).get(expert_key, {})
                for metric_key, value in expert_data.items():
                    if value is None:
                        continue
                    if metric_key not in tier_sums:
                        tier_sums[metric_key] = 0
                        tier_counts[metric_key] = 0
                    tier_sums[metric_key] += value
                    tier_counts[metric_key] += 1

            expert_gradients[expert_key] = {
                k: tier_sums[k] / tier_counts[k] if tier_counts[k] > 0 else 0
                for k in tier_sums.keys()
            }

        # Average divergence scores
        for dyn in dynamics_list:
            for div_key, value in dyn.get('divergence_scores', {}).items():
                if value is None:
                    continue
                if div_key not in divergence_scores:
                    divergence_scores[div_key] = []
                divergence_scores[div_key].append(value)

        divergence_scores = {
            k: sum(v) / len(v) if len(v) > 0 else 0
            for k, v in divergence_scores.items()
        }

        return {
            'expert_gradients': expert_gradients,
            'divergence_scores': divergence_scores
        }

    def on_train_end(self, trainer, pl_module):
        """Close logger on training end."""
        self.dynamics_logger.close()
