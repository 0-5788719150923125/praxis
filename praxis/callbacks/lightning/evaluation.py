"""Periodic evaluation callback for Praxis training."""

from lightning.pytorch.callbacks import Callback


class PeriodicEvaluation(Callback):
    """Callback to perform periodic evaluation during training using the lighteval test suite."""

    def __init__(
        self,
        eval_every=None,
        eval_tasks=None,
        model=None,
        device=None,
        vocab_size=None,
        debug=False,
    ):
        super().__init__()
        self.counter = 0
        self.eval_every = eval_every
        self.eval_tasks = eval_tasks
        self.model = model
        self.device = device
        self.vocab_size = vocab_size
        self.debug = debug

    def on_validation_end(self, trainer, lm):
        super().on_validation_end(trainer, lm)
        self.counter += 1
        self._run_evaluation_suites()

    def _run_evaluation_suites(self):
        if self.eval_every is None or self.counter % self.eval_every != 0:
            return

        try:
            from eval import evaluate_model, get_all_task_metrics
        except:
            return

        metrics = evaluate_model(
            self.model,
            max_samples=250,
            tasks=self.eval_tasks,
            device=self.device,
            vocab_size=self.vocab_size,
            verbose=False,
        )
        parsed = get_all_task_metrics(metrics)

        # Dictionary to collect all metrics
        all_metrics = {}

        # Iterate through metrics and collect them
        for metric in parsed:
            name = metric["task"]
            for key, value in list(metric.items()):
                if key in [
                    "pqem",
                    "pqem_stderr",
                    "acc",
                    "acc_stderr",
                    "f1",
                    "perfect_em",
                ]:
                    metric_name = f"eval_{name}_{key}"
                    metric_value = metric[key]
                    if self.debug:
                        print(f"DEBUG: {name}: {metric_value}")

                    # Add to collected metrics dictionary
                    all_metrics[metric_name] = metric_value

        # Log all metrics at once
        if hasattr(trainer.logger, "log_metrics"):
            # WandB or other loggers that support log_metrics
            trainer.logger.log_metrics(all_metrics, step=trainer.global_step)
        elif hasattr(trainer.logger.experiment, "add_scalar"):
            # TensorBoard logger
            for metric_name, metric_value in all_metrics.items():
                trainer.logger.experiment.add_scalar(
                    metric_name, metric_value, trainer.global_step
                )
        else:
            # Fallback for other loggers
            print(f"Warning: Couldn't log metrics to logger. Metrics: {all_metrics}")

    def state_dict(self):
        # Return the state that should be saved
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        # Restore the state from the saved dictionary
        self.counter = state_dict["counter"]