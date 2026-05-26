"""Terminal interface callback for Praxis training."""

import random
import time
from datetime import datetime, timedelta

from lightning.pytorch.callbacks import Callback

from praxis.generation.streaming import StreamingContext, random_char_seed
from praxis.metrics.ema import LOSS_EMA_ALPHA, compute_ema


class TerminalInterface(Callback):
    """
    A single pane of glass containing charts and information.
    """

    def __init__(
        self,
        tokenizer,
        model_info,  # Required: dict with all model information
        generator=None,
        use_dashboard=False,
        url=None,
        progress_bar=None,
        device=None,
        quiet=False,
        headless=False,
        terminal_output_length=512,
        infer_every=3,
        byte_latent=False,
        debug=False,
        get_memory_info=None,
        api_server=None,
        dashboard=None,  # Accept existing dashboard
    ):
        super().__init__()
        self.alpha = LOSS_EMA_ALPHA
        self.ema_loss = 0
        self.start_time = datetime.now()
        self.last_time = datetime.now()
        self.last_logged_weights = None  # Track weights for change detection
        self.tokenizer = tokenizer
        self.generator = generator
        # Seed factory for the streaming context below: each reset
        # re-rolls a fresh random printable character, so the dashboard
        # never forces a specific priming character across a run.
        self._seed_factory = random_char_seed
        self.text = self._seed_factory()
        self.initial_text = self.text
        self.interval = infer_every
        self.url = url
        self.use_dashboard = use_dashboard
        self.dashboard = dashboard  # Use existing dashboard if provided
        self.progress_bar = progress_bar
        self.device = device
        self.quiet = quiet
        self.headless = headless
        self.terminal_output_length = terminal_output_length
        self.byte_latent = byte_latent
        self.debug = debug
        self.get_memory_info = get_memory_info
        self.api_server = api_server

        # Extract model info from the required dict
        self.optimizer_config = model_info.get("optimizer_config", {})
        self.strategy = model_info.get("strategy")
        self.rl_type = model_info.get("rl_type")
        self.vocab_size = model_info.get("vocab_size")
        self.depth = model_info.get("depth")
        self.num_layers = model_info.get(
            "num_layers"
        )  # Number of layer components for controllers
        self.hidden_size = model_info.get("hidden_size")
        self.embed_size = model_info.get("embed_size")
        self.dropout = model_info.get("dropout")
        self.dev = model_info.get("dev", False)
        self.seed = model_info.get("seed")
        self.truncated_hash = model_info.get("truncated_hash")
        self.total_params = model_info.get("total_params")
        self.target_batch_size = model_info.get("target_batch_size")

        # Track inference timing to detect when it's too slow
        self.inference_time_ema = None
        self.inference_slowdown_threshold = (
            10.0  # Reset if inference is 10x slower than training
        )

        # Streaming context handles stuck-output tracking and degeneracy
        # detection (n-gram repetition, sequential repetition, bracket-pipe
        # patterns, all-whitespace). The ignored_n_grams list filters out
        # special tokens that would otherwise trigger false positives.
        ignored_n_grams = []
        if tokenizer:
            ignored_n_grams = [
                t
                for t in [
                    tokenizer.bos_token,
                    tokenizer.eos_token,
                    tokenizer.pad_token,
                    tokenizer.sep_token,
                    f"{tokenizer.bos_token}system",
                    f"{tokenizer.bos_token}developer",
                    f"{tokenizer.bos_token}user",
                    f"{tokenizer.bos_token}assistant",
                ]
                if t is not None
            ]
        self._streaming = StreamingContext(
            initial_text=self._seed_factory,
            max_length=terminal_output_length,
            unchanged_threshold=30,
            ignored_n_grams=ignored_n_grams,
            repetition_n_gram_size=13 if byte_latent else 7,
            repetition_frequency=50 if byte_latent else 20,
        )
        # Sync local state with whatever the streaming context picked.
        self.text = self._streaming.text
        self.initial_text = self._streaming.initial_text

    def on_fit_start(self, trainer, lm):
        super().on_fit_start(trainer, lm)
        # we limit the context length seen during training, to keep memory
        # usage consistent; very long sequences have a negative impact on training speed.
        self.max_length = self.terminal_output_length

        # Always initialize LiveMetrics for web streaming (works with or without dashboard)
        from praxis.interface.state.live_metrics import LiveMetrics

        self.live_metrics = LiveMetrics()
        self.live_metrics.state.update_seed(self.seed)
        self.live_metrics.state.update_url(self.url or "N/A")
        self.live_metrics.state.update_params(self.total_params)
        self.live_metrics.state.set_start_time(self.start_time)
        self.live_metrics.state.arg_hash = self.truncated_hash

        # In headless mode, capture Python logging output for web log viewer
        if not self.use_dashboard:
            import logging

            class _LiveMetricsLogHandler(logging.Handler):
                def __init__(self, live_metrics):
                    super().__init__()
                    self._lm = live_metrics

                def emit(self, record):
                    self._lm.add_log(self.format(record))

            handler = _LiveMetricsLogHandler(self.live_metrics)
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logging.getLogger().addHandler(handler)

        if self.use_dashboard:
            if self.dashboard is None:
                # Create new dashboard if not provided
                try:
                    # Import here to avoid circular dependency
                    from praxis.interface import TerminalDashboard

                    self.dashboard = TerminalDashboard(self.seed, self.truncated_hash)
                    self.dashboard.start()
                except KeyboardInterrupt:
                    if self.dashboard:
                        self.dashboard.stop()
                    if self.api_server:
                        self.api_server.stop()
            elif not self.dashboard.running:
                # Start existing dashboard if not already running
                self.dashboard.start()

            # Update dashboard info
            self.dashboard.update_seed(self.seed)
            self.dashboard.update_url(self.url)
            self.dashboard.update_params(self.total_params)
            self.dashboard.set_start_time(self.start_time)
            self.print = print
        elif self.progress_bar is not None:
            self.print = self.progress_bar.print
        else:
            self.print = print

    def on_train_batch_start(self, trainer, lm, batch, batch_idx):
        super().on_train_batch_start(trainer, lm, batch, batch_idx)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("train")
        if hasattr(self, "live_metrics"):
            self.live_metrics.state.set_mode("train")

    def on_validation_start(self, trainer, lm):
        super().on_validation_start(trainer, lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("validation")
        if hasattr(self, "live_metrics"):
            self.live_metrics.state.set_mode("validation")
            self.live_metrics._update_count += 1

    def on_validation_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, lm, outputs, batch, batch_idx)
        if not self.quiet and self.generator:
            self._generate_text(lm, batch_idx, self.interval)

    def on_validation_end(self, trainer, lm):
        super().on_validation_end(trainer, lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("train")
        if hasattr(self, "live_metrics"):
            self.live_metrics.state.set_mode("train")
            self.live_metrics._update_count += 1

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        loss = trainer.callback_metrics.get("loss", 0)
        self.ema_loss = self._compute_ema_loss(float(loss), self.ema_loss, self.alpha)

        if not self.quiet and self.generator:
            self._generate_text(lm, batch_idx, self.interval)

        # Handle both tensor and dict batch formats
        if isinstance(batch, dict) and "input_ids" in batch:
            batch_size, seq_length = batch["input_ids"].shape
        else:
            batch_size, seq_length = batch.shape

        # Get metrics from the model (handles compiled models too)
        swarm_info = None
        local_layers = 0
        remote_layers = 0

        # Get metrics from the trainer (which handles compiled/uncompiled models)
        if hasattr(lm, "get_metrics"):
            try:
                swarm_info = lm.get_metrics()
            except Exception:
                pass

        # Extract layer counts if we got metrics
        if swarm_info and "layers" in swarm_info:
            local_layers = swarm_info["layers"].get("local", 0)
            remote_layers = swarm_info["layers"].get("remote", 0)

        data = {
            "step": int(batch_idx // trainer.accumulate_grad_batches),
            "local_layers": int(local_layers),
            "remote_layers": int(remote_layers),
        }

        if swarm_info is not None:
            if "fitness" in swarm_info:
                data.update({"fitness": swarm_info["fitness"]})

            if "churn" in swarm_info:
                data.update({"memory_churn": swarm_info["churn"]})

            if "predictions" in swarm_info:
                data.update(
                    {
                        "acc0": swarm_info["predictions"]["mean"],
                        "acc1": 0,
                    }
                )

        self.log_dict(
            data,
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=True,
        )

        if self.dashboard and hasattr(self.dashboard, "update_batch"):
            self._update_dashboard(
                trainer,
                lm,
                batch_idx,
                batch_size,
                seq_length,
                local_layers,
                remote_layers,
                data,
            )

        # Always update live metrics for web streaming (works with or without dashboard)
        if hasattr(self, "live_metrics"):
            self._update_live_metrics(
                trainer,
                lm,
                batch_idx,
                batch_size,
                seq_length,
                local_layers,
                remote_layers,
                data,
            )

    def _update_dashboard(
        self,
        trainer,
        lm,
        batch_idx,
        batch_size,
        seq_length,
        local_layers,
        remote_layers,
        data,
    ):
        """Update dashboard with current metrics."""
        batch = trainer.callback_metrics.get("batch", 0)
        step = trainer.callback_metrics.get("step", 0)
        rate = trainer.callback_metrics.get("avg_step_time", 0)
        tokens = trainer.callback_metrics.get("num_tokens", 0)
        self.dashboard.update_batch(batch.item() if hasattr(batch, "item") else batch)
        self.dashboard.update_step(step.item() if hasattr(step, "item") else step)
        self.dashboard.update_rate(rate.item() if hasattr(rate, "item") else rate)
        self.dashboard.update_tokens(
            tokens.item() if hasattr(tokens, "item") else tokens
        )
        self.dashboard.update_loss(self.ema_loss)
        self.dashboard.update_layer_count(local_layers, remote_layers)

        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            self.dashboard.update_val(
                val_loss.item() if hasattr(val_loss, "item") else val_loss
            )
        if "fitness" in data:
            self.dashboard.update_fitness(data["fitness"])
        if "memory_churn" in data:
            self.dashboard.update_memory(data["memory_churn"])
        if "acc0" in data:
            self.dashboard.update_accuracy(data["acc0"], data["acc1"])

        # Update the info panel with device and memory information
        if self.get_memory_info:
            memory_info = self.get_memory_info(self.device)
        else:
            memory_info = {}

        info_dict = {
            "device": self.device,
            "ram": f"{memory_info.get('ram_used', 'N/A')}/{memory_info.get('ram_total', 'N/A')}",
        }

        # Add GPU memory info if available
        if self.device and self.device.startswith("cuda:"):
            gpu_idx = int(self.device.split(":")[1])
            # Driver-level GPU usage (mem_get_info) matches what
            # nvidia-smi shows: the full per-device VRAM consumption,
            # CUDA context overhead included. The PyTorch
            # caching-allocator's ``reserved`` counter only sees its
            # own pool and undercounts everything else.
            gpu_actual_key = f"gpu{gpu_idx}_actual_used"
            gpu_total_key = f"gpu{gpu_idx}_total"
            if gpu_actual_key in memory_info and gpu_total_key in memory_info:
                info_dict["vram"] = (
                    f"{memory_info[gpu_actual_key]}/{memory_info[gpu_total_key]}"
                )
            elif "gpu_status" in memory_info:
                info_dict["vram"] = memory_info["gpu_status"]
            else:
                info_dict["vram"] = "N/A"

        info_dict["optimizer"] = self.optimizer_config.get("optimizer_name", "Unknown")
        info_dict["strategy"] = self.strategy
        info_dict["policy"] = self.rl_type
        info_dict["vocab_size"] = self.vocab_size
        info_dict["block_size"] = seq_length
        info_dict["batch_size"] = batch_size
        info_dict["target_batch"] = self.target_batch_size or getattr(
            lm.hparams, "target_batch_size", batch_size
        )
        info_dict["depth"] = self.depth
        info_dict["local_layers"] = local_layers
        info_dict["remote_layers"] = remote_layers
        info_dict["hidden_size"] = self.hidden_size
        info_dict["embed_size"] = self.embed_size
        info_dict["dropout"] = self.dropout
        info_dict["debug"] = self.debug
        info_dict["meta"] = [
            item for item, condition in [("dev", self.dev)] if condition
        ]

        self.dashboard.update_info(info_dict)

    def _update_live_metrics(
        self,
        trainer,
        lm,
        batch_idx,
        batch_size,
        seq_length,
        local_layers,
        remote_layers,
        data,
    ):
        """Update live metrics for web streaming."""
        lm_state = self.live_metrics.state
        batch = trainer.callback_metrics.get("batch", 0)
        step = trainer.callback_metrics.get("step", 0)
        rate = trainer.callback_metrics.get("avg_step_time", 0)
        tokens = trainer.callback_metrics.get("num_tokens", 0)
        lm_state.update_batch(batch.item() if hasattr(batch, "item") else batch)
        lm_state.update_step(step.item() if hasattr(step, "item") else step)
        lm_state.update_rate(rate.item() if hasattr(rate, "item") else rate)
        lm_state.update_tokens(tokens.item() if hasattr(tokens, "item") else tokens)
        lm_state.update_loss(self.ema_loss)
        lm_state.update_layer_count(local_layers, remote_layers)

        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            lm_state.update_val(
                val_loss.item() if hasattr(val_loss, "item") else val_loss
            )
        if "fitness" in data:
            lm_state.update_fitness(data["fitness"])
        if "memory_churn" in data:
            lm_state.update_memory(data["memory_churn"])
        if "acc0" in data:
            lm_state.update_accuracy(data["acc0"], data["acc1"])

        # Build info dict
        if self.get_memory_info:
            memory_info = self.get_memory_info(self.device)
        else:
            memory_info = {}

        info_dict = {
            "device": self.device,
            "ram": f"{memory_info.get('ram_used', 'N/A')}/{memory_info.get('ram_total', 'N/A')}",
        }

        if self.device and self.device.startswith("cuda:"):
            gpu_idx = int(self.device.split(":")[1])
            # Driver-level GPU usage (mem_get_info) matches what
            # nvidia-smi shows: the full per-device VRAM consumption,
            # CUDA context overhead included. The PyTorch
            # caching-allocator's ``reserved`` counter only sees its
            # own pool and undercounts everything else.
            gpu_actual_key = f"gpu{gpu_idx}_actual_used"
            gpu_total_key = f"gpu{gpu_idx}_total"
            if gpu_actual_key in memory_info and gpu_total_key in memory_info:
                info_dict["vram"] = (
                    f"{memory_info[gpu_actual_key]}/{memory_info[gpu_total_key]}"
                )
            elif "gpu_status" in memory_info:
                info_dict["vram"] = memory_info["gpu_status"]
            else:
                info_dict["vram"] = "N/A"

        info_dict["optimizer"] = self.optimizer_config.get("optimizer_name", "Unknown")
        info_dict["strategy"] = self.strategy
        info_dict["policy"] = self.rl_type
        info_dict["vocab_size"] = self.vocab_size
        info_dict["block_size"] = seq_length
        info_dict["batch_size"] = batch_size
        info_dict["target_batch"] = self.target_batch_size or getattr(
            lm.hparams, "target_batch_size", batch_size
        )
        info_dict["depth"] = self.depth
        info_dict["local_layers"] = local_layers
        info_dict["remote_layers"] = remote_layers
        info_dict["hidden_size"] = self.hidden_size
        info_dict["embed_size"] = self.embed_size
        info_dict["dropout"] = self.dropout

        if trainer.world_size > 1:
            info_dict["rank"] = trainer.local_rank
            info_dict["node"] = f"{trainer.node_rank + 1} of {trainer.num_nodes}"

        self.live_metrics.info_dict = info_dict
        self.live_metrics._update_count += 1

    def on_save_checkpoint(self, trainer, lm, checkpoint):
        if self.dev:
            return
        super().on_save_checkpoint(trainer, lm, checkpoint)
        checkpoint["start_time"] = self.start_time

    def on_load_checkpoint(self, trainer, lm, checkpoint):
        super().on_load_checkpoint(trainer, lm, checkpoint)
        self.start_time = checkpoint.get("start_time", datetime.now())

    def _generate_text(self, lm, batch_idx=0, interval=10):
        if not self.generator:
            return

        # Warmup: start with 120s interval at step 0, linearly decrease to
        # base interval over 250 steps so early-training inference (which is
        # slow and unpredictable) doesn't dominate wall-clock time.
        warmup_steps = 250
        trainer = lm.trainer if hasattr(lm, "trainer") else None
        accum = trainer.accumulate_grad_batches if trainer else 1
        step = batch_idx // accum
        if step < warmup_steps:
            progress = step / warmup_steps
            effective_interval = 120 - (120 - self.interval) * progress
        else:
            effective_interval = self.interval

        if not self._is_trigger_passed(self.last_time, effective_interval):
            return

        # Check if we should reset based on previous inference times
        if self.inference_time_ema is not None:
            trainer = lm.trainer if hasattr(lm, "trainer") else None
            if trainer and hasattr(trainer, "callback_metrics"):
                train_step_time = trainer.callback_metrics.get("avg_step_time", None)
                if train_step_time:
                    train_step_time = float(train_step_time)
                    if (
                        self.inference_time_ema
                        > train_step_time * self.inference_slowdown_threshold
                    ):
                        self._streaming.reset()
                        self.text = self._streaming.text
                        self.inference_time_ema = None

        max_new_tokens = 1

        # Chance to generate extra tokens
        while random.random() < 0.1:
            max_new_tokens += 1

        # Time the inference call
        inference_start = time.time()

        # Count tokens in the prompt
        if self.tokenizer:
            prompt_tokens = len(self.tokenizer.encode(self.text))
            if self.dashboard and hasattr(self.dashboard, "update_context_tokens"):
                self.dashboard.update_context_tokens(prompt_tokens)
            if hasattr(self, "live_metrics"):
                self.live_metrics.state.update_context_tokens(prompt_tokens)

        request_id = self.generator.request_generation(
            self.text,
            dict(
                max_new_tokens=max_new_tokens,
                temperature=0.4,
                repetition_penalty=1.15,
                skip_special_tokens=False,
                truncate_to=self.max_length,
                use_cache=False,
            ),
        )
        while True:
            time.sleep(0.1)
            self.generator.fulfill_requests(max_requests=5)
            result = self.generator.get_result(request_id)
            if result is not None:
                did_reset = self._streaming.update(result)
                self.text = self._streaming.text
                if did_reset:
                    print(
                        f"[WARNING] Context reset (stuck or degenerate), "
                        f"reverting to seed."
                    )
                break

        # Track inference time
        inference_time = time.time() - inference_start
        if self.inference_time_ema is None:
            self.inference_time_ema = inference_time
        else:
            self.inference_time_ema = (
                0.9 * self.inference_time_ema + 0.1 * inference_time
            )

        # Update display
        if self.dashboard and hasattr(self.dashboard, "update_status"):
            self.dashboard.update_status(self.text)
            if self._streaming.text == self._streaming.initial_text:
                self.dashboard.force_redraw()
        elif not self.headless:
            self.print(self.text)

        # Always update live metrics status text for web streaming.
        # Pass through untouched - the browser renders Unicode natively
        # and doesn't need the CLI's fixed-width sanitization.
        if hasattr(self, "live_metrics"):
            self.live_metrics.status_text = self.text

        self.last_time = datetime.now()

    def _is_trigger_passed(self, original_time, x_seconds):
        time_difference = datetime.now() - original_time
        return time_difference > timedelta(seconds=x_seconds)

    def _compute_ema_loss(self, current_loss, prev_avg_loss, alpha=LOSS_EMA_ALPHA):
        return compute_ema(current_loss, prev_avg_loss or None, alpha)

    def on_fit_end(self, trainer, pl_module):
        """Called when training ends - clean up dashboard."""
        if self.dashboard:
            self.dashboard.stop()
            # Use context manager's __exit__ to ensure terminal restoration
            self.dashboard.__exit__(None, None, None)
            self.dashboard = None

    def on_exception(self, trainer, pl_module, exception):
        """Called when an exception occurs - clean up dashboard."""
        if self.dashboard:
            try:
                self.dashboard.stop()
                # Use context manager's __exit__ to ensure terminal restoration
                self.dashboard.__exit__(None, None, None)
            except AttributeError:
                # Dashboard might already be None or partially cleaned up
                pass
            finally:
                self.dashboard = None
