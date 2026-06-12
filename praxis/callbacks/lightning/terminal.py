"""Terminal interface callback for Praxis training."""

import random
import time
from datetime import datetime, timedelta

from lightning.pytorch.callbacks import Callback

from praxis.generation.context_blocks import ContextStreams
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
        byte_level=False,
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
        self.byte_level = byte_level
        self.debug = debug
        self.get_memory_info = get_memory_info
        self.api_server = api_server

        # Extract model info from the required dict
        self.optimizer_config = model_info.get("optimizer_config", {})
        self.strategy = model_info.get("strategy")
        _rl = model_info.get("rl_type")
        self.rl_type = ", ".join(_rl) if isinstance(_rl, (list, tuple)) else _rl
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

        # One rolling context per ContextBlock (default: 3 temperature experiments).
        # The factory stamps each with this run's tuning; the primary (chance 1.0)
        # context drives the CLI display + the back-compat status_text.
        def _make_streaming(block):
            return StreamingContext(
                initial_text=self._seed_factory,
                max_length=int(terminal_output_length * block.context_scale),
                unchanged_threshold=30,
                ignored_n_grams=ignored_n_grams,
                repetition_n_gram_size=13 if byte_level else 7,
                repetition_frequency=50 if byte_level else 20,
            )

        def _count_tokens(text):
            return len(self.tokenizer.encode(text)) if self.tokenizer and text else 0

        self._context_streams = ContextStreams(
            _make_streaming, token_counter=_count_tokens
        )
        # Largest per-block scale; the shared generate path truncates prompts to
        # this so a double-length block isn't clipped back to the base length.
        self._max_context_scale = max(
            b.context_scale for b in self._context_streams.blocks
        )
        self._streaming = self._context_streams.primary
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

    def _current_stage(self, lm):
        """Semantic training stage from the model's encoder, if it names one.

        Defaults to "pretrain" (standard LM training). An encoder may report a
        finer stage (e.g. CALM's "preflight" while its codec pretrains)."""
        model = getattr(lm, "model", None)
        model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        encoder = getattr(model, "encoder", None)
        if encoder is not None and hasattr(encoder, "training_stage"):
            try:
                stage = encoder.training_stage()
            except Exception:
                stage = None
            if stage:
                return stage
        return "pretrain"

    def on_train_batch_start(self, trainer, lm, batch, batch_idx):
        super().on_train_batch_start(trainer, lm, batch, batch_idx)
        stage = self._current_stage(lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("train")
            self.dashboard.set_stage(stage)
        if hasattr(self, "live_metrics"):
            self.live_metrics.state.set_mode("train")
            self.live_metrics.state.set_stage(stage)
            # Announce stage transitions (e.g. preflight -> pretrain) once.
            # Every run starts in "preflight" (the state's initial stage), so
            # the first batch of a plain LM announces preflight -> pretrain.
            last = getattr(self, "_last_stage", "preflight")
            if stage != last:
                msg = f"Training stage: {last} → {stage}"
                self.live_metrics.add_event(msg)
                if self.dashboard:
                    self.dashboard.add_log(msg)
            self._last_stage = stage

    def on_validation_start(self, trainer, lm):
        super().on_validation_start(trainer, lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("validation")
            self.dashboard.set_stage("validation")
        if hasattr(self, "live_metrics"):
            self.live_metrics.state.set_mode("validation")
            self.live_metrics.state.set_stage("validation")
            self.live_metrics._update_count += 1

    def on_validation_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, lm, outputs, batch, batch_idx)
        if not self.quiet and self.generator:
            self._generate_text(lm, batch_idx, self.interval)

    def on_validation_end(self, trainer, lm):
        super().on_validation_end(trainer, lm)
        stage = self._current_stage(lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("train")
            self.dashboard.set_stage(stage)
        if hasattr(self, "live_metrics"):
            self.live_metrics.state.set_mode("train")
            self.live_metrics.state.set_stage(stage)
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

        # An active expert pool (orchestration) contributes its alive experts to
        # the remote-layers count, so the dashboards show the swarm growing as
        # sidecar/browser peers join.
        from praxis.orchestration import status as _pool_status

        _pool = _pool_status.snapshot()
        if _pool:
            remote_layers = max(remote_layers, _pool.get("experts_alive", 0))

        data = {
            # The global optimizer step - Lightning restores it from the
            # checkpoint, unlike batch_idx // accum, which is per-epoch and
            # reported 0 after every resume.
            "step": int(trainer.global_step),
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
        # Snapshot the reference: shutdown (ctrl+c) nulls self.dashboard from
        # another thread, which can land between the caller's guard and these
        # calls. Work off the local so the method can't dereference None.
        dashboard = self.dashboard
        if dashboard is None:
            return

        batch = trainer.callback_metrics.get("batch", 0)
        step = trainer.callback_metrics.get("step", 0)
        rate = trainer.callback_metrics.get("avg_step_time", 0)
        tokens = trainer.callback_metrics.get("num_tokens", 0)
        dashboard.update_batch(batch.item() if hasattr(batch, "item") else batch)
        dashboard.update_step(step.item() if hasattr(step, "item") else step)
        dashboard.update_rate(rate.item() if hasattr(rate, "item") else rate)
        dashboard.update_tokens(tokens.item() if hasattr(tokens, "item") else tokens)
        dashboard.update_loss(self.ema_loss)
        dashboard.update_layer_count(local_layers, remote_layers)

        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            dashboard.update_val(
                val_loss.item() if hasattr(val_loss, "item") else val_loss
            )
        if "fitness" in data:
            dashboard.update_fitness(data["fitness"])
        if "memory_churn" in data:
            dashboard.update_memory(data["memory_churn"])
        if "acc0" in data:
            dashboard.update_accuracy(data["acc0"], data["acc1"])

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

        dashboard.update_info(info_dict)

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
        # Warmup tracks the run's optimizer step, NOT batch_idx: validation
        # batches restart at 0 and would re-enter warmup, stalling Terminal
        # inference (~120s interval) for the whole loop.
        step = (
            trainer.global_step
            if trainer and getattr(trainer, "global_step", 0)
            else batch_idx // accum
        )
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

        # Count tokens in the (primary) prompt for the context-tokens chip.
        if self.tokenizer:
            prompt_tokens = len(self.tokenizer.encode(self.text))
            if self.dashboard and hasattr(self.dashboard, "update_context_tokens"):
                self.dashboard.update_context_tokens(prompt_tokens)
            if hasattr(self, "live_metrics"):
                self.live_metrics.state.update_context_tokens(prompt_tokens)

        # Generate one block: request from its running text at its temperature and
        # poll the queue until the result lands. Bounded (~30s) so a stuck block
        # never hangs the others. Returns the new full passage, or None on timeout.
        def _generate(prompt, temperature):
            request_id = self.generator.request_generation(
                prompt,
                dict(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=1.15,
                    skip_special_tokens=False,
                    truncate_to=int(self.max_length * self._max_context_scale),
                    use_cache=False,
                ),
            )
            for _ in range(300):
                time.sleep(0.1)
                self.generator.fulfill_requests(max_requests=5)
                result = self.generator.get_result(request_id)
                if result is not None:
                    return result
            return None

        # Each block rolls its `chance`; the always-on primary fires every time.
        contexts = self._context_streams.step(_generate)
        self.text = self._streaming.text

        # Track inference time
        inference_time = time.time() - inference_start
        if self.inference_time_ema is None:
            self.inference_time_ema = inference_time
        else:
            self.inference_time_ema = (
                0.9 * self.inference_time_ema + 0.1 * inference_time
            )

        # Update display (the CLI shows the primary context)
        if self.dashboard and hasattr(self.dashboard, "update_status"):
            self.dashboard.update_status(self.text)
            if self._streaming.text == self._streaming.initial_text:
                self.dashboard.force_redraw()
        elif not self.headless:
            self.print(self.text)

        # Web streaming: every block as `contexts`, plus the primary as the
        # back-compat status_text. The browser renders Unicode natively.
        if hasattr(self, "live_metrics"):
            self.live_metrics.status_text = self.text
            self.live_metrics.contexts = contexts

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
