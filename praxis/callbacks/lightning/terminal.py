"""Terminal interface callback for Praxis training."""

import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta

from lightning.pytorch.callbacks import Callback


class TerminalInterface(Callback):
    """
    A single pane of glass containing charts and information.
    """

    def __init__(
        self,
        tokenizer,
        generator=None,
        use_dashboard=False,
        url=None,
        progress_bar=None,
        device=None,
        quiet=False,
        terminal_output_length=512,
        byte_latent=False,
        debug=False,
        get_memory_info=None,
        api_server=None,
        model_info=None,
        # Legacy parameters for backward compatibility
        optimizer_config=None,
        strategy=None,
        rl_type=None,
        vocab_size=None,
        depth=None,
        hidden_size=None,
        embed_size=None,
        dropout=None,
        use_source_code=False,
        dev=False,
        seed=None,
        truncated_hash=None,
        total_params=None,
        target_batch_size=None,
    ):
        super().__init__()
        self.alpha = 1e-2
        self.ema_loss = 0
        self.start_time = datetime.now()
        self.last_time = datetime.now()
        self.last_logged_weights = None  # Track weights for change detection
        self.tokenizer = tokenizer
        self.generator = generator
        self.initial_text = tokenizer.bos_token if tokenizer else "<s>"
        self.text = self.initial_text
        self.interval = 3
        self.url = url
        self.use_dashboard = use_dashboard
        self.dashboard = None  # Will be initialized in on_fit_start if needed
        self.progress_bar = progress_bar
        self.device = device
        self.quiet = quiet
        self.terminal_output_length = terminal_output_length
        self.byte_latent = byte_latent
        self.debug = debug
        self.get_memory_info = get_memory_info
        self.api_server = api_server

        # Use model_info dict if provided, otherwise fall back to individual params
        if model_info is not None:
            # Modern usage: single dict with all model info
            self.optimizer_config = model_info.get("optimizer_config", {})
            self.strategy = model_info.get("strategy")
            self.rl_type = model_info.get("rl_type")
            self.vocab_size = model_info.get("vocab_size")
            self.depth = model_info.get("depth")
            self.hidden_size = model_info.get("hidden_size")
            self.embed_size = model_info.get("embed_size")
            self.dropout = model_info.get("dropout")
            self.use_source_code = model_info.get("use_source_code", False)
            self.dev = model_info.get("dev", False)
            self.seed = model_info.get("seed")
            self.truncated_hash = model_info.get("truncated_hash")
            self.total_params = model_info.get("total_params")
            self.target_batch_size = model_info.get("target_batch_size")
        else:
            # Legacy usage: individual parameters
            self.optimizer_config = optimizer_config or {}
            self.strategy = strategy
            self.rl_type = rl_type
            self.vocab_size = vocab_size
            self.depth = depth
            self.hidden_size = hidden_size
            self.embed_size = embed_size
            self.dropout = dropout
            self.use_source_code = use_source_code
            self.dev = dev
            self.seed = seed
            self.truncated_hash = truncated_hash
            self.total_params = total_params
            self.target_batch_size = target_batch_size

        # Track inference timing to detect when it's too slow
        self.inference_time_ema = None
        self.inference_slowdown_threshold = (
            10.0  # Reset if inference is 10x slower than training
        )
        # Track unchanged context to detect when model is stuck
        self.previous_texts = []  # History of generated texts
        self.unchanged_count = 0  # Count of unchanged predictions
        self.unchanged_threshold = 30  # Reset after this many unchanged predictions

    def on_fit_start(self, trainer, lm):
        super().on_fit_start(trainer, lm)
        # we limit the context length seen during training, to keep memory
        # usage consistent; very long sequences have a negative impact on training speed.
        self.max_length = self.terminal_output_length
        if self.use_dashboard:
            try:
                # Import here to avoid circular dependency
                from interface import TerminalDashboard

                self.dashboard = TerminalDashboard(self.seed, self.truncated_hash)
                self.dashboard.start()
                self.dashboard.update_seed(self.seed)
                self.dashboard.update_url(self.url)
            except KeyboardInterrupt:
                self.dashboard.stop()
                if self.api_server:
                    self.api_server.stop()
            self.print = print
            self.dashboard.update_params(self.total_params)
            self.dashboard.set_start_time(self.start_time)
        elif self.progress_bar is not None:
            self.print = self.progress_bar.print
        else:
            self.print = print

    def on_train_batch_start(self, trainer, lm, batch, batch_idx):
        super().on_train_batch_start(trainer, lm, batch, batch_idx)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("train")

    def on_validation_start(self, trainer, lm):
        super().on_validation_start(trainer, lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("validation")

    def on_validation_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, lm, outputs, batch, batch_idx)
        if not self.quiet and self.generator:
            self._generate_text(lm, batch_idx, self.interval)

    def on_validation_end(self, trainer, lm):
        super().on_validation_end(trainer, lm)
        if self.dashboard and hasattr(self.dashboard, "set_mode"):
            self.dashboard.set_mode("train")

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
        swarm_info = None  # TODO: Add get_metrics() when available
        local_experts = 0
        remote_experts = 0

        data = {
            "step": int(batch_idx // trainer.accumulate_grad_batches),
            "local_experts": int(local_experts),
            "remote_experts": int(remote_experts),
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
            sync_dist=False,
        )

        if self.dashboard and hasattr(self.dashboard, "update_batch"):
            self._update_dashboard(
                trainer,
                lm,
                batch_idx,
                batch_size,
                seq_length,
                local_experts,
                remote_experts,
                data,
            )

    def _update_dashboard(
        self,
        trainer,
        lm,
        batch_idx,
        batch_size,
        seq_length,
        local_experts,
        remote_experts,
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
        self.dashboard.update_expert_count(local_experts, remote_experts)

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
            "ram": f"{memory_info.get('ram_percent', 'N/A')}",
        }

        # Add GPU memory info if available
        if self.device and self.device.startswith("cuda:"):
            gpu_idx = int(self.device.split(":")[1])
            gpu_percent_key = f"gpu{gpu_idx}_percent"
            gpu_reserved_key = f"gpu{gpu_idx}_reserved"
            gpu_total_key = f"gpu{gpu_idx}_total"
            if gpu_percent_key in memory_info:
                info_dict["vram"] = f"{memory_info[gpu_percent_key]}"
                if gpu_reserved_key in memory_info and gpu_total_key in memory_info:
                    info_dict["vram_gb"] = (
                        f"{memory_info[gpu_reserved_key]}/{memory_info[gpu_total_key]}"
                    )
            elif "gpu_status" in memory_info:
                info_dict["vram"] = memory_info["gpu_status"]
            else:
                info_dict["vram"] = "0%"

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
        info_dict["hidden_size"] = self.hidden_size
        info_dict["embed_size"] = self.embed_size
        info_dict["dropout"] = self.dropout
        info_dict["debug"] = self.debug
        info_dict["meta"] = [
            item
            for item, condition in [("src", self.use_source_code), ("dev", self.dev)]
            if condition
        ]

        self.dashboard.update_info(info_dict)

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

        if not self._is_trigger_passed(self.last_time, self.interval):
            return

        # Check if we should reset based on previous inference times
        if self.inference_time_ema is not None:
            trainer = lm.trainer if hasattr(lm, "trainer") else None
            if trainer and hasattr(trainer, "callback_metrics"):
                train_step_time = trainer.callback_metrics.get("avg_step_time", None)
                if train_step_time:
                    train_step_time = float(train_step_time)
                    # Reset if inference is much slower than training
                    if (
                        self.inference_time_ema
                        > train_step_time * self.inference_slowdown_threshold
                    ):
                        self.text = self.initial_text
                        self.unchanged_count = 0
                        self.previous_texts = []
                        self.inference_time_ema = (
                            None  # Reset timing after context reset
                        )

        max_new_tokens = 1 if not self.byte_latent else self._biased_randint(1, 7)

        # Chance to generate extra tokens
        while random.random() < 0.1:
            max_new_tokens += 1 if not self.byte_latent else self._biased_randint(1, 7)

        # Time the inference call
        inference_start = time.time()

        # Count tokens in the prompt
        if self.tokenizer:
            prompt_tokens = len(self.tokenizer.encode(self.text))
            if self.dashboard and hasattr(self.dashboard, "update_context_tokens"):
                self.dashboard.update_context_tokens(prompt_tokens)

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
                # Track context changes to detect stuck generation
                if len(self.previous_texts) > 0 and result == self.previous_texts[-1]:
                    self.unchanged_count += 1
                    # Log periodically when context is stuck
                    if self.unchanged_count % 10 == 0 and self.unchanged_count > 0:
                        print(
                            f"[INFO] Context unchanged for {self.unchanged_count} predictions..."
                        )
                        # Debug: show what tokens were attempted to be generated
                        if self.debug:
                            print(
                                f"[DEBUG] Last text length: {len(result)}, First 100 chars: {repr(result[:100])}"
                            )
                else:
                    self.unchanged_count = 0

                # Keep history of last few texts for debugging
                self.previous_texts.append(result)
                if len(self.previous_texts) > 5:
                    self.previous_texts.pop(0)

                # Reset context if it hasn't changed for too many predictions
                if self.unchanged_count >= self.unchanged_threshold:
                    print(
                        f"[WARNING] Context stuck for {self.unchanged_count} predictions, resetting..."
                    )
                    print(
                        f"[INFO] Stuck text sample (last 300 chars): {repr(self.text[-300:])}"
                    )
                    self.text = self.initial_text
                    self.unchanged_count = 0
                    self.previous_texts = []
                else:
                    self.text = result
                break

        # Track inference time
        inference_time = time.time() - inference_start
        if self.inference_time_ema is None:
            self.inference_time_ema = inference_time
        else:
            # Use exponential moving average to smooth out timing
            self.inference_time_ema = (
                0.9 * self.inference_time_ema + 0.1 * inference_time
            )

        n_gram_size = 13 if self.byte_latent else 7
        frequency = 50 if self.byte_latent else 20
        ignored_n_grams = []
        if self.tokenizer:
            ignored_n_grams = [
                self.tokenizer.bos_token,
                self.tokenizer.eos_token,
                self.tokenizer.pad_token,
                self.tokenizer.sep_token,
                f"{self.tokenizer.bos_token}system",
                f"{self.tokenizer.bos_token}developer",
                f"{self.tokenizer.bos_token}user",
                f"{self.tokenizer.bos_token}assistant",
            ]

        if (
            self._detect_repetition(n_gram_size, frequency)
            or self._detect_sequential_repetition(threshold=5, min_segment_length=8)
            or self._is_degenerated_text(self.text)
            or self._is_all_whitespace()
        ):
            self.text = self.initial_text
            self.unchanged_count = 0
            self.previous_texts = []
            if self.dashboard and hasattr(self.dashboard, "update_status"):
                self.dashboard.update_status(self.initial_text)
                self.dashboard.force_redraw()
        elif self.dashboard and hasattr(self.dashboard, "update_status"):
            self.dashboard.update_status(self.text)
        else:
            self.print(self.text)

        self.last_time = datetime.now()

    def _biased_randint(self, low, high):
        # Take average of multiple random numbers to create center bias
        # Using 3 numbers gives a nice bell curve shape
        avg = sum(random.randint(low, high) for _ in range(3)) / 3
        # Round to nearest integer since we want whole numbers
        return round(avg)

    def _detect_repetition(self, top_n, threshold, excluded_ngrams=None):
        text = self.text
        if excluded_ngrams is None:
            excluded_ngrams = set()
        else:
            excluded_ngrams = set(excluded_ngrams)  # Convert to set for O(1) lookup

        # Step 1: Generate n-grams based on characters
        n_grams = [text[i : i + top_n] for i in range(len(text) - top_n + 1)]

        # Step 2: Filter out excluded n-grams and count frequencies
        filtered_ngrams = [ng for ng in n_grams if ng not in excluded_ngrams]
        n_gram_counts = Counter(filtered_ngrams)

        # Step 3: Check if any n-gram exceeds the threshold
        for count in n_gram_counts.values():
            if count > threshold:
                return True

        return False

    def _detect_sequential_repetition(self, threshold, min_segment_length=3):
        """
        Detect unbroken/sequential repetitions of any character sequence in text,
        only if the total repeated segment exceeds a minimum length.
        """
        text = self.text

        # Early return for very short texts
        if len(text) < min_segment_length:
            return False

        # Try all possible pattern lengths, from 1 up to half the text length
        max_pattern_length = len(text) // 2

        for pattern_length in range(1, max_pattern_length + 1):
            # Skip if pattern_length * threshold would be too short
            if pattern_length * threshold < min_segment_length:
                continue

            # Check each possible starting position
            for start in range(len(text) - pattern_length * threshold + 1):
                pattern = text[start : start + pattern_length]

                # Count sequential repetitions
                repeat_count = 1
                current_pos = start + pattern_length

                while (
                    current_pos + pattern_length <= len(text)
                    and text[current_pos : current_pos + pattern_length] == pattern
                ):
                    repeat_count += 1
                    current_pos += pattern_length

                    # Only return True if the total repeated segment is long enough
                    if (
                        repeat_count >= threshold
                        and pattern_length * repeat_count >= min_segment_length
                    ):
                        return True

        return False

    def _is_all_whitespace(self):
        return self.text.isspace()

    def _is_degenerated_text(self, text):
        """
        Detects if text shows signs of bracket-pipe degeneration pattern.
        Returns True if the text appears to be degenerated, False otherwise.
        """
        if not text or len(text.strip()) == 0:
            return False

        # Split the text into lines
        lines = text.strip().split("\n")

        # Skip detection if there's just a single line
        if len(lines) <= 1:
            return False

        # Count lines with the degeneration pattern
        pattern_lines = 0
        bracket_pipe_pattern = r"\[.+?\](\||\s*$)"

        for line in lines:
            # Check if line contains bracketed items separated by pipes
            if re.search(bracket_pipe_pattern, line):
                # Additional check: count brackets and pipes to confirm pattern
                brackets = line.count("[") + line.count("]")
                pipes = line.count("|")

                # If a line has multiple brackets and pipes, it matches our pattern
                if (
                    brackets >= 4 and pipes >= 1
                ):  # At least 2 sets of brackets and 1 pipe
                    pattern_lines += 1

        # Calculate what percentage of lines show the degeneration pattern
        pattern_percentage = pattern_lines / len(lines)

        # If more than 50% of lines show the pattern, consider it degenerated
        return pattern_percentage >= 0.5

    def _is_trigger_passed(self, original_time, x_seconds):
        time_difference = datetime.now() - original_time
        return time_difference > timedelta(seconds=x_seconds)

    def _compute_ema_loss(self, current_loss, prev_avg_loss, alpha=0.01):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (alpha * current_loss) + (1 - alpha) * prev_avg_loss
    
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
            self.dashboard.stop()
            # Use context manager's __exit__ to ensure terminal restoration
            self.dashboard.__exit__(None, None, None)
            self.dashboard = None
