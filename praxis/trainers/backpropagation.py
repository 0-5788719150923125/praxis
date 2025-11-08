"""Backpropagation training module for Praxis models."""

import re
from datetime import datetime

import torch
from lightning.pytorch import LightningModule
from torcheval.metrics.functional import perplexity

from praxis.trainers.compile import try_compile


class BackpropagationTrainer(LightningModule):
    """
    A standard backpropagation training module with automatic torch.compile support.
    """

    def __init__(
        self, model, optimizer, scheduler, hparams, tokenizer=None, byte_latent=False
    ):
        super(BackpropagationTrainer, self).__init__()
        self.scheduler = scheduler
        self.automatic_optimization = True
        self.num_tokens = 0
        self.last_train_step_time = None
        self.train_step_ema = None
        self.tokenizer = tokenizer
        # Check if model has an encoder with aligned outputs
        self.outputs_are_aligned = (
            hasattr(model, "encoder")
            and getattr(model.encoder, "outputs_are_aligned", False)
        ) or byte_latent  # Keep byte_latent for backward compatibility
        self.last_logged_step = -1  # Track last step we logged a document
        self.save_hyperparameters(
            ignore=["model", "optimizer", "scheduler", "tokenizer"]
        )

        # Try to compile the model automatically with fallback
        self.model = try_compile(model, hparams)

        # Try to compile the optimizer as well
        self.optimizer = try_compile(optimizer, hparams)

    def forward(self, **kwargs):
        """Forward pass that accepts keyword arguments directly."""
        return self.model(**kwargs)

    def get_metrics(self):
        """Get metrics from the underlying model if available."""
        # Check if the model has get_metrics (handles both compiled and uncompiled)
        if hasattr(self.model, "get_metrics"):
            return self.model.get_metrics()
        # For torch.compile wrapped models, check _orig_mod
        elif hasattr(self.model, "_orig_mod") and hasattr(
            self.model._orig_mod, "get_metrics"
        ):
            return self.model._orig_mod.get_metrics()
        return {}

    def on_train_start(self):
        super().on_train_start()
        self.last_train_step_time = datetime.now()

    def training_step(self, batch, batch_idx):
        current_time = datetime.now()

        input_ids, rewards, token_weights, should_skip = self._handle_batch_format(
            batch, batch_idx, is_training=True
        )

        if should_skip:
            return torch.tensor(0.0, requires_grad=True)

        # Check if encoder outputs are already aligned
        if self.outputs_are_aligned:
            labels = input_ids.contiguous()
        else:
            labels = input_ids[..., 1:].contiguous()

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            rewards=rewards,
            token_weights=token_weights,
        )
        loss = outputs.loss
        softmax_collapse = self._compute_softmax_collapse(outputs.logits)

        batch_size, num_tokens = input_ids.shape
        self.num_tokens += batch_size * num_tokens

        step_time = current_time - self.last_train_step_time
        self.train_step_ema = self._update_ema(self.train_step_ema, step_time)
        self.last_train_step_time = current_time

        # Log document at 10% intervals (every 10 steps)
        if batch_idx > 0 and batch_idx % 10 == 0 and batch_idx != self.last_logged_step:
            # self._log_training_document(input_ids, batch_idx)
            self.last_logged_step = batch_idx

        # Prepare metrics dict
        metrics = {
            "loss": loss,
            "batch": int(batch_idx),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "num_tokens": self.num_tokens / 1_000_000_000,  # convert to billions
            "avg_step_time": self.train_step_ema,
            "softmax_collapse": softmax_collapse,
        }

        # Add RL-specific metrics if available
        if rewards is not None:
            non_zero_rewards = (rewards > 0).sum().item()
            if non_zero_rewards > 0:
                metrics["rl_reward_mean"] = rewards[rewards > 0].mean()
                metrics["rl_reward_max"] = rewards.max()
                metrics["rl_sequences_pct"] = 100.0 * non_zero_rewards / len(rewards)

                # Extract RL loss if available
                if hasattr(outputs, "rl_loss") and outputs.rl_loss is not None:
                    metrics["rl_loss"] = outputs.rl_loss

        # Collect router convergence metrics (SMEAR, Prismatic)
        model_metrics = self.get_metrics()
        if model_metrics:
            # Extract expert convergence metrics and add to logging
            for key, value in model_metrics.items():
                if key not in metrics and key not in [
                    "experts"
                ]:  # Skip duplicate keys and nested dicts
                    metrics[key] = value

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=False,
        )

        return loss

    def on_after_backward(self):
        """
        Hook called after backward pass but before optimizer step.

        This is where we apply gradient modifications for Prismatic routers,
        enabling diverse optimization trajectories across experts.
        """
        # Apply gradient modifications for all Prismatic modules
        for module in self.model.modules():
            # Handle both compiled and uncompiled models
            actual_module = module._orig_mod if hasattr(module, "_orig_mod") else module

            # Check if this is a Prismatic router
            if hasattr(actual_module, "modify_expert_gradients"):
                actual_module.modify_expert_gradients()

    def _generate_and_evaluate_rl_batch(self, prompt_ids, metadata):
        """
        Generate responses for RL prompts and evaluate them.

        Returns:
            input_ids: New batch with generated responses
            rewards: Computed rewards for each response
        """
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device

        # Get GRPO group size
        group_size = getattr(self.model.config, "grpo_group_size", 8)

        all_input_ids = []
        all_rewards = []

        # Generate multiple responses per prompt
        with torch.no_grad():
            for i in range(batch_size):
                prompt = prompt_ids[i : i + 1]
                ground_truth = (
                    metadata[i].get("ground_truth", "") if i < len(metadata) else ""
                )

                # Generate group_size responses
                prompt_rewards = []
                prompt_sequences = []

                for _ in range(group_size):
                    # Generate response
                    generated = self.model.generate(
                        prompt,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=(
                            self.tokenizer.pad_token_id if self.tokenizer else 0
                        ),
                        eos_token_id=(
                            self.tokenizer.eos_token_id if self.tokenizer else 1
                        ),
                    )

                    # Decode to extract answer
                    if self.tokenizer:
                        generated_text = self.tokenizer.decode(
                            generated[0, prompt.shape[1] :], skip_special_tokens=True
                        )
                    else:
                        generated_text = ""

                    # Evaluate response
                    reward = self._evaluate_math_response(generated_text, ground_truth)

                    prompt_rewards.append(reward)
                    prompt_sequences.append(generated[0])

                # Add all sequences and rewards for this prompt
                all_input_ids.extend(prompt_sequences)
                all_rewards.extend(prompt_rewards)

        # Stack into batch
        if all_input_ids:
            # Pad sequences to same length
            max_len = max(seq.shape[0] for seq in all_input_ids)
            padded_sequences = []
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0

            for seq in all_input_ids:
                if seq.shape[0] < max_len:
                    padding = torch.full(
                        (max_len - seq.shape[0],),
                        pad_token_id,
                        dtype=seq.dtype,
                        device=seq.device,
                    )
                    seq = torch.cat([seq, padding])
                padded_sequences.append(seq)

            input_ids = torch.stack(padded_sequences)
            rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)

            # Log generation results
            successful = (rewards > 0).sum().item()
            print(
                f"[RL Generation] Generated {len(all_rewards)} responses, {successful} correct"
            )

            return input_ids, rewards
        else:
            return None, None

    def _evaluate_math_response(self, response, ground_truth):
        """Evaluate if the response contains the correct answer."""
        # Extract number from response
        patterns = [
            r"answer\s*(?:is|=|:)?\s*([+-]?\d*\.?\d+)",
            r"=\s*([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*$",
        ]

        response = response.lower().strip()
        extracted = None

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    extracted = float(match.group(1))
                    break
                except:
                    continue

        if extracted is None:
            return 0.0

        # Check if correct
        try:
            true_answer = float(ground_truth)
            if abs(extracted - true_answer) < 1e-6:
                return 1.0
            # Partial credit for being close
            elif abs(true_answer) > 0:
                rel_error = abs(extracted - true_answer) / abs(true_answer)
                if rel_error < 0.1:
                    return 0.5
                elif rel_error < 0.5:
                    return 0.2
        except:
            pass

        return 0.1  # Small reward for extracting any number

    def _log_training_document(self, input_ids, batch_idx):
        """Log a single document from the training batch to see chat template rendering."""
        try:
            # Take the first document from the batch
            first_doc = input_ids[0]

            if self.tokenizer:
                # Decode the document
                text = self.tokenizer.decode(first_doc, skip_special_tokens=False)

                # Create a visual separator
                print("\n" + "=" * 80)
                print(f"[DOCUMENT LOG] Step {batch_idx} - Chat Template Sample")
                print("=" * 80)

                # Print the decoded text (limit to first 500 chars for readability)
                print(text)

                print("=" * 80 + "\n")
            else:
                # If no tokenizer, just show token IDs
                print("\n" + "=" * 80)
                print(f"[DOCUMENT LOG] Step {batch_idx} - Token IDs (first 50)")
                print("=" * 80)
                print(first_doc[:50].tolist())
                print("=" * 80 + "\n")

        except Exception as e:
            print(f"[DOCUMENT LOG] Error logging document at step {batch_idx}: {e}")

    def _handle_batch_format(self, batch, batch_idx, is_training=True):
        """
        Handle batch format and RL generation for both training and validation.

        This method unifies the batch processing logic for both training and validation steps,
        ensuring consistent handling of RL generation, CoT token weights, and other batch formats.

        Returns:
            input_ids, rewards, token_weights, should_skip
        """
        step_type = "Training" if is_training else "Validation"

        # Handle RL/CoT batch format (dict with input_ids, rewards, token_weights, etc.)
        if isinstance(batch, dict) and "input_ids" in batch:
            input_ids = batch["input_ids"]
            rewards = batch.get("rewards", None)
            token_weights = batch.get("token_weights", None)

            # Log interesting batch events (only for generation batches to avoid spam)
            if batch.get("needs_generation", False):
                rewards_debug = batch.get("rewards", torch.tensor([]))
                generation_flags = (rewards_debug == -1).sum().item()
                print(
                    f"[RL] {step_type} step {batch_idx}: Processing generation batch with {generation_flags} sequences"
                )

            # Check if this batch needs generation for RL
            if batch.get("needs_generation", False) and rewards is not None:
                print(
                    f"[RL] {step_type} - Generating responses for batch {batch_idx}..."
                )
                # This is a proper RL batch - generate responses
                input_ids, rewards = self._generate_and_evaluate_rl_batch(
                    input_ids, batch.get("metadata", [])
                )
                if input_ids is None:
                    # Generation failed, skip this batch
                    print(
                        f"[RL] {step_type} - Generation failed for batch {batch_idx}, skipping..."
                    )
                    return None, None, None, True

        else:
            # Regular batch format (just tensor of input_ids)
            input_ids = batch
            rewards = None
            token_weights = None

        return input_ids, rewards, token_weights, False

    def validation_step(self, batch, batch_idx):
        # Check if we should stop early
        if hasattr(self.trainer, "should_stop") and self.trainer.should_stop:
            # Return minimal tensor to avoid errors
            return torch.tensor(0.0, device=self.device)

        input_ids, rewards, token_weights, should_skip = self._handle_batch_format(
            batch, batch_idx, is_training=False
        )

        if should_skip:
            return torch.tensor(0.0, requires_grad=True)

        # Check if encoder outputs are already aligned
        if self.outputs_are_aligned:
            labels = input_ids.contiguous()
        else:
            labels = input_ids[..., 1:].contiguous()

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            rewards=rewards,
            token_weights=token_weights,
        )

        stats = {}

        loss = outputs.loss
        stats["val_loss"] = loss

        if self.outputs_are_aligned:
            stats["val_bits_per_byte"] = self._compute_bits_per_byte(input_ids, loss)
        else:
            # Detach logits to prevent memory accumulation from computation graph
            stats["val_perplexity"] = perplexity(
                outputs.logits[..., :-1, :].detach(), labels
            )

        self.log_dict(
            stats,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=input_ids.size(0),
            prog_bar=True,
            sync_dist=False,  # Don't sync across distributed processes
        )

    def on_validation_end(self):
        super().on_validation_end()
        self.last_train_step_time = datetime.now()

        # Clear CUDA cache after validation to free memory from logits and intermediate tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["num_tokens"] = self.num_tokens

        # Explicitly save DataModule state if it exists
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            if hasattr(self.trainer.datamodule, "state_dict"):
                checkpoint["datamodule_state"] = self.trainer.datamodule.state_dict()
                print("[Checkpoint] Saved DataModule state including dataset positions")

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.num_tokens = checkpoint.get("num_tokens", 0)

        # Explicitly load DataModule state if it exists
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            if "datamodule_state" in checkpoint and hasattr(
                self.trainer.datamodule, "load_state_dict"
            ):
                self.trainer.datamodule.load_state_dict(checkpoint["datamodule_state"])
                print(
                    "[Checkpoint] Restored DataModule state including dataset positions"
                )

    def _update_ema(self, ema, new_value):
        if ema is None:
            return new_value.total_seconds()
        alpha = 0.1
        return alpha * new_value.total_seconds() + (1 - alpha) * ema

    def _compute_bits_per_byte(self, batch, loss):
        """
        From "Byte Latent Transformer: Patches Scale Better Than Tokens":
        https://arxiv.org/abs/2412.09871
        """
        batch_size, seq_length = batch.shape
        # Calculate number of bytes
        num_bytes = batch_size * seq_length
        # Convert mean loss back to sum loss
        sum_loss = loss * num_bytes
        # Calculate bits per byte using sum loss
        return sum_loss / (torch.log(torch.tensor(2.0)) * num_bytes)

    def _compute_softmax_collapse(self, output):
        """
        From "Grokking at the Edge of Stability".
        https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/0cc9e8dc62ce5ed66d29d80eebbaf14da2f71c67/logger.py#L154
        """
        output_off = output - output.amax(dim=1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_exp = torch.sum(exp_output, dim=-1, keepdim=True)
        log_softmax = output_off.amax(dim=1, keepdim=True) - torch.log(sum_exp)
        softmax_collapse = (sum_exp == 1).float().mean().item()
        return softmax_collapse
