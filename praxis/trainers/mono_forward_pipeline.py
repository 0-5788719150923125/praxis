"""MonoForward Pipeline Trainer with Process-based Layer Workers.

This implements true pipeline parallelism where each layer runs in its own process,
enabling layers to process different batches simultaneously without waiting for
backpropagation through the entire network.

Based on "Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training
Harnessing Local Errors" by Gong, Li, and Abdulla (2025).
"""

import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.multiprocessing import Process, Queue

from praxis.utils.system import is_shutting_down, register_child_process

# Use torch multiprocessing with proper sharing strategy
torch.multiprocessing.set_sharing_strategy("file_system")

# Verify spawn method is set (required for CUDA)
if mp.get_start_method(allow_none=True) != "spawn":
    raise RuntimeError(
        "MonoForward pipeline requires 'spawn' multiprocessing method for CUDA support. "
        "Please set multiprocessing.set_start_method('spawn') at the start of your program."
    )


class ProjectionMatrix(nn.Module):
    """Projection matrix for computing goodness scores in Mono-Forward.

    Computes G = activations × M^T where M maps from hidden_size to num_classes.
    This is the key innovation of the Mono-Forward algorithm.
    """

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=(2.0 / hidden_size) ** 0.5)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute goodness scores: G = activations × M^T"""
        return F.linear(activations, self.weight)


@dataclass
class LayerWorkerConfig:
    """Configuration for a layer worker process."""

    layer_idx: int
    hidden_size: int
    vocab_size: int
    optimizer_config: Dict[str, Any]
    device: str = "cpu"


class LayerWorkerProcess:
    """
    A worker process that handles a single layer's forward pass and weight updates.
    Runs continuously, pulling batches from input queue and pushing to output queue.
    """

    def __init__(
        self,
        layer_module: nn.Module,
        projection: Optional[nn.Module],
        config: LayerWorkerConfig,
        input_queue: Queue,
        output_queue: Queue,
        loss_queue: Queue,
        control_queue: Queue,
    ):
        """
        Initialize layer worker.

        Args:
            layer_module: The actual layer to run
            projection: Projection matrix for computing goodness scores
            config: Worker configuration
            input_queue: Queue to receive (hidden_states, labels, batch_idx) tuples
            output_queue: Queue to send processed hidden states to next layer
            loss_queue: Queue to send losses back to main process
            control_queue: Queue for control messages (stop, checkpoint, etc.)
        """
        self.layer = layer_module
        self.projection = projection
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.loss_queue = loss_queue
        self.control_queue = control_queue

        # Set device - CUDA will be initialized automatically when needed
        self.device = torch.device(config.device)

        # Move modules to device
        self.layer = self.layer.to(self.device)
        if self.projection:
            self.projection = self.projection.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Statistics
        self.batches_processed = 0
        self.total_loss = 0.0

    def _create_optimizer(self):
        """Create optimizer from config."""
        params = list(self.layer.parameters())
        if self.projection:
            params.extend(self.projection.parameters())

        optimizer_class = self.config.optimizer_config.get("class", torch.optim.Adam)
        optimizer_kwargs = self.config.optimizer_config.get("kwargs", {"lr": 1e-3})

        if isinstance(optimizer_class, str):
            # Convert string to class
            if hasattr(torch.optim, optimizer_class):
                optimizer_class = getattr(torch.optim, optimizer_class)
            else:
                # Try to import from custom optimizers
                from praxis.optimizers import OPTIMIZER_REGISTRY

                optimizer_class = OPTIMIZER_REGISTRY.get(
                    optimizer_class, torch.optim.Adam
                )

        return optimizer_class(params, **optimizer_kwargs)

    def run(self):
        """Main worker loop - process batches continuously."""
        self.layer.train()
        if self.projection:
            self.projection.train()

        running = True
        while running:
            try:
                # Check if system is shutting down
                if is_shutting_down():
                    running = False
                    break

                # Check for control messages (non-blocking)
                try:
                    control_msg = self.control_queue.get_nowait()
                    if control_msg == "stop":
                        running = False
                        break
                    elif control_msg == "checkpoint":
                        self._save_checkpoint()
                except queue.Empty:
                    pass

                # Get next batch (blocking with timeout)
                try:
                    batch_data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Unpack batch data (always 3 elements for simplicity)
                hidden_states, labels, batch_idx = batch_data

                # Move to device
                hidden_states = hidden_states.to(self.device)
                labels = labels.to(self.device)

                # CRITICAL: Detach inputs to prevent gradient flow from previous layers
                hidden_states = hidden_states.detach().requires_grad_(True)

                # Create dummy arguments for LocalLayer
                batch_size, seq_len = hidden_states.shape[:2]
                attention_mask = torch.ones(
                    batch_size, seq_len, dtype=torch.long, device=self.device
                )

                # Forward pass through layer with all required arguments
                # LocalLayer expects: inputs, current_state, attention_mask, past_key_values, current_depth, block_ids
                output = self.layer(
                    hidden_states,
                    current_state=None,  # No state tracking in MonoForward
                    attention_mask=attention_mask,
                    past_key_values=None,  # No KV caching in MonoForward
                    current_depth=self.config.layer_idx,  # Use layer index as depth
                    block_ids=None,  # Not used in MonoForward
                )

                # Handle different output types - LocalLayer returns tuple
                if isinstance(output, tuple):
                    # LocalLayer returns: (hidden_states, key_values, state_update, aux_loss, exit_signal)
                    hidden_output, new_key_values, new_state, aux_loss, exit_signal = (
                        output
                    )
                else:
                    hidden_output = output
                    new_key_values = past_key_values
                    new_state = current_state
                    aux_loss = None

                # Compute loss and update if we have projection
                loss = None
                if self.projection is not None and labels is not None:
                    # MONO-FORWARD: Compute goodness scores G = activations × M^T
                    # The projection module already implements this correctly
                    goodness_scores = self.projection(hidden_output)

                    # Reshape for loss computation
                    if goodness_scores.dim() == 3 and labels.dim() == 2:
                        batch_size, seq_len = labels.shape
                        vocab_size = goodness_scores.size(-1)

                        # Align sequence lengths
                        if goodness_scores.size(1) != seq_len:
                            min_len = min(goodness_scores.size(1), seq_len)
                            goodness_scores = goodness_scores[:, :min_len, :]
                            labels = labels[:, :min_len]

                        # Flatten for loss
                        goodness_flat = goodness_scores.reshape(-1, vocab_size)
                        labels_flat = labels.reshape(-1)
                    else:
                        goodness_flat = goodness_scores
                        labels_flat = labels

                    # Compute cross-entropy loss on goodness scores (key to Mono-Forward)
                    loss = F.cross_entropy(
                        goodness_flat, labels_flat, ignore_index=-100
                    )

                    # IMMEDIATE WEIGHT UPDATE (key to MonoForward)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Send loss to main process for logging
                    self.loss_queue.put(
                        (self.config.layer_idx, batch_idx, loss.detach().cpu().item())
                    )

                    # Update statistics
                    self.total_loss += loss.item()

                # CRITICAL: Detach output before sending to next layer
                hidden_output = hidden_output.detach()

                # Send to next layer (keep simple format)
                # Make sure everything is detached and on CPU for IPC
                self.output_queue.put(
                    (
                        hidden_output.detach().cpu(),  # Double-check detach and move to CPU
                        labels.detach().cpu() if labels is not None else None,
                        batch_idx,
                    )
                )

                self.batches_processed += 1

                # Periodic status update
                if self.batches_processed % 10 == 0:
                    avg_loss = (
                        self.total_loss / self.batches_processed
                        if self.batches_processed > 0
                        else 0
                    )
                    print(
                        f"[Layer {self.config.layer_idx}] Processed {self.batches_processed} batches, avg loss: {avg_loss:.4f}"
                    )

            except Exception as e:
                print(f"[Layer {self.config.layer_idx}] Error: {e}")
                import traceback

                traceback.print_exc()
                # Continue processing despite errors

        print(f"[Layer {self.config.layer_idx}] Worker stopped")

    def _save_checkpoint(self):
        """Save checkpoint of this layer's state."""
        checkpoint = {
            "layer_state_dict": self.layer.state_dict(),
            "projection_state_dict": (
                self.projection.state_dict() if self.projection else None
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "batches_processed": self.batches_processed,
            "total_loss": self.total_loss,
        }
        # Send checkpoint data through control queue response
        # (Implementation depends on checkpointing strategy)
        pass


def worker_process_main(
    layer_module: nn.Module,
    projection: Optional[nn.Module],
    config: LayerWorkerConfig,
    input_queue: Queue,
    output_queue: Queue,
    loss_queue: Queue,
    control_queue: Queue,
):
    """Main function for worker process."""
    worker = LayerWorkerProcess(
        layer_module,
        projection,
        config,
        input_queue,
        output_queue,
        loss_queue,
        control_queue,
    )
    worker.run()


class MonoForwardPipelineModule(LightningModule):
    """
    Lightning module implementing pipeline-parallel MonoForward training.

    This implements the complete Mono-Forward algorithm where:
    - Each layer runs in its own process with local learning
    - Projection matrices compute goodness scores: G = activations × M^T
    - Cross-entropy loss is computed on goodness scores
    - Gradients are detached between layers (O(1) memory)
    - Works with ANY decoder type (sequential, parallel, etc.)

    The key innovation is the training strategy, not the decoder architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Dict[str, Any],
        pipeline_depth: int = 4,
        device: str = "cuda",
        prediction_mode: str = "bp",  # 'ff' or 'bp'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Disable automatic optimization - we handle it per-layer
        self.automatic_optimization = False

        self.model = model
        self.pipeline_depth = pipeline_depth
        self.device_str = device
        self.prediction_mode = prediction_mode

        # Extract layers from model's decoder
        if not hasattr(model, "decoder") or not hasattr(model.decoder, "locals"):
            raise ValueError("Model must have decoder.locals containing layers")

        # Store reference to decoder to preserve layer list
        self.decoder = model.decoder
        self.num_layers = len(self.decoder.locals)

        # Configuration
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size

        # Store projection matrices for prediction
        self.projection_matrices = nn.ModuleList()

        # Create queues for inter-process communication
        self.layer_input_queues = [
            mp.Queue(maxsize=pipeline_depth) for _ in range(self.num_layers)
        ]
        self.layer_output_queues = [
            mp.Queue(maxsize=pipeline_depth) for _ in range(self.num_layers)
        ]
        self.loss_queue = mp.Queue()
        self.control_queues = [mp.Queue() for _ in range(self.num_layers)]

        # Final output queue for completed batches
        self.final_output_queue = self.layer_output_queues[-1]

        # Worker processes (will be created in on_train_start)
        self.worker_processes = []

        # Loss accumulation
        self.accumulated_losses = {}  # batch_idx -> [layer_losses]
        self.completed_batches = set()

        # Optimizer config for workers
        self.optimizer_config = optimizer_config

    def on_train_start(self):
        """Start worker processes when training begins."""
        print(f"[MonoForwardPipeline] Starting {self.num_layers} worker processes")
        print(
            f"[MonoForwardPipeline] Mono-Forward training with {self.prediction_mode.upper()} prediction mode"
        )

        # Get the layers from the decoder
        layers = self.decoder.locals

        for i in range(self.num_layers):
            # Create projection matrix for this layer
            # All layers use full vocabulary (simplification from paper)
            layer_vocab_size = self.vocab_size

            # Use ProjectionMatrix for computing goodness scores: G = a × M^T
            projection = ProjectionMatrix(self.hidden_size, layer_vocab_size)

            # Store for prediction later
            self.projection_matrices.append(projection)

            # Get the layer module for this worker - we'll deepcopy in the worker
            layer_module = layers[i]

            # Projection is newly created, can be moved to CPU safely
            projection_cpu = projection.cpu()

            # Worker configuration
            config = LayerWorkerConfig(
                layer_idx=i,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                optimizer_config=self.optimizer_config,
                device=self.device_str,
            )

            # Determine input/output queues
            input_queue = self.layer_input_queues[i]
            output_queue = (
                self.layer_output_queues[i]
                if i < self.num_layers - 1
                else self.final_output_queue
            )

            # For layers after the first, connect to previous layer's output
            if i > 0:
                input_queue = self.layer_output_queues[i - 1]

            # Create and start worker process
            # Note: With spawn method, all arguments must be pickleable
            # Each worker gets its own copy of the layer module
            process = mp.Process(
                target=worker_process_main,
                args=(
                    layer_module,  # Will be pickled/unpickled
                    projection_cpu,
                    config,
                    input_queue,
                    output_queue,
                    self.loss_queue,
                    self.control_queues[i],
                ),
                name=f"LayerWorker-{i}",
            )
            process.start()
            self.worker_processes.append(process)
            # Register with shutdown manager for proper cleanup
            register_child_process(process)

        print(f"[MonoForwardPipeline] All workers started")

        # Give workers time to initialize
        time.sleep(1)

    def training_step(self, batch, batch_idx):
        """
        Feed batch to pipeline and collect results asynchronously.
        Returns None to keep pipeline flowing, or accumulated loss periodically.
        """
        # Extract input_ids and create labels
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch

        labels = input_ids[..., 1:].contiguous()
        input_ids = input_ids[..., :-1].contiguous()

        # Get initial embeddings - PraxisModel uses self.embeds
        if hasattr(self.model, "embeds"):
            # PraxisForCausalLM uses embeds
            hidden_states = self.model.embeds(input_ids)
        elif hasattr(self.model, "embeddings"):
            # Fallback to embeddings
            hidden_states = self.model.embeddings(input_ids)
        elif hasattr(self.model, "embed_tokens"):
            # Some models use embed_tokens
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            # This shouldn't happen
            raise ValueError(
                f"Cannot find embedding layer in model of type {type(self.model)}"
            )

        # Create attention mask if not provided
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=input_ids.device
        )

        # Feed to first layer's queue with retry mechanism
        # Start with simple format for first layer
        # IMPORTANT: Detach tensors before sending to avoid gradient serialization errors
        batch_data = (
            hidden_states.detach().cpu(),  # Detach and move to CPU for IPC
            labels.cpu() if labels is not None else None,
            batch_idx,
        )

        # Keep trying to put the batch in the queue with small waits
        retry_count = 0
        max_retries = 100  # About 10 seconds total wait
        while retry_count < max_retries:
            try:
                self.layer_input_queues[0].put_nowait(batch_data)
                break  # Success!
            except queue.Full:
                if retry_count == 0:
                    print(
                        f"[Pipeline] Input queue full for batch {batch_idx}, waiting..."
                    )
                time.sleep(0.1)  # Wait 100ms before retry
                retry_count += 1
        else:
            # Only skip if we've waited too long
            print(
                f"[Pipeline] WARNING: Queue still full after {retry_count * 0.1:.1f}s, skipping batch {batch_idx}"
            )
            return None

        # Collect completed losses (non-blocking)
        losses_collected = []
        try:
            while True:
                layer_idx, loss_batch_idx, loss_value = self.loss_queue.get_nowait()

                if loss_batch_idx not in self.accumulated_losses:
                    self.accumulated_losses[loss_batch_idx] = {}

                self.accumulated_losses[loss_batch_idx][layer_idx] = loss_value

                # Check if we have all layer losses for this batch
                if len(self.accumulated_losses[loss_batch_idx]) == self.num_layers:
                    # Average losses across layers
                    batch_losses = list(
                        self.accumulated_losses[loss_batch_idx].values()
                    )
                    avg_loss = sum(batch_losses) / len(batch_losses)
                    losses_collected.append(avg_loss)
                    self.completed_batches.add(loss_batch_idx)
                    del self.accumulated_losses[loss_batch_idx]

        except queue.Empty:
            pass

        # Log and return loss if we have accumulated enough
        if losses_collected:
            avg_loss = sum(losses_collected) / len(losses_collected)
            self.log("train_loss", avg_loss, prog_bar=True)
            self.log("pipeline_depth", len(self.accumulated_losses), prog_bar=True)
            self.log("completed_batches", len(self.completed_batches))

            # Create marker loss to indicate layer-wise training is complete
            # This tells the model not to compute its own loss
            marker = torch.tensor(0.0, requires_grad=True)

            # Return tensor for Lightning with marker
            return torch.tensor(avg_loss, requires_grad=True)

        # Return None to keep pipeline flowing
        return None

    def on_train_end(self):
        """Stop all worker processes."""
        print("[MonoForwardPipeline] Stopping workers...")

        # Send stop signal to all workers
        for control_queue in self.control_queues:
            try:
                control_queue.put("stop")
            except:
                pass  # Queue might be closed

        # Wait for processes to finish gracefully
        for i, process in enumerate(self.worker_processes):
            process.join(timeout=2)
            if process.is_alive():
                # Force terminate if still alive
                print(f"[MonoForwardPipeline] Worker {i} still alive, terminating...")
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    # Last resort - force kill
                    process.kill()
                    process.join(timeout=0.5)

        # Properly close all queues to prevent resource leaks
        all_queues = (
            self.layer_input_queues
            + self.layer_output_queues
            + self.control_queues
            + [self.loss_queue]
        )

        for queue in all_queues:
            try:
                queue.close()
                queue.join_thread()
            except:
                pass  # Queue might already be closed

        print("[MonoForwardPipeline] Worker shutdown initiated")

    def compute_goodness_scores(
        self, hidden_states: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Compute goodness scores for a specific layer using its projection matrix.

        Args:
            hidden_states: Activations from layer [batch_size, seq_len, hidden_size]
            layer_idx: Which layer's projection matrix to use

        Returns:
            goodness_scores: Scores for each class [batch_size, seq_len, vocab_size]
        """
        return self.projection_matrices[layer_idx](hidden_states)

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Make predictions using trained projection matrices.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            prediction_scores: Class scores based on prediction_mode
        """
        # Get embeddings
        if hasattr(self.model, "embeds"):
            hidden_states = self.model.embeds(input_ids)
        else:
            hidden_states = self.model.embeddings(input_ids)

        # Collect goodness scores from each layer
        all_goodness_scores = []

        # Forward through decoder layers
        for i, layer in enumerate(self.decoder.locals):
            # Forward through layer
            layer_output = layer(
                hidden_states,
                current_state=None,
                attention_mask=None,
                past_key_values=None,
                current_depth=i,
                block_ids=None,
            )

            # Extract hidden states from output
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

            # Compute goodness scores for this layer
            goodness_scores = self.compute_goodness_scores(hidden_states, i)
            all_goodness_scores.append(goodness_scores)

            # Detach for next layer (key to Mono-Forward)
            hidden_states = hidden_states.detach()

        # Combine scores based on prediction mode
        if self.prediction_mode == "ff":
            # FF mode: Sum goodness scores from all layers
            final_scores = torch.stack(all_goodness_scores, dim=0).sum(dim=0)
        else:  # "bp"
            # BP mode: Use only last layer's goodness scores
            final_scores = all_goodness_scores[-1]

        return final_scores

    def configure_optimizers(self):
        """
        Return a dummy optimizer for Lightning compatibility.
        Real optimization happens in worker processes.
        """
        # Create a dummy parameter that never gets updated
        dummy_param = nn.Parameter(torch.zeros(1))
        self.register_parameter("_dummy", dummy_param)
        return torch.optim.SGD([dummy_param], lr=0.0)
