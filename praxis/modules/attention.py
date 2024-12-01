import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.encoding import ENCODING_REGISTRY
from praxis.modules.memory import PraxisCompressiveMemory


class PraxisAttention(nn.Module):
    """
    This class is akin to a wrapper, which implements a number of interesting attention
    mechanisms, and makes them optional with feature flags. By toggling features, one can
    essentially blend components from various kinds of attention.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.causal = config.causal
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.head_dim = self.hidden_size // self.num_heads

        # Set the core attention mechanism
        self.linear = config.linear
        self.differential = config.differential
        self.stickbreaking = config.stickbreaking
        assert (
            sum([self.differential, self.stickbreaking, self.linear]) <= 1
        ), "Only one of differential, stickbreaking, or linear attention can be used at a time."

        self.gating = config.mega
        if self.gating:
            self.gating = PraxisGatedEMA(config)

        # Query and key projections for differential heads
        self.multiplier = 2 if self.differential else 1
        self.query = nn.Linear(
            self.hidden_size,
            self.num_query_heads * self.head_dim * self.multiplier,
            bias=False,
        )
        self.key = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.multiplier,
            bias=False,
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        self.memory = config.memory
        self.chunk_size = 0
        if self.memory:
            self.chunk_size = 256
            self.memory = PraxisCompressiveMemory(config)

        # The core attention mechanism
        scale_encoding = True
        if self.stickbreaking:
            self.algorithm = Stickbreaking(config)
            config.encoding = "nope"
            scale_encoding = False
        elif self.differential:
            self.algorithm = Differential(config)
        elif self.linear:
            self.algorithm = LinearAttention(config)
        else:
            self.algorithm = ScaledDotProduct(config)

        # For handling length extrapolation
        self.encoding = ENCODING_REGISTRY[config.encoding](config, scale_encoding)

        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

        # Standard output projection
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.shape

        if self.gating:
            # Compute an exponential moving average-based gating mechanism
            inputs = self.gating(inputs)

        # Determine chunk size
        chunk_size = self.chunk_size if self.chunk_size > 0 else seq_len
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Initialize QKV projections
        q = (
            self.query(inputs)
            .view(batch_size, -1, self.num_query_heads, self.head_dim * self.multiplier)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim * self.multiplier)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        outputs = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            current_chunk_size = end_idx - start_idx

            # Extract current chunk
            chunk_q = q[:, :, start_idx:end_idx]
            chunk_k = k[:, :, start_idx:end_idx]
            chunk_v = v[:, :, start_idx:end_idx]
            chunk_mask = attention_mask[:, start_idx:end_idx]

            # Process chunk with position offset
            chunk_output = self._process_chunk(
                chunk_q,
                chunk_k,
                chunk_v,
                chunk_mask,
                current_chunk_size,
                start_idx,
            )

            outputs.append(chunk_output)

        # Concatenate all chunks
        output = torch.cat(outputs, dim=1)

        if self.memory:
            self.memory.reset_states()

        # Final output projection
        return self.output(output)

    def _process_chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor,
        chunk_size: int,
        offset: int = 0,
    ) -> Tensor:
        batch_size = q.size(0)

        # Handle multiple queries
        if self.num_queries > 1:
            k = k.repeat_interleave(self.num_queries, dim=1)
            v = v.repeat_interleave(self.num_queries, dim=1)

        # Apply positional encoding with offset
        q, k, v = self.encoding.before_scores(q, k, v, offset=offset)

        # Compute attention scores
        q, k, v, scores = self.algorithm.compute_scores(q, k, v)
        hist_len = k.size(2)

        # Apply positional encoding to scores
        scores = self.encoding.after_scores(scores, offset=offset)

        # Apply masking
        scores, causal_mask, chunk_attention_mask = self.algorithm.apply_masking(
            scores, attention_mask, chunk_size, hist_len, self.causal
        )

        # Get attention output
        attention_output = self.algorithm.compute_weights(
            q, k, v, scores, causal_mask, chunk_attention_mask
        )

        if self.memory:
            # Get memory output and blend (memory doesn't need position info)
            attention_output = self.memory(q, k, v, attention_output)

        attention_output = self.dropout(attention_output)

        # Reshape for output projection
        chunk_output = attention_output.transpose(1, 2).reshape(
            batch_size, chunk_size, -1
        )

        return chunk_output


class ScaledDotProduct(nn.Module):
    """
    This class implements scaled dot-product attention:
    https://paperswithcode.com/method/scaled
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.num_query_heads = self.num_heads * config.num_queries
        self.head_dim = self.hidden_size // self.num_heads

    def _compute_score(self, q, k):
        scaling = 1.0 / math.sqrt(self.head_dim)
        return torch.matmul(q, k.transpose(-2, -1)) * scaling

    def compute_scores(self, q, k, v):
        scores = [self._compute_score(q, k)]
        return q, k, v, scores

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores,
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        weights = [F.softmax(score, dim=-1) for score in scores]
        return self._compute_outputs(weights[0], v)

    def _compute_outputs(self, weights, v):
        # Compute attention output
        return weights @ v  # Shape: (batch_size, num_heads, seq_len, head_dim)

    def apply_masking(self, scores, attention_mask, seq_len, hist_len, causal):
        causal_mask = None
        if causal:
            causal_mask = (
                torch.triu(
                    torch.full((seq_len, hist_len), -1e9, device=scores[0].device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = [score + causal_mask for score in scores]

        # Expand attention mask to historical length
        attention_mask = F.pad(
            attention_mask, (hist_len - attention_mask.size(-1), 0), value=1
        )

        attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        scores = [score + attention_mask for score in scores]
        return scores, causal_mask, attention_mask


class LinearAttention(ScaledDotProduct):
    """
    Implements Linear Attention using kernel feature maps.
    Based on 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention'
    """

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.epsilon = 1e-6
        self.causal = config.causal

        # Feature map for positive definite kernel
        self.feature_map = lambda x: F.elu(x) + 1

    def compute_scores(self, q: Tensor, k: Tensor, v: Tensor):
        """
        Instead of returning attention scores, we return the feature-mapped queries and keys.
        """
        # Apply the feature map
        q = self.feature_map(q)  # Shape: (B, H, L, D)
        k = self.feature_map(k)  # Shape: (B, H, L, D)

        return q, k, v, []  # Return an empty list for scores

    def apply_masking(self, scores, attention_mask, seq_len, hist_len, causal):
        return scores, None, attention_mask

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores: List[Tensor],
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        """
        Perform the linear attention computation here, using the feature-mapped q and k.
        """
        # Now, perform the linear attention computation
        B, H, L, D = v.size()

        # Apply mask to k and v if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # Shape: (B, 1, L, 1)
            k = k * mask
            v = v * mask

        if self.causal:
            # Implement causal linear attention using cumulative sums
            k_cumsum = torch.cumsum(k.transpose(2, 1), dim=2).transpose(
                2, 1
            )  # (B, H, L, D)
            kv_cumsum = torch.cumsum((k * v).transpose(2, 1), dim=2).transpose(
                2, 1
            )  # (B, H, L, D)

            # Compute denominator z
            z = torch.einsum("bhld,bhld->bhl", q, k_cumsum) + self.epsilon  # (B, H, L)

            # Apply attention mask to z
            if attention_mask is not None:
                z = z * attention_mask.unsqueeze(1)  # Shape: (B, 1, L)

            # Compute numerator
            output = torch.einsum("bhld,bhld->bhld", q, kv_cumsum)  # (B, H, L, D)
        else:
            # Non-causal linear attention
            k_sum = k.sum(dim=2)  # (B, H, D)
            kv_sum = torch.einsum("bhld,bhld->bhd", k, v)  # (B, H, D)

            # Compute denominator z
            z = torch.einsum("bhld,bhd->bhl", q, k_sum) + self.epsilon  # (B, H, L)

            # Apply attention mask to z
            if attention_mask is not None:
                z = z * attention_mask.unsqueeze(1)  # Shape: (B, 1, L)

            # Compute numerator
            output = torch.einsum("bhld,bhd->bhld", q, kv_sum)  # (B, H, L, D)

        # Normalize output
        output = output / z.unsqueeze(-1)  # (B, H, L, D)

        return output


class Differential(ScaledDotProduct):
    """
    This class implements Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.lambda_init = 0.8  # A good default, per the paper
        self.lambdas = nn.ParameterDict(
            dict(
                q1=nn.Parameter(torch.randn(self.head_dim)),
                q2=nn.Parameter(torch.randn(self.head_dim)),
                k1=nn.Parameter(torch.randn(self.head_dim)),
                k2=nn.Parameter(torch.randn(self.head_dim)),
            )
        )
        self.norm = nn.GroupNorm(
            num_groups=self.num_query_heads,
            num_channels=self.num_query_heads * self.head_dim,
            eps=config.epsilon,
        )

    def compute_scores(self, q, k, v):
        # Split queries and keys
        q_chunks = q.chunk(q.size(-1) // self.head_dim, dim=-1)
        k_chunks = k.chunk(k.size(-1) // self.head_dim, dim=-1)
        # Compute differential attention scores
        scores = [
            self._compute_score(q_chunks[i], k_chunks[i]) for i in range(len(q_chunks))
        ]
        return q, k, v, scores

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores,
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        weights = [F.softmax(score, dim=-1) for score in scores]
        # Compute scalar lambda
        lambda_scalar = (
            torch.exp(torch.dot(self.lambdas["q1"], self.lambdas["k1"]))
            - torch.exp(torch.dot(self.lambdas["q2"], self.lambdas["k2"]))
            + self.lambda_init
        )
        weights = weights[0] - lambda_scalar * weights[1]
        outputs = self._compute_outputs(weights, v)
        return self._normalize_outputs(outputs)

    def _normalize_outputs(self, weights):
        batch_size, num_heads, seq_len, head_dim = weights.shape
        # Reshape for GroupNorm
        attention_output = (
            weights.permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, num_heads * head_dim)
            .permute(0, 2, 1)
            .contiguous()
        )  # Shape: (batch_size, num_heads * head_dim, seq_len)
        # Apply GroupNorm
        attention_output = self.norm(attention_output)
        # Permute to original shape
        attention_output = (
            attention_output.permute(0, 2, 1)
            .view(batch_size, seq_len, num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)
        # Apply scaling factor
        attention_output = attention_output * (1 - self.lambda_init)
        return attention_output


class Stickbreaking(ScaledDotProduct):
    """
    Implements Stickbreaking Attention mechanism.
    https://github.com/IBM/ModuleFormer
    """

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.register_buffer("key_history", None)
        self.register_buffer("value_history", None)
        self.capacity = config.capacity
        self.use_history = True

    def compute_scores(self, q, k, v):
        if self.training and self.use_history:
            k, v = self._update_history(k, v)
        return super().compute_scores(q, k, v)

    def compute_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scores: List[Tensor],
        causal_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        logits = scores[0]
        batch_size, num_heads, seq_len, hist_len = logits.shape

        # Get cumulative weight matrix of appropriate size and expand it
        cum_weight = torch.tril(torch.ones(hist_len, hist_len, device=logits.device))

        # Compute stick-breaking weights
        z = torch.sigmoid(logits)
        log_beta = F.logsigmoid(-logits)
        if causal_mask is not None:
            z = z + causal_mask
            log_beta = log_beta + causal_mask

        # Compute cumulative log beta terms
        re_cum_log_beta = torch.einsum(
            "bhij,jk->bhik", log_beta, cum_weight.type_as(logits)
        )

        # Final attention weights
        weights = z * re_cum_log_beta.exp()

        return self._compute_outputs(weights, v)

    def _get_random_slice(self, tensor: Tensor) -> Tensor:
        """Get random slice of history with size history_slice_size"""
        _, _, seq_len, _ = tensor.shape
        seg_len = int(seq_len * self.capacity)
        if seq_len <= seg_len:
            return tensor

        # Random starting point that ensures we can get history_slice_size tokens
        start_idx = torch.randint(0, seq_len - seg_len + 1, (1,)).item()
        return tensor[:, :, start_idx : start_idx + seg_len, :]

    def _update_history(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        # First forward pass - initialize history
        if self.key_history is None or self.value_history is None:
            self.key_history = k.detach()
            self.value_history = v.detach()
            return k, v

        # Get current and history batch sizes
        curr_batch = k.size(0)
        hist_batch = self.key_history.size(0)

        # If current batch is smaller than history batch,
        # this means we have a longer sequence - return unmodified
        if curr_batch < hist_batch:
            return k, v

        # If current batch is larger than history batch,
        # this means we have shorter sequences - reset history
        if curr_batch > hist_batch:
            self.key_history = k.detach()
            self.value_history = v.detach()
            return k, v

        try:
            # Get random slice from history
            hist_k = self._get_random_slice(self.key_history)
            hist_v = self._get_random_slice(self.value_history)

            # Concatenate [history slice, current sequence]
            new_k = torch.cat([hist_k, k], dim=2)
            new_v = torch.cat([hist_v, v], dim=2)

        except RuntimeError:
            # Safety fallback
            self.key_history = k.detach()
            self.value_history = v.detach()
            return k, v

        # Update history
        self.key_history = new_k.detach()
        self.value_history = new_v.detach()

        return new_k, new_v

    # def _update_history(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    #     """
    #     Update and return concatenated history for keys and values.
    #     Uses batch size as primary decision metric for history management,
    #     since larger batches correspond to smaller sequences that we want to concatenate.
    #     """
    #     # First forward pass - initialize history
    #     if self.key_history is None or self.value_history is None:
    #         self.key_history = k.detach()
    #         self.value_history = v.detach()
    #         return k, v

    #     # Get current and history batch sizes
    #     curr_batch = k.size(0)
    #     hist_batch = self.key_history.size(0)

    #     # If current batch is smaller than history batch,
    #     # this means we have a longer sequence - return unmodified
    #     if curr_batch < hist_batch:
    #         return k, v

    #     # If current batch is larger than history batch,
    #     # this means we have shorter sequences - reset history
    #     if curr_batch > hist_batch:
    #         self.key_history = k.detach()
    #         self.value_history = v.detach()
    #         return k, v

    #     # At this point batch sizes match, safe to concatenate
    #     try:
    #         new_k = torch.cat([self.key_history, k], dim=2)
    #         new_v = torch.cat([self.value_history, v], dim=2)
    #     except RuntimeError:
    #         # Safety fallback
    #         self.key_history = k.detach()
    #         self.value_history = v.detach()
    #         return k, v

    #     # Update history
    #     self.key_history = new_k.detach()
    #     self.value_history = new_v.detach()

    #     return new_k, new_v


class PraxisGatedEMA(nn.Module):
    """
    Inspired by MEGA, this class implements a simple EMA into an attention mechanism,
    encouraging inductive biases in the model.
    Reference: https://arxiv.org/abs/2209.10655
    Original Code: https://github.com/facebookresearch/mega/blob/main/fairseq/modules/exponential_moving_average.py
    """

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.embed_dim = config.num_dims
        self.ndim = 3  # Adjust as needed
        self.scale = math.sqrt(1.0 / self.ndim)

        # Truncation parameter to limit kernel size
        self.truncation = None  # Set to a value like 256 if needed

        # EMA parameters
        self.delta = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.omega = nn.Parameter(torch.Tensor(self.embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def forward(self, x: Tensor) -> Tensor:
        # Compute residual
        residual = x * self.omega  # Shape: (batch_size, seq_len, embed_dim)

        # Compute EMA
        ema_x = self._compute_ema(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Combine EMA output with residual and apply activation function
        y = F.silu(ema_x + residual)  # Shape: (batch_size, seq_len, embed_dim)

        return y

    def _calc_coeffs(self):
        p = torch.sigmoid(self.delta)  # (embed_dim, ndim, 1)
        alpha = torch.sigmoid(self.alpha)  # (embed_dim, ndim, 1)
        q = 1.0 - p * alpha  # (embed_dim, ndim, 1)
        return p, q

    def _compute_kernel(self, seq_len: int) -> Tensor:
        kernel_size = (
            seq_len if self.truncation is None else min(self.truncation, seq_len)
        )
        # Compute coefficients
        p, q = self._calc_coeffs()
        # Compute kernel
        t = torch.arange(kernel_size, device=p.device).view(1, 1, kernel_size)
        log_q = torch.log(q)
        vander = t * log_q  # (embed_dim, ndim, kernel_size)
        kernel = (p * self.beta) * torch.exp(vander)  # (embed_dim, ndim, kernel_size)
        kernel = torch.einsum(
            "dnl,dn->dl", kernel, self.gamma * self.scale
        )  # (embed_dim, kernel_size)
        return kernel

    def _compute_ema(self, x: Tensor) -> Tensor:
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # Compute kernel
        kernel = self._compute_kernel(seq_len)  # (embed_dim, kernel_size)
        kernel_size = kernel.size(1)

        # Zero-pad kernel to match seq_len if necessary
        if kernel_size < seq_len:
            padding = seq_len - kernel_size
            kernel = F.pad(kernel, (0, padding))

        # Perform convolution using FFT
        fft_len = 2 * seq_len
        x_f = torch.fft.rfft(
            x.float(), n=fft_len, dim=2
        )  # (batch_size, embed_dim, fft_len//2+1)
        k_f = torch.fft.rfft(
            kernel.float(), n=fft_len, dim=1
        )  # (embed_dim, fft_len//2+1)

        # Multiply in frequency domain
        y_f = x_f * k_f.unsqueeze(0)  # Broadcasting over batch_size
        y = torch.fft.irfft(y_f, n=fft_len, dim=2)[
            ..., :seq_len
        ]  # (batch_size, embed_dim, seq_len)
        y = y.type_as(x)

        # Transpose back to (batch_size, seq_len, embed_dim)
        y = y.transpose(1, 2)
        return y


def test_memory_scaling():
    BATCH_SIZE = 1
    HIDDEN_SIZE = 512
    NUM_HEADS = 8
    NUM_QUERIES = 2
    SEQUENCE_LENGTHS = [256, 512, 1024, 2048, 4096]
    NUM_RUNS = 3  # Average over multiple runs for stability

    def run_test(config, seq_len):
        # Reset memory tracking
        torch.cuda.reset_peak_memory_stats()

        # Create model
        model = PraxisAttention(config).cuda()

        # Create dummy inputs
        inputs = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE).cuda()
        attention_mask = torch.ones(BATCH_SIZE, seq_len).cuda()

        # Forward pass
        output = model(inputs, attention_mask)

        # Dummy loss and backward
        loss = output.mean()
        loss.backward()

        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB

        # Clear memory
        del model, inputs, attention_mask, output
        torch.cuda.empty_cache()

        return peak_mem

    # Test configurations
    class DummyConfig:
        def __init__(self):
            self.debug = True
            self.causal = True
            self.linear = False
            self.stickbreaking = False
            self.differential = False
            self.dropout = 0.0
            self.encoding = "yarn"
            self.context_length = 8192

    base_config = DummyConfig()
    base_config.num_dims = HIDDEN_SIZE
    base_config.num_heads = NUM_HEADS
    base_config.num_queries = NUM_QUERIES
    base_config.memory = False

    memory_config = DummyConfig()
    memory_config.num_dims = HIDDEN_SIZE
    memory_config.num_heads = NUM_HEADS
    memory_config.num_queries = NUM_QUERIES
    memory_config.memory = True

    print("\nMemory Usage Analysis (in MB):")
    print("=" * 60)
    print(
        f"{'Sequence Length':<15} {'Without Memory':<20} {'With Memory':<20} {'Ratio':<10}"
    )
    print("-" * 60)

    for seq_len in SEQUENCE_LENGTHS:
        base_mems = []
        memory_mems = []

        for _ in range(NUM_RUNS):
            base_mems.append(run_test(base_config, seq_len))
            memory_mems.append(run_test(memory_config, seq_len))

        avg_base = sum(base_mems) / NUM_RUNS
        avg_memory = sum(memory_mems) / NUM_RUNS
        ratio = avg_memory / avg_base

        print(
            f"{seq_len:<15} {avg_base:,.2f}{'MB':<14} {avg_memory:,.2f}{'MB':<14} {ratio:.2f}x"
        )

    print("=" * 60)


if __name__ == "__main__":
    test_memory_scaling()
