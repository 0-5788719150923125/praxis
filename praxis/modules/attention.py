import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


class PraxisAttention(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.causal = config.causal
        self.max_seq_len = config.context_length
        self.hidden_size = config.n_dim
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.query = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.key = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.output = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        self.register_buffer("slopes", slopes)

        # Store positions up to max_seq_len
        self.register_buffer(
            "positions", torch.arange(self.max_seq_len, dtype=torch.float32)
        )
        # slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        # positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        # alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * positions.unsqueeze(
        #     0
        # ).unsqueeze(0)
        # self.register_buffer("alibi_bias", alibi_bias)

    def forward(self, inputs, attention_mask=None, token_indices=None):
        batch_size, seq_len, _ = inputs.size()
        q = (
            self.query(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * torch.rsqrt(
            torch.tensor(self.head_dim, device=inputs.device)
        )

        # Compute ALiBi biases
        if token_indices is not None:
            # positions: [batch_size, seq_len]
            positions = self.positions[token_indices]
        else:
            # positions: [batch_size, seq_len]
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )

        # Compute position differences
        position_differences = positions.unsqueeze(2) - positions.unsqueeze(
            1
        )  # [batch_size, seq_len, seq_len]

        # Compute biases
        slopes = self.slopes.view(1, self.num_heads, 1, 1)  # [1, num_heads, 1, 1]
        biases = slopes * position_differences.unsqueeze(
            1
        )  # [batch_size, num_heads, seq_len, seq_len]

        # Subtract biases from the scores
        scores -= biases

        # Apply the causal mask
        if self.causal:
            causal_mask = (
                torch.triu(
                    torch.ones(seq_len, seq_len, device=inputs.device) * float("-inf"),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores += causal_mask

        if attention_mask is not None:
            scores *= attention_mask.unsqueeze(1).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        attention = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.hidden_size)
        )

        return self.output(attention)


# @torch.jit.script
# def stickbreaking_att(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     mask: torch.Tensor,
#     cum_weight: torch.Tensor,
#     att_mask: Optional[torch.FloatTensor] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     logits = torch.einsum("bkhid,bkhjd->bkhij", q, k) / math.sqrt(k.size(-1))

#     # Instead of using bool(), we'll use a comparison
#     mask = mask == 0

#     if att_mask is not None:
#         logits = logits + att_mask

#     z = torch.sigmoid(logits).masked_fill(mask, 0)
#     log_beta = F.logsigmoid(-logits).masked_fill(mask, 0)
#     re_cum_log_beta = torch.einsum("bkhij,bkhij->bkhij", log_beta, cum_weight)
#     att = z * torch.exp(re_cum_log_beta)
#     y = torch.einsum("bkhij,bkhjd->bkhid", att, v)
#     return y, att


# class PraxisAttention(nn.Module):
#     def __init__(self, config):
#         """
#         Initialize the PraxisAttention module.

#         Args:
#             config: Configuration object with model hyperparameters.
#         """
#         super().__init__()

#         # history_length = 512
#         # block_size = 512
#         # self.context_length = history_length + block_size
#         self.context_length = 1024
#         self.n_head = config.n_head
#         self.top_k = 2
#         self.n_dim = config.n_dim
#         self.head_dim = self.n_dim // self.n_head

#         # self.query = MoE(
#         #     input_size=config.n_embd,
#         #     head_size=config.att_hidden,
#         #     num_experts=config.n_att_experts,
#         #     top_k=config.k_att,
#         #     acc_aux_loss=config.acc_aux_loss,
#         #     bias=False,
#         #     gating_dropout=config.moe_pdrop,
#         #     sample_topk=config.sample_topk,
#         #     gating_size=config.gating_size,
#         #     aux_loss=config.aux_loss_type,
#         #     gate_type=config.gate_type,
#         # )
#         self.query = nn.Linear(
#             self.n_dim, self.n_head * self.head_dim * self.top_k, bias=True
#         )
#         if self.head_dim == config.n_emb and config.n_head == 1:
#             self.key = nn.Identity()
#             self.value = nn.Identity()
#         else:
#             self.key = nn.Linear(self.n_dim, self.n_head * self.head_dim)
#             self.value = nn.Linear(self.n_dim, self.n_head * self.head_dim)

#         self.output = nn.Linear(
#             self.n_head * self.head_dim * self.top_k, self.n_dim, bias=False
#         )

#         # regularization
#         # self.attn_dropout = nn.Dropout(config.attn_pdrop)
#         # causal mask to ensure that attention is only applied to the left in the input sequence

#         self.register_buffer(
#             "mask",
#             torch.tril(
#                 torch.ones(self.context_length, self.context_length, dtype=torch.int8)
#             ),
#         )
#         self.register_buffer(
#             "cum_weight",
#             torch.tril(torch.ones(self.context_length, self.context_length), -1),
#         )

#     def add_history(self, k, v, hidden, use_cache=False):
#         """
#         Add history to key and value tensors.

#         Args:
#             k (torch.Tensor): Key tensor.
#             v (torch.Tensor): Value tensor.
#             hidden: Hidden state.
#             use_cache (bool): Whether to use cached history.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Updated key, value, and history.
#         """
#         if hidden is None or not use_cache:
#             new_k = k
#             new_v = v
#         else:
#             k_history, v_history = hidden
#             new_k = torch.cat([k_history, k], dim=1)
#             new_v = torch.cat([v_history, v], dim=1)
#         k_history = new_k.detach()
#         v_history = new_v.detach()

#         return new_k, new_v, (k_history, v_history)

#     def forward(
#         self,
#         hidden_states: Optional[torch.FloatTensor],
#         attention_mask: Optional[torch.FloatTensor] = None,
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Union[
#         Tuple[torch.Tensor, Tuple[torch.Tensor]],
#         Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
#     ]:
#         """
#         Forward pass of the ModuleFormerAttention module.

#         Args:
#             hidden_states (Optional[torch.FloatTensor]): Input hidden states.
#             attention_mask (Optional[torch.FloatTensor]): Attention mask.
#             layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
#             head_mask (Optional[torch.FloatTensor]): Head mask.
#             use_cache (Optional[bool]): Whether to use cached states.
#             output_attentions (Optional[bool]): Whether to output attention weights.

#         Returns:
#             Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[...]]]: Tuple containing outputs.
#         """

#         if hidden_states.dim() == 2:
#             hidden_states = hidden_states.unsqueeze(0)
#         B, T, C = (
#             hidden_states.size()
#         )  # batch size, sequence length, embedding dimensionality (n_embd)

#         # print(f"Input hidden_states shape: {hidden_states.shape}")
#         # print(
#         #     f"self.n_dim: {self.n_dim}, self.n_head: {self.n_head}, self.head_dim: {self.head_dim}, self.top_k: {self.top_k}"
#         # )

#         # calculate query, key, values
#         # q, aux_loss = self.query.map(hidden_states)
#         # q, aux_loss = self.query(hidden_states)
#         q = self.query(hidden_states)
#         k = self.key(hidden_states)
#         v = self.value(hidden_states)

#         k, v, hidden = self.add_history(k, v, layer_past, use_cache)
#         context_length = k.size(1)

#         q = q.view(B, T, self.top_k, self.n_head, self.head_dim)  # (B, T, k, nh, hs)
#         k = k.view(B, context_length, self.n_head, self.head_dim)  # (B, T, nh, hs)
#         v = v.view(B, context_length, self.n_head, self.head_dim)  # (B, T, nh, hs)

#         mask = torch.tril(
#             torch.ones(T, context_length, dtype=torch.int8, device=q.device)
#         )
#         cum_weight = torch.tril(
#             torch.ones(T, context_length, device=q.device), -1
#         ).type_as(q)

#         # print(f"Hidden states shape: {hidden_states.shape}")
#         # print(f"Original attention_mask shape: {attention_mask.shape}")
#         # print(f"Reshaped attention_mask shape: {attention_mask.shape}")
#         # print(f"q shape: {q.shape}")
#         # print(f"k shape: {k.shape}")
#         # print(f"v shape: {v.shape}")
#         # print(f"mask shape: {mask.shape}")
#         # print(f"cum_weight shape: {cum_weight.shape}")

#         if attention_mask is not None:
#             # Reshape attention_mask to match logits dimensions
#             attention_mask = (
#                 attention_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
#             )  # Add dimensions for top_k and n_head
#             attention_mask = attention_mask[
#                 :, :, :, :T, :context_length
#             ]  # Slice to match the current sequence length and context length
#             attention_mask = attention_mask.expand(
#                 B, self.top_k, self.n_head, T, context_length
#             )
#             attention_mask = attention_mask.contiguous()
#         else:
#             attention_mask = torch.ones(
#                 (B, self.top_k, self.n_head, T, context_length), device=q.device
#             )

#         # print(f"B: {B}, T: {T}, context_length: {context_length}")
#         # print(f"attention_mask shape before expansion: {attention_mask.shape}")

#         # Modify k and v to include the top_k dimension
#         k = k.unsqueeze(2).expand(
#             B, context_length, self.top_k, self.n_head, self.head_dim
#         )
#         v = v.unsqueeze(2).expand(
#             B, context_length, self.top_k, self.n_head, self.head_dim
#         )

#         # Transpose dimensions for stickbreaking_att
#         q = q.permute(0, 2, 3, 1, 4)  # [B, top_k, n_head, T, head_dim]
#         k = k.permute(0, 2, 3, 1, 4)  # [B, top_k, n_head, context_length, head_dim]
#         v = v.permute(0, 2, 3, 1, 4)  # [B, top_k, n_head, context_length, head_dim]

#         # Modify mask and cum_weight to include batch, top_k, and n_head dimensions
#         mask = mask.view(1, 1, 1, T, context_length).expand(
#             B, self.top_k, self.n_head, T, context_length
#         )
#         cum_weight = cum_weight.view(1, 1, 1, T, context_length).expand(
#             B, self.top_k, self.n_head, T, context_length
#         )

#         y, attn_weights = stickbreaking_att(
#             q, k, v, mask=mask, cum_weight=cum_weight, att_mask=attention_mask
#         )

#         # Reshape y back to the original dimensions
#         y = y.permute(0, 3, 1, 2, 4).contiguous()
#         # print(f"After attention, y shape: {y.shape}")

#         # Reshape y based on input dimensions
#         if hidden_states.dim() == 2:
#             y = y.view(T, C)
#         elif hidden_states.dim() == 3:
#             y = y.reshape(B, T, C)
#         else:
#             raise ValueError(f"Unexpected input shape: {hidden_states.shape}")

#         # Apply output projection
#         y = self.output(y)
#         # print(f"After output projection, y shape: {y.shape}")

#         outputs = y
#         return outputs


# def reshape_for_broadcast(self, freqs: torch.Tensor, x: torch.Tensor):
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     assert freqs.shape == (x.shape[1], x.shape[-1])
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs.view(*shape)
