import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import Optional, OrderedDict, Tuple, Union
from .configuration_thorns import ThornsConfig


class ThornsModel(PreTrainedModel):
    config_class = ThornsConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([ThornsBlock(config) for _ in range(config.n_layer)])
        self.ln_f = ScaledRMSNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        input_embeds = self.wte(input_ids)
        hidden_states = input_embeds

        # Always create an attention mask if it's not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=hidden_states.device)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states.view(*output_shape),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class ThornsAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
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
        self.out = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.register_buffer("m", self._get_alibi_slope(self.num_heads))
        self.causal = config.causal

    def _get_relative_positions(self, seq_len: int) -> torch.tensor:
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return x - y

    def _get_alibi_slope(self, num_heads):
        x = (2**8) ** (1 / num_heads)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply ALiBi bias
        bias = (self.m * self._get_relative_positions(seq_len)).unsqueeze(0)
        scores = scores - bias

        # Apply the causal mask
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.hidden_size
        )
        attn_output = self.out(attn_output)
        return attn_output


class ThornsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = ScaledRMSNorm(
            config.n_embd, eps=config.layer_norm_epsilon
        )
        self.attention = ThornsAttention(config)
        self.post_attention_layernorm = ScaledRMSNorm(
            config.n_embd, eps=config.layer_norm_epsilon
        )
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("in_proj", nn.Linear(config.n_embd, 4 * config.n_embd)),
                    ("act", nn.Mish()),
                    ("out_proj", nn.Linear(4 * config.n_embd, config.n_embd)),
                ]
            )
        )

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class ScaledRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class ThornsForCausalLM(ThornsModel):
    def __init__(self, config):
        config.causal = True
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)

        if past:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        transformer_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past
        )
