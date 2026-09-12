# Foreign models: training any `transformers` checkpoint in Praxis

Audit + design, 2026-09-11.

The ask: load and train any model available through the standard `transformers`
registry, not just `PraxisForCausalLM`. First proof of concept is
`HuggingFaceTB/SmolLM2-135M`, LoRA-tuned on `Anthropic/hh-rlhf` through the
existing `PreferencePolicy`.

---

## 1. What already works

The seams are mostly in the right places. Nothing below needed changing:

| Surface | Why it already works |
|---|---|
| `assemble_model` | Already `AutoModelForCausalLM.from_config(config)` (`praxis/trainers/setup.py`). |
| `register_praxis_models()` | Already registers Praxis into the `Auto*` mappings. |
| `ModelBackend` (inference) | Written to the `PreTrainedModel` contract - `model.generate`, `.device`, `config.max_position_embeddings`. Zero Praxis assumptions. |
| `hf_native` ChatFormat | Already exists, with foreign-tokenizer detection and the halting contract. |
| Optimizer routing | `_split_muon_params` partitions by tensor shape + `nn.Embedding` membership, never by module name. |
| Dynamics / metrics callbacks | Every extractor is `getattr`/`hasattr` guarded; a foreign model yields an empty dynamics section rather than a crash. |
| Ghost transforms, precision cast, lazy init, `try_compile` | Generic `nn.Module` walks. |
| Web layer | Talks to `Generator` and Flask app config, never to `PraxisModel`. |

Peripheral `config.*` reads outside the model internals are ~15 fields, most of
them HF-standard (`eos_token_id`, `vocab_size`, `hidden_size`, `pad_token_id`,
`tie_word_embeddings`).

## 2. What breaks

Probed against a real `LlamaForCausalLM`, not reasoned about:

1. **Label convention.** Praxis pre-shifts (`labels = input_ids[..., 1:]`); every
   HF model shifts internally. Passing Praxis labels to a Llama raises
   `ValueError: Expected input batch_size (32) to match target batch_size (30)`.
   Where lengths happen to line up it would instead train on off-by-one targets.

2. **Extra forward kwargs are silently swallowed.** transformers 5.x accepts
   `task_type_ids=<tensor>`, `assistant_mask`, `rewards`, `block_ids`,
   `row_continues` without error and drops them. The trainer passes all of them,
   so a naive model swap looks healthy while task weighting, prompt masking and
   every RL policy are quietly off.

3. **The objectives live inside `PraxisForCausalLM.forward`.** `criterion`,
   `tasker`, `policy`, `mtp`, `strategy`, regularizers, `_build_loss_weights`.
   A foreign model has none of it, so `--loss-func`, `--task-weights`,
   `--rl-type`, `--mtp-type`, `--regularizers` all become no-ops - including the
   preference policy the PoC is built on.

4. **The assistant mask goes all-zero.** A template without `{% generation %}`
   makes `return_assistant_tokens_mask=True` return nothing;
   `message_queue.py` fills zeros; `_build_loss_weights` multiplies by zero; and
   `weighted_reduce` returns `0.0` *with the autograd graph intact*. Loss is
   exactly zero, gradients are zero, nothing raises.

5. **No PEFT.** Not a dependency, not referenced anywhere.

## 3. Approaches considered

**A. Swap the model object.** Cheapest. Praxis becomes a training harness around
an opaque HF model and every Praxis mechanism silently degrades. The PoC cannot
be built on it (see breakage 3).

**B. A wrapper that hosts the foreign model and keeps the Praxis objective
path.** Chosen. See below.

**C. Foreign backbone as a `decoders` registry entry.** The decoder contract is
already `hidden_states -> hidden_states`, so `LlamaModel(inputs_embeds=h)` fits.
Very Praxis-native - you keep halting, width and routing slots wrapped around a
pretrained trunk - but it discards the foreign embeddings and output projection
unless matching `embeddings`/`classifiers` entries are registered too. Good
*later* idea ("borrow a pretrained trunk as an ablation arm"); wrong way to run
SmolLM2 as SmolLM2. Kept out of scope deliberately.

## 4. The design (B)

`PraxisForCausalLM.forward` is two things stapled together:

1. `input_ids` -> embeddings -> trunk -> `hidden_states` (Praxis-specific)
2. `hidden_states` + `scorer` + `input_ids` + `labels` -> logits -> objectives,
   policies, MTP, regularizers, conflict metrics

Half 2 is already model-agnostic *in its arguments*. Extracting it into mixins
(`praxis/objectives.py`) makes it reusable, and the foreign wrapper is then
thin:

```
praxis/objectives.py        ObjectiveMixin       task-agnostic: criterion, tasker,
                                                 strategy, conflict, loss weights,
                                                 fold, metric drains
                            CausalObjectiveMixin causal specifics: the label shift,
                                                 recall/RL policies, MTP, aux losses

praxis/models/__init__.py   registry.declare("model_tasks")   causal_lm today;
                                                 seq_cls / masked_lm / seq2seq are
                                                 registry slots, not rewrites
praxis/models/base.py       ForeignModel         hosts any Auto* class; kwargs pass
                                                 through verbatim
praxis/models/causal.py     ForeignCausalLM      ForeignModel + CausalObjectiveMixin
praxis/models/peft.py       registry.declare("peft_profiles")   adapters; the
                                                 optional dependency installs at
                                                 runtime, Ray-style
```

The task dimension is load-bearing: `--model-task` selects both the `Auto*`
class and the wrapper, so adding sequence classification later is one registry
entry plus one mixin, not a second integration.

## 5. Kwargs: pass through, do not translate

Decided: **the foreign model's own kwargs, verbatim. No alias layer.**

The config is the model's identity - it is what lands in `config.json`, feeds the
run hash, and tells someone a year later what ran. An alias layer means `depth`
and `num_hidden_layers` both work, disagree in the artifacts, and eventually one
of them is a lie. Aliases are where a translation framework starts.

So:

- `--model-name org/repo` (+ `--model-revision`) loads the config from the hub,
  untouched.
- `--model-kwarg k=v`, repeatable, forwarded verbatim to `from_pretrained`.
  That is the whole escape hatch.
- Praxis **architecture** flags (`--attention-type`, `--depth`, `--encoder-type`,
  `--transform-type`, ...) are a **hard error** under `--model-name`, not
  ignored. The registry-is-the-architecture contract cannot hold for a model
  whose architecture came from somebody else, and saying so beats a `config.json`
  that claims `attention_type: modular` on a Llama.
- Praxis **run** flags (optimizer, batch, precision, losses, RL, data, task
  weights) are unchanged. They describe the run, not the model, and they are the
  entire reason to do this.

`--model-name` and `--model-revision` are hashed; the rejected flags cannot
contaminate the identity.

## 6. Per-family quirks

Discovery first, registry second. Everything the wrapper needs is read off the
loaded model and tokenizer (hidden size, output projection, chat template, EOS).
`model_adapters` exists for the cases where discovery is wrong, keyed on the HF
`model_type` with a wildcard default, so a quirk is one registry entry and never
an `if model_type == "..."` in the forward.

## 7. Chat template

The foreign tokenizer's own `chat_template` is used verbatim, paired with the
`hf_native` ChatFormat (its own EOS ends a reply; no Praxis tool layout claimed).

The assistant mask is the one place a template edit is needed: HF derives the
mask from `{% generation %}` blocks, and most published templates have none. The
fix is mechanical - locate the assistant-content expression in the template and
wrap it - so it is done as a *template rewrite of the model's own template*
rather than a Praxis template substitution, which keeps the rendered text
byte-identical to what the model was trained on. When the rewrite cannot find a
spot and prompt masking is on, that is a hard error, never a zero loss.

## 8. Status

- [x] Audit
- [x] `praxis/objectives.py` extraction (`ObjectiveMixin` + `CausalObjectiveMixin`)
- [x] `praxis/models/` + the `model_tasks` / `model_adapters` registries
- [x] Foreign tokenizer loading + the `{% generation %}` template rewrite
- [x] `--model-name` / `--model-revision` / `--model-kwarg` / `--model-task` / `--no-train`
- [x] `peft_profiles` + adapter-only checkpoints (runtime install, Ray-style)
- [x] PoC: SmolLM2-135M-Instruct + LoRA + hh-rlhf + `preference` - first step
      loss 3.92 against ~9.2 from a random init at the same shape
- [x] `experiments/smol.yml` - the reference config, Discord-attached as "Karma"
- [ ] The other task families: `seq_cls`, `masked_lm`, `seq2seq`
- [ ] MTP on a foreign output projection (a named hard error until then)

## 9. What landed where

| Path | What |
|---|---|
| `praxis/objectives.py` | The objective half, extracted from `modeling.py` (-552 lines there). `ObjectiveMixin` is task-agnostic; `CausalObjectiveMixin` adds the label shift, the policies, MTP and the aux sweep. `supervise()` runs the whole thing in one call. |
| `praxis/models/base.py` | `ForeignModel`: hosts any `Auto*` output, keeps its config as `self.config`, exposes the run's `PraxisConfig` as `objective_config`, delegates `generate`, and answers every Praxis `getattr` probe the way a standalone model does. |
| `praxis/models/causal.py` | `ForeignCausalLM`. Asks the hosted model for hidden states and logits and NEVER for its loss. |
| `praxis/models/peft.py` | `peft_profiles` + `ensure_peft` + `apply_peft`. The adapter goes on the hosted model, not on the wrapper. |
| `praxis/tokenizers/generation_spans.py` | The template rewrite, verified byte-identical against the original render. |
| `praxis/tokenizers/pretrained.py` | Loads the checkpoint's tokenizer; `hf_native` when it has a template, `prose` when it does not. |
| `experiments/smol.yml` | The reference config. `tests/models/test_registry.py` parses it, so a flag classification that breaks it fails in CI rather than at launch. |

## 11. Measured, on an RTX 5060 Ti

SmolLM2-135M-Instruct + LoRA, bf16 (`bf16-true`), batch 4 x 512:

| | ms/step | peak VRAM |
|---|---|---|
| eager | 321 | 3681 MiB |
| `torch.compile` | 140 | 3216 MiB |

A 2.3x speedup AND less memory, for a one-time ~207s compile - so compilation
stays on (the Praxis default) rather than being disabled for the foreign path.
Batch 8 x 512 eager peaks at 7.0 GiB.

`sdpa` is what transformers already picks for Llama-family models, and it is
the fast path: PyTorch's fused kernel dispatches to the flash and cuDNN backends
itself (forcing `FLASH_ATTENTION`/`CUDNN_ATTENTION` only changed the loss by
exactly zero). `flash_attn` is not installed and is not worth a dependency.
`smol.yml` names `attn_implementation=sdpa` anyway, so an upstream default
change cannot quietly alter the run.

## 10. Decisions worth remembering

**Never hand `labels` to the hosted model.** Praxis pre-shifts; every HF model
shifts internally. The wrapper takes logits only and runs the Praxis criterion
itself, which is also what makes `--loss-func` and the regularizers apply.

**`hidden_states[-1]` is the post-final-norm activation** the model's own
`lm_head` consumes - verified identical to recomputing it - so cut-CE and the
representation regularizers see exactly the tensor they see on a Praxis model.

**A gap is a named error, never a no-op.** transformers accepts unknown forward
kwargs and drops them, so the default failure of this whole path is a run that
looks healthy with half its mechanisms off. Hence `UNSUPPORTED` on the wrapper,
the parser-derived architecture-flag rejection, and the hard failure when a
template's assistant spans cannot be found.

**`boundary_style="native"`.** A third value, meaning "the checkpoint's own
layout, which Praxis does not describe". The chat validator stands down on it
rather than reporting violations of a contract the data was never written
against.

**A base model gets `prose`, not `default`.** No control tokens, so nothing
assumed about a vocabulary Praxis did not build; halting is a trained stop
string and the assistant mask works.
