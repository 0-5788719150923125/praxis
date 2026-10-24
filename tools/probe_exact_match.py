#!/usr/bin/env python3
"""Corpus exact-match as a reward signal: is there anything for RL to climb?

Indexes a sample of a run's training mix with a suffix array, then scores text
by L_t, the longest substring ending at byte t that occurs in that sample. That
is the per-byte reward next/exact_match_reward.md proposes. Before building any
RL, this asks whether the signal separates real text from the model's samples,
whether it varies between samples of one prompt (no variance, no advantage),
and whether the samples it prefers are the ones an independent LM prefers too.

    python tools/probe_exact_match.py index --mb 32
    python tools/probe_exact_match.py sample --per-prompt 8
    python tools/probe_exact_match.py score

Arms scored against the same index:
    human     the held-out document's true continuation
    shuffled  that continuation with its words shuffled - word salad
    model     the run's own samples, default decoding
"""

import argparse
import bisect
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "build" / "exact_match"
SEP = b"\xff"  # never valid in UTF-8, so no match can cross a document boundary
WIDTH = 256  # longest match measured; longer ones count as WIDTH
DOC_CAP = 8192  # bytes kept per indexed document, so no one file dominates
CONT_BYTES = 256  # continuation length scored per prompt


def bootstrap(experiment, device):
    """Parse a run's flags the way main.py does, minus the CLI's side effects."""
    argv = sys.argv
    sys.argv = [argv[0]]  # importing praxis.cli with a real argv parses it
    from praxis.cli.loaders.experiments import ExperimentLoader

    # Every CLI init rewrites the tracked experiment ymls in its own env.
    ExperimentLoader.format_tracked_files = lambda self: None
    sys.argv = ["main.py", f"--{experiment}", "--no-compile", "--device", device]
    from praxis.cli import RunConfig, parse_cli
    from praxis.tokenizers import create_tokenizer

    args, processed = parse_cli()
    sys.argv = argv
    cfg = RunConfig.from_args(processed, args)
    tokenizer = create_tokenizer(
        vocab_size=cfg.vocab_size,
        encoder_type=cfg.encoder_type,
        tokenizer_type=cfg.tokenizer_type,
        chat_format=cfg.chat_format,
    )
    return args, processed, cfg, tokenizer


def char_cut(text_bytes, at):
    """Move a byte offset back to the start of the UTF-8 character it lands in."""
    while at > 0 and (text_bytes[at] & 0xC0) == 0x80:
        at -= 1
    return at


# ---------------------------------------------------------------- index stage


def mix_weights(run_dir):
    """The run's latest logged sampling weights, keyed by normalized name."""
    import sqlite3

    db = Path(run_dir) / "data_metrics.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute(
        "select sampling_weights from data_metrics order by step desc limit 1"
    ).fetchone()
    return {norm(k): v for k, v in json.loads(row[0]).items()} if row else {}


def norm(name):
    name = str(name).rstrip("/")
    name = "praxis" if name in (".", "") else name.split("/")[-1] if name.startswith(".") else name
    return name.replace("-", "_")


def samplers_for(processed, tokenizer, seed, run_dir):
    """One sampler per entry of the run's training mix, weighted as the run
    last sampled them (static config weights where it logged none).

    The KB spider is left out."""
    from praxis.data.utils import get_dataset, get_dataset_configs

    configs = get_dataset_configs(
        processed.get("train_datasets") or ["base"], [], processed.get("rl_type")
    )
    out = []
    for c in configs["primary"]:
        kind = c.get("type", "huggingface")
        try:
            s = get_dataset(kind, tokenizer, seed, c)
        except Exception as e:
            print(f"[index] skipping {c.get('path', c.get('_id'))}: {e!r:.120}")
            continue
        if s is not None:
            out.append((c.get("path", c.get("_id", kind)), s, c.get("weight", 1.0)))
    paths = processed.get("data_path") or []
    for p in paths if isinstance(paths, list) else [paths]:
        s = get_dataset("directory", tokenizer, seed, data_path=p)
        out.append((p, s, getattr(s, "weight", 1.0)))
    logged = mix_weights(run_dir)
    return [(n, s, logged.get(norm(n), w)) for n, s, w in out]


def is_prose(name):
    """Hub corpora, where the word-salad complaint lives; not code or binaries."""
    return "/" in str(name) and not str(name).startswith(".")


def render(sampler, tokenizer):
    """One training document as the model sees it, or None."""
    doc = sampler.get_document()
    if not isinstance(doc, dict) or not doc.get("messages"):
        return None
    try:
        text = tokenizer.apply_chat_template(
            doc["messages"], tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None
    return text or None


SA_BYTES_PER_BYTE = 40  # peak working set of build_suffix_array, measured loosely


def available_memory():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def build_suffix_array(corpus, width):
    """Suffixes sorted by their first ``width`` bytes, by prefix doubling."""
    rank = np.frombuffer(corpus, dtype=np.uint8).astype(np.int32) + 1
    n = len(rank)
    k = 1
    while k < width:
        key = rank.astype(np.int64) * (max(n, 256) + 2)  # > any rank
        key[: n - k] += rank[k:]
        sa = np.argsort(key)
        sk = key[sa]
        del key
        rank = np.empty(n, np.int32)
        rank[sa[0]] = 1
        rank[sa[1:]] = 1 + np.cumsum(sk[1:] != sk[:-1], dtype=np.int32)
        del sk
        print(f"[index]   doubled to {2 * k} bytes, {int(rank.max()):,} classes")
        if rank.max() == n:
            break
        k *= 2
    return sa.astype(np.int32)


def draw(a):
    """Sample the corpus and the held-out prompts through the run's samplers."""
    _, processed, cfg, tokenizer = bootstrap(a.experiment, "cpu")
    random.seed(a.seed)
    samplers = samplers_for(processed, tokenizer, cfg.seed, a.run_dir)
    names = [n for n, _, _ in samplers]
    weights = [w for _, _, w in samplers]
    print("[index] mix: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))
    prose = [i for i, n in enumerate(names) if is_prose(n)]

    budget = int(a.mb * 2**20)
    parts, size, seen, per_source = [], 0, set(), {}
    t0 = time.time()
    while size < budget:
        i = random.choices(range(len(samplers)), weights=weights)[0]
        text = render(samplers[i][1], tokenizer)
        if not text:
            continue
        b = text.encode("utf-8", errors="replace")
        h = hashlib.sha1(b).digest()
        if h in seen:
            continue
        seen.add(h)
        b = b[: char_cut(b, min(len(b), DOC_CAP))] if len(b) > DOC_CAP else b
        parts.append(b)
        size += len(b) + 1
        per_source[names[i]] = per_source.get(names[i], 0) + len(b)
        if len(parts) % 500 == 0:
            print(f"[index] {size / 2**20:6.1f} MB, {len(parts)} docs")
    corpus = SEP.join(parts) + SEP
    print(f"[index] drew {len(corpus) / 2**20:.1f} MB in {time.time() - t0:.0f}s")

    # Held-out prompts: later in the same streams, never in the index.
    heldout, tries = [], 0
    while len(heldout) < a.prompts and tries < a.prompts * 200:
        tries += 1
        i = random.choices(prose, weights=[weights[j] for j in prose])[0]
        text = render(samplers[i][1], tokenizer)
        if not text:
            continue
        b = text.encode("utf-8", errors="replace")
        if hashlib.sha1(b).digest() in seen or len(b) < 384 + CONT_BYTES + 64:
            continue
        seen.add(hashlib.sha1(b).digest())
        cut = char_cut(b, random.randint(384, min(len(b) - CONT_BYTES - 32, 1536)))
        end = char_cut(b, cut + CONT_BYTES)
        heldout.append(
            dict(
                source=str(names[i]),
                prompt=b[:cut].decode("utf-8", errors="replace"),
                continuation=b[cut:end].decode("utf-8", errors="replace"),
            )
        )

    (OUT / "corpus.bin").write_bytes(corpus)
    with open(OUT / "heldout.jsonl", "w") as f:
        for row in heldout:
            f.write(json.dumps(row) + "\n")
    (OUT / "sources.json").write_text(json.dumps(per_source, indent=2))
    for f in OUT.glob("sa_*.npy"):
        f.unlink()
    return corpus, heldout


def cmd_index(a):
    OUT.mkdir(parents=True, exist_ok=True)
    if a.reuse:
        corpus = (OUT / "corpus.bin").read_bytes()
        heldout = [json.loads(l) for l in open(OUT / "heldout.jsonl")]
    else:
        corpus, heldout = draw(a)
    per_source = json.loads((OUT / "sources.json").read_text()) if (OUT / "sources.json").exists() else None

    # Nested prefixes, each its own index: how the signal moves with corpus
    # size says what a full-size index would do. Docs were drawn at random, so
    # every prefix is a sample of the same mix.
    ends = np.flatnonzero(np.frombuffer(corpus, np.uint8) == SEP[0]) + 1
    indexes = []
    for k in range(a.nested - 1, -1, -1):
        cut = int(ends[min(np.searchsorted(ends, len(corpus) / 2**k), len(ends) - 1)])
        if (OUT / f"sa_{cut}.npy").exists():
            indexes.append(cut)
            continue
        need = cut * SA_BYTES_PER_BYTE + 2**30
        if available_memory() < need:
            # This box runs other jobs; an OOM here takes them down with it.
            print(f"[index] stopping: {cut / 2**20:.0f} MB index needs ~{need / 2**30:.1f} GB free")
            break
        t0 = time.time()
        sa = build_suffix_array(corpus[:cut], WIDTH)
        np.save(OUT / f"sa_{cut}.npy", sa)
        indexes.append(cut)
        print(f"[index] suffix array over {cut / 2**20:.1f} MB in {time.time() - t0:.0f}s")
        del sa
    meta = dict(
        experiment=a.experiment,
        bytes=len(corpus),
        docs=len(ends),
        per_source=per_source,
        prompts=len(heldout),
        indexes=indexes,
    )
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[index] {len(heldout)} held-out prompts; wrote {OUT}")


# --------------------------------------------------------------- sample stage


def load_model(experiment, device, ckpt):
    """Build the run's model from its flags and load a checkpoint's weights."""
    import torch

    args, processed, cfg, tokenizer = bootstrap(experiment, device)
    from praxis.cli import create_praxis_config
    from praxis.trainers.setup import (
        assemble_model,
        configure_torch_precision,
        register_praxis_models,
    )

    configure_torch_precision(processed.get("precision"), device)
    register_praxis_models()
    config = create_praxis_config(args, tokenizer)
    model = assemble_model(cfg, config).model
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    weights = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"[sample] missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval(), tokenizer, cfg


def own_bits_per_byte(model, seqs, n, lengths):
    """The model's own bits per byte on what it generated, prompt excluded,
    over each row's first ``lengths[i]`` generated tokens (what was kept)."""
    import torch

    out = model(input_ids=seqs)
    logits = (out.logits if hasattr(out, "logits") else out[0]).float()
    lp = torch.log_softmax(logits[:, n - 1 : -1], -1).gather(-1, seqs[:, n:, None])[..., 0]
    pos = torch.arange(lp.shape[1], device=lp.device)
    keep = (pos[None] < torch.tensor(lengths, device=lp.device)[:, None]).float()
    bits = -(lp * keep).sum(1) / keep.sum(1).clamp(min=1) / np.log(2)
    return [float(b) for b in bits]


def cmd_sample(a):
    import torch

    from praxis.inference import Generator
    from praxis.inference.request import GenerationRequest

    ckpt = Path(a.ckpt).resolve()
    model, tokenizer, cfg = load_model(a.experiment, a.device, ckpt)
    gen = Generator(model, tokenizer, device=a.device, synchronous=True)
    stops = gen._stop_strings() or []
    rows = [json.loads(l) for l in open(OUT / "heldout.jsonl")][: a.prompts]
    kwargs = dict(max_new_tokens=CONT_BYTES)
    if a.temperature is not None:
        kwargs["temperature"] = a.temperature
    t0 = time.time()
    with open(OUT / a.out, "w") as f:
        for p, row in enumerate(rows):
            if a.sequential:
                # The served path, one request at a time: the reference the
                # batched path below is checked against.
                outs = []
                for _ in range(a.per_prompt):
                    out = str(gen.get_result(gen.request_generation(row["prompt"], dict(kwargs))) or "")
                    outs.append(out[len(row["prompt"]) :] if out.startswith(row["prompt"]) else "")
            else:
                # Same inputs and decoding settings as a served request, with
                # the prompt repeated so all samples decode in one batch.
                req = GenerationRequest(id=str(p), prompt=row["prompt"], kwargs=dict(kwargs))
                ids, gk, _, _ = gen._prepare_inputs(req)
                with gen.backend.eval_mode(), torch.no_grad():
                    seqs = gen.backend.generate_until_halt(ids.repeat(a.per_prompt, 1), gk)
                    # Rows that hit a stop string keep decoding until the
                    # whole batch halts; cut each back to its own stop.
                    outs = []
                    for s in seqs:
                        text = tokenizer.decode(s[ids.shape[1] :], skip_special_tokens=True)
                        for stop in stops:
                            text = text.split(stop)[0]
                        outs.append(text)
                    lengths = [len(t.encode("utf-8")) for t in outs]
                    own = own_bits_per_byte(model, seqs, ids.shape[1], lengths)
            for j, cont in enumerate(outs):
                row_out = dict(prompt_id=p, sample=j, text=cont)
                if not a.sequential:
                    row_out["own_bpb"] = own[j]
                f.write(json.dumps(row_out) + "\n")
            print(f"[sample] prompt {p + 1}/{len(rows)}  {time.time() - t0:.0f}s")
    print(f"[sample] wrote {OUT / a.out} from {ckpt.name}")


# ---------------------------------------------------------------- score stage


class Suffixes:
    """The suffix array viewed as a sorted sequence of WIDTH-byte prefixes."""

    def __init__(self, corpus, sa):
        self.corpus, self.sa = corpus, sa

    def __len__(self):
        return len(self.sa)

    def __getitem__(self, i):
        s = int(self.sa[i])
        return self.corpus[s : s + WIDTH]


def lcp(x, y):
    n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return i
    return n


def match_lengths(index, b, start):
    """L[t] for each byte t >= start of ``b``: the longest corpus substring
    ending at t. Built from forward matches: the earliest i whose match covers t
    moves only forward, so one pass over t suffices."""
    lo = max(0, start - WIDTH)
    ms = np.zeros(len(b), np.int64)
    for i in range(lo, len(b)):
        q = b[i : i + WIDTH]
        pos = bisect.bisect_left(index, q)
        ms[i] = max(
            lcp(index[pos - 1], q) if pos > 0 else 0,
            lcp(index[pos], q) if pos < len(index) else 0,
        )
    L = np.zeros(len(b) - start, np.int64)
    i = lo
    for t in range(start, len(b)):
        while i <= t and i + ms[i] <= t:
            i += 1
        L[t - start] = t - i + 1
    return L


def self_copy(b, start, n=8):
    """Fraction of continuation bytes whose trailing n-gram already appeared."""
    seen, hits, total = set(), 0, 0
    for t in range(n, len(b) + 1):
        g = b[t - n : t]
        if t - 1 >= start:
            total += 1
            hits += g in seen
        seen.add(g)
    return hits / max(total, 1)


def shuffle_words(text, rng):
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


class Judge:
    """An independent LM's bits per byte on a continuation, given its prompt."""

    def __init__(self, name, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.lm = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float32
        ).to(device).eval()
        self.device, self.torch = device, torch

    def bpb(self, prompt, cont):
        torch = self.torch
        enc = self.tok(prompt + cont, return_offsets_mapping=True, add_special_tokens=False)
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        with torch.no_grad():
            logp = torch.log_softmax(self.lm(ids).logits[0, :-1].float(), -1)
        nll = -logp.gather(1, ids[0, 1:, None])[:, 0]
        keep = torch.tensor(
            [s >= len(prompt) for s, _ in enc["offset_mapping"][1:]], device=self.device
        )
        n_bytes = max(len(cont.encode("utf-8")), 1)
        return float(nll[keep].sum() / np.log(2) / n_bytes)


def summarize(Ls):
    pooled = np.concatenate(Ls) if Ls else np.zeros(1)
    row = dict(mean=float(pooled.mean()), median=float(np.median(pooled)))
    for k in (4, 8, 16, 32, 64):
        row[f">={k}"] = float((pooled >= k).mean())
    row["<=2"] = float((pooled <= 2).mean())
    return row


def reward(L):
    return float(np.log2(1 + L).mean())


def within_prompt(Ls, feats):
    """What a group-relative advantage would see for one model arm: the spread
    of reward between samples of one prompt, and what that spread prefers."""
    within, between, rho_j, rho_c, rho_o, best, worst, mean, by_own = ([] for _ in range(9))
    gain = []
    for p, group in Ls.items():
        if len(group) < 3:
            continue
        R = np.array([reward(L) for L in group])
        J = np.array([f["bpb"] for f in feats[p]])
        C = np.array([f["copy"] for f in feats[p]])
        O = np.array([f["own"] for f in feats[p]])
        within.append(R.std())
        between.append(R.mean())
        rho_j.append(spearman(R, -J))
        rho_c.append(spearman(R, C))
        best.append(J[R.argmax()])
        worst.append(J[R.argmin()])
        mean.append(J.mean())
        if not np.isnan(O).any():
            rho_o.append(spearman(R, -O))
            by_own.append(J[O.argmin()])
            gain.append(J[O.argmin()] - J[R.argmax()])
    return dict(
        prompts=len(within),
        reward_std_within=float(np.mean(within)),
        reward_std_between=float(np.std(between)),
        spearman_reward_vs_fluency=float(np.nanmean(rho_j)),
        spearman_reward_vs_self_copy=float(np.nanmean(rho_c)),
        judge_bpb_best_of_n=float(np.mean(best)),
        judge_bpb_mean_of_n=float(np.mean(mean)),
        judge_bpb_worst_of_n=float(np.mean(worst)),
        spearman_reward_vs_own_likelihood=float(np.nanmean(rho_o)) if rho_o else np.nan,
        judge_bpb_best_of_n_by_own=float(np.mean(by_own)) if by_own else np.nan,
        # > 0 when picking by reward beats picking by own likelihood, per prompt.
        gain_over_own=float(np.mean(gain)) if gain else np.nan,
        gain_over_own_se=float(np.std(gain) / np.sqrt(len(gain))) if gain else np.nan,
    )


def cmd_score(a):
    meta = json.loads((OUT / "meta.json").read_text())
    corpus = (OUT / "corpus.bin").read_bytes()
    sizes = [c for c in meta["indexes"] if (OUT / f"sa_{c}.npy").exists()]
    rows = [json.loads(l) for l in open(OUT / "heldout.jsonl")]
    files = dict(spec.split("=", 1) for spec in a.arms)
    samples = {k: [json.loads(l) for l in open(OUT / f)] for k, f in files.items()}
    prompt_ids = sorted({s["prompt_id"] for ss in samples.values() for s in ss})
    rng = random.Random(a.seed)

    texts = {
        "human": {p: [rows[p]["continuation"]] for p in prompt_ids},
        "shuffled": {
            p: [shuffle_words(rows[p]["continuation"], rng)] for p in prompt_ids
        },
    }
    own = {}
    for k, ss in samples.items():
        kept = {p: [s for s in ss if s["prompt_id"] == p and s["text"]] for p in prompt_ids}
        texts[k] = {p: [s["text"] for s in g] for p, g in kept.items()}
        own[k] = {p: [s.get("own_bpb", np.nan) for s in g] for p, g in kept.items()}

    # What does not depend on the index: length, self-repetition, the judge.
    t0 = time.time()
    judge = None if a.no_judge else Judge(a.judge, a.device)
    feats = {}
    for arm, by_p in texts.items():
        feats[arm] = {}
        for p, conts in by_p.items():
            prompt = rows[p]["prompt"]
            feats[arm][p] = [
                dict(
                    bytes=len(c.encode("utf-8")),
                    copy=self_copy((prompt + c).encode("utf-8"), len(prompt.encode("utf-8"))),
                    bpb=judge.bpb(prompt, c) if judge else np.nan,
                    own=own[arm][p][j] if arm in own else np.nan,
                )
                for j, c in enumerate(conts)
            ]
    print(f"[score] judged {sum(len(v) for b in texts.values() for v in b.values())} texts in {time.time() - t0:.0f}s")

    # L for every text against every index size.
    Ls = {}
    for cut in sizes:
        t0 = time.time()
        index = Suffixes(corpus[:cut], np.load(OUT / f"sa_{cut}.npy", mmap_mode="r"))
        Ls[cut] = {
            arm: {
                p: [
                    match_lengths(
                        index,
                        (rows[p]["prompt"] + c).encode("utf-8"),
                        len(rows[p]["prompt"].encode("utf-8")),
                    )
                    for c in conts
                ]
                for p, conts in by_p.items()
            }
            for arm, by_p in texts.items()
        }
        print(f"[score] matched against {cut / 2**20:.1f} MB in {time.time() - t0:.0f}s")
    print()

    arms = list(texts)
    models = list(samples)
    report = dict(index_bytes=sizes, scaling=[], arms={}, within_prompt={})

    # How the reward moves with the size of the index.
    print("reward = mean log2(1 + L) over the continuation, by index size")
    print(f"{'index MB':>9} " + " ".join(f"{k:>10}" for k in arms) + f" {'human-' + models[0]:>14} {'rho(fluency)':>13}")
    for cut in sizes:
        r = {k: float(np.mean([reward(L) for g in Ls[cut][k].values() for L in g])) for k in arms}
        w = within_prompt(Ls[cut][models[0]], feats[models[0]])
        report["scaling"].append(dict(index_bytes=cut, reward=r, within=w))
        print(
            f"{cut / 2**20:9.1f} " + " ".join(f"{r[k]:10.3f}" for k in arms)
            + f" {r['human'] - r[models[0]]:+14.3f} {w['spearman_reward_vs_fluency']:+13.3f}"
        )
    print()

    # Does the corpus pick better samples than the model's own likelihood does?
    print("judge bits/byte saved by picking best-of-n by reward instead of by own likelihood")
    print(f"{'index MB':>9} " + " ".join(f"{k:>16}" for k in models))
    for cut in sizes:
        ws = [within_prompt(Ls[cut][k], feats[k]) for k in models]
        print(
            f"{cut / 2**20:9.1f} "
            + " ".join(f"{w['gain_over_own']:+9.3f} ± {w['gain_over_own_se']:.3f}" for w in ws)
        )
    print()

    big = sizes[-1]
    head = f"{'arm':10} {'n':>5} {'bytes':>6} {'L mean':>7} {'L med':>6} " + " ".join(
        f"{c:>6}" for c in (">=4", ">=8", ">=16", ">=32", ">=64", "<=2")
    ) + f" {'anchor':>7} {'copy8':>6} {'judge':>6}"
    print(f"at {big / 2**20:.1f} MB")
    print(head)
    for k in arms:
        flat_L = [L for g in Ls[big][k].values() for L in g]
        flat_f = [f for g in feats[k].values() for f in g]
        # A match reaches back into the prompt while L_t > t + 1.
        anchor = [int(np.argmax(np.append(L <= np.arange(len(L)) + 1, True))) for L in flat_L]
        s = summarize(flat_L)
        s.update(
            n=len(flat_L),
            bytes=float(np.mean([f["bytes"] for f in flat_f])),
            anchored=float(np.mean(anchor)),
            copy=float(np.mean([f["copy"] for f in flat_f])),
            bpb=float(np.nanmean([f["bpb"] for f in flat_f])),
        )
        report["arms"][k] = s
        print(
            f"{k:10} {s['n']:5d} {s['bytes']:6.0f} {s['mean']:7.2f} {s['median']:6.1f} "
            + " ".join(f"{s[c]:6.3f}" for c in (">=4", ">=8", ">=16", ">=32", ">=64", "<=2"))
            + f" {s['anchored']:7.1f} {s['copy']:6.3f} {s['bpb']:6.2f}"
        )
    print("  L = longest corpus match ending at each byte; anchor = bytes a match from")
    print("  the prompt survives into the continuation; copy8 = 8-grams already seen in")
    print("  the context; judge = independent LM bits/byte (lower = more fluent)\n")

    for k in models:
        w = report["within_prompt"][k] = within_prompt(Ls[big][k], feats[k])
        print(f"{k}: within one prompt ({w['prompts']} prompts)")
        print(
            f"  reward spread within a prompt {w['reward_std_within']:.3f}, "
            f"between prompts {w['reward_std_between']:.3f}"
        )
        print(f"  spearman(reward, judge fluency) {w['spearman_reward_vs_fluency']:+.3f}")
        print(f"  spearman(reward, self-copy)     {w['spearman_reward_vs_self_copy']:+.3f}")
        print(f"  spearman(reward, own likelihood) {w['spearman_reward_vs_own_likelihood']:+.3f}")
        print(
            f"  judge bits/byte: best-of-n by reward {w['judge_bpb_best_of_n']:.3f}, "
            f"by own likelihood {w['judge_bpb_best_of_n_by_own']:.3f}, "
            f"mean {w['judge_bpb_mean_of_n']:.3f}, worst {w['judge_bpb_worst_of_n']:.3f}\n"
        )

    # Examples for eyeballing: highest- and lowest-reward samples of a few prompts.
    ex = []
    for p in prompt_ids[: a.examples]:
        group = sorted(
            (reward(L), c) for L, c in zip(Ls[big][models[0]][p], texts[models[0]][p])
        )
        if group:
            ex.append(
                dict(prompt=rows[p]["prompt"][-200:], human=rows[p]["continuation"],
                     best=group[-1], worst=group[0])
            )
    report["examples"] = ex
    (OUT / a.report).write_text(json.dumps(report, indent=2, default=float))
    for e in ex:
        print("-" * 72)
        print(f"...{e['prompt']!r}")
        print(f"  human           {e['human'][:160]!r}")
        print(f"  best  ({e['best'][0]:.2f})  {e['best'][1][:160]!r}")
        print(f"  worst ({e['worst'][0]:.2f})  {e['worst'][1][:160]!r}")
    print(f"\n[score] wrote {OUT / a.report}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", default="byte-latent-h")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--mb", type=float, default=32)
    i.add_argument("--prompts", type=int, default=64)
    i.add_argument("--run-dir", default=str(REPO_ROOT / "build/current"))
    i.add_argument("--nested", type=int, default=5, help="index sizes, halving")
    i.add_argument("--reuse", action="store_true", help="keep the drawn corpus")
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", default=str(REPO_ROOT / "build/current/model/last.ckpt"))
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--prompts", type=int, default=64)
    s.add_argument("--per-prompt", type=int, default=8)
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--sequential", action="store_true")
    s.add_argument("--out", default="samples.jsonl")
    c = sub.add_parser("score")
    c.add_argument("--judge", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--no-judge", action="store_true")
    c.add_argument("--examples", type=int, default=4)
    c.add_argument(
        "--arms", nargs="+", default=["model=samples.jsonl"], help="name=samples file"
    )
    c.add_argument("--report", default="report.json")
    a = ap.parse_args()
    {"index": cmd_index, "sample": cmd_sample, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
    # Streaming samplers leave non-daemon threads behind that block exit.
    sys.stdout.flush()
    os._exit(0)
