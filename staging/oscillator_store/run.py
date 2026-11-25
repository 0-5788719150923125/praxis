"""Train and measure the oscillator store end to end:  python run.py [--smoke]

1. Count the store text's n-grams (the exact-match reader).
2. Train the oscillator as an address, against held-out text, through the store and n-gram mixture.
3. Key every stored position of the training text with it and index the keys (s-means if SMEANS_REPO
   points at a clone of github.com/suayptalha/s-means, else k-means).
4. Train the parametric readers on the same text: 1024 tiny SMEAR experts on the stored keys, and a GRU-256.
5. Fit a gate per set of readers on other held-out text, every reader frozen, and score Test 3's evaluation
   set (TinyStories validation, 167K predictions) with each reader removed in turn.
6. Repeat with a 1M-character store (Test 6's record), searched exactly, for the untrained oscillator, the
   trained one, and both kinds of index.
7. Sample, and save results.json, run.log and checkpoint.pt next to this file (suffixed _fineweb with --data fineweb).
"""
import argparse
import importlib
import json
import math
import os
import pathlib
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

import data
from model import (
    NEIGHBORS, V, WARMUP, GRU, Gate, Oscillator, TextStore, TinyExperts, bits, confidence, count, entries,
    knn_probs, lloyd, nearest,
)
from ngram import KneserNey

HERE = pathlib.Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOORS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
SETS = {
    "store + n-gram": ("store", "n-gram"),
    "store + n-gram + experts": ("store", "n-gram", "experts"),
    "store + n-gram + GRU": ("store", "n-gram", "GRU"),
    "all four": ("store", "n-gram", "experts", "GRU"),
}
FULL = dict(store=9700, osc_store=500, osc=200, gate=100, eval=1000, small=1000, osc_steps=400, osc_batch=2048,
            experts=1024, expert_steps=3000, gate_steps=500, gru=256, gru_steps=20000, order=9, sample=400)
SMOKE = dict(store=300, osc_store=100, osc=30, gate=30, eval=100, small=150, osc_steps=20, osc_batch=256,
             experts=64, expert_steps=50, gate_steps=100, gru=64, gru_steps=50, order=6, sample=40)
LOG = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)


def clusterer():
    """s-means from a local clone if SMEANS_REPO is set (its package __init__ needs scikit-learn, so it is skipped)."""
    path = os.environ.get("SMEANS_REPO")
    if not path:
        return "k-means (25 passes)", lloyd
    pkg = types.ModuleType("smeans")
    pkg.__path__ = [str(pathlib.Path(path) / "smeans")]
    sys.modules["smeans"] = pkg
    core = importlib.import_module("smeans.core")

    def fit(x, lists):
        try:
            return core.smeans(x, n_clusters=lists, nepochs=3.0, seed=0, compute_labels=False)
        except Exception:  # its Triton kernels may not build on every GPU
            return core.smeans(x, n_clusters=lists, nepochs=3.0, seed=0, backend="native", compute_labels=False)

    return "s-means (3 epochs)", fit


def unigram(text):
    counts = np.bincount(text.reshape(-1), minlength=V) + 1.0
    return torch.tensor(counts / counts.sum(), device=DEVICE).float()


def train_oscillator(osc, store_text, query_text, steps, batch, ngram_truth):
    """Move the oscillator so held-out positions land near stored positions followed by the same character.

    Every step re-keys the store with the current oscillator (gradients flow through both sides), takes each
    query's 64 nearest stored positions plus the nearest one followed by the right character, and minimizes
    -log(lambda p_store + (1 - lambda) p_ngram) with a learned temperature and mixing weight, so the address is
    trained for the job it has in the system. (Trained on the store alone it ends in the same place once the
    n-gram is mixed in: results_solo_address.json.)
    """
    st, qt = torch.from_numpy(store_text).to(DEVICE), torch.from_numpy(query_text).to(DEVICE)
    log_t = torch.nn.Parameter(torch.tensor(math.log(10.0), device=DEVICE))
    mix = torch.nn.Parameter(torch.zeros((), device=DEVICE))
    opt = torch.optim.Adam([*osc.parameters(), log_t, mix], lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    probe = None
    for step in range(steps + 1):
        ks, y_store = entries(osc(st), store_text)
        mu, sd = ks.mean(0).detach(), ks.std(0).detach().clamp_min(1e-6)
        ks = (ks - mu) / sd
        kq, y_query = entries(osc(qt), query_text)
        kq = (kq - mu) / sd
        if probe is None:
            probe = torch.randperm(len(kq), generator=torch.Generator().manual_seed(0))[:4096].to(DEVICE)
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                d, i = nearest(kq[probe], ks, NEIGHBORS)
                p = (torch.softmax(-d / log_t.exp(), 1) * (y_store[i] == y_query[probe, None])).sum(1)
                lam = torch.sigmoid(mix)
                say(f"  oscillator step {step:4d}: held-out bpc with the n-gram {bits(lam * p + (1 - lam) * ngram_truth[probe]):.3f}"
                    f" (n-gram alone {bits(ngram_truth[probe]):.3f}), T={log_t.exp().item():.2f}, lambda={lam.item():.2f}")
        if step == steps:
            break
        sel = torch.randint(len(kq), (batch,), device=DEVICE)
        q, y = kq[sel], y_query[sel]
        with torch.no_grad():
            top = torch.empty(batch, NEIGHBORS, dtype=torch.long, device=DEVICE)
            right = torch.empty(batch, dtype=torch.long, device=DEVICE)
            kn = (ks * ks).sum(1)
            rows = max(16, 2**29 // (4 * len(ks)))
            for a in range(0, batch, rows):
                dist = torch.addmm(kn[None], q[a : a + rows], ks.T, alpha=-2)
                top[a : a + rows] = torch.topk(dist, NEIGHBORS, 1, largest=False).indices
                right[a : a + rows] = dist.masked_fill_(y_store[None, :] != y[a : a + rows, None], float("inf")).argmin(1)
        idx = torch.cat([top, right[:, None]], 1)
        logits = -((q[:, None, :] - ks[idx]) ** 2).sum(-1) / log_t.exp()
        repeat = torch.cat([torch.zeros_like(top, dtype=torch.bool), (top == right[:, None]).any(1, keepdim=True)], 1)
        p = (torch.softmax(logits.masked_fill(repeat, float("-inf")), 1) * (y_store[idx] == y[:, None])).sum(1)
        lam = torch.sigmoid(mix)
        loss = -torch.log(lam * p + (1 - lam) * ngram_truth[sel]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return log_t.exp().item()


def train_experts(experts, keys, nexts, steps, batch=8192, lr=3e-3):
    opt = torch.optim.Adam(experts.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    experts.train()
    for _ in range(steps):
        idx = torch.randint(len(keys), (batch,))
        loss = F.cross_entropy(experts(keys[idx].to(DEVICE, non_blocking=True).float()), nexts[idx.to(nexts.device)])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    experts.eval()


def train_gru(hidden, text, steps, batch=64, seq=128, lr=3e-3):
    """Test 3's recipe on the store text: random 128-character windows, one-cycle schedule."""
    torch.manual_seed(0)
    model = GRU(hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    tr = torch.from_numpy(text).to(DEVICE)
    for _ in range(steps):
        rows = torch.randint(0, tr.shape[0], (batch,), device=DEVICE)
        start = torch.randint(0, tr.shape[1] - seq - 1, (1,)).item()
        chunk = tr[rows, start : start + seq + 1]
        loss = F.cross_entropy(model(chunk[:, :-1]).reshape(-1, V), chunk[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def gru_probs(model, text, streams=100):
    return torch.cat([torch.softmax(model(torch.from_numpy(text[i : i + streams]).to(DEVICE))[:, WARMUP:-1], -1).reshape(-1, V) for i in range(0, len(text), streams)])


@torch.no_grad()
def expert_probs(experts, q, chunk=32768):
    return torch.cat([torch.softmax(experts(q[i : i + chunk]), -1) for i in range(0, len(q), chunk)])


@torch.no_grad()
def read(text, osc, mu, sd, store, kn, experts=None, gru=None):
    """What the readers say about every stored position of `text` (the store's distribution needs T: mixture_inputs)."""
    q, y = osc.keys(text, mu, sd)
    dist, idx = store.search(q)
    p = {"n-gram": torch.from_numpy(kn.table(text)[:, WARMUP + 1 :].reshape(-1, V)).float().to(DEVICE)}
    if experts is not None:
        p["experts"] = expert_probs(experts, q)
    if gru is not None:
        p["GRU"] = gru_probs(gru, text)
    return dict(q=q, y=y, dist=dist, chars=store.nexts[idx], p=p)


def fit_temperature(r, uni):
    """Temperature and unigram floor for the store reader alone (grid search, as in Test 6)."""
    match = (r["chars"] == r["y"][:, None]).float()
    base = float(r["dist"][:, 0].median().clamp_min(1e-6))
    best = (float("inf"), None, None)
    for T in base * 2.0 ** np.linspace(-8, 6, 57):
        p = (torch.softmax(-r["dist"] / T, 1).nan_to_num(0.0) * match).sum(1)
        for eps in FLOORS:
            b = bits((1 - eps) * p + eps * uni[r["y"]])
            if b < best[0]:
                best = (b, float(T), eps)
    return best[1], best[2]


def mixture_inputs(r, T, names):
    p = dict(r["p"], store=knn_probs(r["dist"], r["chars"], T))
    dists = [p[n] for n in names]
    rows = torch.arange(len(r["y"]), device=DEVICE)
    return confidence(r["dist"][:, 0], dists), torch.stack([d[rows, r["y"]] for d in dists], 1)


def train_gate(f, truth, steps):
    gate = Gate(truth.shape[1]).to(DEVICE)
    gate.mu.copy_(f.mean(0))
    gate.sd.copy_(f.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(gate.parameters(), lr=0.03)
    for _ in range(steps):
        loss = -torch.log((gate(f) * truth).sum(1).clamp_min(1e-12)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return gate


def train_fixed(truth, steps):
    logits = torch.zeros(truth.shape[1], device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(steps):
        loss = -torch.log((torch.softmax(logits, 0) * truth).sum(1).clamp_min(1e-12)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(logits, 0).detach()


@torch.no_grad()
def score(gate, fixed, f, truth, y, uni, eps, names):
    w = gate(f)
    out = {"system": bits((w * truth).sum(1)), "fixed mixture": bits((fixed * truth).sum(1))}
    for m, name in enumerate(names):
        out[f"{name} alone"] = bits((1 - eps) * truth[:, m] + eps * uni[y] if name == "store" else truth[:, m])
    for m, name in enumerate(names if len(names) > 2 else ()):  # with two readers, "without" is the other "alone"
        wm = w.clone()
        wm[:, m] = 0
        out[f"without {name}"] = bits((wm / wm.sum(1, keepdim=True).clamp_min(1e-12) * truth).sum(1))
    out["mean weights"] = {name: round(float(x), 3) for name, x in zip(names, w.mean(0))}
    return out


def fit_and_score(held, ev, T, eps, names, steps, uni):
    f, truth = mixture_inputs(held, T, names)
    gate, fixed = train_gate(f, truth, steps), train_fixed(truth, steps)
    fe, te = mixture_inputs(ev, T, names)
    return score(gate, fixed, fe, te, ev["y"], uni, eps, names), gate


@torch.no_grad()
def recall(store, q, k=NEIGHBORS):
    _, exact = store.exact(q, k)
    _, found = store.search(q, k)
    return float((found[:, :, None] == exact[:, None, :]).any(2).float().mean())


@torch.no_grad()
def generate(osc, mu, sd, store, kn, experts, gru, gate, T, prompt, n, seed=0):
    """Sample from all four readers, mixed by the gate."""
    g = torch.Generator().manual_seed(seed)
    context = list(data.encode(prompt))
    lam, emb = osc.lam(), osc.embedding()
    h = torch.zeros(64, dtype=torch.complex64, device=DEVICE)
    for c in context:
        h = lam * h + emb[c]
    y_g, h_g = gru.rnn(F.one_hot(torch.tensor(context, device=DEVICE), V).float()[None])
    out = []
    for _ in range(n):
        q = ((torch.cat([h.real, h.imag]) - mu) / sd)[None]
        dist, idx = store.search(q)
        dists = [knn_probs(dist, store.nexts[idx], T), torch.tensor(kn.dist(context), device=DEVICE).float()[None],
                 torch.softmax(experts(q), -1), torch.softmax(gru.out(y_g[:, -1]), -1)]
        w = gate(confidence(dist[:, 0], dists))
        p = sum(w[:, i : i + 1] * d for i, d in enumerate(dists))
        c = int(torch.multinomial(p[0].cpu(), 1, generator=g))
        context.append(c)
        out.append(c)
        h = lam * h + emb[c]
        y_g, h_g = gru.rnn(F.one_hot(torch.tensor([[c]], device=DEVICE), V).float(), h_g)
    return data.normalize(prompt) + data.decode(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, to check the pipeline runs")
    parser.add_argument("--data", choices=("tinystories", "fineweb"), default="tinystories")
    args = parser.parse_args()
    S = SMOKE if args.smoke else FULL
    suffix = "" if args.data == "tinystories" else f"_{args.data}"
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    train = data.load("train", 10_000_000, args.data).reshape(10_000, 1_000).astype(np.int64)
    store_text = train[: S["store"]]
    osc_text = train[9700 : 9700 + S["osc"]]
    gate_text = train[9900 : 9900 + S["gate"]]
    eval_text = data.load("val", 1_000_000, args.data)[:200_000].reshape(1000, 200)[: S["eval"]].astype(np.int64)
    uni = unigram(store_text)
    name, cluster = clusterer()
    res = {"data": args.data, "sizes": S, "store_chars": int(store_text.size), "clusterer": name}
    say(f"store {store_text.size:,} chars, oscillator queries {osc_text.size:,}, gate queries {gate_text.size:,}, evaluation predictions {S['eval'] * (200 - WARMUP - 1):,}; index by {name}")

    # 1. Exact matches.
    t1 = time.time()
    kn = KneserNey(store_text.reshape(-1), V, S["order"])
    say(f"Kneser-Ney {S['order']}: {kn.stored:,} counts, {time.time() - t1:.0f}s")

    # 2. The address.
    t1 = time.time()
    ngram_truth = torch.from_numpy(kn.prob(osc_text.reshape(-1)).reshape(osc_text.shape)[:, WARMUP + 1 :].reshape(-1)).float().to(DEVICE)
    osc = Oscillator().to(DEVICE)
    T_train = train_oscillator(osc, store_text[: S["osc_store"]], osc_text, S["osc_steps"], S["osc_batch"], ngram_truth)
    with torch.no_grad():
        timescale = 1 / (1 - torch.exp(-torch.exp(osc.nu)))
    res["oscillator"] = {"trained_numbers": count(osc), "T": T_train, "seconds": time.time() - t1,
                         "timescales": sorted(round(float(x), 1) for x in timescale),
                         "phases": sorted(round(float(x), 3) for x in osc.theta.detach().abs())}
    say(f"oscillator: {count(osc):,} trained numbers, timescales {min(res['oscillator']['timescales'])}-{max(res['oscillator']['timescales'])} characters, {time.time() - t1:.0f}s")

    # 3. The record: every stored position of the training text, keyed and indexed.
    t1 = time.time()
    mu, sd = osc.stats(store_text)
    keys, nexts = osc.keys(store_text, mu, sd, dtype=torch.float16, device="cpu")
    store = TextStore(keys, nexts, DEVICE)
    del keys
    lists = 2 ** round(math.log2(4 * math.sqrt(len(store))))
    store.build_index(lists, cluster)
    res["store"] = {"entries": len(store), "lists": lists, "probed": store.nprobe, "index_mse": store.mse,
                    "ngram_order": S["order"], "ngram_counts": kn.stored, "seconds": time.time() - t1}
    say(f"store: {len(store):,} entries in {lists:,} lists (search {store.nprobe}), {time.time() - t1:.0f}s")

    # 4. The parametric readers, on the same text.
    t1 = time.time()
    experts = TinyExperts(experts=S["experts"]).to(DEVICE)
    train_experts(experts, store.keys, store.nexts, S["expert_steps"])
    res["experts"] = {"experts": S["experts"], "hidden": 4, "trained_numbers": count(experts), "seconds": time.time() - t1}
    say(f"experts: {S['experts']} x hidden 4, {count(experts):,} trained numbers, {time.time() - t1:.0f}s")
    t1 = time.time()
    gru = train_gru(S["gru"], store_text, S["gru_steps"])
    res["gru"] = {"hidden": S["gru"], "steps": S["gru_steps"], "trained_numbers": count(gru), "seconds": time.time() - t1}
    say(f"GRU-{S['gru']}: {count(gru):,} trained numbers, {time.time() - t1:.0f}s")

    # 5. Gates on held-out text the store, the n-gram and the parametric readers never saw; evaluation.
    held = read(gate_text, osc, mu, sd, store, kn, experts, gru)
    T, eps = fit_temperature(held, uni)
    ev = read(eval_text, osc, mu, sd, store, kn, experts, gru)
    res["readout"] = {"T": T, "floor": eps}
    res["evaluation"] = {}
    gates = {}
    for label, names in SETS.items():
        res["evaluation"][label], gates[label] = fit_and_score(held, ev, T, eps, names, S["gate_steps"], uni)
        say(f"evaluation, {label}:", json.dumps(res["evaluation"][label]))
    probe = ev["q"][torch.randperm(len(ev["q"]), generator=torch.Generator().manual_seed(0))[:2048].to(DEVICE)]
    res["evaluation"]["index recall@64"] = recall(store, probe)
    say(f"index recall@64 (searching {store.nprobe} of {lists:,} lists): {res['evaluation']['index recall@64']:.4f}")

    # 6. A 1M-character store (Test 6's record), searched exactly; the untrained oscillator; both indexes.
    #    The experts were trained on the trained oscillator's keys, so the untrained row mixes store and n-gram only.
    small_text = store_text[: S["small"]]
    kn_small = KneserNey(small_text.reshape(-1), V, S["order"])
    small = {}
    for label, o in (("untrained oscillator", Oscillator().to(DEVICE)), ("trained oscillator", osc)):
        m_, s_ = o.stats(small_text)
        st = TextStore(*o.keys(small_text, m_, s_), DEVICE)
        r_ho = read(gate_text, o, m_, s_, st, kn_small, experts if o is osc else None)
        r_ev = read(eval_text, o, m_, s_, st, kn_small, experts if o is osc else None)
        T_s, eps_s = fit_temperature(r_ho, uni)
        small[label] = {"store + n-gram": fit_and_score(r_ho, r_ev, T_s, eps_s, SETS["store + n-gram"], S["gate_steps"], uni)[0]}
        if o is osc:
            small[label]["store + n-gram + experts"] = fit_and_score(r_ho, r_ev, T_s, eps_s, SETS["store + n-gram + experts"], S["gate_steps"], uni)[0]
            for cname, cfit in [("k-means (25 passes)", lloyd)] + ([(name, cluster)] if name.startswith("s-means") else []):
                ivf = TextStore(st.keys.clone(), st.nexts.clone(), DEVICE)
                ivf.build_index(2 ** round(math.log2(4 * math.sqrt(len(ivf)))), cfit)
                r_i = read(eval_text, o, m_, s_, ivf, kn_small)
                small[f"index by {cname}"] = {"recall@64": recall(ivf, r_ev["q"][:2048]), "lists": len(ivf.centroids), "probed": ivf.nprobe, "mse": ivf.mse,
                                              "store alone": bits((1 - eps_s) * mixture_inputs(r_i, T_s, ("store",))[1][:, 0] + eps_s * uni[r_i["y"]])}
                say(f"1M store, index by {cname}:", json.dumps(small[f"index by {cname}"]))
        say(f"1M store, {label}:", json.dumps(small[label]))
    res["store_1M"] = small

    # 7. Sample and save.
    prompt = "Once upon a time, " if args.data == "tinystories" else "The history of "
    res["sample"] = generate(osc, mu, sd, store, kn, experts, gru, gates["all four"], T, prompt, S["sample"])
    say("sample (all four readers):", repr(res["sample"]))
    torch.save({"oscillator": osc.state_dict(), "mu": mu, "sd": sd, "experts": experts.state_dict(), "gru": gru.state_dict(),
                "gates": {k: g.state_dict() for k, g in gates.items()}, "T": T, "floor": eps}, HERE / f"checkpoint{suffix}.pt")
    res["seconds"] = time.time() - t0
    (HERE / f"results{suffix}.json").write_text(json.dumps(res, indent=1, default=float))
    (HERE / f"run{suffix}.log").write_text("\n".join(LOG) + "\n")
    say(f"done in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
