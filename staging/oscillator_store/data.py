"""TinyStories as arrays of character indices, cached under data/."""
import pathlib
import re

import numpy as np
import pyarrow as pa

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path.home() / (
    ".cache/huggingface/datasets/roneneldan___tiny_stories/default/0.0.0/"
    "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
)
SYMBOLS = "\n !\"',.?abcdefghijklmnopqrstuvwxyz"  # 34 symbols


def _stories(name, max_chars):
    out, n = [], 0
    with pa.memory_map(str(CACHE / name)) as src:
        for batch in pa.ipc.open_stream(src):
            for s in batch.column("text").to_pylist():
                out.append(s)
                n += len(s)
                if n >= max_chars:
                    return out
    return out


def normalize(text):
    text = text.replace("â€™", "'").replace("â€˜", "'")
    text = re.sub("â€.?", '"', text).lower()
    keep = set(SYMBOLS)
    text = "".join(c if c in keep else " " for c in text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return re.sub(r"[ \n]{2,}", lambda m: "\n" if "\n" in m.group() else " ", text)


def fineweb(split, train_chars=10_000_000, val_chars=1_000_000):
    """FineWeb (sample-10BT, streamed in order) in the same alphabet. Whole documents go to train until it
    holds train_chars, then the following documents to val; no document is split between them."""
    path = ROOT / "data" / f"fineweb_{split}_{train_chars if split == 'train' else val_chars}.npy"
    if path.exists():
        return np.load(path)
    import datasets

    lookup = np.zeros(256, dtype=np.uint8)
    for i, c in enumerate(SYMBOLS):
        lookup[ord(c)] = i
    stream = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    out, parts, n, want = {}, [], 0, train_chars
    for doc in stream:
        text = normalize(doc["text"]).strip()
        if not text:
            continue
        parts.append(text + "\n")
        n += len(text) + 1
        if n >= want:
            out["train" if "train" not in out else "val"] = lookup[np.frombuffer("".join(parts)[:want].encode("ascii"), dtype=np.uint8)]
            if len(out) == 2:
                break
            parts, n, want = [], 0, val_chars
    path.parent.mkdir(exist_ok=True)
    for name, idx in out.items():
        np.save(ROOT / "data" / f"fineweb_{name}_{len(idx)}.npy", idx)
    return out[split]


def load(split, max_chars, source="tinystories"):
    """Character indices (uint8) for the first max_chars of a split ("train" or "val")."""
    if source == "fineweb":
        return fineweb(split)[:max_chars]
    path = ROOT / "data" / f"{split}_{max_chars}.npy"
    if path.exists():
        return np.load(path)
    name = "tiny_stories-validation.arrow" if split == "val" else "tiny_stories-train-00000-of-00004.arrow"
    text = normalize("\n".join(_stories(name, int(max_chars * 1.1))))[:max_chars]  # normalizing shrinks the text
    lookup = np.zeros(256, dtype=np.uint8)
    for i, c in enumerate(SYMBOLS):
        lookup[ord(c)] = i
    idx = lookup[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    path.parent.mkdir(exist_ok=True)
    np.save(path, idx)
    return idx


def encode(text):
    return np.array([SYMBOLS.index(c) for c in normalize(text)], dtype=np.int64)


def decode(idx):
    return "".join(SYMBOLS[int(i)] for i in idx)
