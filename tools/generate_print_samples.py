#!/usr/bin/env python3
"""Materialize default `print` data and trace the engagement reward over it.

Generates the 25 broad-spectrum question formats x N random instances, dumps them
as JSON, and (unless --no-trace) runs the homeostatic energy / activation / recall
over a shuffled stream so you can eyeball the dynamics before any model exists.

Usage:
    python tools/generate_print_samples.py                  # 4 per format -> stdout summary
    python tools/generate_print_samples.py -n 8 -o data/print_samples.json
    python tools/generate_print_samples.py --no-trace       # just dump the data
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.data.formatters.print import PRINT_FORMATS, generate_print_samples
from praxis.policies.engagement_reward import HomeostaticEnergy, activation, recall


def _char_tokens(s: str):
    """Cheap, tokenizer-free token proxy for the offline trace: lowercased
    whitespace words. (Live training uses real token ids.)"""
    return s.lower().split()


def trace(samples):
    """Run the reward/energy as if the model predicted each ground-truth answer
    perfectly (upper bound) - sanity-checks that energy climbs and satiates."""
    energy = HomeostaticEnergy()
    acts, recs = [], []
    print(f"\nTrace over {len(samples)} samples (perfect-prediction upper bound):")
    print(f"{'step':>5} {'category':<16} {'act':>4} {'recall':>7} {'energy':>7}")
    for i, s in enumerate(samples):
        pred = _char_tokens(s["answer"])  # model "predicts" the answer
        resp = _char_tokens(s["answer"])  # simulated user says the answer
        a = activation(pred, resp)
        r = recall(pred, resp)
        e = energy.update(a)
        acts.append(a)
        recs.append(r)
        if i < 8 or i % max(1, len(samples) // 10) == 0:
            print(f"{i:>5} {s['category']:<16} {a:>4.0f} {r:>7.2f} {e:>7.3f}")
    print(
        f"\nactivation_rate={sum(acts)/len(acts):.3f}  "
        f"mean_recall={sum(recs)/len(recs):.3f}  final_energy={energy.value:.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--per-format", type=int, default=4)
    ap.add_argument("-o", "--out", type=str, default=None, help="write JSON here")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    samples = generate_print_samples(n_per_format=args.per_format)
    print(f"Generated {len(samples)} samples across {len(PRINT_FORMATS)} formats.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(samples, indent=2))
        print(f"Wrote {args.out}")

    if not args.no_trace:
        random.shuffle(samples)
        trace(samples)


if __name__ == "__main__":
    main()
