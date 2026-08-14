#!/usr/bin/env python3
"""Expose a Piper voice's duration predictor, so we get exact phoneme timings.

A VITS model already decides how long every phoneme lasts - that is what the
stochastic duration predictor does - but the exported ONNX graph keeps that
tensor internal and returns only audio. Promoting it to a graph output makes the
synthesizer hand us its own plan.

That is strictly better than forced alignment, which is what everyone reaches
for: a second model, a second dependency, and less accurate by construction,
since it guesses at boundaries the synthesizer already knows exactly.

The mutation is small: find the node whose op_type is "Ceil" - the ceiling of
the duration predictor, `w_ceil`, in frames per phoneme id - and add its output
to the graph's outputs. Patched models stay backward compatible, so an unpatched
loader still works.

This is the same transformation `piper.patch_voice_with_alignment` performs.
Doing it ourselves with the `onnx` package (Apache 2.0) means the GPL-licensed
piper-tts package is never installed, not even at build time.

    python patch_alignment.py <voice.onnx> [more.onnx ...]

Idempotent: a model that already exposes the tensor is left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch(path: Path) -> str:
    import onnx

    model = onnx.load(str(path))
    graph = model.graph
    existing = {o.name for o in graph.output}

    ceil_nodes = [n for n in graph.node if n.op_type == "Ceil"]
    if not ceil_nodes:
        return f"{path.name}: no Ceil node - not a VITS graph we recognize, skipped"
    if len(ceil_nodes) > 1:
        # Never seen in Piper's exports, but if it happens, guessing is worse
        # than stopping: the wrong tensor would give plausible nonsense timings.
        return (
            f"{path.name}: {len(ceil_nodes)} Ceil nodes, expected 1 - "
            "skipped rather than guess"
        )

    tensor = ceil_nodes[0].output[0]
    if tensor in existing:
        return f"{path.name}: already patched"

    graph.output.append(
        onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, None)
    )
    tmp = path.with_suffix(path.suffix + ".part")
    onnx.save(model, str(tmp))
    tmp.replace(path)  # atomic, so a reader never sees a half-written graph
    return f"{path.name}: patched, exposing '{tensor}'"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        p = Path(arg).expanduser()
        if not p.exists():
            print(f"{arg}: not found")
            return 1
        print(patch(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
