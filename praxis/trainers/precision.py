"""Numeric precision profiles for a training run.

A single ``--precision`` flag drives three knobs that otherwise drift apart:

1. the dtype the model's parameters and buffers are built in,
2. the precision string Lightning runs the step at (which is what actually
   decides whether gradients are low-precision),
3. the float32 matmul kernel policy torch uses for whatever stays in fp32.

Profiles live in a registry rather than as branches at the call sites, so a
new level is an entry here and nothing else. ``resolve_precision`` applies the
device heuristics - a request the hardware cannot honor is downgraded to the
nearest thing that runs, loudly, instead of dying mid-run.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

DEFAULT_PRECISION = "float32"


@dataclass(frozen=True)
class PrecisionProfile:
    """One coherent (model dtype, trainer precision, matmul policy) triple."""

    name: str
    # Lightning ``Trainer(precision=...)``. The "-true" variants put weights,
    # activations and gradients in that dtype; "-mixed" keeps fp32 master
    # weights and autocasts the forward.
    lightning: str
    # torch.set_float32_matmul_precision(...) - governs the ops that remain
    # fp32 regardless of the above (TF32 kernels on Ampere+).
    #
    # "high" rather than "medium" for every non-fp64 profile. On CUDA the two
    # currently resolve identically (torch 2.10: both give allow_tf32=True,
    # fp32_precision='tf32'), so this costs nothing measurable - but "medium"
    # is DEFINED as permitting a bf16 internal datatype, an 8-bit mantissa,
    # and a mode named float32 should not be leaving that door open for
    # whichever backend decides to walk through it.
    matmul: str
    # Cast the assembled model here. None leaves it in fp32, which is correct
    # for the mixed-precision profiles: their master weights are fp32.
    param_dtype: Optional[str] = None
    note: str = ""

    @property
    def torch_dtype(self) -> Optional[Any]:
        """The torch dtype to cast parameters to, or None to leave them be."""
        import torch

        return None if self.param_dtype is None else getattr(torch, self.param_dtype)


PRECISION_REGISTRY: Dict[str, PrecisionProfile] = {
    "float64": PrecisionProfile(
        name="float64",
        lightning="64-true",
        matmul="highest",
        param_dtype="float64",
        note="double precision end-to-end; ~1/64 throughput on consumer GPUs",
    ),
    "float32": PrecisionProfile(
        name="float32",
        lightning="32-true",
        matmul="high",
        note="fp32 weights and gradients, TF32 matmul kernels where available",
    ),
    "bfloat16": PrecisionProfile(
        name="bfloat16",
        lightning="bf16-true",
        matmul="high",
        param_dtype="bfloat16",
        note="weights, activations and gradients all in bf16",
    ),
    "float16": PrecisionProfile(
        name="float16",
        lightning="16-mixed",
        matmul="high",
        # Deliberately not "16-true": fp16 has no exponent headroom for
        # gradients, so pure fp16 training needs loss scaling to survive.
        # The mixed plugin gives fp16 compute with an fp32 master copy and a
        # GradScaler; bf16 is the profile to reach for if you want the weights
        # themselves halved.
        note="fp16 compute with fp32 master weights and loss scaling",
    ),
}

# Shorthand people actually type. Canonical names stay the registry keys.
PRECISION_ALIASES: Dict[str, str] = {
    "fp64": "float64",
    "64": "float64",
    "double": "float64",
    "fp32": "float32",
    "32": "float32",
    "tf32": "float32",
    "full": "float32",
    "bf16": "bfloat16",
    "bfloat": "bfloat16",
    "fp16": "float16",
    "16": "float16",
    "half": "float16",
}

PRECISION_CHOICES = sorted(PRECISION_REGISTRY) + sorted(PRECISION_ALIASES)


def canonical_precision(name: Optional[str]) -> str:
    """Registry key for a user-supplied precision name (or the default)."""
    if name is None:
        return DEFAULT_PRECISION
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    key = PRECISION_ALIASES.get(key, key)
    if key not in PRECISION_REGISTRY:
        raise ValueError(
            f"Unknown precision '{name}'. Choose one of: "
            + ", ".join(sorted(PRECISION_REGISTRY))
        )
    return key


def _cuda_supports_bf16() -> bool:
    """Whether the visible CUDA device has real bf16 support (Ampere+)."""
    import torch

    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def resolve_precision(
    name: Optional[str] = None, device: str = "cpu", verbose: bool = True
) -> PrecisionProfile:
    """The profile this machine can actually run for the requested level.

    Downgrades are announced rather than silent: a run that asked for fp16 and
    got bf16 is a different experiment, and the log is the only place that
    difference is recoverable after the fact.
    """
    profile = PRECISION_REGISTRY[canonical_precision(name)]
    on_cuda = str(device).startswith("cuda")

    def _swap(target: str, why: str) -> PrecisionProfile:
        if verbose:
            print(f"[INIT] {profile.name} unavailable ({why}); using {target}.")
        return PRECISION_REGISTRY[target]

    if not on_cuda:
        # CPU kernels for fp16 are largely unimplemented ("addmm_impl_cpu_"
        # not implemented for 'Half'), while bf16 has coverage.
        if profile.name == "float16":
            return _swap("bfloat16", "no fp16 kernels on CPU")
    elif profile.name == "bfloat16" and not _cuda_supports_bf16():
        return _swap("float16", "device predates bf16 support")

    return profile


def apply_precision(profile: PrecisionProfile, verbose: bool = True) -> None:
    """Install the process-wide half of a profile: the float32 matmul policy."""
    import torch

    try:
        torch.set_float32_matmul_precision(profile.matmul)
    except Exception as e:  # older torch / exotic backends
        print(f"[INIT] Could not set float32 matmul precision: {e}")

    if verbose:
        print(
            f"[INIT] Precision: {profile.name} "
            f"(trainer={profile.lightning}, matmul={profile.matmul}) - {profile.note}"
        )


@contextmanager
def init_context(profile: PrecisionProfile):
    """Build a model under the profile's dtype, then restore the default.

    Scoped rather than global on purpose: parameters and buffers created inside
    modules should follow the profile, but the loss containers, metrics and
    dataset tensors built elsewhere are fp32/int by intent. Lightning's own
    precision plugin re-enters the same dtype around each forward, so nothing
    downstream depends on this staying set.
    """
    import torch

    dtype = profile.torch_dtype
    if dtype is None:
        yield
        return

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def cast_module(model: Any, profile: PrecisionProfile) -> Any:
    """Put an assembled model in the profile's parameter dtype.

    Lightning's true-precision plugins would do this at setup time anyway, but
    doing it here means the optimizer is built against the dtype the model will
    actually train in, and that inference paths which never touch the Lightning
    trainer (the Generator, mono_forward) see the same model.
    """
    dtype = profile.torch_dtype
    if dtype is None:
        return model
    return model.to(dtype)
