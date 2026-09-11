"""One home for every loss the model folds into its objective."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch.nn as nn

from praxis.losses.regularizer_base import BaseRegularizer


class Objectives(nn.Module):
    """Every loss term a run carries, keyed by the tag it reports under.

    The model used to hold two of these - ``criterion`` for the main loss and
    ``reg`` for the representation-shaping regularizers - while the terms
    computed inside classifiers and the MTP stack were bare ``F.cross_entropy``
    calls that appeared in neither. They are all the same kind of thing: a
    term the strategy sums into the objective. Registering them here under
    that tag puts every one in a single place, so the blueprint says what a
    run is actually optimizing.

    Terms are OWNED here. A producer that computes its own term (the MTP
    stack, a parallel classifier's per-arm CE) reads it back out of this container
    instead of holding it as a child, so no term is printed - or
    checkpointed - twice.
    """

    def __init__(self) -> None:
        super().__init__()
        self._register_load_state_dict_pre_hook(self._migrate_split_criterion)

    def register(self, name: str, loss: nn.Module) -> nn.Module:
        """Take ownership of ``loss`` under ``name`` and hand it back."""
        if name in self._modules:
            raise KeyError(f"objective '{name}' is already registered")
        if name != "main" and hasattr(type(self), name):
            raise KeyError(f"objective '{name}' collides with an attribute")
        if not isinstance(loss, nn.Module):
            raise TypeError(f"objective '{name}' is not a module: {type(loss)}")
        # Written straight into ``_modules`` rather than through
        # ``add_module``, which refuses any name the class already defines -
        # and ``main`` is the property just below, which has to stay a
        # property so it can answer None for an encoder-owned loss.
        self._modules[name] = loss
        return loss

    def claim(self, root: nn.Module) -> None:
        """Register the terms every module under ``root`` declares.

        A module that computes its own loss exposes ``objectives()`` ->
        ``{name: loss module}`` and reads the term back out of here (see
        :meth:`require`) instead of holding it as a child. Walked once at
        build time, so the blueprint is complete before a step runs.
        """
        for module in root.modules():
            declare = getattr(module, "objectives", None)
            if not callable(declare):
                continue
            for name, loss in declare().items():
                self.register(name, loss)

    @property
    def main(self) -> Optional[nn.Module]:
        """The primary criterion, or None when an encoder owns the loss."""
        return self._modules.get("main")

    def get(self, name: str) -> Optional[nn.Module]:
        return self._modules.get(name)

    def require(self, name: str) -> nn.Module:
        """The term registered under ``name``, or a hard failure.

        A producer whose term never got registered would otherwise silently
        fall back to computing its own - which is the arrangement this
        container exists to end.
        """
        term = self._modules.get(name)
        if term is None:
            raise KeyError(
                f"no objective registered as '{name}'; "
                f"registered: {sorted(self._modules)}"
            )
        return term

    def terms(self) -> Iterator[Tuple[str, nn.Module]]:
        return iter(self._modules.items())

    def regularizers(self) -> List[BaseRegularizer]:
        """The additive representation-shaping terms, in registration order."""
        return [m for m in self._modules.values() if isinstance(m, BaseRegularizer)]

    def reset(self) -> None:
        """Drop state the regularizers collected during a forward.

        Called unconditionally at the start of every model forward; see
        :meth:`BaseRegularizer.reset` for why it cannot wait for their own
        call.
        """
        for term in self.regularizers():
            term.reset()

    def training_metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for term in self._modules.values():
            fn = getattr(term, "training_metrics", None)
            if fn is not None:
                out.update(fn() or {})
        return out

    def dashboard_snapshots(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for term in self._modules.values():
            fn = getattr(term, "dashboard_snapshots", None)
            if fn is not None:
                out.update(fn() or {})
        return out

    def metric_descriptions(self) -> Iterator[Dict[str, Any]]:
        """Each term's chart hints, declared as a class attr on the loss."""
        for term in self._modules.values():
            descs = getattr(type(term), "metric_descriptions", None)
            if isinstance(descs, dict):
                yield descs

    def _migrate_split_criterion(self, state_dict, prefix, *args, **kwargs) -> None:
        """Load the main term from a checkpoint that predates this container.

        The main loss used to sit directly at ``criterion.*``. Only a term
        with parameters of its own ever wrote a key (HALO's gamma), so this is
        short - but without it those keys resume as missing.
        """
        names = set(self._modules)
        if "main" not in names:
            return
        for key in [k for k in state_dict if k.startswith(prefix)]:
            rest = key[len(prefix) :]
            if rest.split(".", 1)[0] in names:
                continue
            state_dict[prefix + "main." + rest] = state_dict.pop(key)

    def migrate_regularizer_keys(self, state_dict, prefix, *args, **kwargs) -> None:
        """Fold a pre-container checkpoint's ``reg.<index>.*`` onto its terms.

        Register this on the module that HOLDS the container as ``criterion``,
        not on the container: ``load_state_dict`` hands each child only the
        keys already under its own prefix, so by the time this container is
        reached the old top-level ``reg.*`` keys are out of view.
        """
        for index, term in enumerate(self.regularizers()):
            name = next(n for n, m in self._modules.items() if m is term)
            old = f"{prefix}reg.{index}."
            for key in [k for k in state_dict if k.startswith(old)]:
                new = f"{prefix}criterion.{name}.{key[len(old):]}"
                state_dict[new] = state_dict.pop(key)
