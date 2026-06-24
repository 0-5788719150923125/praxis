"""Git-history training data: the repository's own evolution as a self-lineage signal.

Samples commits recency-weighted (recent code is drawn more often, the free proxy
for loss influence) and emits per-file **before -> after** transitions, each
grounded with an in-band timestamp: a normalized lifespan position ``t in [0,1]``
(0 = first commit, 1 = HEAD) plus the ISO date. This gives a model temporal
ordering between otherwise-shuffled samples - the phase along the project lifespan.

Selected via the dataset registry (``DATASETS["git-history"]`` / the ``git_history``
collection), not a CLI flag - in keeping with the tuning-free, registry-driven
design. Reads the current repo by default; ``config["repo"]`` points elsewhere.

Git extraction mirrors ``praxis/pillars/evolution.py`` (the recency kernel turned
on the repo), swapped from ``--numstat`` to a parsed ``git show`` patch. Design +
decisions: ``next/self_lineage.md``.

v1 scope: recency = sampling frequency; value = HEAD-as-truth. A true per-sample
loss weight (the smooth fixed continuum) and survival-weighting are deferred to v2.
"""

import math
import random
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from praxis.data.datasets.base import PraxisSampler

# Tunables (baked in - fixed model-agnostic constants, not per-experiment knobs).
MAX_COMMITS = 4000  # cap history scan for speed, like evolution.py
RECENCY_DECAY = 3.0  # exp falloff into the past; newest weight 1.0, oldest ~0.05
MAX_FILES_PER_COMMIT = 6  # don't let one large commit flood the cache
MAX_SIDE_BYTES = 6000  # per-side cap; the 2MB-OOM lesson, scaled to a sample
MAX_PICK_ATTEMPTS = 12  # tries to find a commit with usable (non-binary) changes
DIFF_CONTEXT = 3  # unified-diff context lines around each hunk


# --------------------------------------------------------------------------- #
# git plumbing
# --------------------------------------------------------------------------- #
def _run_git(args: List[str], repo: str) -> str:
    """Run a git command in ``repo``, returning stdout ('' on any failure)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        ).stdout
    except Exception:
        return ""


def _commit_list(repo: str) -> List[Tuple[str, int]]:
    """[(sha, unix_ts)] oldest-first, no merges, capped. [] if git is unusable."""
    out = _run_git(["log", "--no-merges", f"-n{MAX_COMMITS}", "--format=%H %ct"], repo)
    commits: List[Tuple[str, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                commits.append((parts[0], int(parts[1])))
            except ValueError:
                continue
    commits.reverse()  # oldest -> newest
    return commits


def _clean_path(p: str) -> Optional[str]:
    """Resolve a diff path; None for /dev/null (an absent side)."""
    p = p.strip().strip('"')
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def _parse_patch(text: str) -> List[Tuple[str, str, str, bool, bool]]:
    """Parse a ``git show`` patch into per-file (path, before, after, is_new,
    is_deleted). Reconstructs before/after *states* from the hunks (context+'-'
    -> before, context+'+' -> after), so the model sees code states, not diff
    syntax. Binary/empty files are skipped."""
    files: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal cur
        if cur and (cur["before"] or cur["after"]) and not cur["binary"]:
            files.append(cur)
        cur = None

    for line in text.split("\n"):
        if line.startswith("diff --git "):
            flush()
            cur = {
                "pre": "",
                "post": "",
                "before": [],
                "after": [],
                "binary": False,
                "in_hunk": False,
            }
            continue
        if cur is None:
            continue
        if line.startswith("Binary files") or line.startswith("GIT binary patch"):
            cur["binary"] = True
        elif line.startswith("--- "):
            cur["pre"] = line[4:]
        elif line.startswith("+++ "):
            cur["post"] = line[4:]
        elif line.startswith("@@"):
            cur["in_hunk"] = True
            # blank separator marks a discontinuity between hunks
            if cur["before"]:
                cur["before"].append("")
            if cur["after"]:
                cur["after"].append("")
        elif not cur["in_hunk"]:
            continue  # index / mode / rename headers
        elif line.startswith("-") and not line.startswith("---"):
            cur["before"].append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            cur["after"].append(line[1:])
        elif line.startswith("\\"):
            continue  # "\ No newline at end of file"
        else:  # context line (leading space) or empty
            content = line[1:] if line.startswith(" ") else line
            cur["before"].append(content)
            cur["after"].append(content)
    flush()

    out: List[Tuple[str, str, str, bool, bool]] = []
    for f in files:
        pre, post = _clean_path(f["pre"]), _clean_path(f["post"])
        path = post or pre or "unknown"
        out.append(
            (path, "\n".join(f["before"]), "\n".join(f["after"]), pre is None, post is None)
        )
    return out


def _changed_pairs(repo: str, sha: str) -> List[Tuple[str, str, str, bool, bool]]:
    out = _run_git(
        ["show", sha, "--no-color", f"--unified={DIFF_CONTEXT}", "--format="], repo
    )
    return _parse_patch(out)


# --------------------------------------------------------------------------- #
# sample formatting
# --------------------------------------------------------------------------- #
def _cap(s: str) -> str:
    s = s.strip("\n")
    return s if len(s) <= MAX_SIDE_BYTES else s[:MAX_SIDE_BYTES] + "\n... [truncated]"


def _format_sample(
    path: str, before: str, after: str, is_new: bool, is_deleted: bool, iso: str, t: float, short: str
) -> str:
    before_block = "(new file)" if is_new else _cap(before)
    after_block = "(deleted file)" if is_deleted else _cap(after)
    if not before_block.strip() and not after_block.strip():
        return ""
    return (
        f"[git transition] {path}\n"
        f"when {iso} · lifespan {t:.3f} · commit {short}\n"
        f"--- before ---\n{before_block}\n"
        f"--- after ---\n{after_block}\n"
    )


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class GitHistoryDataset(PraxisSampler):
    """Yields recency-weighted per-file before/after transitions from a repo's
    git history, each stamped with its lifespan position and date.

    Registry-driven: ``DATASETS["git-history"]`` (type ``git_history``). The
    optional ``config["repo"]`` overrides the default (the current repo)."""

    def __init__(self, tokenizer: Any, seed: int, config: Optional[Dict] = None):
        super().__init__(tokenizer)
        config = config or {}
        self.dataset_path = "git_history"  # name for metrics tracking
        self.repo = config.get("repo", ".")
        self.rng = random.Random(seed)
        self.commits = _commit_list(self.repo)
        self._pending: List[str] = []  # raw before/after samples awaiting emission
        if self.commits:
            self._t0 = self.commits[0][1]
            self._span = max(self.commits[-1][1] - self._t0, 1)
            self._weights = [
                math.exp(-RECENCY_DECAY * (1.0 - (ts - self._t0) / self._span))
                for _, ts in self.commits
            ]
        else:
            self._t0, self._span, self._weights = 0, 1, []
            print(f"[GIT_HISTORY] no commits found in {self.repo!r}; dataset is empty.")

    def _refill_pending(self) -> None:
        """Pick a recency-weighted commit and queue its files' before/after pairs."""
        for _ in range(MAX_PICK_ATTEMPTS):
            idx = self.rng.choices(range(len(self.commits)), weights=self._weights, k=1)[0]
            sha, ts = self.commits[idx]
            pairs = _changed_pairs(self.repo, sha)
            if not pairs:
                continue
            if len(pairs) > MAX_FILES_PER_COMMIT:
                pairs = self.rng.sample(pairs, MAX_FILES_PER_COMMIT)
            t = (ts - self._t0) / self._span
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            short = sha[:9]
            for path, before, after, is_new, is_deleted in pairs:
                sample = _format_sample(path, before, after, is_new, is_deleted, iso, t, short)
                if sample:
                    self._pending.append(sample)
            if self._pending:
                return

    def _next_raw(self) -> str:
        """One before/after transition as raw text ('' if history is unusable)."""
        if not self.commits:
            return ""
        if not self._pending:
            self._refill_pending()
        return self._pending.pop(0) if self._pending else ""

    def get_document(self) -> Dict:
        """A document for the message-queue manager: the transition as raw text
        wrapped in the same system/developer/assistant shape as plain-text
        sources (format_simple), but verbatim - no prose reformatting, which
        would mangle code indentation."""
        text = self._next_raw()
        if not text:
            return {"messages": [], "metadata": {}}
        # Deferred import keeps the datasets package free of a config import cycle.
        from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "developer", "content": sample_developer_prompt("continue_text")},
            {"role": "assistant", "content": text},
        ]
        return {"messages": messages, "metadata": {"format": "git_history"}}

    def fill_sequence_cache(self) -> None:
        """Legacy text path: one raw transition per call."""
        self.sequence_cache.append(self._next_raw())


# Smoke test: `python -m praxis.data.datasets.git_history [repo]`
if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    ds = GitHistoryDataset(tokenizer=None, seed=42, config={"repo": repo})
    print(f"commits found: {len(ds.commits)}")
    if ds.commits:
        print(f"lifespan: {ds._span / 86400.0:.1f} days across {len(ds.commits)} commits\n")
        for s in ds.get_sequences(3):
            clip = s[:1400] + ("\n... [clipped]" if len(s) > 1400 else "")
            print("=" * 72)
            print(clip)
