"""Per-band LSM shard store: manifest, append, iterate, tiered compaction.

Each fuzzy-dedup band gets its own directory of *immutable, sorted-unique*
int64 shards rather than one globally-sorted file. This avoids the O(N)
read-whole-index + union + rewrite-whole-index cost on every ingested
release:

  * **Append** (``append_shard``): a release writes ONE new level-0 shard
    holding only its own sorted-unique band keys — O(|D_k|), no rewrite of
    prior history. This is "commit-once L0": the shard is written exactly
    once per dataset tag and is never touched again except by compaction
    (which folds it into a merged run) or withdrawal (which unlinks it).
    A resumed/retried ingest overwrites its own shard by tag, so retries are
    idempotent rather than duplicating keys.
  * **Screen**: handled by ``puffer.screen`` — per-shard ``searchsorted``,
    OR-ed across shards. Membership in the union of shards is identical to
    membership in one combined sorted array, so results never change; only
    the cost of computing them does.
  * **Compact** (``compact_band``): size-tiered ONLY. Shards are bucketed by
    manifest ``level``; once a level accumulates >= ``T`` (``tier_fanout``)
    unprotected runs, exactly ``T`` of the smallest are merged into one run
    at ``level + 1``. Levels only increase on merge, so any given key is
    rewritten at most ``O(log_T(#releases))`` times regardless of how much
    it overlaps with other releases — this is the base-T counter argument:
    a level-``L`` run has survived >= ``T`` merges at every level below it,
    so after ``k`` releases there are at most ``T-1`` runs per level and
    ``O(log_T k)`` levels, bounding both shard count (screen cost) and total
    write volume (compaction cost). The policy is stamped into the band
    manifest on first use (``compaction_policy: {"name": "tiered", "T": T}``)
    and enforced on every later call; passing a different ``T`` later raises
    rather than silently changing the fanout bound the persisted state
    already relies on.

Shards are raw little-endian int64 (band keys are signed-reinterpreted
xxh64 digests — NOT reinterpreted as uint64, which would flip sort order and
break ``searchsorted``). No Parquet, no header: read back via ``np.memmap``
so even multi-GB bands cost near-zero resident RAM.

Withdrawal (``puffer.withdraw``) is index-STATE removal: it deletes/rebuilds
the shards that carry a dataset's keys so the index behaves as if that
dataset's keys were never combined into a merged run. It does NOT
retroactively re-derive any other dataset's already-written dedup outputs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"[^A-Za-z0-9_.=-]")


def sanitize_tag(tag: str) -> str:
    """Make a dataset name safe + deterministic for use as a shard filename."""
    safe = _TAG_RE.sub("_", tag.strip())
    return safe or "shard"


def band_dir(index_dir: Path, band_id: int) -> Path:
    return Path(index_dir) / f"band_{band_id:02d}"


def _manifest_path(index_dir: Path, band_id: int) -> Path:
    return band_dir(index_dir, band_id) / "manifest.json"


def load_manifest(index_dir: Path, band_id: int) -> dict:
    mp = _manifest_path(index_dir, band_id)
    if not mp.exists():
        return {"band": band_id, "seq": 0, "shards": []}
    try:
        return json.loads(mp.read_text())
    except (OSError, ValueError):
        return {"band": band_id, "seq": 0, "shards": []}


def write_manifest(index_dir: Path, band_id: int, manifest: dict) -> None:
    mp = _manifest_path(index_dir, band_id)
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp.with_name(f"{mp.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(manifest))
        os.replace(tmp, mp)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_shard_bin(arr, path: Path) -> None:
    """Write a sorted-unique int64 array as a raw .bin shard, atomically.

    Written to a temp file and ``os.replace``-d into place so a kill never
    leaves a truncated/corrupt shard visible under its final name.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        np.ascontiguousarray(arr, dtype=np.int64).tofile(str(tmp))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_shard_bin(path: Path, mmap: bool = True):
    """Read a raw int64 .bin shard. ``mmap=True`` returns a memory-mapped view."""
    import numpy as np

    path = Path(path)
    try:
        nbytes = path.stat().st_size
    except OSError:
        return np.empty(0, dtype=np.int64)
    if nbytes == 0:
        return np.empty(0, dtype=np.int64)
    if mmap:
        return np.memmap(str(path), dtype=np.int64, mode="r")
    return np.fromfile(str(path), dtype=np.int64)


def iter_shards(
    index_dir: Path, band_id: int, exclude_tag: str | None = None,
) -> list[tuple[dict, Path]]:
    """Return ``(meta, path)`` for each shard in a band, honoring ``exclude_tag``.

    ``exclude_tag`` drops the current dataset's own shard so a resumed/retried
    ingest never screens a dataset against itself.
    """
    manifest = load_manifest(index_dir, band_id)
    bdir = band_dir(index_dir, band_id)
    skip = sanitize_tag(exclude_tag) if exclude_tag else None
    out: list[tuple[dict, Path]] = []
    for s in manifest.get("shards", []):
        if skip is not None and s.get("dataset") == skip:
            continue
        out.append((s, bdir / s["file"]))
    return out


def append_shard(new_unique_np, index_dir: Path, band_id: int, tag: str) -> None:
    """Append one dataset's unique keys as a new level-0 sorted .bin shard.

    ``new_unique_np`` must already be sorted-unique for this dataset. This is
    the commit-once L0 write: exactly one shard per ``tag`` at level 0.
    Idempotent on ``tag`` — a re-run overwrites its own shard + manifest
    entry instead of duplicating it, so retries after a crash never grow the
    index.
    """
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(new_unique_np), dtype=np.int64)
    if arr.size == 0:
        return
    safe_tag = sanitize_tag(tag)
    manifest = load_manifest(index_dir, band_id)
    bdir = band_dir(index_dir, band_id)
    name = f"{safe_tag}.bin"
    write_shard_bin(arr, bdir / name)
    shards = [s for s in manifest.get("shards", []) if s.get("dataset") != safe_tag]
    shards.append({
        "file": name,
        "level": 0,
        "dataset": safe_tag,
        "source_datasets": [safe_tag],
        "count": int(arr.size),
        "min": int(arr[0]),
        "max": int(arr[-1]),
    })
    manifest["shards"] = shards
    write_manifest(index_dir, band_id, manifest)


def read_band_union(index_dir: Path, band_id: int, exclude_tag: str | None = None):
    """Combined sorted-unique band keys (numpy int64), or ``None`` if empty.

    Test/maintenance helper — unions all shards in RAM. The hot screening
    path (``puffer.screen``) searches shards individually instead, so it
    never pays this full-history union/sort.
    """
    import numpy as np

    shards = iter_shards(index_dir, band_id, exclude_tag)
    if not shards:
        return None
    parts = []
    for _meta, path in shards:
        a = read_shard_bin(path, mmap=False)
        if a.size:
            parts.append(np.asarray(a))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.unique(np.concatenate(parts))


def _protected_tags_path(index_dir: Path) -> Path:
    return Path(index_dir) / "protected_tags.json"


def load_protected_tags(index_dir: Path) -> set[str]:
    p = _protected_tags_path(index_dir)
    if not p.exists():
        return set()
    try:
        return {sanitize_tag(t) for t in json.loads(p.read_text())}
    except (OSError, ValueError):
        return set()


def set_protected_tags(index_dir: Path, tags) -> None:
    """Persist dataset tags that compaction must never merge.

    A protected dataset's per-band shard stays an unmerged level-0 tag shard
    forever, reserving a perpetual O(1) withdrawal path for it. Overwrites
    the whole list; pass the full set.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    p = _protected_tags_path(index_dir)
    data = sorted({sanitize_tag(t) for t in tags})
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def compact_band(
    index_dir: Path,
    band_id: int,
    tier_fanout: int,
    ram_budget: int,
    protect_tag: str | None = None,
) -> int:
    """Size-tiered compaction: merge ``T`` same-level unprotected runs into
    one run at ``level + 1``, looping until every level holds fewer than
    ``T`` unprotected runs.

    The policy is stamped into the band manifest the first time a band is
    compacted (``compaction_policy: {"name": "tiered", "T": tier_fanout}``)
    and honored on every later call. Requesting a different ``T`` for an
    already-stamped band raises — changing T mid-life would silently break
    the fanout / write-amplification bound the persisted shard levels
    already assume; rebuild the index instead.

    Terminates because each merge round removes ``T`` runs and adds one, so
    the number of unprotected runs at levels <= the merge target strictly
    decreases. Returns the total number of shards merged (across all merge
    rounds this call performed).
    """
    from puffer.merge import merge_runs

    if tier_fanout < 2:
        raise ValueError(f"tier_fanout must be >= 2, got {tier_fanout}")
    manifest = load_manifest(index_dir, band_id)
    pol = manifest.get("compaction_policy") or {}
    if pol.get("name") == "tiered":
        stamped_t = int(pol["T"])
        if int(tier_fanout) != stamped_t:
            raise ValueError(
                f"band {band_id} is stamped tiered T={stamped_t}; got "
                f"tier_fanout={tier_fanout} (changing T mid-life breaks the "
                "fanout/write-amplification bound; rebuild the index to "
                "change T)"
            )
    else:
        manifest["compaction_policy"] = {"name": "tiered", "T": int(tier_fanout)}
        write_manifest(index_dir, band_id, manifest)

    protect = sanitize_tag(protect_tag) if protect_tag else None
    protected = load_protected_tags(index_dir)
    merged_total = 0
    while True:
        manifest = load_manifest(index_dir, band_id)
        by_level: dict[int, list[dict]] = {}
        for s in manifest.get("shards", []):
            if s.get("dataset") == protect or s.get("dataset") in protected:
                continue
            by_level.setdefault(int(s.get("level", 0)), []).append(s)
        full = [lvl for lvl, runs in by_level.items() if len(runs) >= tier_fanout]
        if not full:
            return merged_total
        lvl = min(full)
        group = sorted(by_level[lvl], key=lambda s: s.get("count", 0))[:tier_fanout]
        merged_total += merge_runs(
            index_dir, band_id, manifest, group, lvl + 1, ram_budget,
        )
