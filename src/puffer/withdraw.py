"""Dataset withdrawal: remove one dataset's keys from the LSM band index.

Withdrawal is INDEX-STATE removal, not data-lineage undo: it restores the
band index to the state it would be in had the withdrawn dataset's keys
never been folded into any merged run, and it deletes that dataset's own
artifacts (sidecars, removals manifest). It does NOT retroactively re-derive
any OTHER dataset's already-written dedup output parquets — a document that
an earlier release dropped because it collided with the withdrawn dataset
stays dropped in that release's output; only the index (and hence future
ingests) forgets the withdrawn dataset ever existed.

Per band there are two cases:

  * **Standalone (O(1))**: the withdrawn tag's own level-0 shard is still
    present (it was never folded into a merged run) — unlink it and drop its
    manifest entry. No other shard is touched.
  * **Compacted**: one or more merged runs carry the tag's keys (explicitly,
    via ``source_datasets``, or — for provenance-unknown legacy shards —
    conservatively). Those runs must be REBUILT from the union of their
    SURVIVING contributors, never derived from any dataset's dedup output
    (dedup outputs have already dropped rows that collided with keys the
    rebuild must still represent). ``withdraw_dataset`` rebuilds affected
    runs by streaming-merging the surviving contributors' per-dataset
    SIDECAR files (``state/datasets/<t>/band_XX.bin`` — the exact array each
    contributor appended for that band, kept precisely so this rebuild never
    needs to re-read or re-hash any dataset's original text).

``plan_band_withdrawal`` is a pure, read-only classification step (safe to
call repeatedly); ``apply_band_withdrawal`` performs the crash-safe
write — new rebuilt shard (if any) written and the manifest swapped BEFORE
any old shard file is unlinked, so a crash before the swap leaves the prior
state fully intact and a re-applied plan is then a no-op (the tag's files
are already gone from the manifest, so ``remove_files`` recomputed against
the current manifest is empty).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _shard_source_datasets(entry: dict) -> list[str] | None:
    v = entry.get("source_datasets")
    if v is not None:
        return list(v)
    # Legacy manifests predate per-shard provenance. A level-0 shard is always
    # one dataset's standalone run (see index.append_shard), so its provenance
    # is intrinsically its own ``dataset`` -- infer it rather than treating the
    # shard as unknown, which would over-conservatively force full rebuilds when
    # withdrawing an already-compacted tag.
    if int(entry.get("level", 0)) == 0 and entry.get("dataset"):
        return [entry["dataset"]]
    return None


def plan_band_withdrawal(index_dir: Path, band_id: int, tag: str) -> dict:
    """Classify one band's shards for withdrawing ``tag`` (read-only).

    Returns ``{"tag_files", "affected", "contributors"}``:

    - ``tag_files``: the dataset's own level-0 shard file(s) — withdrawal is
      an O(1) unlink + manifest edit.
    - ``affected``: merged shard entries that (may) contain the dataset's
      keys and must be rebuilt from their surviving contributors. When the
      tag's own shard is still present, only merged shards *explicitly*
      listing the tag are affected; when it is absent (already compacted
      away), shards with unknown provenance are conservatively affected too.
    - ``contributors``: surviving dataset tags to rebuild ``affected`` from,
      or ``None`` when any affected shard has unknown provenance (caller
      must rebuild from every remaining dataset).
    """
    from puffer.index import load_manifest, sanitize_tag

    manifest = load_manifest(index_dir, band_id)
    safe_tag = sanitize_tag(tag)
    tag_files: list[str] = []
    explicit: list[dict] = []
    unknown: list[dict] = []
    for s in manifest.get("shards", []):
        if s.get("dataset") == safe_tag:
            tag_files.append(s["file"])
            continue
        src = _shard_source_datasets(s)
        if src is None:
            unknown.append(s)
        elif safe_tag in src:
            explicit.append(s)
    # Own tag shard present => never compacted in this band (compaction
    # removes merged inputs' entries), so unknown-provenance shards cannot
    # carry its keys.
    affected = explicit if tag_files else explicit + unknown
    if not affected:
        contributors: list[str] | None = []
    else:
        srcs = [_shard_source_datasets(s) for s in affected]
        if any(v is None for v in srcs):
            contributors = None
        else:
            contributors = sorted({t for v in srcs for t in v} - {safe_tag})
    return {"tag_files": tag_files, "affected": affected, "contributors": contributors}


def apply_band_withdrawal(
    index_dir: Path,
    band_id: int,
    tag: str,
    plan: dict,
    *,
    rebuilt_path: Path | None = None,
    rebuilt_count: int = 0,
    rebuilt_min: int = 0,
    rebuilt_max: int = 0,
    rebuilt_sources: list[str] | None = None,
) -> None:
    """Apply a :func:`plan_band_withdrawal` plan to one band.

    Unlinks the tag's own shard(s); when the plan has ``affected`` merged
    shards, replaces them with the single rebuilt shard already written at
    ``rebuilt_path`` (a bounded-memory sorted-unique union file, published by
    an atomic rename), using ``rebuilt_count``/``rebuilt_min``/``rebuilt_max``
    for the manifest entry -- the union is NEVER loaded into RAM here.
    ``rebuilt_path=None``/``rebuilt_count==0`` just removes the affected shards
    outright. Crash-safe: the new shard is renamed into place and the manifest
    is swapped BEFORE any old shard file is unlinked. Applying the same plan
    twice is a no-op the second time: ``remove_files`` is computed from the
    plan's file *names*, and by the second call those names are no longer in
    the manifest and the files are already gone, so nothing changes. Any unused
    ``rebuilt_path`` temp is removed on exit.
    """
    from puffer.index import band_dir, load_manifest, write_manifest

    try:
        manifest = load_manifest(index_dir, band_id)
        bdir = band_dir(index_dir, band_id)
        remove_files = set(plan["tag_files"]) | {s["file"] for s in plan["affected"]}
        if not remove_files:
            return
        shards = [s for s in manifest.get("shards", []) if s["file"] not in remove_files]
        if plan["affected"] and rebuilt_path is not None and rebuilt_count > 0:
            seq = int(manifest.get("seq", 0)) + 1
            new_name = f"rb_{seq:06d}.bin"
            os.replace(rebuilt_path, bdir / new_name)   # publish before manifest swap
            rebuilt_path = None
            shards.append({
                "file": new_name,
                "level": max((int(s.get("level", 0)) for s in plan["affected"]), default=1),
                "dataset": new_name[:-4],
                "source_datasets": sorted(rebuilt_sources or []),
                "count": int(rebuilt_count),
                "min": int(rebuilt_min),
                "max": int(rebuilt_max),
            })
            manifest["seq"] = seq
        manifest["shards"] = shards
        write_manifest(index_dir, band_id, manifest)
        for f in remove_files:
            try:
                (bdir / f).unlink()
            except FileNotFoundError:
                pass
    finally:
        if rebuilt_path is not None:
            try:
                Path(rebuilt_path).unlink()
            except FileNotFoundError:
                pass


def _sidecar_path(state_dir: Path, tag: str, band_id: int) -> Path:
    from puffer.index import sanitize_tag

    return Path(state_dir) / "datasets" / sanitize_tag(tag) / f"band_{band_id:02d}.bin"


def _rebuild_from_sidecars(
    index_dir: Path, band_id: int, sidecar_paths: list[Path], ram_budget: int,
):
    """Union the surviving contributor sidecars into one sorted-unique file
    under a hard ``ram_budget`` via the compiled loser-tree (bounded RAM, no
    mmap). Returns ``(out_path | None, count, min, max)``; ``out_path`` is a
    temp file in the band directory for the caller to rename into place, or
    ``None`` when the union is empty."""
    from puffer.bounded_merge import bounded_merge_unique
    from puffer.index import band_dir

    existing = [
        Path(p) for p in sidecar_paths
        if Path(p).exists() and Path(p).stat().st_size > 0
    ]
    if not existing:
        return None, 0, 0, 0
    bdir = band_dir(index_dir, band_id)
    out_path = bdir / f"_withdraw_rebuild.tmp.{os.getpid()}.b{band_id:02d}"
    count, mn, mx = bounded_merge_unique(existing, out_path, ram_budget)
    if count == 0:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        return None, 0, 0, 0
    return out_path, count, mn, mx


def withdraw_dataset(state_dir: Path, tag: str, cfg) -> dict:
    """Withdraw ``tag`` from every band of the LSM index and delete its
    artifacts (sidecars, removals manifest).

    For each band: O(1) unlink when the tag's shard is still standalone;
    otherwise rebuild every affected merged run by streaming-merging the
    SURVIVING contributors' sidecar files (``state/datasets/<t>/band_XX.bin``
    — never dedup output text, which has already dropped collided rows).
    When a merged shard's provenance is unknown, falls back to rebuilding
    from every OTHER dataset currently registered under ``state/datasets/``.
    When any band needs a rebuild but the compiled loser-tree kernel cannot
    load (no C compiler / unwritable package dir), refuses up front and
    modifies NO band; purely standalone (O(1)) withdrawals never need it.

    Returns ``{"bands_o1": int, "bands_rebuilt": int, "elapsed_s": float}``.
    """
    import shutil

    from puffer.index import sanitize_tag

    start = time.monotonic()
    state_dir = Path(state_dir)
    index_dir = state_dir / "hash_index"
    datasets_dir = state_dir / "datasets"
    safe_tag = sanitize_tag(tag)

    other_tags = sorted(
        p.name for p in datasets_dir.iterdir()
        if p.is_dir() and p.name != safe_tag
    ) if datasets_dir.exists() else []

    bands_o1 = 0
    bands_rebuilt = 0
    plans = [plan_band_withdrawal(index_dir, b, safe_tag) for b in range(cfg.num_bands)]

    # Refuse up front -- BEFORE any band is mutated -- when a rebuild is
    # needed but the compiled loser-tree merge cannot load. Raising mid-loop
    # would leave earlier bands already withdrawn (recoverable by retry, but
    # avoidably messy); a standalone-only withdrawal proceeds without it.
    if any(p["affected"] for p in plans):
        from puffer.bounded_merge import _load_lib

        if not _load_lib():
            raise RuntimeError(
                f"withdrawing {tag!r} requires rebuilding merged index "
                "segments with the compiled loser-tree kernel (_kway.c): "
                "install a C compiler (gcc) and ensure the package directory "
                "is writable. No band was modified."
            )

    for band_id in range(cfg.num_bands):
        plan = plans[band_id]
        if not plan["tag_files"] and not plan["affected"]:
            continue
        if not plan["affected"]:
            apply_band_withdrawal(index_dir, band_id, safe_tag, plan)
            bands_o1 += 1
            continue

        contributors = plan["contributors"]
        if contributors is None:
            contributors = other_tags
        sidecar_paths = [_sidecar_path(state_dir, t, band_id) for t in contributors]
        out_path, cnt, mn, mx = _rebuild_from_sidecars(
            index_dir, band_id, sidecar_paths, cfg.ram_budget_bytes,
        )
        apply_band_withdrawal(
            index_dir, band_id, safe_tag, plan,
            rebuilt_path=out_path, rebuilt_count=cnt,
            rebuilt_min=mn, rebuilt_max=mx, rebuilt_sources=contributors,
        )
        bands_rebuilt += 1

    tag_dir = datasets_dir / safe_tag
    if tag_dir.exists():
        shutil.rmtree(tag_dir, ignore_errors=True)
    removal_path = state_dir / "removals" / f"{safe_tag}.parquet"
    if removal_path.exists():
        removal_path.unlink()

    return {
        "bands_o1": bands_o1,
        "bands_rebuilt": bands_rebuilt,
        "elapsed_s": time.monotonic() - start,
    }


def rebuild_survivors(state_dir, drop_tag, dest_state_dir, cfg=None, *, output_root=None):
    """Faithfulness ORACLE / engine of faithful withdrawal.

    Re-ingest every SURVIVING dataset (all ledger ``ingest`` events except
    ``drop_tag``) in original ledger order, from each one's recorded replay
    manifest (interned input ids + content digests), into a FRESH state at
    ``dest_state_dir``. The result is the corpus "as if ``drop_tag`` was never
    ingested" -- the exact semantics a full rebuild of the survivors gives.

    Requires ``record_replay_manifest`` to have been on at ingest (else a
    survivor has no ``inputs`` to replay). Refuses to proceed if any survivor's
    input bytes have drifted from ingest time (digest mismatch) -- faithful
    replay must run against the same content it recorded.

    The source state's persisted config is authoritative (a different
    ``num_perm``/``num_bands``/``ngram``/``seed``/``exact_dedup`` would give a
    valid but NON-faithful rebuild); a ``cfg`` argument, if given, is validated
    against it and rejected on any immutable-field mismatch. The rebuilt state
    is forced to keep ``record_replay_manifest`` on so it remains replayable.
    """
    from dataclasses import replace

    from puffer import paths as _paths
    from puffer.config import PufferConfig
    from puffer.index import sanitize_tag
    from puffer.pipeline import Deduper, _read_ledger

    state_dir = Path(state_dir)
    src_cfg = PufferConfig.load(state_dir)
    if src_cfg is None:
        raise ValueError(f"source state {state_dir} has no persisted config; cannot faithfully rebuild")
    if cfg is not None:
        cfg.validate_against(src_cfg)  # reject a cfg that changes dedup semantics
    rebuild_cfg = replace(src_cfg, record_replay_manifest=True)
    dest = Path(dest_state_dir)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"dest_state_dir {dest} must be fresh/empty for a faithful rebuild")
    drop = sanitize_tag(drop_tag)
    events = [e for e in _read_ledger(state_dir) if e.get("op") == "ingest"]
    survivors = [e for e in events if sanitize_tag(e["dataset"]) != drop]
    if len(survivors) == len(events):
        raise ValueError(f"dataset {drop_tag!r} was never ingested; nothing to withdraw")

    out_root = Path(output_root) if output_root is not None else dest / "outputs"
    dd = Deduper(dest, rebuild_cfg)
    for e in survivors:
        inputs = e.get("inputs")
        if inputs is None:
            raise ValueError(
                f"dataset {e['dataset']!r} has no replay manifest; faithful "
                "withdrawal needs record_replay_manifest=True at ingest time"
            )
        drift = _paths.verify_ids(state_dir, inputs)
        if drift:
            raise RuntimeError(f"input drift for {e['dataset']!r}, refusing faithful replay: {drift}")
        uris = [entry["uri"] for entry in _paths.resolve_ids(state_dir, inputs)]
        column = e.get("column", rebuild_cfg.text_column)
        dd.ingest(
            uris, dataset=e["dataset"],
            output_dir=str(out_root / sanitize_tag(e["dataset"])), text_column=column,
        )
    return dd


def _link_or_copy_file(src: Path, dst: Path) -> Path:
    """Create an immutable-state hard link when possible, else copy."""
    import shutil

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def rebuild_survivors_suffix(
    state_dir, drop_tag, dest_state_dir, cfg=None, *, output_root=None,
):
    """Corpus-faithful withdrawal with prefix reuse and ordered suffix replay.

    Unlike :func:`rebuild_survivors` (the from-scratch oracle), this engine does
    not recompute the unaffected prefix. It hard-links/copies the prefix's
    immutable per-dataset state, removals and outputs into the fresh
    destination; reconstructs only the prefix index from compact sidecars;
    then replays every release after ``drop_tag`` in ledger order.

    When a suffix release has ``row_band_keys.i64``, replay feeds that
    row-ordered matrix through the *same* within/cross/commit/output stages,
    avoiding shingling and MinHash recomputation. Legacy releases without the
    artifact fall back to ordinary text-to-signature ingest. Original suffix
    inputs are still digest-verified and read to materialize corrected output
    rows. Prefix inputs need not remain online because their committed outputs
    and compact state are reused.

    The destination must be fresh. The returned :class:`~puffer.pipeline.Deduper`
    points at the corrected state; publication remains the caller's concern.
    """
    import json
    import shutil
    from dataclasses import replace

    from puffer import paths as _paths
    from puffer.config import PufferConfig
    from puffer.index import (
        append_shard,
        compact_band,
        read_shard_bin,
        sanitize_tag,
    )
    from puffer.pipeline import (
        Deduper,
        _append_ledger_event,
        _read_ledger,
        _write_json_atomic,
    )

    source = Path(state_dir)
    dest = Path(dest_state_dir)
    src_cfg = PufferConfig.load(source)
    if src_cfg is None:
        raise ValueError(f"source state {source} has no persisted config; cannot faithfully rebuild")
    if cfg is not None:
        cfg.validate_against(src_cfg)
    # Regenerated suffix releases retain their compact signatures, so the new
    # state stays eligible for the same fast path on a later withdrawal.
    replay_cfg = replace(
        src_cfg, record_replay_manifest=True, record_row_signatures=True,
    )
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"dest_state_dir {dest} must be fresh/empty for faithful suffix replay")

    drop = sanitize_tag(drop_tag)
    events = [e for e in _read_ledger(source) if e.get("op") == "ingest"]
    drop_positions = [
        i for i, e in enumerate(events) if sanitize_tag(e["dataset"]) == drop
    ]
    if not drop_positions:
        raise ValueError(f"dataset {drop_tag!r} was never ingested; nothing to withdraw")
    drop_at = drop_positions[0]
    prefix = [
        e for e in events[:drop_at] if sanitize_tag(e["dataset"]) != drop
    ]
    suffix = [
        e for e in events[drop_at + 1:] if sanitize_tag(e["dataset"]) != drop
    ]
    survivors = prefix + suffix
    for e in survivors:
        if e.get("inputs") is None:
            raise ValueError(
                f"dataset {e['dataset']!r} has no replay manifest; faithful "
                "withdrawal needs record_replay_manifest=True at ingest time"
            )

    paths_path = source / "paths.json"
    if not paths_path.exists():
        raise ValueError(f"source state {source} has no content-addressed path table")

    # Preflight the required prefix artifacts, suffix inputs, and compact
    # replay-artifact sizes before creating output.
    prefix_meta: dict[str, dict] = {}
    for e in prefix:
        safe = sanitize_tag(e["dataset"])
        dataset_dir = source / "datasets" / safe
        meta_path = dataset_dir / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"prefix dataset {e['dataset']!r} has no metadata to reuse")
        meta = json.loads(meta_path.read_text())
        output_files = [Path(p) for p in meta.get("output_files", [])]
        if not output_files or any(not p.exists() for p in output_files):
            raise ValueError(
                f"prefix outputs for {e['dataset']!r} are unavailable; "
                "use rebuild_survivors() for a full replay"
            )
        for band_id in range(replay_cfg.num_bands):
            sidecar = dataset_dir / f"band_{band_id:02d}.bin"
            if not sidecar.exists():
                raise ValueError(f"prefix sidecar missing: {sidecar}")
        prefix_meta[safe] = meta

    resolved_suffix: list[tuple[dict, list[dict]]] = []
    for e in suffix:
        inputs = e["inputs"]
        drift = _paths.verify_ids(source, inputs)
        if drift:
            raise RuntimeError(
                f"input drift for {e['dataset']!r}, refusing faithful replay: {drift}"
            )
        safe = sanitize_tag(e["dataset"])
        artifacts = source / "datasets" / safe
        row_keys = artifacts / "row_band_keys.i64"
        if row_keys.exists():
            n_docs = int(e.get("n_docs", 0))
            expected_keys = n_docs * replay_cfg.num_bands * 8
            if row_keys.stat().st_size != expected_keys:
                raise ValueError(
                    f"row-signature artifact {row_keys} has {row_keys.stat().st_size} "
                    f"bytes; expected {expected_keys}"
                )
        resolved_suffix.append((e, _paths.resolve_ids(source, inputs)))

    out_root = Path(output_root) if output_root is not None else dest / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    _link_or_copy_file(paths_path, dest / "paths.json")
    dd = Deduper(dest, replay_cfg)
    dd._ensure_config_persisted()

    def _copy_for_tree(src, dst):
        return str(_link_or_copy_file(Path(src), Path(dst)))

    for e in prefix:
        safe = sanitize_tag(e["dataset"])
        src_dataset = source / "datasets" / safe
        dst_dataset = dest / "datasets" / safe
        shutil.copytree(src_dataset, dst_dataset, copy_function=_copy_for_tree)

        src_removal = source / "removals" / f"{safe}.parquet"
        if src_removal.exists():
            _link_or_copy_file(src_removal, dest / "removals" / src_removal.name)

        dst_output_dir = out_root / safe
        dst_outputs: list[str] = []
        for src_output in prefix_meta[safe]["output_files"]:
            src_output = Path(src_output)
            dst_output = dst_output_dir / src_output.name
            if not dst_output.exists():
                _link_or_copy_file(src_output, dst_output)
            dst_outputs.append(str(dst_output))
        meta = dict(prefix_meta[safe])
        meta["output_dir"] = str(dst_output_dir)
        meta["output_files"] = dst_outputs
        _write_json_atomic(dst_dataset / "meta.json", meta)

        for band_id in range(replay_cfg.num_bands):
            keys = read_shard_bin(dst_dataset / f"band_{band_id:02d}.bin", mmap=False)
            append_shard(keys, dest / "hash_index", band_id, e["dataset"])
            compact_band(
                dest / "hash_index", band_id, replay_cfg.tier_fanout,
                replay_cfg.ram_budget_bytes, protect_tag=e["dataset"],
            )
        copied_event = dict(e)
        copied_event["output_dir"] = str(dst_output_dir)
        _append_ledger_event(dest, copied_event)


    for e, resolved in resolved_suffix:
        safe = sanitize_tag(e["dataset"])
        uris = [Path(entry["uri"]) for entry in resolved]
        artifacts = source / "datasets" / safe
        replay_artifacts = artifacts if (artifacts / "row_band_keys.i64").exists() else None
        dd._ingest_fresh(
            uris,
            dataset=e["dataset"],
            safe_tag=safe,
            output_dir=out_root / safe,
            column=e.get("column", replay_cfg.text_column),
            start=time.monotonic(),
            replay_artifacts_dir=replay_artifacts,
            replay_file_ids=[int(i) for i in e["inputs"]],
        )
    return dd
