"""Append-only content-addressed path table for replay manifests + removals.

Motivation (dictionary-encoding of ``source_file``): an input path repeats
in every removals row and
every replay manifest entry; storing a small integer id + one shared table,
instead of the full path string per occurrence, is a large saving at corpus
scale (int32 code vs a multi-hundred-byte URI).

Identity is ``(canonical_uri, content_digest)`` where ``content_digest`` is a
FULL streamed SHA-256 of the file's bytes -- not a sampled fingerprint -- so a
recorded id always denotes exactly one immutable content. Ids are append-only
and NEVER reused (even after withdrawal): a mutated file at the same URI is a
NEW id, and historical replay manifests keep resolving to the content they were
recorded against. Replay MUST re-verify the digest and reject a same-URI
mismatch rather than silently replaying different bytes.

Safety: a malformed/unreadable existing table is a hard error (never silently
overwritten from id 0 -- that would reuse ids and corrupt every historical
manifest), and interning is serialized under an ``fcntl`` lock so concurrent
ingests cannot race load->append->replace and lose/duplicate ids.

Ingest-time assumption: an input's bytes are stable for the duration of the
ingest that records them (PUFFER's workload is *released, immutable* datasets).
The digest is taken once at intern; callers wanting to close the intra-ingest
TOCTOU window can re-run :func:`verify_ids` after processing and fail the ingest
on drift. Replay ALWAYS re-verifies, which is the load-bearing guard.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

_TABLE = "paths.json"
_CHUNK = 8 << 20  # 8 MiB streamed hash chunk


def _table_path(state_dir: Path) -> Path:
    return Path(state_dir) / _TABLE


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def _locked(state_dir: Path):
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    lock_path = _table_path(state_dir).with_suffix(".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def content_digest(path: str | Path) -> str:
    """Full streamed SHA-256 hex of the file's bytes (the content identity)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_strict(state_dir: Path) -> list[dict]:
    """Load the table, FAILING CLOSED on any corruption (never returns [] for a
    present-but-broken table -- that would let the next intern reuse ids)."""
    p = _table_path(state_dir)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"path table {p} is present but unreadable/corrupt; refusing to "
            f"intern (would reuse ids and corrupt historical manifests): {exc}"
        ) from exc
    paths = data.get("paths")
    if not isinstance(paths, list):
        raise RuntimeError(f"path table {p} malformed: missing 'paths' list")
    for i, entry in enumerate(paths):
        if (
            not isinstance(entry, dict)
            or entry.get("id") != i
            or "uri" not in entry
            or "digest" not in entry
        ):
            raise RuntimeError(f"path table {p} malformed at entry {i}: {entry!r}")
    return paths


def intern_paths(state_dir: str | Path, files) -> list[int]:
    """Append-only intern of resolved inputs -> stable ids, one per file in order.

    Reuse an id only on an exact ``(uri, digest)`` match; a same-uri file whose
    content digest differs gets a fresh id. Digests are computed OUTSIDE the
    lock (pure, expensive); only the short load->append->replace transaction is
    serialized.
    """
    state_dir = Path(state_dir)
    keys = [(str(Path(f).resolve()), content_digest(f)) for f in files]
    with _locked(state_dir):
        table = _load_strict(state_dir)
        by_key: dict[tuple[str, str], int] = {(e["uri"], e["digest"]): e["id"] for e in table}
        ids: list[int] = []
        changed = False
        for uri, digest in keys:
            key = (uri, digest)
            if key in by_key:
                ids.append(by_key[key])
                continue
            new_id = len(table)
            table.append(
                {"id": new_id, "uri": uri, "digest": digest, "size": Path(uri).stat().st_size}
            )
            by_key[key] = new_id
            ids.append(new_id)
            changed = True
        if changed:
            _write_json_atomic(_table_path(state_dir), {"paths": table})
    return ids


def resolve_ids(state_dir: str | Path, ids) -> list[dict]:
    """Return the path-table entries for ``ids`` (order preserved)."""
    table = _load_strict(Path(state_dir))
    out = []
    for i in ids:
        i = int(i)
        if i < 0 or i >= len(table):
            raise KeyError(f"path id {i} absent from table")
        out.append(table[i])
    return out


def verify_ids(state_dir: str | Path, ids) -> list[dict]:
    """Re-check each id's file still has its recorded digest. Returns the list
    of drifted/missing entries (empty == all inputs byte-identical to ingest
    time). Faithful replay must refuse to proceed on a non-empty result."""
    drifted = []
    for entry in resolve_ids(state_dir, ids):
        try:
            if content_digest(entry["uri"]) != entry["digest"]:
                drifted.append({**entry, "reason": "digest_mismatch"})
        except OSError:
            drifted.append({**entry, "reason": "missing"})
    return drifted
