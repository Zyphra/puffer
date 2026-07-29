"""Optional Ray dispatch for the distributed portions of PUFFER ingestion.

The driver remains the sole writer of state: it appends LSM runs, compacts
bands, and performs withdrawal.  Ray workers only calculate release-local
signatures or screen immutable-once-opened index runs on a shared filesystem.
That boundary keeps the local executor's ordering and crash semantics intact.

Ray is deliberately a soft dependency.  This module is safe to import in a
normal installation; importing Ray, starting a cluster, and constructing remote
tasks happen only when one of the public execution functions is called.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from puffer.config import PufferConfig

logger = logging.getLogger(__name__)

_RAY_INSTALL_HINT = "Ray execution requires the optional dependency; install it with `pip install puffer-dedup[ray]`."


def _require_ray() -> Any:
    """Return Ray without making it an import-time dependency."""
    try:
        import ray
    except ImportError as exc:
        raise ImportError(_RAY_INSTALL_HINT) from exc
    return ray


def _ensure_ray() -> Any:
    """Return an initialized Ray runtime, joining a cluster when one is present.

    Resolution order:
      1. ``RAY_ADDRESS`` set -> connect to that cluster (hard error if unreachable
         -- the user asked for a specific one).
      2. an existing local cluster (``ray start`` on this node) -> ``address="auto"``
         (the multi-node path: ``ray start --head`` here + ``--address`` elsewhere).
      3. otherwise -> a fresh local single-node cluster (no address) so ``ray``
         execution works on one box with no manual ``ray start``.
    """
    ray = _require_ray()
    if ray.is_initialized():
        return ray
    explicit = os.environ.get("RAY_ADDRESS")
    if explicit:
        ray.init(address=explicit, log_to_driver=False)
    else:
        try:
            ray.init(address="auto", log_to_driver=False)
        except Exception as exc:  # noqa: BLE001 -- diagnose before dropping to 1 node
            if "find any running Ray" not in str(exc):
                raise RuntimeError(
                    "Found a Ray cluster but could not connect to it "
                    f"({type(exc).__name__}: {exc or 'no message'}). This is almost "
                    "always a Ray version mismatch between this process and the "
                    "cluster, or an unreachable head: make `ray --version` match on "
                    "every node and ensure `puffer` is importable there, or set "
                    "RAY_ADDRESS to target a cluster explicitly."
                ) from exc
            ray.init(log_to_driver=False)
    logger.info(
        "Ray initialized: nodes=%d, cluster CPUs=%s",
        len(ray.nodes()), ray.cluster_resources().get("CPU"),
    )
    return ray


def cluster_cpus(default: int = 8) -> int:
    """Total CPU slots across the Ray cluster (floor 1)."""
    try:
        ray = _require_ray()
        return max(1, int(ray.cluster_resources().get("CPU", default)))
    except Exception:  # noqa: BLE001
        return default


def num_nodes(default: int = 1) -> int:
    """Count alive Ray nodes (floor 1)."""
    try:
        ray = _require_ray()
        return max(1, len([n for n in ray.nodes() if n.get("Alive")]))
    except Exception:  # noqa: BLE001
        return default

def resolve_max_in_flight(configured: int = 0) -> int:
    """Resolve PUFFER's cluster-wide cap for one-CPU Ray tasks.

    A positive configured value is an explicit total cap across the cluster.
    Otherwise ``PUFFER_RAY_MAX_IN_FLIGHT`` wins, then the live cluster's CPU
    slots are used. If four nodes each advertise 64 CPUs, the automatic cap
    is therefore 256, allowing Ray to place up to 64 tasks on every node.

    Initializes/joins the Ray runtime if needed: ``cluster_cpus()`` on an
    uninitialized Ray silently returns its default (8), which once capped an
    entire run at 8 in-flight tasks.
    """
    if configured > 0:
        return int(configured)
    env = os.environ.get("PUFFER_RAY_MAX_IN_FLIGHT", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    _ensure_ray()
    return max(1, cluster_cpus())


def ray_map_unordered(
    fn: Callable[[Any], Any],
    jobs: Iterable[Any],
    *,
    num_cpus: float = 1,
    memory_bytes: float | None = None,
    scheduling_strategy: Any = None,
    label: str = "",
    max_in_flight: int | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Run ``fn(job)`` as a Ray task per job; yield ``(job, result_or_exc)`` as they finish.

    Submission is bounded by ``max_in_flight``. Callers that need the
    cluster-aware PUFFER policy should pass :func:`resolve_max_in_flight`;
    otherwise this generic helper retains a three-waves default. A task
    that raises yields the exception object (not raised) so the caller
    reconciles failures exactly like the local pool path. PUFFER extension: an
    optional ``scheduling_strategy`` (we pass ``"SPREAD"`` so tasks fan across
    worker nodes rather than packing onto the head).
    """
    ray = _ensure_ray()
    opts: dict[str, Any] = {"num_cpus": num_cpus}
    if memory_bytes and memory_bytes > 0:
        opts["memory"] = int(memory_bytes)
    if scheduling_strategy is not None:
        opts["scheduling_strategy"] = scheduling_strategy
    remote_fn = ray.remote(**opts)(fn)

    if max_in_flight is None:
        env = os.environ.get("PUFFER_RAY_MAX_IN_FLIGHT", "")
        max_in_flight = int(env) if env.isdigit() and int(env) > 0 else max(8, cluster_cpus() * 3)
    harvest_batch = max(1, min(max_in_flight, 256))
    logger.info(
        "Ray dispatch [%s]: streaming tasks (max_in_flight=%d, harvest_batch=%d, "
        "num_cpus=%s%s each)",
        label or "?", max_in_flight, harvest_batch, num_cpus,
        f", memory={memory_bytes / 1024 ** 3:.2f}GiB" if memory_bytes else "",
    )

    jobs_iter = iter(jobs)
    pending: dict[Any, Any] = {}

    def _submit_next() -> bool:
        try:
            job = next(jobs_iter)
        except StopIteration:
            return False
        pending[remote_fn.remote(job)] = job
        return True

    for _ in range(max_in_flight):
        if not _submit_next():
            break
    while pending:
        ready, _ = ray.wait(list(pending.keys()), num_returns=min(len(pending), harvest_batch))
        for ref in ready:
            job = pending.pop(ref)
            try:
                yield job, ray.get(ref)
            except Exception as exc:  # noqa: BLE001 -- surface to caller for reconcile
                yield job, exc
            _submit_next()


def split_row_chunks(n_rows: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Return contiguous half-open row intervals covering ``range(n_rows)``.

    Keeping partitioning independent of Ray makes its ordering contract easy to
    test and prevents completion order from leaking into release row order.
    """
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    return [(start, min(n_rows, start + chunk_rows)) for start in range(0, n_rows, chunk_rows)]


def combine_screen_chunks(
    n_rows: int,
    results: Iterable[tuple[int, int, Any, dict[str, int] | None]],
) -> tuple[Any, dict[str, int]]:
    """Place ordered-or-unordered worker results into release row order.

    Every result is ``(start, stop, cross_mask, counters)``.  Intervals must be
    a complete non-overlapping partition of the release; rejecting gaps avoids
    silently attaching a worker's decisions to the wrong documents.
    """
    import numpy as np

    ordered = sorted(results, key=lambda result: result[0])
    expected = 0
    masks: list[Any] = []
    totals = {"probes_scheduled": 0, "probes_done": 0}
    for start, stop, mask, counters in ordered:
        if start != expected or stop < start or stop > n_rows:
            raise ValueError("screen chunk intervals must partition the release in row order")
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim != 1 or len(arr) != stop - start:
            raise ValueError("screen chunk mask length does not match its interval")
        masks.append(arr)
        expected = stop
        if counters is not None:
            for key in totals:
                totals[key] += int(counters.get(key, 0))
    if expected != n_rows:
        raise ValueError("screen chunk intervals do not cover the release")
    return (np.concatenate(masks) if masks else np.zeros(0, dtype=bool), totals)


def _signature_file_task(pq_file: str, cfg: PufferConfig, text_column: str) -> dict[str, Any]:
    """Per-document band keys for one parquet.

    Text is streamed in ``cfg.sig_chunk_rows`` batches, so the worker's resident
    text is O(chunk) rather than O(file) -- the same wall-avoiding bound as the
    local ingest. The returned dict is compact per-document state (band keys),
    O(file rows); the driver writes it into its disk-backed memmap and frees it,
    and the bounded producer keeps only ``max_in_flight`` of these.
    """
    from dataclasses import replace

    import numpy as np
    import pyarrow.parquet as pq

    from puffer.signature import compute_band_keys

    pf = pq.ParquetFile(str(pq_file))
    if text_column not in pf.schema_arrow.names:
        raise ValueError(f"text column {text_column!r} is absent from {pq_file}")
    scfg = replace(cfg, n_workers=1)  # one thread per file task; Ray owns cross-file parallelism
    bk_parts: list = []
    n = 0
    for batch in pf.iter_batches(batch_size=max(1, cfg.sig_chunk_rows), columns=[text_column]):
        texts = ["" if v is None else str(v) for v in batch.column(0).to_pylist()]
        if not texts:
            continue
        bk_parts.append(compute_band_keys(texts, scfg))
        n += len(texts)
        del texts
    band_keys = (
        np.concatenate(bk_parts, axis=0) if bk_parts
        else np.empty((0, cfg.num_bands), dtype=np.int64)
    )
    return {"source_file": str(pq_file), "n_rows": n, "band_keys": band_keys}


def ray_compute_band_keys(
    pq_files: Sequence[str | Path],
    cfg: PufferConfig,
    *,
    text_column: str | None = None,
) -> list[dict[str, Any]]:
    """Compute signatures in one Ray task per parquet, preserving file order.

    Results contain ``source_file``, ``n_rows``, and ``band_keys``.
    ``source_row`` remains an implicit ``range(n_rows)`` per input file. The
    caller continues to own parquet output and all mutable deduplication
    state. Input paths must be visible to every Ray worker.
    """
    ray = _ensure_ray()
    column = text_column or cfg.text_column
    paths = [str(Path(path)) for path in pq_files]
    if not paths:
        return []
    # SPREAD so tasks fan across worker nodes instead of packing onto whichever
    # node (often the driver/head) has spare CPU slots under default scheduling.
    remote_task = ray.remote(num_cpus=1, scheduling_strategy="SPREAD")(_signature_file_task)
    refs = [remote_task.remote(path, cfg, column) for path in paths]
    # ray.get over the submission-order list is intentionally ordered even when
    # workers finish out of order: source-row tie breaking depends on it.
    return list(ray.get(refs))


def ray_iter_band_keys(
    pq_files: "Sequence[str | Path]",
    cfg: PufferConfig,
    *,
    text_column: str | None = None,
    max_in_flight: int = 0,
):
    """Yield ``(file_index, result)`` in file order with bounded in-flight tasks.

    Unlike :func:`ray_compute_band_keys`, this never materializes all file
    results at once: at most ``max_in_flight`` signature tasks are in flight and
    the driver holds at most that many result dictionaries. Results are yielded
    in submission (file) order so the caller writes the memmap in stable file
    order. ``max_in_flight=0`` uses the live Ray cluster's total CPU slots.
    This is the bounded-memory producer the streaming ingest relies on.
    """

    ray = _ensure_ray()
    column = text_column or cfg.text_column
    paths = [str(Path(path)) for path in pq_files]
    if not paths:
        return
    cap = resolve_max_in_flight(max_in_flight)
    remote_task = ray.remote(num_cpus=1, scheduling_strategy="SPREAD")(_signature_file_task)
    inflight: deque = deque()
    nxt = 0
    while nxt < len(paths) and len(inflight) < cap:
        inflight.append((nxt, remote_task.remote(paths[nxt], cfg, column)))
        nxt += 1
    while inflight:
        fi, ref = inflight.popleft()
        yield fi, ray.get(ref)  # ordered: blocks on the front of the window
        if nxt < len(paths):
            inflight.append((nxt, remote_task.remote(paths[nxt], cfg, column)))
            nxt += 1


# ``ray_iter_band_keys`` (object-store return) is the only Ray transport; its
# results are verified byte-identical to the local executor.


def _screen_chunk_task(
    band_keys: Any,
    index_dir: str,
    exclude_tag: str | None,
    cfg: PufferConfig,
    start: int,
    stop: int,
) -> tuple[int, int, Any, dict[str, int]]:
    """Screen one contiguous release interval, including all bands in order."""
    from puffer.screen import screen_release

    counters: dict[str, int] = {}
    mask = screen_release(band_keys, Path(index_dir), exclude_tag, cfg, counters)
    return start, stop, mask, counters


def _screen_chunk_job(job: tuple) -> tuple[int, int, Any, dict[str, int]]:
    """Unpack a :func:`ray_map_unordered` job tuple into :func:`_screen_chunk_task`."""
    return _screen_chunk_task(*job)


def ray_screen_chunks(
    band_keys: Any,
    index_dir: str | Path,
    exclude_tag: str | None,
    cfg: PufferConfig,
    *,
    chunk_rows: int | None = None,
) -> tuple[Any, dict[str, int]]:
    """Screen release band keys remotely and return its ordered cross-history mask.

    Each task receives one contiguous row interval and calls ``screen_release``
    on that whole slice, so its early-stop state spans every band exactly as it
    does locally. Dispatch goes through :func:`ray_map_unordered` (bounded
    submission), and completion order is discarded by
    :func:`combine_screen_chunks`; both mask and counters therefore equal local
    execution. In-flight work uses the configured cluster-wide cap, defaulting
    to the live Ray cluster's CPU slots. Large releases produce three scheduling
    waves per usable slot.
    """
    import numpy as np

    _ensure_ray()  # initialize before deriving cluster-wide task capacity

    keys = np.asarray(band_keys)
    if keys.ndim != 2:
        raise ValueError("band_keys must be a two-dimensional (rows, bands) array")
    rows = len(keys)
    if chunk_rows is None:
        cap = resolve_max_in_flight(cfg.ray_max_in_flight)
        usable_slots = max(1, min(cap, cluster_cpus()))
        target_chunks = usable_slots * 3
        chunk_rows = max(1, (rows + target_chunks - 1) // target_chunks)
    else:
        cap = resolve_max_in_flight(cfg.ray_max_in_flight)
    intervals = split_row_chunks(rows, chunk_rows)
    if not intervals:
        return combine_screen_chunks(0, [])
    shared_index = str(Path(index_dir).resolve())  # absolute: workers lack driver CWD
    jobs = [(keys[start:stop], shared_index, exclude_tag, cfg, start, stop) for start, stop in intervals]
    # ``cap`` is cluster-wide. Ray's per-node ``--num-cpus`` resource controls
    # how those one-CPU tasks are distributed across live nodes.
    results: list = []
    for _job, res in ray_map_unordered(
        _screen_chunk_job, jobs, num_cpus=1, scheduling_strategy="SPREAD",
        label="screen", max_in_flight=cap,
    ):
        if isinstance(res, BaseException):
            raise res
        results.append(res)
    return combine_screen_chunks(rows, results)
