"""PUFFER configuration.

One frozen dataclass carries every knob that affects dedup decisions or index
state. The first ``Deduper.ingest`` persists it to ``state_dir/puffer_config.json``;
every later run validates that the decision-relevant fields are unchanged
(changing them mid-life would silently alter collision semantics or break the
tiered-compaction bounds — rebuild the index instead).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

#: Fields that are frozen for the lifetime of a state directory.
IMMUTABLE_FIELDS = (
    "num_perm",
    "num_bands",
    "ngram_type",
    "ngram_size",
    "seed",
    "tier_fanout",
)


@dataclass(frozen=True)
class PufferConfig:
    """Pipeline + index configuration.

    MinHash / shingling (decision-relevant, immutable per state dir):

    - ``num_perm``: MinHash permutations (signature length).
    - ``num_bands``: LSH bands; ``num_perm % num_bands == 0``.
    - ``ngram_type``: ``"char"`` or ``"word"`` shingles.
    - ``ngram_size``: shingle length (chars or words).
    - ``seed``: MinHash seed.

    Index (immutable per state dir):

    - ``tier_fanout``: T of the tiered compaction policy. A level holding T
      unprotected runs merges them into one run at level+1 (base-T counter).

    Operational (may vary run to run):

    - ``ram_budget_bytes``: advisory working-set budget. Sizes compaction
      merge-eligibility and streaming-merge buffers, and caps the number of
      in-flight signature chunks in the local streaming ingest (bounding
      resident text). Not a hard RSS cap.
    - ``probe_order``: order in which a band's live runs are probed during
      screening: ``"largest_first"`` (default), ``"smallest_first"``, or
      ``"newest_first"``. Pure heuristic for early-stop probe savings; the
      screening *result* is identical under any order.
    - ``n_workers``: worker threads / processes for signatures and screening.
    - ``text_column``: default text column name in input parquets.
    - ``sig_chunk_rows``: row-chunk size for the streaming ingest. Peak text
      RAM is bounded by ``sig_chunk_rows`` per in-flight chunk (× workers),
      independent of release size; also the signature output batch size.
    - ``ray_max_in_flight``: cluster-wide cap on one-CPU Ray signature and
      screening tasks. ``0`` derives the cap from live Ray cluster CPU slots;
      a positive value is a total cap, not a per-node value.
    - ``screen_batch_bytes``: cross-history screen window, sized by band-key
      bytes (not text). Decouples screen batching from ``sig_chunk_rows`` so
      the screen isn't over-chunked into tiny Ray tasks, while staying bounded
      (a window is ``screen_batch_bytes`` of int64 keys, not O(release)).
    """

    num_perm: int = 64
    num_bands: int = 8
    ngram_type: str = "char"
    ngram_size: int = 20
    seed: int = 42

    tier_fanout: int = 4

    ram_budget_bytes: int = 4 * 1024**3
    probe_order: str = "largest_first"
    n_workers: int = 0  # 0 -> os.cpu_count()
    text_column: str = "text"
    sig_chunk_rows: int = 20_000
    ray_max_in_flight: int = 0  # 0 -> live cluster CPU slots; positive -> cluster-wide total
    screen_batch_bytes: int = 256 * 1024 * 1024  # cross-screen window, sized by band-key bytes
    # Persist a per-release replay manifest (ordered interned input ids +
    # full content digests, via the append-only path table) into the ledger,
    # enabling faithful-mode withdrawal to replay survivors from verified
    # inputs. Costs one full content hash per input file at ingest; set False
    # to skip if you never need faithful withdrawal.
    record_replay_manifest: bool = True
    # Persist the row-ordered B x int64 band-key matrix produced during ingest.
    # This costs 8 * num_bands bytes/document, but lets faithful suffix replay
    # skip text shingling and MinHash recomputation. It is operational (it does
    # not alter decisions), and legacy releases without it fall back to normal
    # text-to-signature replay.
    record_row_signatures: bool = False

    def __post_init__(self) -> None:
        if self.num_perm % self.num_bands != 0:
            raise ValueError(
                f"num_perm ({self.num_perm}) must be divisible by "
                f"num_bands ({self.num_bands})"
            )
        if self.ngram_type not in ("char", "word"):
            raise ValueError(f"ngram_type must be 'char' or 'word', got {self.ngram_type!r}")
        if self.tier_fanout < 2:
            raise ValueError(f"tier_fanout must be >= 2, got {self.tier_fanout}")
        if self.probe_order not in ("largest_first", "smallest_first", "newest_first"):
            raise ValueError(f"unknown probe_order: {self.probe_order!r}")
        if self.ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {self.ngram_size}")
        if self.sig_chunk_rows < 1:
            raise ValueError(f"sig_chunk_rows must be >= 1, got {self.sig_chunk_rows}")
        if self.ray_max_in_flight < 0:
            raise ValueError(f"ray_max_in_flight must be >= 0, got {self.ray_max_in_flight}")
        if self.screen_batch_bytes < 1:
            raise ValueError(f"screen_batch_bytes must be >= 1, got {self.screen_batch_bytes}")

    @property
    def rows_per_band(self) -> int:
        return self.num_perm // self.num_bands

    @property
    def effective_workers(self) -> int:
        return self.n_workers if self.n_workers > 0 else (os.cpu_count() or 4)

    @property
    def screen_batch_rows(self) -> int:
        """Rows per cross-screen window, from ``screen_batch_bytes`` / key size."""
        return max(1, self.screen_batch_bytes // (8 * self.num_bands))

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PufferConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, state_dir: Path) -> Path:
        path = Path(state_dir) / "puffer_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, state_dir: Path) -> "PufferConfig | None":
        path = Path(state_dir) / "puffer_config.json"
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))

    def validate_against(self, persisted: "PufferConfig") -> None:
        """Raise if a decision-relevant field differs from the persisted config."""
        diffs = [
            (f, getattr(persisted, f), getattr(self, f))
            for f in IMMUTABLE_FIELDS
            if getattr(persisted, f) != getattr(self, f)
        ]
        if diffs:
            detail = ", ".join(f"{f}: state={a!r} vs requested={b!r}" for f, a, b in diffs)
            raise ValueError(
                "PufferConfig conflicts with the state directory's persisted "
                f"config ({detail}). These fields are immutable for the life "
                "of an index; rebuild the state dir to change them."
            )
