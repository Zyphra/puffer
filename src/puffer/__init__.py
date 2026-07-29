"""PUFFER: incremental MinHash-LSH deduplication for growing corpora.

Pipeline: parquet releases -> shingles -> MinHash band keys -> exact-membership
LSM band-key index (streaming tiered compaction, early-stop screening) ->
deduplicated parquet outputs. Withdrawal removes a dataset's contribution from
index state via provenance-tracked segments (O(1) unlink for standalone runs,
survivor rebuild for merged runs). Faithful, corpus-level withdrawal (outputs
of later releases recomputed "as if never ingested") is available via the
from-scratch oracle ``rebuild_survivors`` and the prefix-reusing,
artifact-assisted ``rebuild_survivors_suffix``.

Public API:

    from puffer import Deduper, PufferConfig

    dd = Deduper("state/", PufferConfig(num_bands=8, tier_fanout=4))
    report = dd.ingest(["release0/*.parquet"], dataset="release0",
                       output_dir="out/release0")
    dd.withdraw("release0")
"""

from puffer.config import PufferConfig
from puffer.pipeline import Deduper, IngestReport, WithdrawReport
from puffer.withdraw import rebuild_survivors, rebuild_survivors_suffix

__all__ = [
    "Deduper", "PufferConfig", "IngestReport", "WithdrawReport",
    "rebuild_survivors", "rebuild_survivors_suffix",
]
__version__ = "0.1.0"
