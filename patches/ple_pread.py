"""Replace the engram (PLE) row gather's page-fault path with threaded pread.

The shipping implementation gathers decode-sized batches inline with numpy fancy indexing over
an np.memmap, so every row that is not resident becomes a synchronous page fault, serialized
one after another on the calling thread. Measured on this box with the real 25M-row table and
180 random rows per call (scripts/ple_gather_bench2.py):

    strategy                      cold (uniform)   cold (skewed)   fully warm
    memmap fancy index, inline      26.3-41.0 ms      6.8-8.1 ms       3.0 ms
    os.pread, 8 threads              3.9 ms           4.5 ms          4.0 ms

So explicit reads are up to 6.7x faster when rows are cold and about 0.9 ms slower when every
row is already resident. Because the gather sits on the decode critical path (the custom op
blocks the CUDA stream), the tail is what matters.

Activate with VLLM_PLE_PREAD=1; VLLM_PLE_PREAD_THREADS (default 8),
VLLM_PLE_PREAD_MIN_ROWS (default 16, below which inline memmap is cheaper than dispatch).
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger("vllm.ple_pread")

_THREADS = int(os.environ.get("VLLM_PLE_PREAD_THREADS", "8"))
_MIN_ROWS = int(os.environ.get("VLLM_PLE_PREAD_MIN_ROWS", "16"))


def apply(table_cls) -> None:
    if getattr(table_cls, "_ple_pread_patched", False):
        return
    orig_gather = table_cls._gather

    def _ensure_fds(self):
        fds = getattr(self, "_pread_fds", None)
        if fds is None:
            fds = [None] * len(self.paths)
            for i, p in enumerate(self.paths):
                if p is not None:
                    fds[i] = os.open(p, os.O_RDONLY)
            self._pread_fds = fds
            self._pread_pool = ThreadPoolExecutor(max_workers=_THREADS)
            # force thread creation so the first gather does not pay for it
            list(self._pread_pool.map(lambda _: None, range(_THREADS)))
            self._pread_base = [
                (self.mm[i].offset if self.mm[i] is not None else 0)
                for i in range(len(self.paths))
            ]
            logger.warning(
                "PLE pread: %d shards, %d threads, min_rows=%d",
                sum(f is not None for f in fds), _THREADS, _MIN_ROWS,
            )
        return fds

    def _gather(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size < _MIN_ROWS:
            return orig_gather(self, ids)
        if ids.min() < 0 or ids.max() >= self.rows_total:
            raise IndexError(
                f"PLE row id out of range: [{ids.min()}, {ids.max()}] for {self.rows_total} rows"
            )
        fds = _ensure_fds(self)
        uniq, inverse = np.unique(ids, return_inverse=True)
        shard = (uniq // self.shard_size).astype(np.int64)
        local = uniq - shard * self.shard_size
        rb = self.row_bytes
        out = np.empty((uniq.size, rb), dtype=np.uint8)
        base = self._pread_base

        def run(i: int) -> None:
            si = int(shard[i])
            buf = os.pread(fds[si], rb, base[si] + int(local[i]) * rb)
            out[i] = np.frombuffer(buf, dtype=np.uint8)

        n = uniq.size
        if n <= _THREADS:
            for i in range(n):
                run(i)
        else:
            for _ in self._pread_pool.map(run, range(n), chunksize=max(1, n // (_THREADS * 4))):
                pass
        return out[inverse]

    table_cls._ple_pread_orig_gather = orig_gather
    table_cls._gather = _gather
    table_cls._ple_pread_patched = True
    logger.warning("PLE pread: patched %s._gather (threads=%d)", table_cls.__name__, _THREADS)
