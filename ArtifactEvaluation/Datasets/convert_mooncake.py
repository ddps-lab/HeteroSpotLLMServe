#!/usr/bin/env python3
"""
Convert the Mooncake ToolAgent trace (jsonl) to the Azure trace CSV format
so the existing loading utils (e.g. load_trace in statistics.ipynb) work as-is.

Source : mooncake_toolagent_trace.jsonl
         {"timestamp": <ms since trace start>, "input_length": ..., "output_length": ..., "hash_ids": [...]}
Target : MooncakeToolAgentTrace.csv
         TIMESTAMP,ContextTokens,GeneratedTokens

De-bucketing: the source timestamps are bucketed at ~3000 ms granularity
(1,180 buckets, ~20 requests sharing each timestamp). Each request is moved
to a uniformly random offset within its bucket [start, next_bucket_start).
Given the per-bucket counts, i.i.d. uniform placement is exactly the
conditional distribution of a Poisson process, so the de-bucketed trace is
equivalent to a piecewise-homogeneous Poisson arrival process.

A fixed RNG seed makes the output reproducible. Relative ms timestamps are
mapped onto absolute datetimes from an arbitrary base (loading utils only
use deltas). No pruning is applied.

Run: python3 convert_mooncake.py
"""

import csv
import json
import os
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

_d = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_d, "mooncake_toolagent_trace.jsonl")
DST = os.path.join(_d, "MooncakeToolAgentTrace.csv")

BASE = datetime(2024, 1, 1, 0, 0, 0)  # arbitrary; only deltas matter
SEED = 42
GRID_MS = 3000  # nominal bucket grid of the source trace (starts carry ±3 ms jitter)


def verify():
    """
    Compare the source jsonl and the converted CSV and check that the
    de-bucketing preserved the distribution:

    1. request count and (input, output) token multiset are unchanged
    2. timestamps are monotonically non-decreasing
    3. every event stays inside its source bucket and per-bucket counts are
       preserved -> at any aggregation >= bucket width the arrival
       distribution is IDENTICAL to the original
    4. in-bucket relative offsets are U(0,1) (KS test) -> given the bucket
       counts this is exactly the conditional law of a Poisson process
    5. inter-arrival times show the Poisson signature (mean = duration/N,
       coefficient of variation ~ 1)
    """
    src = []
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            src.append((d["timestamp"], d["input_length"], d["output_length"]))

    conv = []
    with open(DST) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["TIMESTAMP"])
            conv.append(((ts - BASE).total_seconds() * 1000.0,
                         int(row["ContextTokens"]), int(row["GeneratedTokens"])))

    print("\n--- verification: original vs converted ---")

    # 1. counts and token multiset
    assert len(src) == len(conv), f"count mismatch: {len(src)} vs {len(conv)}"
    assert Counter((i, o) for _, i, o in src) == Counter((i, o) for _, i, o in conv), \
        "(input, output) token multiset changed"
    print(f"[1] count ({len(conv):,}) and token multiset preserved: OK")

    # 2. monotonic
    ms = np.array([t for t, _, _ in conv])
    assert (np.diff(ms) >= 0).all(), "converted timestamps not monotonic"
    print("[2] timestamps monotonic: OK")

    # 3. per-bucket counts (distribution identity at >= bucket granularity)
    starts = sorted({t for t, _, _ in src})
    edges = starts + [(round(starts[-1] / GRID_MS) + 1) * GRID_MS]
    idx = np.searchsorted(edges, ms, side="right") - 1
    assert (idx >= 0).all() and (idx < len(starts)).all(), "event outside trace span"
    src_counts = Counter(t for t, _, _ in src)
    conv_counts = Counter(starts[i] for i in idx)
    assert src_counts == conv_counts, "per-bucket request counts changed"
    print(f"[3] per-bucket counts preserved over {len(starts):,} buckets: OK "
          "(arrival distribution identical at >= 3 s granularity)")

    # 4. in-bucket offsets uniform (KS one-sample test vs U(0,1))
    width = {s: e - s for s, e in zip(starts, edges[1:])}
    offs = (ms - np.array(starts)[idx]) / np.array([width[starts[i]] for i in idx])
    ks = stats.kstest(offs, "uniform")
    assert ks.pvalue > 0.05, \
        f"in-bucket offsets not uniform (KS D={ks.statistic:.4f}, p={ks.pvalue:.4f})"
    print(f"[4] in-bucket offsets ~ U(0,1): OK "
          f"(KS D={ks.statistic:.4f}, p={ks.pvalue:.3f}, mean={offs.mean():.3f})")

    # 5. Poisson signature of inter-arrival times
    iat = np.diff(ms)
    expected_mean = (edges[-1] - ms[0]) / (len(ms) - 1)
    cv = iat.std() / iat.mean()
    assert abs(iat.mean() - expected_mean) / expected_mean < 0.05, "IAT mean off"
    assert 0.9 < cv < 1.2, f"IAT CV={cv:.2f} not Poisson-like"
    print(f"[5] inter-arrival times: mean={iat.mean():.1f} ms "
          f"(expected ~{expected_mean:.1f}), CV={cv:.2f} (Poisson=1): OK")

    print("--- all verification checks passed ---")


def main():
    rows = []
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["timestamp"], d["input_length"], d["output_length"]))
    assert all(a[0] <= b[0] for a, b in zip(rows, rows[1:])), "source not monotonic"

    # Bucket boundaries: each unique timestamp starts a bucket that extends
    # to the next unique timestamp (measured widths are 2997-3003 ms due to
    # recording jitter). The last bucket has no successor, so its end is
    # snapped to the next nominal grid boundary.
    starts = sorted({ts for ts, _, _ in rows})
    width = {s: nxt - s for s, nxt in zip(starts, starts[1:])}
    width[starts[-1]] = (round(starts[-1] / GRID_MS) + 1) * GRID_MS - starts[-1]

    rng = np.random.default_rng(SEED)
    jittered = [(ts + rng.uniform(0, width[ts]), il, ol) for ts, il, ol in rows]
    jittered.sort(key=lambda r: r[0])

    with open(DST, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        for ms, il, ol in jittered:
            ts = BASE + timedelta(milliseconds=ms)
            writer.writerow([f"{ts:%Y-%m-%d %H:%M:%S.%f}", il, ol])

    print(f"Wrote {len(jittered):,} requests to {DST}")
    print(f"Buckets: {len(starts):,} (~{np.median(list(width.values())):.0f} ms wide), "
          f"seed={SEED}")
    print(f"Time span: {BASE} -> {BASE + timedelta(milliseconds=jittered[-1][0])} "
          f"({jittered[-1][0] / 1000:.1f}s)")

    verify()


if __name__ == "__main__":
    main()
