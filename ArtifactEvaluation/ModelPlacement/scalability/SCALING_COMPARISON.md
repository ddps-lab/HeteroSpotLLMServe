# Optimizer cost vs cluster size — full extraction & first-pipeline time

llama3-70b, k=1, max_stages=None, soft_slo. Two cluster definitions swept over
the node-multiplier M:

- **46-type** (`uswest2_46type/`): us-west-2 full NVIDIA cluster, 46 instance
  types × M nodes (139 GPUs at M=1). Abandoned for the paper — too slow.
- **15-type** (`large_hetero_15type/`): large_hetero cluster, 15 instance types
  × M nodes (76 GPUs at M=1). Used for the paper's sensitivity sweep.

For each config we report **full extraction time** (extract every pipeline until
the cluster is exhausted) and **first-pipeline time** (round 1 alone — the single
most expensive search on the full cluster, isolates per-search cost).

## 46-type cluster (46 instance types)

| M | nodes | GPUs | first-pipeline (round 1) | full extraction | per pipeline | throughput | pipelines |
| - | ----- | ---- | ------------------------ | --------------- | ------------ | ---------- | --------- |
| 1 | 46    | 139  | 23.4 min (1404 s)        | 2.39 h          | 10.3 min     | 232.2      | 14        |
| 2 | 92    | 278  | 39.3 min (2357 s)        | 8.56 h          | 21.4 min     | 468.7      | 24        |
| 3 | 138   | 417  | 44.8 min (2690 s)        | 15.15 h         | 28.4 min     | 704.8      | 32        |
| 4 | 184   | 556  | 47.5 min (2850 s)        | 21.99 h         | 30.0 min     | 939.4      | 44        |
| 6 | 276   | 834  | 51.0 min (3058 s)        | 35.49 h         | 33.8 min     | 1410.7     | 63        |

M=8 (368 nodes) OOM-crashed at round-1 layer 80 on 15 GB boxes; not re-run.

## 15-type cluster (15 instance types) — paper sweep

| M  | nodes | GPUs | first-pipeline (round 1) | full extraction | per pipeline | throughput | pipelines |
| -- | ----- | ---- | ------------------------ | --------------- | ------------ | ---------- | --------- |
| 1  | 15    | 76   | 1.8 min (106 s)          | 0.10 h          | 0.7 min      | 112.3      | 8         |
| 2  | 30    | 152  | 5.5 min (329 s)          | 0.55 h          | 2.2 min      | 225.3      | 15        |
| 3  | 45    | 228  | 9.4 min (564 s)          | 1.49 h          | 4.5 min      | 337.9      | 20        |
| 4  | 60    | 304  | 12.8 min (769 s)         | 2.63 h          | 5.8 min      | 451.8      | 27        |
| 6  | 90    | 456  | 16.9 min (1015 s)        | 5.57 h          | 8.6 min      | 678.4      | 39        |
| 8  | 120   | 608  | 18.9 min (1136 s)        | 8.32 h          | 9.8 min      | 905.9      | 51        |
| 12 | 180   | 912  | 20.5 min (1233 s)        | 13.67 h         | 11.4 min     | 1360.3     | 72        |
| 16 | 240   | 1216 | 21.4 min (1283 s)        | 19.11 h         | 12.1 min     | 1813.8     | 95        |
| 24 | 360   | 1824 | 22.3 min (1340 s)        | 30.18 h         | 12.8 min     | 2722.5     | 141       |
| 32 | 480   | 2432 | 22.4 min (1346 s)        | 41.01 h         | 13.1 min     | 3630.6     | 188       |

## First-pipeline time at larger M (first-pipeline-only sweeps, N < 1000)

Full extraction is too slow to push to large M (esp. the 46-type), so these
runs stop after round 1 — measuring only the time to find the **1st** pipeline.
This extends the first-pipeline curve far past where full extraction is feasible,
and both clusters **saturate**: once node count N exceeds num_layers (80), the
DP depth caps at min(N, 80), so adding nodes no longer increases single-search
cost. (First pipeline picks the same best placement throughout: 46-type
27.74 req/s, 15-type 66.57 req/s.)

| cluster | M | nodes | GPUs | first-pipeline (round 1) |
| ------- | -- | ----- | ---- | ------------------------ |
| 46-type | 8  | 368   | 1112 | 53.3 min (3198 s) |
| 46-type | 10 | 460   | 1390 | 54.9 min (3295 s) |
| 46-type | 12 | 552   | 1668 | 55.6 min (3335 s) |
| 46-type | 14 | 644   | 1946 | 55.8 min (3350 s) |
| 46-type | 16 | 736   | 2224 | 56.7 min (3404 s) |
| 46-type | 20 | 920   | 2780 | 56.8 min (3409 s) |
| 15-type | 40 | 600   | 3040 | 22.3 min (1336 s) |
| 15-type | 48 | 720   | 3648 | 22.2 min (1334 s) |
| 15-type | 56 | 840   | 4256 | 22.5 min (1348 s) |
| 15-type | 64 | 960   | 4864 | 22.4 min (1343 s) |

Combined with the per-cluster tables above, the first-pipeline time plateaus at
~57 min (46-type) and ~22 min (15-type) — see `figures/time_find_1st_pipeline.*`.

## Head-to-head: ≈ equal node count, different type diversity

46 nodes (46 types × 1) vs 45 nodes (15 types × 3) — node count nearly equal,
so the only variable is type diversity (46 vs 15 → branching factor 3×).

| metric                                             | 46-type (M=1)      | 15-type (M=3)     | ratio               |
| -------------------------------------------------- | ------------------ | ----------------- | ------------------- |
| instance types                                     | 46                 | 15                | 3.07×              |
| nodes                                              | 46                 | 45                | ≈ equal            |
| GPUs                                               | 139                | 228               | 0.61×              |
| **first-pipeline time**                      | **23.4 min** | **9.4 min** | **2.49×**    |
| candidates evaluated (round 1)                     | 66,631             | 22,901            | 2.91×              |
| peak RSS (round 1)                                 | 5.27 GB            | 2.10 GB           | 2.51×              |
| **first-pipeline placement (P1 throughput)** | **66.57**    | **66.57**   | **identical** |
| full extraction time                               | 2.39 h             | 1.49 h            | 1.60×              |
| throughput / GPU                                   | 1.670              | 1.482             | 1.13×              |

## Takeaways

1. **First-pipeline cost ∝ type diversity.** At equal node count, round-1 time
   scales with the number of instance types (2.49×), and candidate evaluations /
   memory scale almost exactly with the type ratio (2.9× ≈ 46/15). The DP
   branching factor is `|types_remaining|`.
2. **Same node count → same first pipeline.** The chosen P1 throughput is
   identical (66.57 req/s) — only the *search cost* differs, not the result.
3. **Full extraction narrows the gap (1.60×)** because the 46-type cluster
   exhausts its types in fewer rounds (14 vs 20), so later searches get cheaper.
4. **First-pipeline time saturates with M; full time keeps growing.** Once
   M ≥ ~knee (depth caps at min(N, 80)), round-1 time plateaus (e.g. 15-type:
   20.5 → 22.4 min from M=12 to M=32) while full time grows ∝ #pipelines (∝ N).
