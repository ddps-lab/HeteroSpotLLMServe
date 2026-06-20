# Cluster-size sensitivity sweep — large_hetero cluster (15 instance types)

Optimizer sensitivity vs cluster size on the **15-type** large heterogeneous
cluster (7 GPU types, 15 instance types, 15×M nodes, 76 GPUs at M=1).
llama3-70b, k=1, max_stages=None. This is the cluster used for the paper
(the 46-type sweep in `../uswest2_46type/` was too slow).

Single invocation, M-sweep across parallel workers on one instance:
`--node-multipliers 1 2 3 4 6 8 12 16 24 32` → combined JSON below.

NOTE: JSON cluster key is `uswest2_full_cluster` but the node dict is the
15-type large_hetero definition.

| M | nodes (15×M) | wall | throughput | pipelines |
|---|------|------|------------|-----------|
| 1 |  15  | 0.10 h | 112.3 | 8 |
| 2 |  30  | 0.55 h | 225.3 | 15 |
| 3 |  45  | 1.49 h | 337.9 | 20 |
| 4 |  60  | 2.63 h | 451.8 | 27 |
| 6 |  90  | 5.57 h | 678.5 | 39 |
| 8 | 120  | 8.32 h | 905.9 | 51 |
| 12| 180  | 13.67 h | 1360.3 | 72 |
| 16| 240  | 19.11 h | 1813.8 | 95 |
| 24| 360  | 30.18 h | 2722.5 | 141 |
| 32| 480  | 41.01 h | 3630.6 | 188 |

json/  — one combined results JSON (all 10 M values)
logs/  — main log + workers_<ts>/ per-M per-layer worker logs (M=1..32)
