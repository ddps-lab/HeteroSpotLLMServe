"""
ALL-NODES variant: each group's pipeline must use ALL of the group's nodes (every
node = one stage, >=1 layer). Memory-infeasible all-nodes groups are excluded.
Per group the layer partition is optimized for throughput (cost fixed when the node
set is fixed, so soft_slo == only_throughput); extracted from the DP table at
dp[num_layers][S][full-signature] with a high top_k so the layer partition converges.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_p_common as J
from cluster_pool import ClusterPool
from shuntserve_optimizer import run_test_case

TOP_K_AN = 3
_W = {}

def _init(model_name):
    import logging; logging.disable(logging.CRITICAL)
    _W["cfg"] = J.build_config(model_name)

def _full_sig(group):
    s = []
    for i in range(3):
        s += [J.INSTANCE_ORDER[i]] * group[i]
    return tuple(sorted(s))

def all_nodes_pipeline(group, cfg, top_k=TOP_K_AN):
    """Best pipeline using ALL nodes of `group` (or None if memory-infeasible)."""
    S = sum(group)
    cp = ClusterPool(available_spot_nodes=J.group_to_nodes(group), spot_prices=J.PRICES)
    t0 = time.time()
    _res, opt, ot = run_test_case(cfg, budget=J.BUDGET, latency_slo=J.LATENCY_SLO,
                                  cluster_pool=cp, max_stages=S, top_k=top_k,
                                  optimization_mode="only_throughput")
    pl = opt.dp[cfg["num_layers"]][S].get(_full_sig(group))
    return pl, ot, time.time() - t0

def _task(group):
    cfg = _W["cfg"]
    pl, ot, wall = all_nodes_pipeline(group, cfg)
    feas = pl is not None and pl.throughput > 0
    return (tuple(group), {"group": list(group), "feasible": feas, "opt_time": ot, "wall_time": wall,
                           "throughput": (pl.throughput if feas else 0.0),
                           "tp": ([J.TP_OF[s] for s in pl.stages] if feas else []),
                           "layers": ([int(x) for x in pl.layer_per_stage] if feas else [])})

def compute_an_memo(model_short, procs, cache):
    from multiprocessing import Pool
    groups = J.all_unique_groups()
    recs = {}
    if cache and os.path.exists(cache):
        blob = json.load(open(cache))
        if blob.get("model") == J.MODELS[model_short]:
            recs = {tuple(int(x) for x in k.split("_")): v for k, v in blob["records"].items()}
    todo = [g for g in groups if g not in recs]
    if todo:
        with Pool(procs, initializer=_init, initargs=(J.MODELS[model_short],)) as p:
            done = 0
            for g, rec in p.imap_unordered(_task, todo, chunksize=1):
                recs[g] = rec; done += 1
                if done % 10 == 0: print(f"    {done}/{len(todo)} groups", flush=True)
        if cache:
            json.dump({"model": J.MODELS[model_short],
                       "records": {f"{g[0]}_{g[1]}_{g[2]}": v for g, v in recs.items()}}, open(cache, "w"))
    return recs

def best_all_nodes_partition(memo, p):
    best = None
    for part in J.enumerate_partitions(J.CLUSTER, p):
        if all(memo[g]["feasible"] for g in part):           # 모든 그룹이 all-nodes feasible
            tot = sum(memo[g]["throughput"] for g in part)
            if best is None or tot > best["total"]:
                best = {"total": tot, "partition": [list(g) for g in part]}
    return best

def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 1)
    out = {"experiment": "all_nodes_each_group", "cluster": list(J.CLUSTER),
           "top_k": TOP_K_AN, "models": {}}
    for ms in ["llama3-70b", "qwen3-32b"]:
        print(f"\n==== {ms}: all-nodes optima, {len(J.all_unique_groups())} groups (top_k={TOP_K_AN}, {procs} procs)")
        t0 = time.time()
        memo = compute_an_memo(ms, procs, f"{ms}/joint-p/results/memo_allnodes_k3_{ms}.json")
        print(f"  computed in {time.time()-t0:.1f}s")

        # independent re-check of the 9-node pipeline throughput
        cfg = J.build_config(J.MODELS[ms])
        pl, _, _ = all_nodes_pipeline(J.CLUSTER, cfg)
        if pl is not None:
            from shuntserve_optimizer import Pipeline
            chk = Pipeline(); chk.stages = list(pl.stages); chk.azs = ["dummy-az"]*len(pl.stages)
            chk.layer_per_stage = list(pl.layer_per_stage); chk.calculate_throughput(cfg)
            print(f"  [verify 9-node] stages={len(pl.stages)} (uses all 9? {len(pl.stages)==9}) "
                  f"thr={pl.throughput:.4f}  recompute={chk.throughput:.4f}  layers={list(pl.layer_per_stage)}")

        greedy = J.GREEDY_REF[ms]["total"]; K = J.GREEDY_REF[ms]["K"]
        print(f"  greedy K={K} total={greedy:.4f}")
        per_p = {}
        for p in range(1, 10):
            b = best_all_nodes_partition(memo, p)
            per_p[p] = b
            if b is None:
                print(f"    p={p}: none (all-nodes 구성 가능한 {p}-분할 없음)")
            else:
                cmp = "  >greedy!" if b["total"] > greedy + 1e-6 else ("  =greedy" if abs(b["total"]-greedy)<1e-6 else "")
                print(f"    p={p}: {b['total']:.4f}{cmp}   parts={b['partition']}")
        out["models"][ms] = {"greedy": greedy, "K": K,
                             "per_p": {p: (per_p[p]["total"] if per_p[p] else None) for p in per_p},
                             "per_p_partition": {p: (per_p[p]["partition"] if per_p[p] else None) for p in per_p}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/all_nodes_compare.json", "w"), indent=2)
    print("\nsaved -> results/all_nodes_compare.json")

if __name__ == "__main__":
    main()
