"""
Greedy  vs  Joint(global optimum via memoization + subset-DP) — TIME comparison.

The per-pipeline optimizer (run_test_case) is UNCHANGED.  We only optimize the
*orchestration*:
  - GREEDY   : ~K calls to run_test_case (one per iteration over the shrinking cluster).
  - JOINT    : optimize each UNIQUE sub-cluster once (memoization, parallel), then
               find the GLOBAL optimum over all partitions with a subset-DP — WITHOUT
               enumerating the (super-exponentially many) partitions.

Both should return the SAME total throughput (joint = global optimum >= greedy;
here they coincide).  This script reports the wall-clock / CPU time of each on the
current machine.  Run it on the target instance (e.g. m8a) for the final numbers.

Run:  python3 joint_optimum_dp.py            # both models
      python3 joint_optimum_dp.py <procs>    # cap parallel workers
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_p_common as J  # noqa: E402


def main():
    # silence the optimizer's per-layer logs so (a) output is clean and (b) greedy
    # is timed WITHOUT logging overhead, consistently with the memo workers.
    import logging
    logging.disable(logging.CRITICAL)

    procs = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 1)
    n_groups = len(J.all_unique_groups())
    out = {
        "experiment": "greedy_vs_joint_subset_dp",
        "machine": {"cores": os.cpu_count(), "workers": procs},
        "cluster": list(J.CLUSTER),
        "num_unique_groups": n_groups,
        "optimizer": {"top_k": J.TOP_K, "max_stages": J.MAX_STAGES,
                      "mode": "soft_slo", "untouched": True},
        "models": {},
    }
    bar = "=" * 92
    print(bar)
    print("GREEDY vs JOINT(memoization + subset-DP)  —  orchestration time "
          "(optimizer itself unchanged)")
    print(f"Cluster (g5,g6,g6e) = {J.CLUSTER}   unique groups = "
          f"prod(n_i+1)-1 = {n_groups}   |   machine cores = {os.cpu_count()}, "
          f"workers = {procs}")
    print(bar)

    rows = []
    for model in ["llama3-70b", "qwen3-32b"]:
        print(f"\n[{model}] running greedy ...", flush=True)
        gd = J.run_greedy_timed(model)

        print(f"[{model}] running joint: memoize {n_groups} groups (soft_slo, parallel) "
              f"then subset-DP ...", flush=True)
        t0 = time.time()
        memo = J.compute_memo(model, modes=("soft_slo",), cache_path=None,
                              processes=procs, log=lambda *a: None)
        memo_wall = time.time() - t0
        opt = J.joint_optimum_dp(memo, "soft_slo")
        serial_cpu = sum(memo[(g, "soft_slo")]["opt_time"] for g in J.all_unique_groups())

        # per-group (per-candidate-pipeline) optimizer time for all 59 unique groups
        group_recs = []
        for g in J.all_unique_groups():
            r = memo[(g, "soft_slo")]
            group_recs.append({"group": list(g), "label": J.group_str(g),
                               "opt_time_s": r["opt_time"], "wall_time_s": r["wall_time"],
                               "throughput": r["throughput"], "feasible": r["feasible"],
                               "num_stages": r["num_stages"], "tp": r["tp"]})

        same = abs(gd["total"] - opt["total"]) < 1e-6
        rows.append((model, gd, opt, memo_wall, serial_cpu, same))

        out["models"][model] = {
            "greedy": {"total": gd["total"], "K": gd["K"],
                       "optimizer_calls": gd["n_calls"],
                       "opt_cpu_s": gd["opt_time"], "wall_s": gd["wall"]},
            "greedy_calls": gd["calls"],
            "joint": {"total": opt["total"], "optimizer_calls": n_groups,
                      "num_groups": opt["num_groups"],
                      "num_feasible_groups": opt["num_feasible_groups"],
                      "memo_serial_cpu_s": serial_cpu,
                      "memo_parallel_wall_s": memo_wall,
                      "subset_dp_ms": opt["dp_time"] * 1000.0,
                      "partition": [list(g) for g in opt["partition"]],
                      "per_group_times": group_recs},
            "equal_greedy_eq_joint_optimum": same,
            "ratio_joint_over_greedy": {
                "wall": (memo_wall / gd["wall"]) if gd["wall"] else None,
                "cpu": (serial_cpu / gd["opt_time"]) if gd["opt_time"] else None},
        }

        print(f"\n--- {model} ---")
        print(f"  GREEDY : total={gd['total']:.4f}  K={gd['K']}  "
              f"optimizer_calls={gd['n_calls']}  "
              f"opt_CPU={gd['opt_time']:.1f}s  wall={gd['wall']:.1f}s")
        print(f"  JOINT  : total={opt['total']:.4f}  groups_used={opt['num_groups']}"
              f"({opt['num_feasible_groups']} feasible)  "
              f"optimizer_calls={n_groups} (unique groups)")
        print(f"           memo: serial_CPU={serial_cpu:.1f}s  parallel_wall={memo_wall:.1f}s"
              f"  | subset-DP={opt['dp_time']*1000:.2f}ms")
        print(f"  EQUAL? greedy==joint optimum : {same}  "
              f"(joint is the GLOBAL optimum; greedy matches it here)")

        print(f"  greedy per-call optimizer time:")
        for c in gd["calls"]:
            print(f"    iter{c['iter']:<2} {c['phase']:<20} opt={c['opt_time_s']:8.3f}s  "
                  f"thr={c['throughput']:7.3f}  nodes={c.get('nodes', {})}")
        print(f"  joint per-group optimizer time (all {n_groups} unique groups, slowest first):")
        for gr in sorted(group_recs, key=lambda x: -x["opt_time_s"]):
            print(f"    {gr['label']:<26} opt={gr['opt_time_s']:8.3f}s  "
                  f"thr={gr['throughput']:7.3f}  {'OK' if gr['feasible'] else 'INFEASIBLE'}")

    print("\n" + bar)
    print("SUMMARY  (total throughput identical; only orchestration time differs)")
    print(bar)
    hdr = (f"{'model':<14}{'total':>9}{'greedy_calls':>13}{'greedy_wall':>13}"
           f"{'joint_calls':>13}{'joint_CPU':>11}{'joint_wall':>12}{'DP':>9}")
    print(hdr)
    for model, gd, opt, memo_wall, serial_cpu, same in rows:
        print(f"{model:<14}{gd['total']:>9.4f}{gd['n_calls']:>13}{gd['wall']:>12.1f}s"
              f"{n_groups:>13}{serial_cpu:>10.1f}s{memo_wall:>11.1f}s"
              f"{opt['dp_time']*1000:>7.1f}ms")
    print("\nNote: greedy is inherently serial (~K calls). Joint memoizes "
          f"{n_groups} unique-group optimizations (parallelizable) + a subset-DP "
          "(O(prod (n_i+1)(n_i+2)/2), sub-ms). Joint = global optimum but does far "
          "more optimizer calls; this is the cost of exhaustive joint optimization.")

    out_path = os.environ.get("JOINT_DP_OUT") or os.path.join(
        HERE, "results", "joint_optimum_dp.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results JSON -> {out_path}")


if __name__ == "__main__":
    main()
