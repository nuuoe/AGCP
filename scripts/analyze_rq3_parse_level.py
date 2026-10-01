"""Score the RQ3 correction-layer runs at the parse level (smart F1 vs GT).

The RQ3 chain is reported by execution success; this rescores the same
runs (K=1, K=3, K=3 repro, refinement) by parse F1 of final_atoms on the
114 unique paired tasks and prints McNemar p-values for both metrics.
Output: runs/rq3_parse_level.json.
"""
from __future__ import annotations

import json
import sys
from math import comb

sys.path.insert(0, ".")
from scripts.alfworld_enum_constrained import smart_f1_extended  # noqa: E402

RUNS = {
    "K1_one_shot": "runs/alfworld_iterative_e4_K1.json",
    "LayerA_K3": "runs/alfworld_iterative_e4.json",
    "LayerA_K3_repro": "runs/alfworld_iterative_e4_repro.json",
    "LayerB_refine": "runs/alfworld_iterative_e4_with_refinement.json",
}
THRESHOLDS = (0.9, 0.95)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def uniq(rows):
    d = {}
    for r in rows:
        d[r["task_id"]] = r  # last occurrence wins (the 114-task paired set)
    return d


def main() -> None:
    data = {k: uniq(json.load(open(p))["results"]) for k, p in RUNS.items()}
    common = set.intersection(*(set(v) for v in data.values()))
    common = sorted(common)
    print(f"paired unique tasks: {len(common)}")

    out = {"n_paired_unique": len(common), "runs": {}, "pairs": {}}

    per_run_flags = {}
    for name, rows in data.items():
        flags = {}
        for t in common:
            r = rows[t]
            f1 = smart_f1_extended(r.get("final_atoms") or {},
                                   r.get("gt_for_logging") or {})
            flags[t] = {
                "f1": f1,
                **{f"acc{th}": f1 >= th for th in THRESHOLDS},
                "exec": bool(r.get("execution_success")),
            }
        per_run_flags[name] = flags
        n = len(common)
        summary = {
            "exec": sum(1 for t in common if flags[t]["exec"]),
            "mean_f1": sum(flags[t]["f1"] for t in common) / n,
            **{f"parse_acc_{th}": sum(1 for t in common if flags[t][f"acc{th}"])
               for th in THRESHOLDS},
        }
        out["runs"][name] = summary
        print(f"{name:16s} exec {summary['exec']}/{n}"
              + "".join(f"  parse@{th} {summary[f'parse_acc_{th}']}/{n}"
                        for th in THRESHOLDS)
              + f"  meanF1 {summary['mean_f1']:.3f}")

    def pair(a: str, b: str, metric: str):
        fa, fb = per_run_flags[a], per_run_flags[b]
        bb = sum(1 for t in common if fb[t][metric] and not fa[t][metric])
        cc = sum(1 for t in common if fa[t][metric] and not fb[t][metric])
        p = mcnemar_exact(bb, cc)
        key = f"{a}->{b}:{metric}"
        out["pairs"][key] = {"wins_b_only": bb, "wins_a_only": cc,
                             "mcnemar_p": p}
        print(f"  {key:44s} +{bb} / -{cc}   p={p:.4g}")

    print("\npaired contrasts:")
    for metric in ["exec"] + [f"acc{th}" for th in THRESHOLDS]:
        pair("K1_one_shot", "LayerA_K3", metric)
        pair("LayerA_K3", "LayerB_refine", metric)
        pair("K1_one_shot", "LayerB_refine", metric)

    # repro check at parse level
    same = sum(
        1 for t in common
        if per_run_flags["LayerA_K3"][t]["acc0.9"]
        == per_run_flags["LayerA_K3_repro"][t]["acc0.9"])
    out["repro_identical_at_0.9"] = same
    print(f"\nLayerA vs repro identical accept@0.9 on {same}/{len(common)}")

    with open("runs/rq3_parse_level.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote: runs/rq3_parse_level.json")


if __name__ == "__main__":
    main()
