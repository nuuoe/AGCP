"""Compare the K=3 iterative-E4 run, its reproduction and a K=1 ablation.

Reads runs/alfworld_iterative_e4{,_repro,_K1}.json (same task order) and
prints Wilson CIs, the reproducibility delta, exact McNemar p-values and
Cohen's h for K=1 vs K=3, per-task-type stratification and sample tasks.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest


def wilson_ci(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (0.0, 0.0)
    from math import sqrt
    z = 1.959963984540054  # 95% normal
    phat = k / n
    denom = 1 + (z * z) / n
    centre = (phat + (z * z) / (2 * n)) / denom
    spread = z * sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load_run(path: str) -> dict:
    if not Path(path).exists():
        return None
    return json.load(open(path))


def task_wins(run: dict) -> dict[str, bool]:
    """Map task_id -> won; duplicate ids collapse to the last occurrence."""
    return {r["task_id"]: bool(r.get("execution_success")) for r in run["results"]}


def task_wins_indexed(run: dict) -> list[tuple[str, str, bool]]:
    """Return (task_id, task_type, won) triples in file order, so duplicate ids stay distinct."""
    return [(r["task_id"], r["task_type"], bool(r.get("execution_success")))
             for r in run["results"]]


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact binomial McNemar p-value, two-sided.
    H0: prob of (b) = prob of (c) = 0.5 among discordant pairs.
    Test statistic: min(b, c), n = b + c.
    """
    n = b + c
    if n == 0:
        return 1.0
    res = binomtest(min(b, c), n, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def per_task_type(run: dict) -> dict[str, tuple[int, int]]:
    by = defaultdict(lambda: [0, 0])
    for r in run["results"]:
        by[r["task_type"]][1] += 1
        if r.get("execution_success"):
            by[r["task_type"]][0] += 1
    return {k: (v[0], v[1]) for k, v in by.items()}


def main():
    p_orig = "runs/alfworld_iterative_e4.json"
    p_repro = "runs/alfworld_iterative_e4_repro.json"
    p_k1 = "runs/alfworld_iterative_e4_K1.json"

    orig = load_run(p_orig)
    repro = load_run(p_repro)
    k1 = load_run(p_k1)

    if orig is None or repro is None or k1 is None:
        print("Missing one of the runs:")
        print(f"  orig: {p_orig} -> {'OK' if orig else 'MISSING'}")
        print(f"  repro: {p_repro} -> {'OK' if repro else 'MISSING'}")
        print(f"  k1: {p_k1} -> {'OK' if k1 else 'MISSING'}")
        sys.exit(1)

    # Pair runs by position so duplicate task_ids stay distinct.
    idx_orig = task_wins_indexed(orig)
    idx_repro = task_wins_indexed(repro)
    idx_k1 = task_wins_indexed(k1)
    if not (len(idx_orig) == len(idx_repro) == len(idx_k1)):
        sys.exit(f"Length mismatch: orig={len(idx_orig)}, "
                  f"repro={len(idx_repro)}, k1={len(idx_k1)}")
    for i, (o, r, k_) in enumerate(zip(idx_orig, idx_repro, idx_k1)):
        if o[0] != r[0] or o[0] != k_[0]:
            sys.exit(f"Position {i} mismatch: orig={o[0]}, "
                      f"repro={r[0]}, k1={k_[0]}")
    n = len(idx_orig)
    print(f"# Tasks (index-paired, handles 5 duplicates): {n}")

    won_orig = [o[2] for o in idx_orig]
    won_repro = [r[2] for r in idx_repro]
    won_k1 = [k_[2] for k_ in idx_k1]
    task_types = [o[1] for o in idx_orig]
    task_ids = [o[0] for o in idx_orig]

    w_orig = task_wins(orig)
    w_repro = task_wins(repro)
    w_k1 = task_wins(k1)
    ids_all = set(w_orig) & set(w_repro) & set(w_k1)

    # Reproducibility: K3-repro vs K3-original.
    n_orig_win = sum(won_orig)
    n_repro_win = sum(won_repro)
    n_k1_win = sum(won_k1)

    ci_orig = wilson_ci(n_orig_win, n)
    ci_repro = wilson_ci(n_repro_win, n)
    ci_k1 = wilson_ci(n_k1_win, n)

    p_orig_rate = n_orig_win / n
    p_repro_rate = n_repro_win / n
    p_k1_rate = n_k1_win / n

    # Tasks whose outcome flipped between the two K=3 runs.
    flipped_repro_win_orig_lose_idx = [i for i, (a, b) in enumerate(zip(won_repro, won_orig)) if a and not b]
    flipped_repro_lose_orig_win_idx = [i for i, (a, b) in enumerate(zip(won_repro, won_orig)) if not a and b]

    # K=1 vs K=3-repro (McNemar).
    a = sum(1 for r, k in zip(won_repro, won_k1) if r and k)
    b_repro = sum(1 for r, k in zip(won_repro, won_k1) if r and not k)
    c_k1 = sum(1 for r, k in zip(won_repro, won_k1) if not r and k)
    d = sum(1 for r, k in zip(won_repro, won_k1) if not r and not k)

    mcnemar_p_repro = mcnemar_exact(b_repro, c_k1)
    cohens_h_repro = cohens_h(p_repro_rate, p_k1_rate)

    # The same contrast against the original K=3 run.
    a2 = sum(1 for o, k in zip(won_orig, won_k1) if o and k)
    b_orig = sum(1 for o, k in zip(won_orig, won_k1) if o and not k)
    c_k1_orig = sum(1 for o, k in zip(won_orig, won_k1) if not o and k)
    d2 = sum(1 for o, k in zip(won_orig, won_k1) if not o and not k)
    mcnemar_p_orig = mcnemar_exact(b_orig, c_k1_orig)
    cohens_h_orig = cohens_h(p_orig_rate, p_k1_rate)

    print("\n" + "=" * 72)
    print("REPRODUCIBILITY: K=3-repro vs K=3-original (same N=119 tasks)")
    print("=" * 72)
    print(f"K=3-original:  {n_orig_win}/{n} = {100 * p_orig_rate:.1f}% "
          f"(95% CI {100 * ci_orig[0]:.1f}–{100 * ci_orig[1]:.1f}%)")
    print(f"K=3-repro:     {n_repro_win}/{n} = {100 * p_repro_rate:.1f}% "
          f"(95% CI {100 * ci_repro[0]:.1f}–{100 * ci_repro[1]:.1f}%)")
    delta_pp_repro = 100 * (p_repro_rate - p_orig_rate)
    print(f"Delta:         {delta_pp_repro:+.2f} pp")
    print(f"Tasks flipped (repro win, orig lose): {len(flipped_repro_win_orig_lose_idx)}")
    print(f"Tasks flipped (repro lose, orig win): {len(flipped_repro_lose_orig_win_idx)}")

    print("\n" + "=" * 72)
    print("K=1 vs K=3-repro: does iteration help?")
    print("=" * 72)
    print(f"K=1:           {n_k1_win}/{n} = {100 * p_k1_rate:.1f}% "
          f"(95% CI {100 * ci_k1[0]:.1f}–{100 * ci_k1[1]:.1f}%)")
    print(f"K=3-repro:     {n_repro_win}/{n} = {100 * p_repro_rate:.1f}% "
          f"(95% CI {100 * ci_repro[0]:.1f}–{100 * ci_repro[1]:.1f}%)")
    delta_pp_k = 100 * (p_repro_rate - p_k1_rate)
    print(f"Delta:         {delta_pp_k:+.2f} pp")
    print(f"\nPaired confusion (K=1 rows x K=3-repro cols):")
    print(f"                  K3 win   K3 lose")
    print(f"   K1 win:        {a:4d}    {c_k1:4d}")
    print(f"   K1 lose:       {b_repro:4d}    {d:4d}")
    print(f"\nMcNemar discordant: b={b_repro} (K3 win, K1 lose), "
          f"c={c_k1} (K1 win, K3 lose)")
    print(f"McNemar exact binomial p-value: {mcnemar_p_repro:.4f}")
    print(f"Cohen's h:                       {cohens_h_repro:.3f}")
    print(f"  (interp: 0.2=small, 0.5=medium, 0.8=large)")

    print("\n" + "=" * 72)
    print("CHECK: K=1 vs K=3-original")
    print("=" * 72)
    print(f"Paired confusion (K=1 rows x K=3-orig cols):")
    print(f"                  K3 win   K3 lose")
    print(f"   K1 win:        {a2:4d}    {c_k1_orig:4d}")
    print(f"   K1 lose:       {b_orig:4d}    {d2:4d}")
    print(f"McNemar exact p:  {mcnemar_p_orig:.4f}")
    print(f"Cohen's h:        {cohens_h_orig:.3f}")
    print(f"Delta:            {100*(p_orig_rate - p_k1_rate):+.2f} pp")

    # Per-task-type breakdown (index-paired, includes duplicates).
    print("\n" + "=" * 72)
    print(f"PER-TASK-TYPE BREAKDOWN (full N={n}, index-paired)")
    print("=" * 72)
    by_type_w: dict[str, dict[str, list]] = {}
    for i, tt in enumerate(task_types):
        by_type_w.setdefault(tt, {"orig": [], "repro": [], "k1": []})
        by_type_w[tt]["orig"].append(won_orig[i])
        by_type_w[tt]["repro"].append(won_repro[i])
        by_type_w[tt]["k1"].append(won_k1[i])

    print(f"{'task_type':<35} {'N':>4} {'K=1':>9} {'K3-rep':>9} {'K3-org':>9}  delta(K3-K1)  McN-p")
    print("-" * 105)
    for tt in sorted(by_type_w):
        rec = by_type_w[tt]
        n_t = len(rec["orig"])
        s_k1 = sum(rec["k1"])
        s_rep = sum(rec["repro"])
        s_orig = sum(rec["orig"])
        d_t = (s_rep - s_k1) / n_t if n_t else 0
        # McNemar within this task type
        b_t = sum(1 for r, k in zip(rec["repro"], rec["k1"]) if r and not k)
        c_t = sum(1 for r, k in zip(rec["repro"], rec["k1"]) if not r and k)
        p_t = mcnemar_exact(b_t, c_t)
        print(f"{tt:<35} {n_t:>4d} "
              f"{s_k1:>4d}/{n_t:<4d} "
              f"{s_rep:>4d}/{n_t:<4d} "
              f"{s_orig:>4d}/{n_t:<4d}  "
              f"{100*d_t:+6.1f} pp     "
              f"{p_t:.3f}")

    # Sample tasks where iteration helped or hurt.
    print("\n" + "=" * 72)
    print("SAMPLE: Tasks where iteration HELPED (K1 lose, K3-repro win)")
    print("=" * 72)
    helped_idx = [i for i in range(n) if not won_k1[i] and won_repro[i]]
    print(f"# helped = {len(helped_idx)}")
    repro_records = repro["results"]
    k1_records = k1["results"]
    helped = [task_ids[i] for i in helped_idx]
    repro_by_id = {(r["task_id"], idx): r for idx, r in enumerate(repro["results"])}
    k1_by_id = {(r["task_id"], idx): r for idx, r in enumerate(k1["results"])}
    repro_by_pos = {idx: r for idx, r in enumerate(repro["results"])}
    k1_by_pos = {idx: r for idx, r in enumerate(k1["results"])}
    repro_by_id_only = {r["task_id"]: r for r in repro["results"]}
    k1_by_id_only = {r["task_id"]: r for r in k1["results"]}
    for i_pos in helped_idx[:5]:
        tid = task_ids[i_pos]
        r3 = repro_records[i_pos]
        r1 = k1_records[i_pos]
        print(f"\n  task_id={tid}  (slot #{i_pos})")
        print(f"    task_type={r3['task_type']}")
        errs_k1 = r1.get('errors_per_iteration', [[]])[0] if r1.get('errors_per_iteration') else []
        print(f"    K=1 iter_used={r1.get('iterations_used')} "
              f"verified={r1.get('verified')} "
              f"errors_iter1={errs_k1[:2] if errs_k1 else []}")
        print(f"    K=1 final_atoms={r1.get('final_atoms')}")
        print(f"    K=1 stage_failed={r1.get('stage_failed')}  "
              f"plan_len={r1.get('plan_length')}")
        print(f"    K=3 iter_used={r3.get('iterations_used')} "
              f"verified={r3.get('verified')}")
        atoms_k3 = r3.get('atoms_per_iteration', [{}])
        if atoms_k3:
            print(f"    K=3 atoms_iter1={atoms_k3[0]}")
        if r3.get('iterations_used', 0) >= 2:
            errs_k3 = r3.get('errors_per_iteration', [[]])[0] if r3.get('errors_per_iteration') else []
            print(f"    K=3 errors_iter1={errs_k3[:2]}")
            print(f"    K=3 atoms_iter2={atoms_k3[1] if len(atoms_k3) > 1 else None}")
        print(f"    K=3 final_atoms={r3.get('final_atoms')}")

    print("\n" + "=" * 72)
    print("SAMPLE: Tasks where iteration HURT (K1 win, K3-repro lose)")
    print("=" * 72)
    hurt_idx = [i for i in range(n) if won_k1[i] and not won_repro[i]]
    print(f"# hurt = {len(hurt_idx)}")
    for i_pos in hurt_idx[:5]:
        tid = task_ids[i_pos]
        r3 = repro_records[i_pos]
        r1 = k1_records[i_pos]
        print(f"\n  task_id={tid}  (slot #{i_pos})")
        print(f"    task_type={r3['task_type']}")
        print(f"    K=1 final={r1.get('final_atoms')}  won=True")
        print(f"    K=3 iter_used={r3.get('iterations_used')} verified={r3.get('verified')}")
        print(f"    K=3 final={r3.get('final_atoms')}  won=False  "
              f"stage_failed={r3.get('stage_failed')}")

    print("\n" + "=" * 72)
    print("SAMPLE: Tasks where K=1 == K=3-repro (both won)")
    print("=" * 72)
    both_won_idx = [i for i in range(n) if won_k1[i] and won_repro[i]]
    print(f"# both_won = {len(both_won_idx)}")
    for i_pos in both_won_idx[:5]:
        tid = task_ids[i_pos]
        r3 = repro_records[i_pos]
        r1 = k1_records[i_pos]
        verified_iter1 = not (r3.get('errors_per_iteration', [[]])[0] if r3.get('errors_per_iteration') else [])
        print(f"  task_id={tid}  type={r3['task_type']:<35s}  "
              f"K=1_iter_used={r1.get('iterations_used')}, "
              f"K=3_iter_used={r3.get('iterations_used')} (verified iter1: "
              f"{verified_iter1})")

    print("\n" + "=" * 72)
    print("K=3-repro: iterations used distribution")
    print("=" * 72)
    iter_dist: dict[int, int] = defaultdict(int)
    for r in repro_records:
        iter_dist[r.get("iterations_used", 0)] += 1
    for k in sorted(iter_dist):
        print(f"  iterations_used={k}: {iter_dist[k]} tasks")

    print("\n" + "=" * 72)
    print("K=3-original: iterations used distribution")
    print("=" * 72)
    iter_dist_o: dict[int, int] = defaultdict(int)
    for r in orig["results"]:
        iter_dist_o[r.get("iterations_used", 0)] += 1
    for k in sorted(iter_dist_o):
        print(f"  iterations_used={k}: {iter_dist_o[k]} tasks")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    # A |delta| of at most 3pp counts as reproducible.
    reproducible = abs(delta_pp_repro) <= 3.0
    print(f"Reproducibility (|delta| ≤ 3pp):  "
          f"{'YES' if reproducible else 'NO'}  "
          f"(observed: {delta_pp_repro:+.2f} pp)")
    print(f"Iteration causes lift (>3pp + p<0.05):  "
          f"{'YES' if (delta_pp_k > 3.0 and mcnemar_p_repro < 0.05) else 'NO'}")
    print(f"  K=3-repro lift over K=1:  {delta_pp_k:+.2f} pp")
    print(f"  McNemar p:                {mcnemar_p_repro:.4f}")
    print(f"  Cohen's h:                {cohens_h_repro:.3f}")

    if delta_pp_k > 3.0 and mcnemar_p_repro < 0.05:
        print("\nK=3 lift over K=1 exceeds 3pp and is significant (p<0.05).")
    elif delta_pp_k > 3.0:
        print("\nK=3 lift over K=1 exceeds 3pp but is not significant "
              "at this N.")
    elif delta_pp_k > 0:
        print("\nK=3 lift over K=1 is under 3pp: within sampling noise.")
    else:
        print("\nK=3 does not improve on K=1.")


if __name__ == "__main__":
    main()
