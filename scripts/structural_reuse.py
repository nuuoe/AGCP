"""COMPOSITIONAL ABSTRACTION REUSE on Depots (IPC-2002).

Depots is the canonical compositional planning benchmark: its
action set is literally Blocksworld (stack/unstack via hoists)
PLUS Logistics (drive trucks between depots). Each action uses
predicates from a clearly identifiable subdomain.

Pipeline:
  1. Induce lifted schemas on LOGISTICS problems → SchemaMemory.
  2. Induce lifted schemas on BLOCKSWORLD problems → SchemaMemory.
  3. Collect Depots transitions from random rollouts.
  4. For each Depots action: attempt STRUCTURAL reuse from
     SchemaMemory (cross-action-name, signature-subset match)
     against held-out Depots transitions. Accept if verifier
     accuracy ≥ threshold. Else fresh-induce from Depots
     transitions.
  5. Compare resulting Depots model F1 vs ground-truth PDDL.
  6. Report: actions reused (and from which source domain),
     actions fresh-induced, total compute saved.

This demonstrates the compositional-abstraction-reuse claim on
a REAL IPC benchmark, not a hand-crafted hybrid.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from agplan.adaptive_pipeline import SchemaMemory
from scripts.induce_pddl_generic import (
    collect_transitions, induce_lifted_models,
    load_task, ground_truth_models, f1,
)
from scripts.grounding import ground_lifted


def grounded_verifier(schema: dict, held_out: list,
                       min_n: int = 5) -> float:
    """Apply lifted schema to each held-out transition. Score =
    fraction where predicted state_after matches observed.

    Returns 0 if held_out has < min_n samples (insufficient evidence
    — refuse to claim reuse on too-few examples).
    """
    if len(held_out) < min_n:
        return 0.0
    add_t = set(schema.get("eff_add", []))
    del_t = set(schema.get("eff_del", []))
    ok = 0
    for sb, _action, args, sa in held_out:
        add_g = ground_lifted(add_t, args)
        del_g = ground_lifted(del_t, args)
        predicted = (sb - del_g) | add_g
        if predicted == sa:
            ok += 1
    return ok / max(len(held_out), 1)


def try_structural_reuse(memory: SchemaMemory,
                          held_out: list,
                          accept_threshold: float = 0.9,
                          ) -> tuple:
    """Try every memory schema (regardless of original name)
    against held_out. Return (matched_entry, score) for the best
    match, or (None, 0.0) if none meets threshold.
    """
    obs_names = set()
    for sb, _act, _args, sa in held_out:
        for p in sb | sa:
            name = p.split("(")[0]
            obs_names.add(name)

    best = (None, 0.0)
    for cand in memory.structural_candidates(obs_names):
        score = grounded_verifier(cand, held_out)
        if score > best[1]:
            best = (cand, score)
    if best[1] >= accept_threshold:
        return best
    return (None, best[1])


def _models_to_lists(models: dict) -> dict:
    return {a: {k: sorted(v) if isinstance(v, (set, frozenset))
                else v
                for k, v in m.items()}
            for a, m in models.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw_domain", required=True)
    ap.add_argument("--bw_problems", required=True,
                    help="Glob for BW problem instances")
    ap.add_argument("--log_domain", required=True)
    ap.add_argument("--log_problems", required=True,
                    help="Glob for Logistics problem instances")
    ap.add_argument("--depots_domain", required=True)
    ap.add_argument("--depots_problems", required=True,
                    help="Glob for Depots problem instances")
    ap.add_argument("--n_problems_per_domain", type=int, default=5)
    ap.add_argument("--steps_per_problem", type=int, default=80)
    ap.add_argument("--accept_threshold", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("COMPOSITIONAL ABSTRACTION REUSE — Depots = BW + Logistics")
    print("=" * 60)

    # ---- STEP 1: Logistics → memory ----
    print("\n[1] Inducing on LOGISTICS...")
    log_paths = sorted(glob.glob(args.log_problems))[
        :args.n_problems_per_domain]
    log_trs = collect_transitions(args.log_domain, log_paths,
                                    steps_per_problem=args.steps_per_problem)
    log_models = induce_lifted_models(log_trs)
    print(f"  {len(log_trs)} transitions → {len(log_models)} schemas: "
          f"{sorted(log_models.keys())}")

    memory = SchemaMemory()
    for op, m in log_models.items():
        memory.add(op, m, provenance="logistics")

    # ---- STEP 2: Blocksworld → memory ----
    print("\n[2] Inducing on BLOCKSWORLD...")
    bw_paths = sorted(glob.glob(args.bw_problems))[
        :args.n_problems_per_domain]
    bw_trs = collect_transitions(args.bw_domain, bw_paths,
                                   steps_per_problem=args.steps_per_problem)
    bw_models = induce_lifted_models(bw_trs)
    print(f"  {len(bw_trs)} transitions → {len(bw_models)} schemas: "
          f"{sorted(bw_models.keys())}")

    for op, m in bw_models.items():
        memory.add(op, m, provenance="blocksworld")

    print(f"\n  Total memory: {len(memory.schemas)} schemas from 2 domains")

    # ---- STEP 3: Depots transitions ----
    print("\n[3] Collecting DEPOTS rollouts...")
    depots_paths = sorted(glob.glob(args.depots_problems))[
        :args.n_problems_per_domain]
    depots_trs = collect_transitions(args.depots_domain, depots_paths,
                                       steps_per_problem=args.steps_per_problem)
    print(f"  {len(depots_trs)} Depots transitions")

    # Group by action name
    by_action = {}
    for sb, action, args_t, sa in depots_trs:
        if sb == sa: continue
        by_action.setdefault(action, []).append((sb, action, args_t, sa))
    print(f"  Depots action set: {sorted(by_action.keys())}")

    # ---- STEP 4: Per-action structural reuse vs fresh induction ----
    print("\n[4] Per-action: try reuse, else fresh-induce...")
    final_models = {}
    reuse_report = {}
    fresh_trs_for_reinduction = []
    for action, trs in by_action.items():
        matched, score = try_structural_reuse(
            memory, trs, accept_threshold=args.accept_threshold,
        )
        if matched is not None:
            final_models[action] = {
                "pre_pos": set(matched.get("pre_pos", [])),
                "eff_add": set(matched.get("eff_add", [])),
                "eff_del": set(matched.get("eff_del", [])),
            }
            reuse_report[action] = {
                "reused": True,
                "from_domain": matched.get("provenance_domain", "?"),
                "score": score,
                "n_held_out": len(trs),
            }
            print(f"  {action}: REUSED from "
                  f"{matched.get('provenance_domain', '?')} "
                  f"(verifier acc={score*100:.1f}%, n={len(trs)})")
        else:
            reuse_report[action] = {
                "reused": False,
                "best_score": score,
                "n_held_out": len(trs),
            }
            fresh_trs_for_reinduction.extend(trs)

    # Fresh-induce remaining actions
    if fresh_trs_for_reinduction:
        print(f"\n  Fresh-inducing {len(fresh_trs_for_reinduction)} transitions "
              f"for {len([k for k,v in reuse_report.items() if not v['reused']])} "
              f"unmatched actions...")
        fresh = induce_lifted_models(fresh_trs_for_reinduction)
        for op, m in fresh.items():
            if op not in final_models:
                final_models[op] = {
                    "pre_pos": set(m["pre_pos"]),
                    "eff_add": set(m["eff_add"]),
                    "eff_del": set(m["eff_del"]),
                }
                print(f"    {op}: FRESH-induced ({m['n_examples']} examples)")

    # ---- STEP 5: F1 vs Depots ground-truth ----
    print("\n[5] Compare composite model vs Depots GT...")
    dom, _task = load_task(args.depots_domain, depots_paths[0])
    gt = ground_truth_models(dom)
    overall = []
    for action in gt:
        if action not in final_models:
            print(f"  {action}: MISSING from final model")
            continue
        fm = final_models[action]
        gtm = gt[action]
        pp = f1(fm["pre_pos"], gtm["pre_pos"])
        aa = f1(fm["eff_add"], gtm["eff_add"])
        dd = f1(fm["eff_del"], gtm["eff_del"])
        overall.append((action, pp, aa, dd))
        marker = ("REUSED" if reuse_report.get(action, {}).get("reused")
                  else "FRESH")
        print(f"  {action} [{marker}]: pre F1={pp[2]:.2f} "
              f"add F1={aa[2]:.2f} del F1={dd[2]:.2f}")

    # ---- STEP 6: Summary ----
    n_reused = sum(1 for v in reuse_report.values() if v["reused"])
    n_total = len(by_action)
    print("\n" + "=" * 60)
    print(f"COMPOSITIONAL REUSE: {n_reused}/{n_total} Depots actions "
          f"reused schemas from prior subdomains")
    print(f"  ({n_total - n_reused} fresh-induced from Depots transitions)")
    if overall:
        mean_pre = sum(pp[2] for _, pp, _, _ in overall) / len(overall)
        mean_add = sum(aa[2] for _, _, aa, _ in overall) / len(overall)
        mean_del = sum(dd[2] for _, _, _, dd in overall) / len(overall)
        print(f"  Final-model mean F1: pre={mean_pre:.2f} "
              f"add={mean_add:.2f} del={mean_del:.2f}")
    print("=" * 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_reused": n_reused,
            "n_total_actions": n_total,
            "reuse_report": reuse_report,
            "final_models": _models_to_lists(final_models),
            "ground_truth": _models_to_lists(gt),
            "comparison": [{
                "action": a, "pre_f1": pp[2],
                "add_f1": aa[2], "del_f1": dd[2],
                "reused": reuse_report.get(a, {}).get("reused", False),
            } for (a, pp, aa, dd) in overall],
        }, f, indent=2, default=str)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
