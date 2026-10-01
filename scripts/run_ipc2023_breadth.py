"""Induction and cross-domain schema reuse across the IPC-2023 learning-track domains.

For each domain: collect random-rollout transitions over N problems, try
structural reuse from SchemaMemory, induce the remaining actions, refine
preconditions, compare to the ground-truth PDDL (per-component F1), and
add the schemas to memory. Writes per-domain results and a summary to --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from agplan.adaptive_pipeline import SchemaMemory
from scripts.induce_pddl_generic import (
    load_task,
    ground_truth_models, f1, random_rollout_pddl,
)
from scripts.induce_positional import (
    induce_lifted_models_positional as induce_lifted_models,
)
from scripts.structural_reuse import try_structural_reuse
from scripts.refine_preconditions import refine_preconditions


def discover_domains(base_dir: str,
                       stripped_dir: str = None,
                       ) -> list[tuple[str, str, list[str]]]:
    """Return (domain_name, domain_path, problem_paths) for each subdirectory
    of base_dir holding a domain.pddl and at least one problem.

    childsnack, ferry and satellite use `(not P)` preconditions; STRIPS
    variants are written to stripped_dir via strip_negative_preconditions.
    All six sub-tracks (training/testing x easy/medium/hard) are pooled."""
    import tempfile
    from scripts.strip_negative_preconditions import strip_negative_pre
    if stripped_dir is None:
        stripped_dir = tempfile.mkdtemp(prefix="ipc2023_strips_")
    os.makedirs(stripped_dir, exist_ok=True)

    needs_strip = {"childsnack", "ferry", "satellite"}
    out = []
    if not os.path.isdir(base_dir):
        return out
    for d in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, d)
        if not os.path.isdir(sub):
            continue
        dom_path = os.path.join(sub, "domain.pddl")
        if not os.path.isfile(dom_path):
            continue
        if d in needs_strip:
            stripped_path = os.path.join(stripped_dir, f"{d}_domain.pddl")
            stripped_text = strip_negative_pre(open(dom_path).read())
            with open(stripped_path, "w") as f:
                f.write(stripped_text)
            dom_path = stripped_path
        probs = []
        for relsub in ["training/easy", "training/medium", "training/hard",
                        "testing/easy", "testing/medium", "testing/hard"]:
            cand = sorted(glob.glob(os.path.join(sub, relsub, "p*.pddl")))
            probs.extend(cand)
        if not probs:
            for relsub in ["training", "testing"]:
                cand = sorted(glob.glob(os.path.join(sub, relsub, "p*.pddl")))
                if cand: probs = cand; break
        if probs:
            out.append((d, dom_path, probs))
    return out


def run_domain(name: str, dom_path: str, prob_paths: list[str],
                memory: SchemaMemory,
                n_problems: int = 5,
                n_rollouts_per_problem: int = 5,
                steps_per_rollout: int = 200,
                accept_threshold: float = 0.9) -> dict:
    """Run induction on one domain. Returns per-action stats."""
    paths = prob_paths[:n_problems]
    trs = []
    for i, p in enumerate(paths):
        try:
            _, task = load_task(dom_path, p)
        except Exception as e:
            return {"error": f"parse: {e}", "n_actions": 0}
        for s in range(n_rollouts_per_problem):
            trs.extend(random_rollout_pddl(
                task, n_steps=steps_per_rollout, seed=i * 100 + s,
            ))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    by_action = {}
    for sb, act, args, sa in trs_ok:
        by_action.setdefault(act, []).append((sb, act, args, sa))

    composed = {}
    reuse = {}
    fresh_trs = []
    for action, items in by_action.items():
        matched, score = try_structural_reuse(memory, items,
                                                 accept_threshold=accept_threshold)
        if matched is not None:
            composed[action] = {
                "pre_pos": set(matched.get("pre_pos", [])),
                "eff_add": set(matched.get("eff_add", [])),
                "eff_del": set(matched.get("eff_del", [])),
            }
            reuse[action] = {"reused": True,
                              "source": matched.get("provenance_domain", "?"),
                              "score": score, "n": len(items)}
        else:
            reuse[action] = {"reused": False, "best_score": score,
                              "n": len(items)}
            fresh_trs.extend(items)

    if fresh_trs:
        fresh = induce_lifted_models(fresh_trs)
        for op, m in fresh.items():
            if op not in composed:
                composed[op] = {
                    "pre_pos": set(m["pre_pos"]),
                    "eff_add": set(m["eff_add"]),
                    "eff_del": set(m["eff_del"]),
                }

    # Precondition refinement: drop induced atoms that always co-occur in
    # rollouts but are not required, using pyperplan's grounded operators
    # as the oracle.
    tasks_cache = {}
    for action in composed:
        if reuse.get(action, {}).get("reused"):
            continue  # reused schemas already verified, skip
        rargs = [(sb, args)
                  for sb, _act, args, _sa in by_action.get(action, [])][:50]
        if not rargs: continue
        # grounded operators from the first problem
        p = prob_paths[0]
        if p not in tasks_cache:
            try:
                _, task = load_task(dom_path, p)
                tasks_cache[p] = task
            except Exception:
                continue
        task = tasks_cache[p]
        new_pre, _dropped = refine_preconditions(
            task, action, composed[action]["pre_pos"], rargs,
        )
        composed[action]["pre_pos"] = new_pre

    dom, _ = load_task(dom_path, paths[0])
    gt = ground_truth_models(dom)
    per_action = {}
    for action in gt:
        if action not in composed:
            per_action[action] = {"missing": True}
            continue
        ind = composed[action]
        gtm = gt[action]
        pp = f1(ind["pre_pos"], gtm["pre_pos"])
        aa = f1(ind["eff_add"], gtm["eff_add"])
        dd = f1(ind["eff_del"], gtm["eff_del"])
        per_action[action] = {
            "pre_f1": pp[2], "add_f1": aa[2], "del_f1": dd[2],
            "reused": reuse.get(action, {}).get("reused", False),
            "reuse_source": reuse.get(action, {}).get("source"),
            "reuse_score": reuse.get(action, {}).get("score", 0),
            "n_examples": len(by_action.get(action, [])),
        }

    # Add every schema under this domain's name, reused ones included,
    # so later lookups by name find them.
    for action, model in composed.items():
        if reuse.get(action, {}).get("reused"):
            memory.add(action, model, provenance=name)
        else:
            memory.add(action, model, provenance=name)

    return {
        "n_transitions": len(trs_ok),
        "n_actions_observed": len(by_action),
        "n_actions_gt": len(gt),
        "per_action": per_action,
        "reuse": reuse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="external/ipc2023-learning")
    ap.add_argument("--n_problems", type=int, default=5)
    ap.add_argument("--n_rollouts_per_problem", type=int, default=5)
    ap.add_argument("--steps_per_rollout", type=int, default=200)
    ap.add_argument("--accept_threshold", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    discovered = discover_domains(args.base_dir)
    print("=" * 60)
    print(f"IPC-2023 LEARNING BREADTH — {len(discovered)} domains found")
    print("=" * 60)

    memory = SchemaMemory()
    results = {}
    for name, dom_path, probs in discovered:
        print(f"\n[{name}] {len(probs)} problems available")
        try:
            res = run_domain(name, dom_path, probs, memory,
                              n_problems=args.n_problems,
                              n_rollouts_per_problem=args.n_rollouts_per_problem,
                              steps_per_rollout=args.steps_per_rollout,
                              accept_threshold=args.accept_threshold)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results[name] = {"error": str(e)}
            continue
        if "error" in res:
            print(f"  parse error: {res['error']}")
            results[name] = res
            continue

        results[name] = res
        n_perfect = sum(1 for s in res["per_action"].values()
                         if not s.get("missing") and
                         s.get("pre_f1", 0) >= 0.9 and
                         s.get("add_f1", 0) >= 0.9 and
                         s.get("del_f1", 0) >= 0.9)
        n_reused = sum(1 for s in res["per_action"].values()
                        if not s.get("missing") and s.get("reused"))
        print(f"  observed {res['n_actions_observed']}/{res['n_actions_gt']} actions, "
              f"{res['n_transitions']} transitions")
        print(f"  {n_perfect} actions at F1>=0.9, {n_reused} reused from memory")
        for action, s in res["per_action"].items():
            if s.get("missing"):
                print(f"    {action}: MISSING")
                continue
            tag = (f"REUSED({s['reuse_source']}, {s['reuse_score']*100:.0f}%)"
                   if s.get("reused") else "fresh")
            print(f"    {action} [{tag}]: pre={s['pre_f1']:.2f} "
                  f"add={s['add_f1']:.2f} del={s['del_f1']:.2f}")

    print("\n" + "=" * 60)
    print("BREADTH SUMMARY")
    print("=" * 60)
    total_actions = 0
    total_perfect = 0
    total_reused = 0
    total_missing = 0
    for name, res in results.items():
        if "error" in res or "per_action" not in res:
            continue
        for action, s in res["per_action"].items():
            if s.get("missing"):
                total_missing += 1
                continue
            total_actions += 1
            if (s.get("pre_f1", 0) >= 0.9 and
                s.get("add_f1", 0) >= 0.9 and
                s.get("del_f1", 0) >= 0.9):
                total_perfect += 1
            if s.get("reused"):
                total_reused += 1
    print(f"Domains parsed: {sum(1 for r in results.values() if 'error' not in r)}/{len(discovered)}")
    print(f"Actions induced: {total_actions} (+{total_missing} missing/never-observed)")
    print(f"Actions at F1>=0.9 on all (pre, add, del): {total_perfect}/{total_actions}"
          f" ({100*total_perfect/max(total_actions,1):.1f}%)")
    print(f"Cross-domain reuses: {total_reused}/{total_actions} "
          f"({100*total_reused/max(total_actions,1):.1f}%)")
    print(f"Final memory size: {len(memory.schemas)} entries")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "discovered_domains": [d[0] for d in discovered],
            "results": results,
            "summary": {
                "n_domains_parsed": sum(1 for r in results.values()
                                          if "error" not in r),
                "n_total_actions": total_actions,
                "n_perfect": total_perfect,
                "n_reused": total_reused,
                "n_missing": total_missing,
                "memory_size_final": len(memory.schemas),
            },
        }, f, indent=2, default=str)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
