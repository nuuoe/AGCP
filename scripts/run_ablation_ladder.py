"""Ablation ladder for the induction pipeline on the IPC-2023 learning domains.

Runs the breadth evaluation under the five CONFIGS: full, and one feature
removed at a time (F1 negative-precondition stripping, F2 keeping
irrelevant operators, F3 positional templating, F4 precondition refinement).
Reports domains parsed, actions induced and F1>=0.9 counts; writes --out.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from scripts.induce_pddl_generic import (
    load_task, ground_truth_models, f1,
    random_rollout_pddl,
)
from scripts.induce_pddl_generic import induce_lifted_models as induce_first_occ
from scripts.induce_positional import (
    induce_lifted_models_positional as induce_positional,
)
from scripts.strip_negative_preconditions import strip_negative_pre
from scripts.refine_preconditions import refine_preconditions


NEEDS_STRIP = {"childsnack", "ferry", "satellite"}


def discover_domains(base_dir: str, strip_adl: bool,
                       stripped_dir: str):
    """Variant of run_ipc2023_breadth.discover_domains with a strip_adl switch (F1)."""
    import glob
    if strip_adl:
        os.makedirs(stripped_dir, exist_ok=True)
    out = []
    for d in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, d)
        if not os.path.isdir(sub): continue
        dom_path = os.path.join(sub, "domain.pddl")
        if not os.path.isfile(dom_path): continue
        if d in NEEDS_STRIP:
            if strip_adl:
                stripped_path = os.path.join(stripped_dir, f"{d}_domain.pddl")
                with open(stripped_path, "w") as f:
                    f.write(strip_negative_pre(open(dom_path).read()))
                dom_path = stripped_path
            else:
                # left unstripped; parsing will fail
                pass
        probs = []
        for relsub in ["training/easy", "testing/easy",
                        "testing/medium"]:
            cand = sorted(glob.glob(os.path.join(sub, relsub, "p*.pddl")))
            probs.extend(cand)
        if probs:
            out.append((d, dom_path, probs))
    return out


def collect_trs(dom_path, prob_paths, n_problems=8,
                  n_rollouts=5, steps_per_rollout=200,
                  keep_irrelevant=True):
    """Collect transitions. keep_irrelevant=False enables F2's
    failure mode (pyperplan strips irrelevant adds)."""
    paths = prob_paths[:n_problems]
    trs = []
    for i, p in enumerate(paths):
        try:
            _, task = load_task(dom_path, p,
                                  keep_irrelevant=keep_irrelevant)
        except Exception as e:
            return None, str(e)
        for s in range(n_rollouts):
            trs.extend(random_rollout_pddl(
                task, n_steps=steps_per_rollout, seed=i * 100 + s,
            ))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    return trs_ok, None


def evaluate_one_domain(name, dom_path, prob_paths,
                          induce_fn, apply_refinement: bool,
                          keep_irrelevant: bool,
                          n_problems=8):
    """Run one config on one domain. Return per-action F1 dict."""
    trs_ok, err = collect_trs(dom_path, prob_paths,
                                 n_problems=n_problems,
                                 keep_irrelevant=keep_irrelevant)
    if err:
        return {"error": err}, 0, 0
    models = induce_fn(trs_ok)

    by_action = {}
    for sb, action, args, sa in trs_ok:
        by_action.setdefault(action, []).append((sb, args))

    if apply_refinement:
        # Use first problem as task oracle
        try:
            _, task = load_task(dom_path, prob_paths[0],
                                  keep_irrelevant=keep_irrelevant)
        except Exception:
            task = None
        if task is not None:
            for action, model in list(models.items()):
                rargs = by_action.get(action, [])[:50]
                if not rargs: continue
                new_pre, _dropped = refine_preconditions(
                    task, action, set(model["pre_pos"]), rargs,
                )
                models[action]["pre_pos"] = new_pre

    try:
        _, t2 = load_task(dom_path, prob_paths[0],
                            keep_irrelevant=keep_irrelevant)
        from pyperplan.pddl.parser import Parser
        gt = ground_truth_models(Parser(dom_path, prob_paths[0]).parse_domain())
    except Exception as e:
        return {"error": f"GT: {e}"}, 0, 0

    per = {}
    perfect = 0
    n_actions = 0
    for action in gt:
        if action not in models:
            per[action] = {"missing": True}; continue
        n_actions += 1
        pp = f1(models[action]["pre_pos"], gt[action]["pre_pos"])[2]
        aa = f1(models[action]["eff_add"], gt[action]["eff_add"])[2]
        dd = f1(models[action]["eff_del"], gt[action]["eff_del"])[2]
        per[action] = {"pre_f1": pp, "add_f1": aa, "del_f1": dd}
        if pp >= 0.9 and aa >= 0.9 and dd >= 0.9:
            perfect += 1
    return per, perfect, n_actions


CONFIGS = {
    "full": dict(strip_adl=True, keep_irrelevant=True,
                  induce_positional=True, refine=True),
    "no_F4": dict(strip_adl=True, keep_irrelevant=True,
                   induce_positional=True, refine=False),
    "no_F3": dict(strip_adl=True, keep_irrelevant=True,
                   induce_positional=False, refine=True),
    "no_F2": dict(strip_adl=True, keep_irrelevant=False,
                   induce_positional=True, refine=True),
    "no_F1": dict(strip_adl=False, keep_irrelevant=True,
                   induce_positional=True, refine=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--n_problems", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 70)
    print("ABLATION LADDER — IPC-2023 learning")
    print("=" * 70)
    results = {}
    for config_name, cfg in CONFIGS.items():
        print(f"\n[CONFIG] {config_name}: {cfg}")
        strip_dir = tempfile.mkdtemp(prefix=f"strip_{config_name}_")
        domains = discover_domains(args.base_dir,
                                     strip_adl=cfg["strip_adl"],
                                     stripped_dir=strip_dir)
        if not cfg["strip_adl"]:
            domains = [(d, p, probs) for (d, p, probs) in domains
                        if d not in NEEDS_STRIP]
            print(f"  (no_F1: excluded ADL domains, "
                  f"{len(domains)} remaining)")
        induce_fn = (induce_positional if cfg["induce_positional"]
                      else induce_first_occ)

        domain_results = {}
        total_perfect = 0
        total_actions = 0
        n_domains_parsed = 0
        for name, dom_path, probs in domains:
            per, perfect, n_actions = evaluate_one_domain(
                name, dom_path, probs, induce_fn,
                apply_refinement=cfg["refine"],
                keep_irrelevant=cfg["keep_irrelevant"],
                n_problems=args.n_problems,
            )
            if isinstance(per, dict) and "error" in per:
                print(f"  {name}: ERROR {per['error'][:60]}")
                domain_results[name] = {"error": per["error"]}
                continue
            n_domains_parsed += 1
            total_perfect += perfect
            total_actions += n_actions
            domain_results[name] = {
                "n_actions": n_actions,
                "n_perfect": perfect,
                "per_action": per,
            }
            print(f"  {name}: {perfect}/{n_actions} perfect")
        results[config_name] = {
            "config": cfg,
            "n_domains_parsed": n_domains_parsed,
            "n_actions_total": total_actions,
            "n_perfect": total_perfect,
            "per_domain": domain_results,
        }
        print(f"  [{config_name}] TOTAL: "
              f"{total_perfect}/{total_actions} F1>=0.9, "
              f"{n_domains_parsed}/10 domains parsed")

    print("\n" + "=" * 70)
    print("ABLATION LADDER — SUMMARY")
    print("=" * 70)
    print(f"{'Config':10s}  {'Domains':10s}  {'F1>=0.9':15s}  {'Pct':6s}")
    for cname in CONFIGS:
        r = results[cname]
        pct = r["n_perfect"] / max(r["n_actions_total"], 1) * 100
        print(f"{cname:10s}  {r['n_domains_parsed']:>3d}/10      "
              f"{r['n_perfect']:>3d}/{r['n_actions_total']:<3d}          "
              f"{pct:5.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "configs": CONFIGS,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
