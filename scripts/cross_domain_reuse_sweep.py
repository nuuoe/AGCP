"""Sweep all (source, target) domain pairs for classical schema reuse.

Sources and targets are the IPC-2023 learning domains plus PlanBench Depots.
A reuse pair is a source schema whose predicate names fit the target
vocabulary and that predicts >= --threshold of held-out target transitions.
No LLM is involved. Output: runs/cross_domain_reuse_sweep.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile
from pathlib import Path


def _strip_adl_if_needed(domain: str, dom_path: str) -> str:
    if domain in {"childsnack", "ferry", "satellite", "sokoban", "spanner"}:
        from scripts.strip_negative_preconditions import strip_negative_pre
        tmpdir = tempfile.mkdtemp()
        stripped = os.path.join(tmpdir, f"{domain}_domain.pddl")
        with open(stripped, "w") as f:
            f.write(strip_negative_pre(open(dom_path).read()))
        return stripped
    return dom_path


def induce_ipc2023_domain(base_dir: str, domain: str,
                            n_problems: int = 5,
                            steps_per_problem: int = 100) -> dict:
    from scripts.induce_pddl_generic import (
        load_task, random_rollout_pddl,
    )
    from scripts.induce_positional import (
        induce_lifted_models_positional,
    )
    dom_dir = os.path.join(base_dir, domain)
    dom_path = os.path.join(dom_dir, "domain.pddl")
    if not os.path.isfile(dom_path):
        return {}
    dom_path = _strip_adl_if_needed(domain, dom_path)

    probs = []
    for sub in ["training/easy", "testing/easy"]:
        probs.extend(sorted(glob.glob(
            os.path.join(dom_dir, sub, "p*.pddl"))))
    if not probs:
        return {}
    probs = probs[:n_problems]

    trs = []
    for i, p in enumerate(probs):
        try:
            _, task = load_task(dom_path, p)
        except Exception:
            continue
        for s in range(3):
            trs.extend(random_rollout_pddl(
                task, n_steps=steps_per_problem, seed=i*100+s))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    return induce_lifted_models_positional(trs_ok)


def induce_planbench_depots(pb_root: str,
                              n_problems: int = 5,
                              steps_per_problem: int = 100) -> dict:
    from scripts.induce_pddl_generic import (
        load_task, random_rollout_pddl,
    )
    from scripts.induce_positional import (
        induce_lifted_models_positional,
    )
    dom_path = os.path.join(pb_root, "instances/depots/generated_domain.pddl")
    if not os.path.isfile(dom_path):
        return {}
    probs = sorted(glob.glob(
        os.path.join(pb_root, "instances/depots/generated_basic/instance-*.pddl")))
    if not probs:
        return {}
    probs = probs[:n_problems]
    trs = []
    for i, p in enumerate(probs):
        try:
            _, task = load_task(dom_path, p)
        except Exception:
            continue
        for s in range(3):
            trs.extend(random_rollout_pddl(
                task, n_steps=steps_per_problem, seed=i*100+s))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    return induce_lifted_models_positional(trs_ok)


def gather_target_rollouts(base_dir: str, domain: str,
                              n_problems: int = 5,
                              steps_per_problem: int = 100) -> dict:
    """Return {action_name: [(sb, args, sa)]}. Suite-agnostic."""
    from scripts.induce_pddl_generic import (
        load_task, random_rollout_pddl,
    )
    is_pb = base_dir.endswith("plan-bench")
    if is_pb and domain == "depots":
        dom_path = os.path.join(base_dir,
                                  "instances/depots/generated_domain.pddl")
        probs = sorted(glob.glob(os.path.join(
            base_dir, "instances/depots/generated_basic/instance-*.pddl")))
    else:
        dom_dir = os.path.join(base_dir, domain)
        dom_path = os.path.join(dom_dir, "domain.pddl")
        if not os.path.isfile(dom_path):
            return {"trs_by_action": {}, "vocab": set()}
        dom_path = _strip_adl_if_needed(domain, dom_path)
        probs = []
        for sub in ["training/easy", "testing/easy"]:
            probs.extend(sorted(glob.glob(
                os.path.join(dom_dir, sub, "p*.pddl"))))
    probs = probs[:n_problems]
    trs_by_action = {}
    vocab = set()
    for i, p in enumerate(probs):
        try:
            _, task = load_task(dom_path, p)
        except Exception:
            continue
        for s in range(3):
            for sb, action, ag, sa in random_rollout_pddl(
                    task, n_steps=steps_per_problem, seed=i*100+s):
                if sb == sa:
                    continue
                trs_by_action.setdefault(action, []).append((sb, ag, sa))
                for atom in sb | sa:
                    nm = atom.split("(")[0]
                    vocab.add(nm)
    return {"trs_by_action": trs_by_action, "vocab": vocab}


def signature_subset(source_schema: dict, target_vocab: set) -> bool:
    def names(model):
        out = set()
        for k in ("pre_pos", "eff_add", "eff_del"):
            for p in model.get(k, set()):
                m = re.match(r"^([\w-]+)", p)
                if m:
                    out.add(m.group(1))
        return out
    return names(source_schema) <= target_vocab


def verify_schema_on_target(schema: dict, target_trs: list) -> float:
    from scripts.grounding import ground_lifted
    if not target_trs:
        return 0.0
    add_t = set(schema.get("eff_add", []))
    del_t = set(schema.get("eff_del", []))
    ok = 0
    for sb, args, sa in target_trs:
        add_g = ground_lifted(add_t, args)
        del_g = ground_lifted(del_t, args)
        predicted = (sb - del_g) | add_g
        if predicted == sa:
            ok += 1
    return ok / max(len(target_trs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc_root", default="external/ipc2023-learning")
    ap.add_argument("--pb_root", default="external/LLMs-Planning/plan-bench")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--n_held_out", type=int, default=30)
    ap.add_argument("--out", default="runs/cross_domain_reuse_sweep.json")
    args = ap.parse_args()

    ipc_domains = ["blocksworld", "childsnack", "ferry", "floortile",
                    "miconic", "rovers", "satellite", "sokoban",
                    "spanner", "transport"]
    pb_domains = ["depots"]

    print(f"Inducing source schemas — IPC2023 ({len(ipc_domains)}) + "
          f"plan-bench ({len(pb_domains)}) ...")
    source_models = {}
    for d in ipc_domains:
        m = induce_ipc2023_domain(args.ipc_root, d)
        if m:
            source_models[("ipc2023", d)] = m
            print(f"  [ipc2023] {d:>14}: {len(m)} schemas")
    for d in pb_domains:
        m = induce_planbench_depots(args.pb_root) if d == "depots" else {}
        if m:
            source_models[("planbench", d)] = m
            print(f"  [planbench] {d:>14}: {len(m)} schemas")

    print(f"\nGathering target rollouts ...")
    target_data = {}
    for d in ipc_domains:
        td = gather_target_rollouts(args.ipc_root, d)
        if td["trs_by_action"]:
            target_data[("ipc2023", d)] = td
    for d in pb_domains:
        td = gather_target_rollouts(args.pb_root, d)
        if td["trs_by_action"]:
            target_data[("planbench", d)] = td

    print(f"\nTesting (source,target) pairs ...")
    reuse_pairs = []
    n_pairs_tested = 0
    n_sig_match = 0
    for (s_suite, s_dom), s_models in source_models.items():
        for (t_suite, t_dom), td in target_data.items():
            if (s_suite, s_dom) == (t_suite, t_dom):
                continue
            for s_act, s_schema in s_models.items():
                if not signature_subset(s_schema, td["vocab"]):
                    continue
                n_sig_match += 1
                for t_act, t_trs in td["trs_by_action"].items():
                    n_pairs_tested += 1
                    if len(t_trs) < 5:
                        continue
                    held = t_trs[:args.n_held_out]
                    f1 = verify_schema_on_target(s_schema, held)
                    if f1 >= args.threshold:
                        cross_suite = (s_suite != t_suite)
                        reuse_pairs.append({
                            "source_suite": s_suite,
                            "source_domain": s_dom,
                            "source_action": s_act,
                            "target_suite": t_suite,
                            "target_domain": t_dom,
                            "target_action": t_act,
                            "f1": f1,
                            "cross_suite": cross_suite,
                        })

    cs_pairs = [r for r in reuse_pairs if r["cross_suite"]]
    print()
    print("=" * 70)
    print(f"Reuse pairs (F1>={args.threshold}):  {len(reuse_pairs)} total")
    print(f"  intra-suite (IPC2023 only):       "
          f"{len(reuse_pairs) - len(cs_pairs)}")
    print(f"  CROSS-SUITE (IPC2023<->planbench): {len(cs_pairs)}")
    print(f"  sig-match candidates checked:     {n_sig_match}")
    print(f"  (source,target) action-pairs tested: {n_pairs_tested}")
    print("=" * 70)
    print("\nReuse pairs:")
    for r in reuse_pairs:
        cs = " [CROSS-SUITE]" if r["cross_suite"] else ""
        print(f"  {r['source_suite']}.{r['source_domain']}."
              f"{r['source_action']} -> "
              f"{r['target_suite']}.{r['target_domain']}."
              f"{r['target_action']}  "
              f"F1={r['f1']:.2f}{cs}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "threshold": args.threshold,
            "ipc_domains": ipc_domains,
            "pb_domains": pb_domains,
            "n_sources_with_schemas": len(source_models),
            "n_targets_with_rollouts": len(target_data),
            "n_sig_match": n_sig_match,
            "n_pairs_tested": n_pairs_tested,
            "n_reuse_total": len(reuse_pairs),
            "n_cross_suite": len(cs_pairs),
            "reuse_pairs": reuse_pairs,
        }, f, indent=2)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
