"""BFS-versus-A* fallback statistics for the G_env compile step.

Re-runs the deterministic compile stage of agcp_plan_one (seeded rollouts,
induction, bounded-BFS compile, A* fallback) without LLM decoding and
records per instance the BFS cap hit, state counts, whether A* fired, and
wall times. Usage: python3 scripts/compile_provenance_stats.py --suite planbench|apb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from scripts.run_agcp_on_apb_simple import _with_timeout  # noqa: E402
from scripts.induce_pddl_generic import (  # noqa: E402
    load_task, random_rollout_pddl, state_to_predicates, parse_op_name,
)
from scripts.induce_positional import (  # noqa: E402
    induce_lifted_models_positional,
)
from agplan.grammars.generic_grammar import (  # noqa: E402
    compile_from_action_models, compile_from_action_models_astar,
)

# PlanBench T1 instance sets exactly as used in the paper
# (planbench_t1_final/*_score.json ids 2-11; agcp_depots_n30 ids 2-31).
PLANBENCH_SETS = {
    "blocksworld_3": range(2, 12),
    "mystery_blocksworld_3": range(2, 12),
    "logistics": range(2, 12),
    "depots": range(2, 32),
}


def _planbench_paths(pb_root: str, config: str):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(pb_root, "configs",
                                           config + ".yaml")))
    domain_file = os.path.join(pb_root, "instances", cfg["domain_file"])
    def inst(i):
        return os.path.join(pb_root, "instances", cfg["instance_dir"],
                            cfg["instances_template"].format(i))
    return domain_file, inst


def _conv_to_raw(atom: str) -> str:
    """'pred(a,b)' -> '(pred a b)'; 'handempty' -> '(handempty)'."""
    m = re.match(r"^([\w-]+)\((.*)\)$", atom)
    if m:
        args = [a for a in m.group(2).split(",") if a]
        return "(" + " ".join([m.group(1)] + args) + ")"
    return f"({atom})"


def _raw_to_conv(atom: str) -> str:
    s = atom.strip()
    if s.startswith("(") and s.endswith(")"):
        parts = s[1:-1].split()
        return (f"{parts[0]}({','.join(parts[1:])})"
                if len(parts) > 1 else parts[0])
    return s


class _FormatFixedOp:
    """Run pyperplan applicability/apply on converted-format states.

    Without it the ground_op_oracle check never fires: states are
    serialised as 'pred(a,b)' while pyperplan operators test '(pred a b)'."""

    def __init__(self, op):
        self._op = op

    def applicable(self, s_conv) -> bool:
        return self._op.applicable(frozenset(_conv_to_raw(a)
                                             for a in s_conv))

    def apply(self, s_conv):
        raw = self._op.apply(frozenset(_conv_to_raw(a) for a in s_conv))
        return frozenset(_raw_to_conv(a) for a in raw)


def compile_one(domain_file: str, instance_file: str,
                n_rollouts: int = 10, steps_per_rollout: int = 50,
                compile_timeout: int = 60, fix_format: bool = False) -> dict:
    """Mirror agcp_plan_one up to compile; return provenance row."""
    row = {"domain_file": domain_file, "instance_file": instance_file}
    try:
        _, task = load_task(domain_file, instance_file)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"load_task: {e}"
        return row
    trs = []
    for s in range(n_rollouts):
        trs.extend(random_rollout_pddl(task, n_steps=steps_per_rollout,
                                       seed=s))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    if not trs_ok:
        row["error"] = "no successful rollouts"
        return row
    models = induce_lifted_models_positional(trs_ok)
    init_state = state_to_predicates(task.initial_state)
    goal_atoms = set()
    for atom in task.goals:
        s = str(atom).strip()
        if s.startswith("(") and s.endswith(")"):
            parts = s[1:-1].split()
            goal_atoms.add(f"{parts[0]}({','.join(parts[1:])})"
                           if len(parts) > 1 else parts[0])
        else:
            goal_atoms.add(s)
    objects = set()
    for atom in init_state | goal_atoms:
        m = re.match(r"^[\w-]+\((.*)\)$", atom)
        if m:
            for a in m.group(1).split(","):
                if a.strip():
                    objects.add(a.strip())
    objects = sorted(objects)
    ground_op_filter, ground_op_oracle = {}, {}
    for op in task.operators:
        action_name, args = parse_op_name(op.name)
        ground_op_filter.setdefault(action_name, []).append(args)
        ground_op_oracle[(action_name, args)] = (
            _FormatFixedOp(op) if fix_format else op)

    stats: dict = {}
    t0 = time.time()
    cfg_obj, terr = _with_timeout(
        compile_timeout, compile_from_action_models,
        models, init_state, goal_atoms,
        objects=objects, max_extra=4,
        ground_op_filter=ground_op_filter,
        ground_op_oracle=ground_op_oracle,
        stats=stats,
    )
    row["bfs_seconds"] = round(time.time() - t0, 2)
    row["bfs_stats"] = stats
    row["bfs_grammar"] = cfg_obj is not None and not terr
    row["bfs_error"] = terr or (None if cfg_obj is not None
                                else stats.get("fail_reason", "none"))
    if not row["bfs_grammar"]:
        t1 = time.time()
        astar_obj, aerr = _with_timeout(
            compile_timeout, compile_from_action_models_astar,
            task, init_state, goal_atoms,
        )
        row["astar_seconds"] = round(time.time() - t1, 2)
        row["astar_grammar"] = astar_obj is not None and not aerr
        row["astar_error"] = aerr or None
        if astar_obj:
            row["astar_plan_len"] = str(astar_obj).count('"\\"') // 2
    return row


def run_planbench(args) -> list[dict]:
    rows = []
    for config, ids in PLANBENCH_SETS.items():
        dom, inst = _planbench_paths(args.pb_root, config)
        for i in ids:
            p = inst(i)
            if not os.path.exists(p):
                rows.append({"suite": "planbench", "domain": config,
                             "instance_id": i, "error": "missing file"})
                continue
            r = compile_one(dom, p, compile_timeout=args.timeout,
                            fix_format=args.fix_format)
            r.update({"suite": "planbench", "domain": config,
                      "instance_id": i})
            rows.append(r)
            print(f"  {config} inst {i}: bfs={r.get('bfs_grammar')} "
                  f"cap={r.get('bfs_stats', {}).get('cap_hit')} "
                  f"reach={r.get('bfs_stats', {}).get('n_reachable')} "
                  f"astar={r.get('astar_grammar', '-')}")
    return rows


def run_apb(args) -> list[dict]:
    rows = []
    domains = sorted(d for d in os.listdir(args.apb_root)
                     if os.path.isfile(os.path.join(args.apb_root, d,
                                                    "domain.pddl")))
    for d in domains:
        dom_path = os.path.join(args.apb_root, d, "domain.pddl")
        prob_dir = os.path.join(args.apb_root, d, "orig_problems")
        if not os.path.isdir(prob_dir):
            prob_dir = os.path.join(args.apb_root, d, "adapted_instances")
        probs = sorted(Path(prob_dir).glob("instance-*.pddl"))
        probs = probs[: args.max_instances]
        for prob in probs:
            inst_id = int(re.findall(r"instance-(\d+)", prob.name)[0])
            r = compile_one(dom_path, str(prob), compile_timeout=args.timeout,
                            fix_format=args.fix_format)
            r.update({"suite": "apb", "domain": d, "instance_id": inst_id})
            rows.append(r)
            print(f"  {d} inst {inst_id}: bfs={r.get('bfs_grammar')} "
                  f"cap={r.get('bfs_stats', {}).get('cap_hit')} "
                  f"reach={r.get('bfs_stats', {}).get('n_reachable')} "
                  f"astar={r.get('astar_grammar', '-')}")
    return rows


def aggregate(rows: list[dict]) -> dict:
    agg: dict = {}
    for r in rows:
        d = r["domain"]
        a = agg.setdefault(d, {"n": 0, "bfs_ok": 0, "cap_hit": 0,
                               "astar_needed": 0, "astar_ok": 0,
                               "load_or_rollout_error": 0})
        a["n"] += 1
        if r.get("error"):
            a["load_or_rollout_error"] += 1
            continue
        if r.get("bfs_grammar"):
            a["bfs_ok"] += 1
        else:
            a["astar_needed"] += 1
            if r.get("astar_grammar"):
                a["astar_ok"] += 1
        if r.get("bfs_stats", {}).get("cap_hit"):
            a["cap_hit"] += 1
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["planbench", "apb"],
                    required=True)
    ap.add_argument("--pb_root",
                    default="external/LLMs-Planning/plan-bench")
    ap.add_argument("--apb_root",
                    default="external/autoplanbench/"
                            "autoplanbench_dataset/apb2.0_dataset")
    ap.add_argument("--max_instances", type=int, default=3,
                    help="APB instances per domain (paper used 3)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--fix_format", action="store_true",
                    help="wrap the pyperplan oracle in a state-format "
                         "adapter so BFS applicability checks fire "
                         "(measures true BFS coverage)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = run_planbench(args) if args.suite == "planbench" else run_apb(args)
    agg = aggregate(rows)
    suffix = "_fixed" if args.fix_format else ""
    out_path = (args.out or
                f"runs/compile_provenance_stats_{args.suite}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"suite": args.suite, "timeout": args.timeout,
                   "fix_format": args.fix_format,
                   "aggregate": agg, "rows": rows}, f, indent=2,
                  default=str)
    print("\nAGGREGATE:")
    for d, a in agg.items():
        print(f"  {d:32s} n={a['n']:3d} bfs_ok={a['bfs_ok']:3d} "
              f"cap_hit={a['cap_hit']:3d} astar_needed={a['astar_needed']:3d} "
              f"astar_ok={a['astar_ok']:3d}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
