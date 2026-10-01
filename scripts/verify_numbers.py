"""Check the headline numbers reported in the paper against the run outputs under runs/.

Covers the mask-isolation ablation, the paraphrase control, the E6 planner
comparison, fallback provenance, SCI-ReDuce end-to-end, the RQ3 chain and
its GPT-4o replication, and the matched Table 1 cells. Needs no GPU or API.
Run from the repository root: python scripts/verify_numbers.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

FAILURES: list[str] = []
CHECKED = 0


def load(rel: str):
    p = Path(rel)
    if not p.exists():
        FAILURES.append(f"missing file: {rel}")
        return None
    return json.loads(p.read_text())


def load_jsonl(rel: str):
    p = Path(rel)
    if not p.exists():
        FAILURES.append(f"missing file: {rel}")
        return None
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def check(label: str, expected, actual) -> None:
    global CHECKED
    CHECKED += 1
    ok = expected == actual
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label:<52} expected={expected!s:<12} actual={actual!s}")
    if not ok:
        FAILURES.append(f"{label}: expected {expected}, got {actual}")


def mcnemar_two_sided(a: int, n: int) -> float:
    """Exact two-sided binomial test on the discordant pairs.

    Computed in exact rationals: a float epsilon large enough to absorb
    rounding also swallows genuine tail terms once n is big enough, which
    inflates the p-value by many orders of magnitude.
    """
    if n == 0:
        return 1.0
    pmf = [Fraction(comb(n, k), 2 ** n) for k in range(n + 1)]
    thr = pmf[a]
    return float(sum(x for x in pmf if x <= thr))


def paired(a_map: dict, b_map: dict) -> tuple[int, int, int, float]:
    ids = sorted(set(a_map) & set(b_map))
    a_only = sum(1 for i in ids if a_map[i] and not b_map[i])
    b_only = sum(1 for i in ids if b_map[i] and not a_map[i])
    return len(ids), a_only, b_only, mcnemar_two_sided(a_only, a_only + b_only)


# Mask isolation
def check_mask_isolation() -> None:
    print("\n[1] Mask-isolation ablation (Sec. 5.1)")
    for tag, domain, expected_success in [
        ("syntax_bw", "Blocksworld syntax-only", 0),
        ("syntax_mystery", "Mystery syntax-only", 0),
        ("genv_bw", "Blocksworld G_env", 80),
        ("genv_mystery", "Mystery G_env", 80),
    ]:
        rows = load_jsonl(f"runs/mask_isolation_{tag}.jsonl")
        if rows is None:
            continue
        plans = [c for r in rows for c in r["candidates"]]
        check(f"{domain}: goal-reaching plans", f"{expected_success}/80",
              f"{sum(1 for c in plans if c.get('success'))}/{len(plans)}")
        if tag.startswith("syntax"):
            check(f"{domain}: all parse as valid", len(plans),
                  sum(1 for c in plans if c.get("parse_ok")))
        else:
            check(f"{domain}: first sample valid on every instance", len(rows),
                  sum(1 for r in rows if r.get("first_success_idx") == 0))


# Paraphrase control
def check_decontamination() -> None:
    print("\n[2] Post-cutoff paraphrase control (Sec. 5.2)")
    for cond, system, expected in [
        ("struct", "regex", "2/251"),
        ("struct", "gpt4o", "158/251"),
        ("struct", "claude_sonnet", "172/251"),
        ("struct", "together_qwen7b", "128/251"),
        ("lex", "regex", "43/251"),
        ("lex", "claude_sonnet", "145/251"),
    ]:
        d = load(f"runs/alfworld_paraphrase_decon_{cond}_{system}.json")
        if d is None:
            continue
        check(f"{cond}/{system} accepts", expected,
              f"{d['n_accept']}/{d['n_total']}")

    # paired discordant counts
    def _acc(path, field, thr=0.9):
        d = load(path)
        return None if d is None else [r[field] >= thr for r in d["rows"]]

    regex_o = _acc("runs/ood/alfworld_regex_v2_unseen.json", "regex_v2_f1")
    regex_p = _acc("runs/alfworld_paraphrase_decon_struct_regex.json", "smart_f1")
    son_p = _acc("runs/alfworld_paraphrase_decon_struct_claude_sonnet.json", "smart_f1")
    if all(x is not None for x in (regex_o, regex_p, son_p)):
        # rows are positionally aligned across files; task_id repeats per template
        lost = sum(1 for a, b in zip(regex_o, regex_p) if a and not b)
        gained = sum(1 for a, b in zip(regex_o, regex_p) if b and not a)
        check("regex collapse under paraphrase", "+78/-0", f"+{lost}/-{gained}")
        won = sum(1 for a, b in zip(son_p, regex_p) if a and not b)
        lost2 = sum(1 for a, b in zip(son_p, regex_p) if b and not a)
        check("Sonnet over regex on paraphrases", "+171/-1", f"+{won}/-{lost2}")

    # the originals the paraphrases are compared against
    for name, path, key, expected in [
        ("original OOD regex", "runs/ood/alfworld_regex_v2_unseen.json", "n_accept", "80/251"),
        ("original OOD gpt4o", "runs/ood/alfworld_cloud_gpt4o.json", "n_accept", "156/251"),
        ("original OOD sonnet", "runs/ood/alfworld_cloud_claude_sonnet.json", "n_accept", "169/251"),
    ]:
        d = load(path)
        if d is None:
            continue
        check(name, expected, f"{d[key]}/{d['n_total']}")


# E6 planners
def _exec_map(path: str) -> dict | None:
    d = load(path)
    if d is None:
        return None
    return {r["task_id"]: bool(r.get("execution_success")) for r in d["results"]}


def check_e6_planners() -> None:
    print("\n[3] Three-planner E6 comparison (Sec. 5.4)")
    induced = _exec_map("runs/alfworld_e6_induced_planner.json")
    llm = _exec_map("runs/alfworld_closed_loop_e6_llm_planner.json")
    llm80 = load("runs/alfworld_closed_loop_e6_llm_planner_cap80.json")
    hand = _exec_map("runs/alfworld_closed_loop_e6_full_unfiltered.json")
    if not all(x is not None for x in (induced, llm, llm80, hand)):
        return

    check("induced planner successes", "41/114",
          f"{sum(induced.values())}/{len(induced)}")

    # The paper quotes the LLM-as-planner twice, on two denominators:
    # 43/119 over the raw stream (Sec. 5.4 text) and 42/114 on the set paired
    # against the induced planner (Table 3b). Both are checked.
    llm_raw = load("runs/alfworld_closed_loop_e6_llm_planner.json")
    check("LLM-as-planner, raw stream (30-step cap)", "43/119",
          f"{llm_raw['n_execution_success']}/{llm_raw['n_total']}")
    check("LLM-as-planner (80-step cap) identical", 43,
          llm80["n_execution_success"])
    check("LLM-as-planner on the paired set", "42/114",
          f"{sum(llm.values())}/{len(llm)}")

    n, ind_only, hand_only, p = paired(induced, hand)
    check("induced vs HandCoded: paired N", 114, n)
    check("induced vs HandCoded: discordant", "+0/-36", f"+{ind_only}/-{hand_only}")
    check("induced vs HandCoded: p < 1e-10", True, p < 1e-10)
    check("HandCoded on the paired set", "77/114",
          f"{sum(hand[i] for i in set(induced) & set(hand))}/{n}")

    n, ind_only, llm_only, p = paired(induced, llm)
    check("induced vs LLM-planner: discordant", "+16/-17", f"+{ind_only}/-{llm_only}")
    check("induced vs LLM-planner: p = 1.0", 1.0, round(p, 6))


# Fallback provenance
def check_fallback_provenance() -> None:
    print("\n[4] Fallback provenance (Sec. 3.3)")
    d = load("runs/compile_provenance_planbench_fixed.json")
    if d is not None:
        agg = d["aggregate"]
        pooled_bfs = sum(agg[k]["bfs_ok"] for k in
                         ("blocksworld_3", "mystery_blocksworld_3", "logistics"))
        pooled_n = sum(agg[k]["n"] for k in
                       ("blocksworld_3", "mystery_blocksworld_3", "logistics"))
        check("BW+Mystery+Logistics compile within budget", "30/30",
              f"{pooled_bfs}/{pooled_n}")
        check("Depots cap-hit on every instance", "30/30",
              f"{agg['depots']['cap_hit']}/{agg['depots']['n']}")
        check("Depots solved under A* fallback", "30/30",
              f"{agg['depots']['astar_ok']}/{agg['depots']['n']}")

    d = load("runs/compile_provenance_apb_fixed.json")
    if d is not None:
        agg = d["aggregate"]
        tot = sum(v["n"] for v in agg.values())
        bfs = sum(v["bfs_ok"] for v in agg.values())
        cap = sum(v["cap_hit"] for v in agg.values())
        ok = sum(v["astar_ok"] for v in agg.values())
        need = sum(v["astar_needed"] for v in agg.values())
        check("AutoPlanBench BFS compiles", f"70/{tot}", f"{bfs}/{tot}")
        check("AutoPlanBench cap-hit", f"120/{tot}", f"{cap}/{tot}")
        check("AutoPlanBench A* recovers", f"93/{need}", f"{ok}/{need}")


# SCI-ReDuce end-to-end
def check_scireduce_e2e() -> None:
    print("\n[5] SCI-ReDuce end-to-end (Sec. 5.4)")
    raw = load("runs/scireduce_e2e_raw.json")
    if raw is not None:
        seeds = raw["per_seed"]
        check("raw grammar: seeds run", 2, len(seeds))
        check("raw grammar: end-to-end successes", 0,
              sum(s["e2e_success"] for s in seeds))
        check("raw grammar: instances", 20, sum(s["n"] for s in seeds))

    bud = load("runs/scireduce_e2e_budget.json")
    if bud is not None:
        seeds = bud["per_seed"]
        n = sum(s["n"] for s in seeds)
        check("budget-indexed: end-to-end", "15/40",
              f"{sum(s['e2e_success'] for s in seeds)}/{n}")
        check("budget-indexed: gold-plan coverage", "13/40",
              f"{sum(s['coverage'] for s in seeds)}/{n}")
        compiled = sum(s["compiled"] for s in seeds)
        check("compile failures (the whole loss)", "25/40", f"{n - compiled}/{n}")


# RQ3 chain and its GPT-4o replication
def check_rq3() -> None:
    print("\n[6] RQ3 uniform paired chain (Sec. 5.3)")
    d = load("runs/rq3_parse_level.json")
    if d is not None:
        runs, pairs = d["runs"], d["pairs"]
        check("paired set is unique-id", 114, d["n_paired_unique"])
        check("one-shot (execution)", 77, runs["K1_one_shot"]["exec"])
        check("Layer A (execution)", 86, runs["LayerA_K3"]["exec"])
        check("Layer A+B (execution)", 92, runs["LayerB_refine"]["exec"])
        check("composed lift = 13.2pp", 13.2,
              round(100 * (runs["LayerB_refine"]["exec"]
                           - runs["K1_one_shot"]["exec"]) / 114, 1))
        check("Layer A dominance 9-0",
              (9, 0), (pairs["K1_one_shot->LayerA_K3:exec"]["wins_b_only"],
                       pairs["K1_one_shot->LayerA_K3:exec"]["wins_a_only"]))
        check("Layer B dominance 6-0",
              (6, 0), (pairs["LayerA_K3->LayerB_refine:exec"]["wins_b_only"],
                       pairs["LayerA_K3->LayerB_refine:exec"]["wins_a_only"]))
        check("Layer A p = 0.0039", 0.0039,
              round(pairs["K1_one_shot->LayerA_K3:exec"]["mcnemar_p"], 4))
        check("Layer B p = 0.031", 0.031,
              round(pairs["LayerA_K3->LayerB_refine:exec"]["mcnemar_p"], 3))
        # parse-level metric
        check("parse level 61.4%", 61.4,
              round(100 * runs["K1_one_shot"]["parse_acc_0.9"] / 114, 1))
        check("parse level 69.3%", 69.3,
              round(100 * runs["LayerA_K3"]["parse_acc_0.9"] / 114, 1))
        check("parse level 75.4%", 75.4,
              round(100 * runs["LayerB_refine"]["parse_acc_0.9"] / 114, 1))
        check("parse-level Layer B +8/-1",
              (8, 1), (pairs["LayerA_K3->LayerB_refine:acc0.9"]["wins_b_only"],
                       pairs["LayerA_K3->LayerB_refine:acc0.9"]["wins_a_only"]))

    print("\n[7] Cross-model RQ3 replication, GPT-4o (Sec. 5.3)")
    for label, path, expected in [
        ("one-shot 67.2%", "runs/alfworld_iterative_e4_gpt4o_K1.json", 80),
        ("Layer A 71.4%", "runs/alfworld_iterative_e4_gpt4o.json", 85),
        ("Layer A+B 75.6%", "runs/alfworld_iterative_e4_gpt4o_refine.json", 90),
    ]:
        d = load(path)
        if d is None:
            continue
        check(f"GPT-4o {label}", expected, d["n_execution_success"])
        check(f"GPT-4o {label} denominator", 119, d["n_total"])


# Table 1 matched denominators
def check_table1_matched() -> None:
    print("\n[8] Table 1 matched denominators (Sec. 5.1)")
    d = load("runs/llm_removal_depots_n30.json")
    if d is not None:
        check("No-LLM Depots at N=30", "30/30",
              f"{d.get('n_solved')}/{d.get('n_total')}")
    pooled_k = pooled_n = 0
    for dom, f in [("blocksworld", "llm_removal_blocksworld_3.json"),
                   ("mystery", "llm_removal_mystery_blocksworld_3.json"),
                   ("logistics", "llm_removal_logistics.json")]:
        d = load(f"runs/{f}")
        if d is None:
            continue
        pooled_k += d["n_solved"]
        pooled_n += d["n_total"]
    check("No-LLM pooled over other three domains", "30/30",
          f"{pooled_k}/{pooled_n}")


# Syntax-only failure modes
def check_syntax_failure_modes() -> None:
    """Sec. 5.1's breakdown of why the syntax-only samples fail."""
    import collections
    pooled: collections.Counter = collections.Counter()
    for tag in ("syntax_bw", "syntax_mystery"):
        rows = load_jsonl(f"runs/mask_isolation_{tag}.jsonl")
        if rows is None:
            return
        per: collections.Counter = collections.Counter()
        for r in rows:
            for c in r["candidates"]:
                per[c.get("error") or "<horizon>"] += 1
        check(f"{tag}: 'not on table'", 39, per["not on table"])
        check(f"{tag}: 'not clear'", 14, per["not clear"])
        check(f"{tag}: gripper already full", 12, per["gripper not empty"])
        pooled.update(per)
    total = sum(pooled.values())
    horizon = pooled["<horizon>"] + pooled["plan exhausted without goal"]
    check("all samples accounted for", 160, total)
    check("precondition violations", 158, total - horizon)
    check("ran to the horizon", 2, horizon)


# Adversarial mask (Ethics section)
def check_adversarial() -> None:
    rows = load_jsonl(
        "runs/adversarial_q7/adversarial.jsonl"
    )
    if rows is None:
        return
    agg: dict[str, list[int]] = {}
    for r in rows:
        a = agg.setdefault(r["mask"], [0, 0])
        a[0] += r["n_sound"]
        a[1] += r["k"]
    check("coarse syntax: sound plans under jailbreak", "0/100",
          f"{agg.get('coarse', [0, 0])[0]}/{agg.get('coarse', [0, 0])[1]}")
    check("G_env: sound plans under jailbreak", "100/100",
          f"{agg.get('genv', [0, 0])[0]}/{agg.get('genv', [0, 0])[1]}")
    check("distinct jailbreak prompts", 5,
          len({r["adversarial"] for r in rows}))


# AutoPlanBench breadth
def check_apb_breadth() -> None:
    import glob
    solved = total = 0
    domains = set()
    for f in sorted(glob.glob("runs/final_apb_*.json")):
        d = json.loads(Path(f).read_text())
        for dom, v in (d.get("summary") or {}).items():
            domains.add(dom)
            solved += v["n_solved"]
            total += v["n_total"]
    check("APB breadth scan: domains", 32, len(domains))
    check("APB breadth scan: solved", "63/96", f"{solved}/{total}")


# SCI-ReDuce sample counts
def check_scireduce_samples() -> None:
    raw = load("runs/scireduce_e2e_raw.json")
    bud = load("runs/scireduce_e2e_budget.json")
    if raw is None or bud is None:
        return
    r_inst = sum(s["n"] for s in raw["per_seed"])
    r_succ = sum(s["e2e_success"] for s in raw["per_seed"])
    r_comp = [x for x in raw["rows"] if x.get("compiled")]
    check("raw grammar: held-out instances solved", "0/20", f"{r_succ}/{r_inst}")
    check("raw grammar: goal-reaching samples", "0/28",
          f"{sum(x.get('n_goal_reaching', 0) for x in r_comp)}"
          f"/{len(r_comp) * raw['k']}")
    b_comp = [x for x in bud["rows"] if x.get("compiled")]
    check("budget grammar: non-empty compiles solved", "15/15",
          f"{sum(1 for x in b_comp if x.get('e2e_success'))}/{len(b_comp)}")
    check("budget grammar: goal-reaching samples", "118/120",
          f"{sum(x.get('n_goal_reaching_samples', 0) for x in b_comp)}"
          f"/{len(b_comp) * bud['k']}")


# Regex NL-parse baseline
def check_regex_baseline() -> None:
    import glob
    acc = tot = 0
    for f in sorted(glob.glob("runs/regex_baseline_*.json")):
        d = json.loads(Path(f).read_text())
        check(f"regex baseline {Path(f).stem.replace('regex_baseline_', '')}",
              f"{d['n_total']}/{d['n_total']}",
              f"{d['n_accept_at_0.9']}/{d['n_total']}")
        acc += d["n_accept_at_0.9"]
        tot += d["n_total"]
    check("regex baseline pooled on templated NL", "120/120", f"{acc}/{tot}")


def main() -> None:
    print("Reported-number check")
    check_mask_isolation()
    check_decontamination()
    check_e6_planners()
    check_fallback_provenance()
    check_scireduce_e2e()
    check_rq3()
    check_table1_matched()
    check_syntax_failure_modes()
    check_adversarial()
    check_apb_breadth()
    check_scireduce_samples()
    check_regex_baseline()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} problem(s) out of {CHECKED} checks:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"All {CHECKED} reported numbers match their source outputs.")
    sys.exit(0)


if __name__ == "__main__":
    main()
