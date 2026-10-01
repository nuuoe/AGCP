"""Compare the numbers in the paper's tables with the run outputs under runs/.

Covers the ALFWorld parse table and its seed counts, the SCI-ReDuce
coverage curve, the LLM+P rescore, the closed-loop coverage variants, the
ground-truth-fallback correction, the schema-reuse count and the PlanBench
NL-parse table. Run from the repository root: python scripts/verify_tables.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

FAILURES: list[str] = []
CHECKED = 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(rel: str):
    p = Path(rel)
    if not p.exists():
        FAILURES.append(f"missing file: {rel}")
        return None
    return json.loads(p.read_text())


def record(label: str, ok: bool, stated, actual) -> bool:
    global CHECKED
    CHECKED += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label:<56} stated={stated!s:<24} actual={actual!s}")
    if not ok:
        FAILURES.append(f"{label}: stated {stated}, got {actual}")
    return ok


def check(label: str, stated, actual) -> bool:
    return record(label, stated == actual, stated, actual)


def column(pattern: str, key: str, domains: list[str]) -> list:
    out = []
    for dom in domains:
        d = load(pattern.format(dom))
        out.append(None if d is None else d.get(key))
    return out


# Table 2
def check_table2() -> None:
    print("\n[Table 2] ALFWorld NL-goal parse accept rate, % with Wilson 95% CI")
    rows = [
        ("Regex IID", "runs/alfworld_regex_v2_full.json", "n_accept", 251, 31, (25, 37)),
        ("Qwen-1.5B IID", "runs/alfworld_fair_eval_full.json", "n_llm_accept", 251, 29, (24, 35)),
        ("Qwen-2.5-7B-Turbo IID", "runs/alfworld_cloud_together_qwen7b.json", "n_accept", 251, 42, (36, 48)),
        ("Llama-3.3-70B-Turbo IID", "runs/alfworld_cloud_together_llama70b.json", "n_accept", 251, 41, (35, 47)),
        ("GPT-3.5-turbo IID", "runs/alfworld_cloud_gpt35.json", "n_accept", 251, 40, (34, 46)),
        ("GPT-4o-mini IID", "runs/alfworld_cloud_openai_mini.json", "n_accept", 251, 44, (38, 50)),
        ("Haiku 4.5 IID", "runs/alfworld_cloud_claude_haiku.json", "n_accept", 251, 49, (42, 55)),
        ("GPT-4o IID", "runs/alfworld_cloud_gpt4o.json", "n_accept", 251, 56, (50, 62)),
        ("Sonnet 4.6 IID", "runs/alfworld_cloud_claude_sonnet.json", "n_accept", 251, 60, (54, 66)),
        ("Regex OOD", "runs/ood/alfworld_regex_v2_unseen.json", "n_accept", 251, 32, (26, 38)),
        ("Qwen-1.5B OOD", "runs/ood/alfworld_fair_eval_unseen.json", "n_llm_accept", 255, 42, (36, 48)),
        ("Qwen-2.5-7B-Turbo OOD", "runs/ood/alfworld_cloud_together_qwen7b.json", "n_accept", 255, 52, (46, 58)),
        ("Llama-3.3-70B-Turbo OOD", "runs/ood/alfworld_cloud_together_llama70b.json", "n_accept", 255, 55, (49, 61)),
        ("GPT-3.5-turbo OOD", "runs/ood/alfworld_cloud_gpt35.json", "n_accept", 255, 50, (44, 56)),
        ("GPT-4o-mini OOD", "runs/ood/alfworld_cloud_openai_mini.json", "n_accept", 251, 53, (47, 59)),
        ("Haiku 4.5 OOD", "runs/ood/alfworld_cloud_claude_haiku.json", "n_accept", 251, 57, (51, 63)),
        ("GPT-4o OOD", "runs/ood/alfworld_cloud_gpt4o.json", "n_accept", 251, 62, (56, 68)),
        ("Sonnet 4.6 OOD", "runs/ood/alfworld_cloud_claude_sonnet.json", "n_accept", 251, 67, (61, 73)),
    ]
    for label, path, key, n_stated, pct, ci in rows:
        d = load(path)
        if d is None:
            continue
        k, n = d[key], d["n_total"]
        lo, hi = wilson(k, n)
        got_pct = round(100 * k / n, 1)
        got_ci = (round(100 * lo), round(100 * hi))
        # The table prints whole percentages; 74/251 = 29.5% appears as 30.
        ok = n == n_stated and abs(got_pct - pct) <= 0.6 and got_ci == ci
        record(label, ok, f"{pct} {list(ci)} N={n_stated}", f"{got_pct} {list(got_ci)} N={n} ({k}/{n})")


# Table 11
def check_table11_seeds() -> None:
    print("\n[Table 11] per-seed accept counts and stdev in pp")
    sets = [
        ("GPT-4o-mini IID", ["runs/alfworld_cloud_openai_mini.json",
                             "runs/seed1/alfworld_cloud_openai_mini.json",
                             "runs/seed2/alfworld_cloud_openai_mini.json",
                             "runs/seed3/alfworld_cloud_openai_mini.json"], [110, 111, 111, 109], 0.4),
        ("GPT-4o IID", ["runs/alfworld_cloud_gpt4o.json", "runs/seed2/alfworld_cloud_gpt4o.json",
                        "runs/seed3/alfworld_cloud_gpt4o.json"], [141, 136, 142], 1.3),
        ("Sonnet 4.6 IID", ["runs/alfworld_cloud_claude_sonnet.json",
                            "runs/seed2/alfworld_cloud_claude_sonnet.json",
                            "runs/seed3/alfworld_cloud_claude_sonnet.json"], [151, 150, 149], 0.4),
        ("GPT-4o-mini OOD", ["runs/ood/alfworld_cloud_openai_mini.json"], [134], None),
        ("GPT-4o OOD", ["runs/ood/alfworld_cloud_gpt4o.json", "runs/seed2/ood/alfworld_cloud_gpt4o.json",
                        "runs/seed3/ood/alfworld_cloud_gpt4o.json"], [156, 158, 158], 0.1),
        ("Sonnet 4.6 OOD", ["runs/ood/alfworld_cloud_claude_sonnet.json",
                            "runs/seed2/ood/alfworld_cloud_claude_sonnet.json",
                            "runs/seed3/ood/alfworld_cloud_claude_sonnet.json"], [169, 173, 173], 0.3),
    ]
    for label, paths, counts, sd in sets:
        runs = [load(p) for p in paths]
        if any(r is None for r in runs):
            continue
        check(f"{label} seed counts", counts, [r["n_accept"] for r in runs])
        if sd is not None:
            pcts = [100 * r["n_accept"] / r["n_total"] for r in runs]
            check(f"{label} stdev", sd, round(statistics.stdev(pcts), 1))


# Table 4
def check_table4_scireduce() -> None:
    print("\n[Table 4] SCI-ReDuce held-out coverage, mean and sd over 4 seeds")
    stated = {10: (2.5, 3.5), 15: (7.9, 3.6), 20: (12.5, 5.0), 25: (17.0, 6.0),
              30: (20.0, 0.0), 35: (30.0, 8.6), 40: (32.5, 9.6)}
    d = load("runs/sci_reduce_coverage_curve.json")
    if d is None:
        return
    got = {r["train_size"]: r for r in d["results"]}
    check("training sizes", sorted(stated), sorted(got))
    check("seeds per setting", 4, d.get("n_seeds_per_setting"))
    check("trajectory pool", 50, d.get("total_trajectories_available"))
    for n, (mean, sd) in stated.items():
        r = got.get(n)
        if r is None:
            continue
        check(f"n={n} mean, sd, seeds, held-out", (mean, sd, 4, 50 - n),
              (r["mean_coverage_pct"], r["stdev"], len(r["values"]), r["held_out"]))
    if 30 in got:
        check("n=30 per-seed coverage (caption)", [20, 20, 20, 20], got[30]["values"])


# Table 8 and Table 1
def check_llm_p() -> None:
    print("\n[Table 8, Table 1] LLM+P on Depots under the non-empty-plan criterion")

    def recount(path):
        d = load(path)
        if d is None:
            return None
        res = d["results"]
        raw = sum(1 for r in res if r.get("llm_success"))
        strict = sum(1 for r in res if r.get("llm_success") and (r.get("plan_length") or 0) > 0)
        return raw, strict, len(res)

    lex = recount("runs/llm_p_depots_lexsorted.json")
    if lex is not None:
        check("Table 8: reported -> non-empty (lexicographic run)", "12/30 -> 0/30",
              f"{lex[0]}/{lex[2]} -> {lex[1]}/{lex[2]}")
    matched = recount("runs/llm_p_depots_matched_n30.json")
    if matched is not None:
        check("Table 1: LLM+P Depots, matched instances 2-31", "0/30",
              f"{matched[1]}/{matched[2]}")
        print(f"         matched run before the criterion: {matched[0]}/{matched[2]}")
    d = load("runs/llm_p_rescored_nonempty.json")
    if d is not None and lex is not None and matched is not None:
        summary = sorted((e["n_orig_success"], e["n_strict_success"], e["n_total"]) for e in d["summary"])
        check("rescore file agrees with the recount", sorted([lex, matched]), summary)


# Sec. 5.4 and App. T
def check_e6_coverage() -> None:
    print("\n[Sec. 5.4, App. T] closed-loop success with the HandCoded expert")
    for label, path, k_stated, pct in [
        ("30 instances at F1 >= 0.95", "runs/alfworld_closed_loop_e6.json", "29/30", 96.7),
        ("93 instances at F1 >= 0.95", "runs/alfworld_closed_loop_e6_full.json", "74/93", 79.6),
        ("all 143 instances, six task types", "runs/alfworld_closed_loop_e6_full_unfiltered.json",
         "81/143", 56.6),
    ]:
        d = load(path)
        if d is None:
            continue
        k, n = d["n_execution_success"], d["n_total"]
        check(label, (k_stated, pct), (f"{k}/{n}", round(100 * k / n, 1)))
    d = load("runs/alfworld_closed_loop_e6_full_unfiltered.json")
    if d is not None:
        excluded = {"pick_two_obj_and_place", "pick_and_place_with_movable_recep"}
        core = [r for r in d["results"] if r["task_type"] not in excluded]
        k = sum(1 for r in core if r.get("execution_success"))
        lo, hi = wilson(k, len(core))
        check("119 core-5 instances, no parse filter", ("81/119", 68.1, (59, 76)),
              (f"{k}/{len(core)}", round(100 * k / len(core), 1), (round(100 * lo), round(100 * hi))))
        check("parameters accepted by the environment policy", "133/143",
              f"{d['n_planner_success']}/{d['n_total']}")


# Table 8
def check_gt_fallback() -> None:
    print("\n[Table 8] unreported ground-truth fallback, Mystery 2-11")
    rejected = reported = n = 0
    for i in range(2, 12):
        d = load(f"runs/n4_closed_loop_mystery_inst{i}.json")
        if d is None:
            continue
        n += 1
        rejected += not d["trace"]["step3_nl_parse"].get("llm_accepted")
        reported += bool(d.get("goal_achieved"))
    check("parse rejected by the verifier", "4/10", f"{rejected}/{n}")
    check("reported end-to-end with the fallback", "10/10", f"{reported}/{n}")
    ok = n = 0
    for i in range(2, 12):
        d = load(f"runs/n4_no_gt_fallback_mystery_blocksworld_3_inst{i}.json")
        if d is None:
            continue
        n += 1
        ok += bool(d.get("llm_e4_succeeded") and d.get("success_against_gt"))
    check("corrected LLM-driven success without fallback", "10/10", f"{ok}/{n}")


# Sec. 5.1 and App. O
def check_reuse() -> None:
    print("\n[Sec. 5.1, App. O] schema library reuse across the IPC-2023 domains")
    d = load("runs/ipc2023_breadth_v4.json")
    if d is None:
        return
    results = d["results"]
    reused = [(dom, act) for dom, body in results.items()
              for act, info in body["per_action"].items() if info.get("reused")]
    n = sum(len(b["per_action"]) for b in results.values())
    check("domains", 10, len(results))
    check("schemas reused from the library", 2, len(reused))
    check("schemas induced afresh", 44, n - len(reused))


# Table 7
def check_table7() -> None:
    print("\n[Table 7] PlanBench NL-goal parse, N=30 per domain")
    doms = ["blocksworld_3", "mystery_blocksworld_3", "logistics", "depots"]
    regex = column("runs/regex_baseline_{}.json", "n_accept_at_0.9", doms)
    check("Regex column", [30, 30, 30, 30], regex)
    check("Regex pooled", "120/120", f"{sum(x or 0 for x in regex)}/120")
    llm = column("runs/n1_parse_corrected_{}.json", "n_llm_accepted", doms)
    check("LLM+v. column (100/100/56.7/100%)", [30, 30, 17, 30], llm)
    check("LLM+v. pooled %", 89.2, round(100 * sum(x or 0 for x in llm) / 120, 1))
    d = load("runs/n1_parse_corrected_canon_logistics.json")
    if d is not None:
        check("+canon Logistics (73.3%, five newly accepted)", (22, 30, 5),
              (d["n_post"], d["n_total"], d["new_accepts"]))
    alias = column("runs/n1_alias_prompted_{}.json", "n_llm_accepted", doms)
    check("+alias column", [30, 30, 30, 30], alias)
    unc = column("runs/n1_parse_uncorrected_{}.json", "n_llm_accepted", doms)
    check("uncorrected extraction (46.7/40.0/30.0/3.3%)", [14, 12, 9, 1], unc)
    check("uncorrected pooled %", 30.0, round(100 * sum(x or 0 for x in unc) / 120, 1))


def main() -> int:
    print("Table check against the run outputs")
    check_table2()
    check_table11_seeds()
    check_table4_scireduce()
    check_llm_p()
    check_e6_coverage()
    check_gt_fallback()
    check_reuse()
    check_table7()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} problem(s) out of {CHECKED} comparisons:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"All {CHECKED} table values match their run outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
