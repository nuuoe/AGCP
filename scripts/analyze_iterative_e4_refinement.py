"""Compare the online-refinement run with the iterative-E4 baseline.

Reads runs/alfworld_iterative_e4.json and
runs/alfworld_iterative_e4_with_refinement.json; prints success rates with
Wilson CIs, per-task-type deltas, early-vs-late accuracy, LLM-call counts,
per-task gains/losses and sample rule hints.
"""
from __future__ import annotations
import json
import sys
import math
from collections import defaultdict
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    radius = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def main():
    base_path = Path("runs/alfworld_iterative_e4.json")
    ref_path = Path("runs/alfworld_iterative_e4_with_refinement.json")
    if not ref_path.exists():
        sys.exit(f"missing: {ref_path}")
    base = json.load(open(base_path))
    ref = json.load(open(ref_path))

    base_results = base["results"]
    ref_results = ref["results"]
    base_by_id = {r["task_id"]: r for r in base_results}
    ref_by_id = {r["task_id"]: r for r in ref_results}

    n_base = len(base_results)
    n_ref = len(ref_results)
    base_won = sum(1 for r in base_results if r.get("execution_success"))
    ref_won = sum(1 for r in ref_results if r.get("execution_success"))

    print("=" * 78)
    print("ONLINE REFINEMENT vs ITERATIVE E4 BASELINE")
    print("=" * 78)
    print(f"\nN (baseline): {n_base}  won: {base_won}  rate: {100*base_won/n_base:.2f}%")
    b_ci = wilson_ci(base_won, n_base)
    print(f"  Wilson 95% CI: [{100*b_ci[0]:.2f}, {100*b_ci[1]:.2f}]")
    print(f"\nN (refinement): {n_ref}  won: {ref_won}  rate: {100*ref_won/n_ref:.2f}%")
    r_ci = wilson_ci(ref_won, n_ref)
    print(f"  Wilson 95% CI: [{100*r_ci[0]:.2f}, {100*r_ci[1]:.2f}]")
    print(f"\nDelta: {100*(ref_won/n_ref - base_won/n_base):+.2f}pp")

    # --- Per-task-type ---
    print("\n" + "-" * 78)
    print("PER-TASK-TYPE BREAKDOWN")
    print("-" * 78)
    print(f"{'task_type':<40s}{'baseline':>16s}{'refinement':>16s}{'delta':>10s}")
    base_by_type = defaultdict(list)
    for r in base_results:
        base_by_type[r["task_type"]].append(r)
    ref_by_type = defaultdict(list)
    for r in ref_results:
        ref_by_type[r["task_type"]].append(r)
    for tt in sorted(base_by_type):
        bn = len(base_by_type[tt])
        bw = sum(1 for r in base_by_type[tt] if r.get("execution_success"))
        rn = len(ref_by_type.get(tt, []))
        rw = sum(1 for r in ref_by_type.get(tt, []) if r.get("execution_success"))
        b_rate = bw / bn if bn else 0
        r_rate = rw / rn if rn else 0
        delta = r_rate - b_rate
        print(f"{tt:<40s}{bw}/{bn} ({100*b_rate:.0f}%)".ljust(57)
              + f"{rw}/{rn} ({100*r_rate:.0f}%)".ljust(16)
              + f"{100*delta:+.1f}pp".rjust(10))

    # --- Early-vs-late within refinement run ---
    print("\n" + "-" * 78)
    print("EARLY-vs-LATE WITHIN REFINEMENT (self-improvement signal)")
    print("-" * 78)
    half = n_ref // 2
    early = ref_results[:half]
    late = ref_results[half:]
    ew = sum(1 for r in early if r.get("execution_success"))
    lw = sum(1 for r in late if r.get("execution_success"))
    eci = wilson_ci(ew, len(early))
    lci = wilson_ci(lw, len(late))
    print(f"first half  ({len(early)} tasks): {ew}/{len(early)} = "
          f"{100*ew/len(early):.1f}%  CI [{100*eci[0]:.1f}, {100*eci[1]:.1f}]")
    print(f"second half ({len(late)} tasks): {lw}/{len(late)} = "
          f"{100*lw/len(late):.1f}%  CI [{100*lci[0]:.1f}, {100*lci[1]:.1f}]")
    delta = lw/len(late) - ew/len(early)
    print(f"delta (late - early): {100*delta:+.1f}pp "
          f"{'POSITIVE' if delta > 0 else ('NULL' if delta == 0 else 'NEGATIVE')}")

    # Baseline early-vs-late for comparison.
    base_ew = sum(1 for r in base_results[:half] if r.get("execution_success"))
    base_lw = sum(1 for r in base_results[half:] if r.get("execution_success"))
    print(f"\nCHECK — baseline early-vs-late (no refinement):")
    print(f"  first half: {base_ew}/{half} = {100*base_ew/half:.1f}%")
    print(f"  second half: {base_lw}/{n_base-half} = "
          f"{100*base_lw/(n_base-half):.1f}%")
    base_delta = base_lw/(n_base-half) - base_ew/half
    print(f"  delta: {100*base_delta:+.1f}pp")
    print(f"\nNET self-improvement = (refinement delta) - (baseline delta)")
    print(f"  = {100*delta:+.1f}pp - ({100*base_delta:+.1f}pp) = "
          f"{100*(delta-base_delta):+.1f}pp")

    # --- Per-task-type early vs late ---
    print("\n" + "-" * 78)
    print("PER-TASK-TYPE: first-third vs last-third within refinement")
    print("-" * 78)
    for tt in sorted(ref_by_type):
        rs = ref_by_type[tt]
        if len(rs) < 6:
            print(f"  {tt}: only {len(rs)} tasks, skipping")
            continue
        q = max(1, len(rs) // 3)
        early_t = rs[:q]
        late_t = rs[-q:]
        e_w = sum(1 for r in early_t if r.get("execution_success"))
        l_w = sum(1 for r in late_t if r.get("execution_success"))
        print(f"  {tt}: first{q}: {e_w}/{q} = {100*e_w/q:.0f}%; "
              f"last{q}: {l_w}/{q} = {100*l_w/q:.0f}%; "
              f"delta: {100*(l_w-e_w)/q:+.0f}pp")

    # --- Call budget ---
    print("\n" + "-" * 78)
    print("LLM call budget")
    print("-" * 78)
    base_calls = sum(r.get("llm_calls", 0) for r in base_results)
    ref_e4 = ref.get("total_e4_calls", sum(r.get("llm_calls", 0) for r in ref_results))
    ref_rule = ref.get("total_rule_calls", 0)
    ref_total = ref.get("total_llm_calls", ref_e4 + ref_rule)
    print(f"  baseline (iter E4):       {base_calls:5d} calls "
          f"({base_calls/n_base:.2f}/task)")
    print(f"  refinement E4:            {ref_e4:5d} calls "
          f"({ref_e4/n_ref:.2f}/task)")
    print(f"  refinement rule-proposal: {ref_rule:5d} calls "
          f"({ref_rule/n_ref:.2f}/task)")
    print(f"  refinement TOTAL:         {ref_total:5d} calls "
          f"({ref_total/n_ref:.2f}/task)")
    print(f"  wall seconds (refinement run): {ref.get('wall_seconds', 0):.1f}s")

    # --- Diff vs baseline at task level ---
    print("\n" + "-" * 78)
    print("PER-TASK DIFF vs BASELINE")
    print("-" * 78)
    gains = []   # baseline failed, refinement won
    losses = []  # baseline won, refinement failed
    same = 0
    for tid, r in ref_by_id.items():
        b = base_by_id.get(tid)
        if b is None:
            continue
        rw = r.get("execution_success")
        bw = b.get("execution_success")
        if rw and not bw:
            gains.append((tid, r))
        elif bw and not rw:
            losses.append((tid, r))
        else:
            same += 1
    print(f"  gains  (base failed -> ref won):  {len(gains)}")
    print(f"  losses (base won  -> ref failed): {len(losses)}")
    print(f"  same: {same}")
    if gains:
        print("\n  GAIN examples (first 5):")
        for tid, r in gains[:5]:
            lessons = r.get("lessons_provided") or []
            n_lessons = len(lessons)
            print(f"    {tid} (had {n_lessons} lessons in context)")
            for L in lessons[-2:]:
                print(f"      lesson: {L.get('rule_hint','')[:120]}")
    if losses:
        print("\n  LOSS examples (first 5):")
        for tid, r in losses[:5]:
            lessons = r.get("lessons_provided") or []
            n_lessons = len(lessons)
            print(f"    {tid} (had {n_lessons} lessons in context)")
            for L in lessons[-2:]:
                print(f"      lesson: {L.get('rule_hint','')[:120]}")

    # --- Sample helpful rule hints (task with lessons -> won) ---
    print("\n" + "-" * 78)
    print("SAMPLE RULE-HINTS")
    print("-" * 78)
    helpful = [r for r in ref_results
                 if r.get("lessons_provided")
                     and r.get("execution_success")
                     and len(r.get("lessons_provided", [])) >= 1]
    unhelpful = [r for r in ref_results
                   if r.get("lessons_provided")
                       and not r.get("execution_success")
                       and len(r.get("lessons_provided", [])) >= 1]
    print(f"\nHELPFUL contexts: {len(helpful)} tasks won with >=1 lesson in context")
    for r in helpful[:5]:
        print(f"  task: {r['task_id']}  (n_lessons: {len(r['lessons_provided'])})")
        for L in r["lessons_provided"][-2:]:
            print(f"    rule: {L['rule_hint'][:140]}")
    print(f"\nUNHELPFUL contexts: {len(unhelpful)} tasks failed with >=1 lesson")
    for r in unhelpful[:5]:
        print(f"  task: {r['task_id']}  (n_lessons: {len(r['lessons_provided'])})")
        for L in r["lessons_provided"][-2:]:
            print(f"    rule: {L['rule_hint'][:140]}")

    # --- ALL rule hints proposed during the run ---
    print("\n" + "-" * 78)
    print("ALL RULE HINTS GENERATED (in order)")
    print("-" * 78)
    fc = ref.get("failure_cache_final", {})
    n_total_rules = sum(len(v) for v in fc.values())
    print(f"  total rules in failure_cache: {n_total_rules}")
    for tt in sorted(fc):
        print(f"  {tt} ({len(fc[tt])} rules):")
        for i, r in enumerate(fc[tt]):
            print(f"    [{i+1}] order={r.get('order_index')}: "
                  f"{r['rule_hint'][:160]}")

    # --- Headline summary ---
    print("\n" + "=" * 78)
    print("HEADLINE SUMMARY")
    print("=" * 78)
    print(f"Refinement: {ref_won}/{n_ref} = {100*ref_won/n_ref:.2f}% "
          f"(CI {100*r_ci[0]:.1f}-{100*r_ci[1]:.1f})")
    print(f"Baseline:   {base_won}/{n_base} = {100*base_won/n_base:.2f}% "
          f"(CI {100*b_ci[0]:.1f}-{100*b_ci[1]:.1f})")
    print(f"Delta: {100*(ref_won/n_ref - base_won/n_base):+.2f}pp")
    print(f"Self-improvement (refinement: late-early): {100*delta:+.1f}pp")
    print(f"Self-improvement (baseline: late-early): {100*base_delta:+.1f}pp")
    print(f"Net self-improvement: {100*(delta-base_delta):+.1f}pp")


if __name__ == "__main__":
    main()
