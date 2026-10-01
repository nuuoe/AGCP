"""Recompute the appendix statistics of the paper from the run files under runs/.

Checks the task-clustered bootstrap intervals of App. I, the paraphrase-control
McNemar p-values of Sec. 5.2, the GPT-4o replication p-value of Sec. 5.3, the
occurrence-rule sensitivity table of App. H and the induced-planner failure
tally of App. R. Standard library only. Run from the repository root; exits
non-zero if any recomputed value differs from the paper's.
"""
from __future__ import annotations

import collections
import decimal
import json
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.alfworld_enum_constrained import smart_f1_extended  # noqa: E402

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
BOOTSTRAP_TOLERANCE = 0.01
ACCEPT_THRESHOLD = 0.9

OOD_FILES = {
    "regex": "runs/ood/alfworld_regex_v2_unseen.json",
    "gpt4o": "runs/ood/alfworld_cloud_gpt4o.json",
    "sonnet": "runs/ood/alfworld_cloud_claude_sonnet.json",
    "regex_para": "runs/alfworld_paraphrase_decon_struct_regex.json",
    "gpt4o_para": "runs/alfworld_paraphrase_decon_struct_gpt4o.json",
    "sonnet_para": "runs/alfworld_paraphrase_decon_struct_claude_sonnet.json",
}

RQ3_CHAIN = {
    "sonnet": ("runs/alfworld_iterative_e4_K1.json",
               "runs/alfworld_iterative_e4.json",
               "runs/alfworld_iterative_e4_with_refinement.json"),
    "gpt4o": ("runs/alfworld_iterative_e4_gpt4o_K1.json",
              "runs/alfworld_iterative_e4_gpt4o.json",
              "runs/alfworld_iterative_e4_gpt4o_refine.json"),
}

INDUCED_PLANNER = "runs/alfworld_e6_induced_planner.json"
INDUCED_PLANNER_PARSES = "runs/alfworld_cloud_claude_sonnet.json"


def load_json(rel_path):
    with open(os.path.join(ROOT, rel_path)) as f:
        return json.load(f)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def rounds_to(value, paper):
    target = decimal.Decimal(paper)
    rounded = decimal.Decimal(repr(value)).quantize(
        target, rounding=decimal.ROUND_HALF_UP)
    if "e" in paper:
        mantissa_decimals = len(paper.split("e")[0].split(".")[1])
        shown = format(rounded, f".{mantissa_decimals}e")
    else:
        shown = str(rounded)
    return rounded == target, shown


def percentile(sorted_values, q):
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def load_accepts(rel_path):
    data = load_json(rel_path)
    rows = data["rows"]
    f1_key = next(k for k in ("smart_f1", "regex_v2_f1", "llm_f1") if k in rows[0])
    desc_key = "task_desc_orig" if "task_desc_orig" in rows[0] else "task_desc"
    seen = collections.Counter()
    out = []
    for r in rows:
        base = (r["task_id"], r[desc_key])
        out.append((base + (seen[base],), int(r[f1_key] >= ACCEPT_THRESHOLD)))
        seen[base] += 1
    n_accept = data["n_llm_accept"] if "n_llm_accept" in data else data["n_accept"]
    if sum(x for _, x in out) != n_accept:
        raise ValueError(f"{rel_path}: recounted accepts differ from n_accept")
    return out


def align(a, b):
    if [k for k, _ in a] != [k for k, _ in b]:
        raise ValueError("row keys do not align")
    return [x for _, x in a], [x for _, x in b]


def discordant(a, b):
    xa, xb = align(a, b)
    gains = sum(1 for p, q in zip(xa, xb) if q and not p)
    losses = sum(1 for p, q in zip(xa, xb) if p and not q)
    return gains, losses


def clustered_bootstrap(a, b, rng):
    xa, xb = align(a, b)
    by_task = collections.defaultdict(lambda: [0, 0, 0])
    for (key, _), p, q in zip(a, xa, xb):
        cell = by_task[key[0]]
        cell[0] += p
        cell[1] += q
        cell[2] += 1
    tasks = sorted(by_task)
    diffs = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        na = nb = n = 0
        for t in rng.choices(tasks, k=len(tasks)):
            sa, sb, m = by_task[t]
            na += sa
            nb += sb
            n += m
        diffs.append((na - nb) / n)
    diffs.sort()
    point = (sum(xa) - sum(xb)) / len(xa)
    return point, percentile(diffs, 0.025), percentile(diffs, 0.975), len(tasks), len(xa)


def check_bootstrap():
    acc = {name: load_accepts(path) for name, path in OOD_FILES.items()}
    comparisons = [
        ("Sonnet - regex, original goals", "sonnet", "regex", (0.27, 0.45)),
        ("GPT-4o - regex, original goals", "gpt4o", "regex", (0.23, 0.38)),
        ("Sonnet - regex, paraphrases", "sonnet_para", "regex_para", (0.60, 0.75)),
        ("Sonnet paraphrases - originals", "sonnet_para", "sonnet", (-0.02, 0.04)),
        ("GPT-4o - Sonnet, original goals", "gpt4o", "sonnet", (-0.11, 0.01)),
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    checks = []
    for label, a, b, (paper_lo, paper_hi) in comparisons:
        point, lo, hi, n_tasks, n_rows = clustered_bootstrap(acc[a], acc[b], rng)
        ok = (abs(lo - paper_lo) <= BOOTSTRAP_TOLERANCE + 1e-12
              and abs(hi - paper_hi) <= BOOTSTRAP_TOLERANCE + 1e-12)
        checks.append((f"{label} ({n_rows} rows, {n_tasks} tasks)",
                       f"[{paper_lo:+.2f}, {paper_hi:+.2f}]",
                       f"[{lo:+.3f}, {hi:+.3f}] point {point:+.3f}", ok))
    return checks


def check_paraphrase():
    acc = {name: load_accepts(path) for name, path in OOD_FILES.items()}
    comparisons = [
        ("GPT-4o, paraphrases vs originals", "gpt4o", "gpt4o_para", "0.75"),
        ("Sonnet, paraphrases vs originals", "sonnet", "sonnet_para", "0.55"),
        ("regex, paraphrases vs originals", "regex", "regex_para", "6.6e-24"),
        ("Sonnet vs regex on paraphrases", "regex_para", "sonnet_para", "5.8e-50"),
    ]
    checks = []
    for label, a, b, paper in comparisons:
        gains, losses = discordant(acc[a], acc[b])
        p = mcnemar_exact(gains, losses)
        ok, shown = rounds_to(p, paper)
        ka = sum(x for _, x in acc[a])
        kb = sum(x for _, x in acc[b])
        checks.append((f"{label} ({ka} -> {kb} of {len(acc[a])})", f"p={paper}",
                       f"p={shown} (+{gains}/-{losses}, {p:.3g})", ok))
    return checks


def dedup(rows, rule):
    out = {}
    for r in rows:
        if rule == "last" or r["task_id"] not in out:
            out[r["task_id"]] = r
    return out


def parse_accept(r):
    return smart_f1_extended(r.get("final_atoms") or {},
                             r.get("gt_for_logging") or {}) >= ACCEPT_THRESHOLD


def chain_flags(system, rule):
    runs = [load_json(p)["results"] for p in RQ3_CHAIN[system]]
    uniq = [dedup(rows, rule) for rows in runs]
    common = sorted(set.intersection(*(set(u) for u in uniq)))
    flags = []
    for u in uniq:
        flags.append({t: (bool(u[t]["execution_success"]), parse_accept(u[t]))
                      for t in common})
    return runs, flags, common


def contrast(flags_a, flags_b, common, index):
    gains = sum(1 for t in common if flags_b[t][index] and not flags_a[t][index])
    losses = sum(1 for t in common if flags_a[t][index] and not flags_b[t][index])
    return gains, losses, mcnemar_exact(gains, losses)


def pair_check(label, gains, losses, p, paper_gains, paper_losses, paper_p):
    ok, shown = rounds_to(p, paper_p)
    ok = ok and gains == paper_gains and losses == paper_losses
    return (label, f"+{paper_gains}/-{paper_losses} (p={paper_p})",
            f"+{gains}/-{losses} (p={shown}, {p:.3g})", ok)


def check_gpt4o():
    runs, flags, common = chain_flags("gpt4o", "last")
    checks = []
    for label, rows, paper in (("GPT-4o one-shot completion, 119-row stream", runs[0], 80),
                               ("GPT-4o A+B completion, 119-row stream", runs[2], 90)):
        k = sum(1 for r in rows if r["execution_success"])
        checks.append((label, f"{paper}/119", f"{k}/{len(rows)}", k == paper and len(rows) == 119))
    gains, losses, p = contrast(flags[0], flags[2], common, 0)
    ok, shown = rounds_to(p, "0.006")
    checks.append((f"GPT-4o A+B vs one-shot, {len(common)} unique tasks (last)",
                   "p=0.006", f"p={shown} (+{gains}/-{losses}, {p:.3g})", ok))
    return checks


def check_sensitivity():
    paper = {
        "last": {"exec": (77, 86, 92), "parse": (70, 79, 86),
                 "exec_b": (6, 0, "0.031"), "parse_b": (8, 1, "0.039"),
                 "exec_ab": (15, 0, "6.1e-5")},
        "first": {"exec": (77, 86, 91), "parse": (70, 79, 85),
                  "exec_b": (5, 0, "0.063"), "parse_b": (7, 1, "0.070"),
                  "exec_ab": (14, 0, "1.2e-4")},
    }
    checks = []
    for rule in ("last", "first"):
        _, flags, common = chain_flags("sonnet", rule)
        n = len(common)
        for index, name in ((0, "exec"), (1, "parse")):
            counts = tuple(sum(1 for t in common if f[t][index]) for f in flags)
            label = {"exec": "Completion", "parse": "Parse >= 0.9"}[name]
            checks.append((f"{label}, {rule}: 1-shot/A/A+B of {n}",
                           "/".join(map(str, paper[rule][name])),
                           "/".join(map(str, counts)), counts == paper[rule][name]))
        for key, index, label, a, b in (("exec_b", 0, "Completion B vs A", 1, 2),
                                        ("parse_b", 1, "Parse B vs A", 1, 2),
                                        ("exec_ab", 0, "Completion A+B vs 1-shot", 0, 2)):
            gains, losses, p = contrast(flags[a], flags[b], common, index)
            checks.append(pair_check(f"{label}, {rule}", gains, losses, p, *paper[rule][key]))
    return checks


def failure_category(r):
    if r.get("stage_failed") == "planner_no_plan":
        return "no plan"
    if r.get("stage_failed") == "no_goal_candidates":
        return "goal construction"
    reason = r.get("fail_reason") or ""
    if reason == "plan exhausted":
        return "divergence"
    if reason.startswith("not admissible:"):
        return "rendering"
    return "unmapped"


def check_failures():
    data = load_json(INDUCED_PLANNER)
    rows = data.get("results")
    if not rows:
        with open(os.path.join(ROOT, INDUCED_PLANNER + ".rows.jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    parses = {r["task_id"]: r for r in load_json(INDUCED_PLANNER_PARSES)["rows"]}
    failures = [r for r in rows if not r["execution_success"]]
    mapping = collections.Counter()
    tally = collections.Counter()
    for r in failures:
        category = failure_category(r)
        raw = r.get("stage_failed") or r.get("fail_reason") or ""
        if raw.startswith("not admissible:"):
            raw = "not admissible: <action>"
        mapping[(raw, category)] += 1
        tally[category] += 1
    bad_parse = sum(1 for r in failures if failure_category(r) == "goal construction"
                    and parses[r["task_id"]]["smart_f1"] < 1.0)
    print("  raw label -> category")
    for (raw, category), n in sorted(mapping.items(), key=lambda kv: -kv[1]):
        print(f"    {raw:<32} -> {category:<18} {n}")
    checks = [(f"Failures of {len(rows)} tasks", "73", str(len(failures)), len(failures) == 73)]
    for category, paper in (("no plan", 37), ("goal construction", 16),
                            ("divergence", 17), ("rendering", 3)):
        checks.append((category, str(paper), str(tally[category]), tally[category] == paper))
    checks.append(("goal construction after an imperfect parse", "15", str(bad_parse), bad_parse == 15))
    checks.append(("unmapped failure labels", "0", str(tally["unmapped"]), tally["unmapped"] == 0))
    return checks


def main():
    items = [
        ("App. I task-clustered bootstrap, 95% percentile intervals "
         f"({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, tolerance {BOOTSTRAP_TOLERANCE})",
         check_bootstrap),
        ("Sec. 5.2 paraphrase control, exact McNemar", check_paraphrase),
        ("Sec. 5.3 GPT-4o replication", check_gpt4o),
        ("App. H occurrence-rule sensitivity", check_sensitivity),
        ("App. R induced-planner failure tally", check_failures),
    ]
    failed = 0
    for title, fn in items:
        print(f"\n{title}")
        checks = fn()
        width = max(len(c[0]) for c in checks)
        print(f"  {'quantity':<{width}}  {'paper':<24} {'recomputed':<40} status")
        for label, paper, recomputed, ok in checks:
            failed += not ok
            print(f"  {label:<{width}}  {paper:<24} {recomputed:<40} {'ok' if ok else 'MISMATCH'}")
    print(f"\n{'all values match the paper' if not failed else f'{failed} value(s) differ from the paper'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
