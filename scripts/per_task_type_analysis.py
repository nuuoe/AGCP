#!/usr/bin/env python3
"""Per-task-type accept rates across the ALFWorld scale-curve systems.

Accept = score field (smart_f1, llm_f1 or regex_v2_f1) >= 0.9. For each
system, computes per-task-type and pooled k/n with Wilson 95% CIs, prints a
wide table and a LaTeX table, and writes runs/per_task_type_stratified.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
OUT_PATH = RUNS_DIR / "per_task_type_stratified.json"

THRESHOLD = 0.9

TASK_TYPES = [
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
]

# Short labels for compact LaTeX column headers.
TASK_SHORT = {
    "look_at_obj_in_light": "look",
    "pick_and_place_simple": "p\\&p",
    "pick_clean_then_place_in_recep": "clean",
    "pick_cool_then_place_in_recep": "cool",
    "pick_heat_then_place_in_recep": "heat",
    "pick_two_obj_and_place": "two-obj",
    "pick_and_place_with_movable_recep": "movable",
}


# (display_name, json_path_relative_to_repo_root, score_field, split)
SYSTEMS: list[tuple[str, str, str, str]] = [
    # IID systems
    ("regex_v2 (IID)", "runs/alfworld_regex_v2_full.json", "regex_v2_f1", "IID"),
    ("fair_eval/LLM (IID)", "runs/alfworld_fair_eval_full.json", "llm_f1", "IID"),
    ("GPT-3.5 (IID)", "runs/alfworld_cloud_gpt35.json", "smart_f1", "IID"),
    ("OpenAI mini (IID)", "runs/alfworld_cloud_openai_mini.json", "smart_f1", "IID"),
    ("GPT-4o (IID)", "runs/alfworld_cloud_gpt4o.json", "smart_f1", "IID"),
    ("Claude Haiku (IID)", "runs/alfworld_cloud_claude_haiku.json", "smart_f1", "IID"),
    ("Claude Sonnet (IID)", "runs/alfworld_cloud_claude_sonnet.json", "smart_f1", "IID"),
    ("Qwen-7B (IID)", "runs/alfworld_cloud_together_qwen7b.json", "smart_f1", "IID"),
    ("Llama-70B (IID)", "runs/alfworld_cloud_together_llama70b.json", "smart_f1", "IID"),
    # OOD systems (unseen split)
    ("regex_v2 (OOD)", "runs/ood/alfworld_regex_v2_unseen.json", "regex_v2_f1", "OOD"),
    ("OpenAI mini (OOD)", "runs/ood/alfworld_cloud_openai_mini.json", "smart_f1", "OOD"),
    ("GPT-4o (OOD)", "runs/ood/alfworld_cloud_gpt4o.json", "smart_f1", "OOD"),
    ("Claude Haiku (OOD)", "runs/ood/alfworld_cloud_claude_haiku.json", "smart_f1", "OOD"),
    ("Claude Sonnet (OOD)", "runs/ood/alfworld_cloud_claude_sonnet.json", "smart_f1", "OOD"),
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval at 95%. Returns (lo, hi); (0, 0) if n == 0."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _get_score(row: dict, field: str) -> float | None:
    """Score lookup with fallback to smart_f1, llm_f1, regex_v2_f1 and f1."""
    for key in (field, "smart_f1", "llm_f1", "regex_v2_f1", "f1"):
        v = row.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def analyze_run(path: Path, score_field: str) -> dict:
    """Return per-task-type and POOLED stats for a single run JSON."""
    with path.open() as fp:
        data = json.load(fp)
    rows = data.get("rows", [])

    buckets: dict[str, dict[str, int]] = {tt: {"k": 0, "n": 0} for tt in TASK_TYPES}
    pooled = {"k": 0, "n": 0}

    for row in rows:
        tt = row.get("task_type")
        if tt not in buckets:
            continue
        score = _get_score(row, score_field)
        if score is None:
            continue
        buckets[tt]["n"] += 1
        pooled["n"] += 1
        if score >= THRESHOLD:
            buckets[tt]["k"] += 1
            pooled["k"] += 1

    result: dict[str, dict] = {}
    for tt, kn in buckets.items():
        lo, hi = wilson_ci(kn["k"], kn["n"])
        rate = (kn["k"] / kn["n"]) if kn["n"] else 0.0
        result[tt] = {
            "n_accept": kn["k"],
            "n_total": kn["n"],
            "accept_rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
        }
    lo, hi = wilson_ci(pooled["k"], pooled["n"])
    result["POOLED"] = {
        "n_accept": pooled["k"],
        "n_total": pooled["n"],
        "accept_rate": (pooled["k"] / pooled["n"]) if pooled["n"] else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
    }
    return result


def fmt_cell(stats: dict) -> str:
    """Cell as 'k/n (pct%)'."""
    return f"{stats['n_accept']}/{stats['n_total']} ({stats['accept_rate'] * 100:.1f}%)"


def fmt_cell_latex(stats: dict) -> str:
    if stats["n_total"] == 0:
        return "--"
    return f"{stats['n_accept']}/{stats['n_total']} ({stats['accept_rate'] * 100:.1f}\\%)"


def print_wide_table(all_stats: dict[str, dict]) -> None:
    """Plain-text wide table to stdout."""
    headers = ["system"] + [TASK_SHORT[t] for t in TASK_TYPES] + ["POOLED"]
    rows_txt = []
    for sysname, stats in all_stats.items():
        row = [sysname]
        for tt in TASK_TYPES:
            row.append(fmt_cell(stats[tt]))
        row.append(fmt_cell(stats["POOLED"]))
        rows_txt.append(row)
    col_w = [max(len(headers[i]), max(len(r[i]) for r in rows_txt)) for i in range(len(headers))]
    sep = "  "
    print(sep.join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
    print(sep.join("-" * col_w[i] for i in range(len(headers))))
    for r in rows_txt:
        print(sep.join(r[i].ljust(col_w[i]) for i in range(len(r))))


def print_latex_table(all_stats: dict[str, dict]) -> None:
    n_cols = 1 + len(TASK_TYPES) + 1  # system + 7 task types + POOLED
    print("")
    print("% ===== LaTeX: per-task-type stratified accept-rate (smart_f1 >= 0.9) =====")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\begin{tabular}{l" + "r" * (n_cols - 1) + "}")
    print("\\toprule")
    hdr = ["System"] + [TASK_SHORT[t] for t in TASK_TYPES] + ["\\textbf{POOLED}"]
    print(" & ".join(hdr) + " \\\\")
    print("\\midrule")

    prev_split = None
    for sysname, stats in all_stats.items():
        split = "OOD" if "OOD" in sysname else "IID"
        if prev_split is not None and split != prev_split:
            print("\\midrule")
        prev_split = split
        row = [sysname.replace("&", "\\&")]
        for tt in TASK_TYPES:
            row.append(fmt_cell_latex(stats[tt]))
        row.append("\\textbf{" + fmt_cell_latex(stats["POOLED"]) + "}")
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        "\\caption{Per-task-type accept rates (k/n, percentage) on ALFWorld with "
        "$\\mathrm{smart\\_f1} \\geq 0.9$. Wilson 95\\% CIs in JSON dump. "
        "Top block: IID (seen) split; bottom block: OOD (unseen) split. "
        "Short labels: look=look\\_at\\_obj\\_in\\_light, p\\&p=pick\\_and\\_place\\_simple, "
        "clean=pick\\_clean\\_then\\_place\\_in\\_recep, cool=pick\\_cool\\_then\\_place\\_in\\_recep, "
        "heat=pick\\_heat\\_then\\_place\\_in\\_recep, two-obj=pick\\_two\\_obj\\_and\\_place, "
        "movable=pick\\_and\\_place\\_with\\_movable\\_recep.}")
    print("\\label{tab:alfworld-per-task-type}")
    print("\\end{table*}")
    print("")


def compute_findings(all_stats: dict[str, dict]) -> dict:
    """Compute (a) best/worst task types for LLMs (mean over LLM systems within split),
    (b) max LLM-vs-regex gap per task type, and (c) any task type where regex ties or beats
    the best LLM."""
    def split_of(name: str) -> str:
        return "OOD" if "OOD" in name else "IID"

    def is_regex(name: str) -> bool:
        return name.startswith("regex_v2")

    findings: dict = {}
    for split in ("IID", "OOD"):
        sys_in_split = [(n, s) for n, s in all_stats.items() if split_of(n) == split]
        llm_systems = [(n, s) for n, s in sys_in_split if not is_regex(n)]
        regex_systems = [(n, s) for n, s in sys_in_split if is_regex(n)]
        if not llm_systems or not regex_systems:
            continue
        regex_name, regex_stats = regex_systems[0]

        # (a) LLM mean accept-rate per task type
        mean_llm: dict[str, float] = {}
        for tt in TASK_TYPES:
            vals = [s[tt]["accept_rate"] for _, s in llm_systems if s[tt]["n_total"] > 0]
            mean_llm[tt] = sum(vals) / len(vals) if vals else 0.0
        best_tt = max(mean_llm, key=mean_llm.get)
        worst_tt = min(mean_llm, key=mean_llm.get)

        # (b) max gap (best LLM accept-rate per tt) - regex accept-rate
        max_gap_tt = None
        max_gap = -2.0
        gaps: dict[str, dict] = {}
        for tt in TASK_TYPES:
            best_llm_rate = max(
                (s[tt]["accept_rate"] for _, s in llm_systems if s[tt]["n_total"] > 0),
                default=0.0,
            )
            best_llm_name = max(
                (n for n, s in llm_systems if s[tt]["n_total"] > 0),
                key=lambda n: dict(llm_systems)[n][tt]["accept_rate"],
                default="",
            )
            regex_rate = regex_stats[tt]["accept_rate"]
            gap = best_llm_rate - regex_rate
            gaps[tt] = {
                "regex_rate": regex_rate,
                "best_llm_rate": best_llm_rate,
                "best_llm_name": best_llm_name,
                "gap": gap,
            }
            if gap > max_gap:
                max_gap = gap
                max_gap_tt = tt

        # (c) task types where regex ties or beats best LLM
        regex_ties_or_wins = {
            tt: g for tt, g in gaps.items() if g["regex_rate"] >= g["best_llm_rate"]
        }

        findings[split] = {
            "llm_mean_accept_rate_by_task_type": mean_llm,
            "best_llm_task_type": {"task_type": best_tt, "mean_rate": mean_llm[best_tt]},
            "worst_llm_task_type": {"task_type": worst_tt, "mean_rate": mean_llm[worst_tt]},
            "max_llm_minus_regex_gap": {
                "task_type": max_gap_tt,
                "gap": max_gap,
                "regex_rate": gaps[max_gap_tt]["regex_rate"],
                "best_llm_rate": gaps[max_gap_tt]["best_llm_rate"],
                "best_llm_name": gaps[max_gap_tt]["best_llm_name"],
            },
            "regex_ties_or_beats_best_llm": regex_ties_or_wins,
            "per_task_gaps": gaps,
            "regex_system_name": regex_name,
            "llm_systems_in_split": [n for n, _ in llm_systems],
        }
    return findings


def print_findings(findings: dict) -> None:
    print("\n===== Findings =====")
    for split, f in findings.items():
        print(f"\n--- {split} ---")
        b = f["best_llm_task_type"]
        w = f["worst_llm_task_type"]
        g = f["max_llm_minus_regex_gap"]
        print(f"(a) LLM mean accept is highest on '{b['task_type']}' ({b['mean_rate']*100:.1f}%) "
              f"and lowest on '{w['task_type']}' ({w['mean_rate']*100:.1f}%).")
        print(f"(b) Largest LLM-over-regex gap: '{g['task_type']}' "
              f"({g['best_llm_name']}: {g['best_llm_rate']*100:.1f}% vs regex: {g['regex_rate']*100:.1f}%, "
              f"gap = {g['gap']*100:+.1f} pp).")
        rtw = f["regex_ties_or_beats_best_llm"]
        if rtw:
            for tt, info in rtw.items():
                print(f"(c) regex ties or beats the best LLM on '{tt}' "
                      f"(regex: {info['regex_rate']*100:.1f}% vs best LLM: {info['best_llm_rate']*100:.1f}%).")
        else:
            print("(c) regex does not tie or beat the best LLM on any task type in this split.")


def main() -> None:
    all_stats: dict[str, dict] = {}
    for sysname, rel_path, score_field, split in SYSTEMS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"# WARN: missing {path}")
            continue
        all_stats[sysname] = analyze_run(path, score_field)

    findings = compute_findings(all_stats)

    out_blob = {
        "threshold": THRESHOLD,
        "task_types": TASK_TYPES,
        "systems": {
            name: {
                "split": ("OOD" if "OOD" in name else "IID"),
                "per_task_type": stats,
            }
            for name, stats in all_stats.items()
        },
        "findings": findings,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fp:
        json.dump(out_blob, fp, indent=2)
    print(f"# Wrote {OUT_PATH}\n")

    print("===== Wide table (k/n, pct%) =====\n")
    print_wide_table(all_stats)
    print_latex_table(all_stats)
    print_findings(findings)


if __name__ == "__main__":
    main()
