"""Pairwise McNemar exact tests and Cohen's h across the ALFWorld scale-curve systems.

For each pair of systems, trials present in both runs are matched by
(task_id, task_desc); the discordant counts get a two-sided exact binomial
test and Cohen's h is computed on the common subset. Writes
runs/pairwise_mcnemar_{iid,ood}.json and prints LaTeX tables.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

from scipy.stats import binomtest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_accepts(path: str) -> Dict[str, int]:
    """Return {trial_key: 0/1} for a run file.

    Accept thresholds: regex_v2_f1 / llm_f1 >= 0.9; smart_f1 >= 1.0 (both
    match the reported n_accept). OOD files repeat each task_id under several
    paraphrases and IID cloud files hold a few duplicate task_ids, so trials
    are keyed by (task_id, task_desc) plus an occurrence index.
    """
    with open(path) as f:
        data = json.load(f)
    rows = data["rows"]
    out: Dict[str, int] = {}
    seen: Dict[Tuple[str, str], int] = {}
    for r in rows:
        tid = r["task_id"]
        desc = r.get("task_desc", "")
        base = (tid, desc)
        n = seen.get(base, 0)
        seen[base] = n + 1
        key = f"{tid}||{desc}||{n}"
        if "regex_v2_f1" in r:
            acc = 1 if r["regex_v2_f1"] >= 0.9 else 0
        elif "llm_f1" in r:
            acc = 1 if r["llm_f1"] >= 0.9 else 0
        elif "smart_f1" in r:
            acc = 1 if r["smart_f1"] >= 1.0 else 0
        else:
            raise ValueError(f"unknown f1 field in row of {path}: keys={list(r.keys())}")
        out[key] = acc
    return out


def cohens_h(p1: float, p2: float) -> float:
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def classify_h(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    res = binomtest(b, n, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def pairwise(systems: Dict[str, Dict[str, int]]) -> Dict[str, dict]:
    names = list(systems.keys())
    out: Dict[str, dict] = {}
    for i, a in enumerate(names):
        for j, bname in enumerate(names):
            if j <= i:
                continue
            A, B = systems[a], systems[bname]
            common = sorted(set(A) & set(B))
            n = len(common)
            b = sum(1 for t in common if A[t] == 1 and B[t] == 0)
            c = sum(1 for t in common if A[t] == 0 and B[t] == 1)
            p1 = sum(A[t] for t in common) / n if n else 0.0
            p2 = sum(B[t] for t in common) / n if n else 0.0
            p_val = mcnemar_exact(b, c)
            h = cohens_h(p1, p2)
            out[f"{a}__vs__{bname}"] = {
                "system_a": a,
                "system_b": bname,
                "n_common": n,
                "accept_a": int(round(p1 * n)),
                "accept_b": int(round(p2 * n)),
                "p_a": p1,
                "p_b": p2,
                "b_only_a": b,
                "c_only_b": c,
                "p_value": p_val,
                "cohens_h": h,
                "effect_class": classify_h(h),
                "sig": sig_stars(p_val),
            }
    return out


def latex_table(systems: Dict[str, Dict[str, int]],
                pair_results: Dict[str, dict],
                caption: str,
                label: str) -> str:
    names = list(systems.keys())
    lookup = {}
    for k, v in pair_results.items():
        lookup[(v["system_a"], v["system_b"])] = v
        lookup[(v["system_b"], v["system_a"])] = v

    header_cols = " & ".join(names)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    col_spec = "l" + "c" * len(names)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & " + header_cols + r" \\")
    lines.append(r"\midrule")
    rates = []
    for n in names:
        A = systems[n]
        p = sum(A.values()) / len(A) if A else 0.0
        rates.append(f"{p*100:.1f}\\%")
    lines.append(r"\textit{accept} & " + " & ".join(rates) + r" \\")
    lines.append(r"\midrule")

    for i, row_name in enumerate(names):
        cells = []
        for j, col_name in enumerate(names):
            if i == j:
                cells.append("-")
            else:
                v = lookup[(row_name, col_name)]
                h = v["cohens_h"]
                # if row -> col reversed from stored a/b, flip sign of h
                if v["system_a"] != row_name:
                    h = -h
                stars = v["sig"]
                cells.append(f"{h:+.2f}{stars}")
        lines.append(row_name + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + r" Cells show Cohen's $h$ (row $-$ column) with McNemar exact-test significance: "
                 r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. Top row is per-system accept rate on the union of evaluated tasks.}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> None:
    # IID systems in scale-curve order (weakest to strongest)
    iid_files: List[Tuple[str, str]] = [
        ("regex_v2",  "runs/alfworld_regex_v2_full.json"),
        ("qwen1.5b",  "runs/alfworld_fair_eval_full.json"),
        ("gpt3.5",    "runs/alfworld_cloud_gpt35.json"),
        ("qwen7b",    "runs/alfworld_cloud_together_qwen7b.json"),
        ("llama70b",  "runs/alfworld_cloud_together_llama70b.json"),
        ("mini",      "runs/alfworld_cloud_openai_mini.json"),
        ("haiku",     "runs/alfworld_cloud_claude_haiku.json"),
        ("gpt4o",     "runs/alfworld_cloud_gpt4o.json"),
        ("sonnet",    "runs/alfworld_cloud_claude_sonnet.json"),
    ]

    iid_systems: Dict[str, Dict[str, int]] = {}
    for name, rel in iid_files:
        iid_systems[name] = load_accepts(os.path.join(REPO_ROOT, rel))
        print(f"[IID] {name}: n={len(iid_systems[name])} accept_rate={sum(iid_systems[name].values())/len(iid_systems[name]):.3f}")

    iid_results = pairwise(iid_systems)
    iid_out = os.path.join(REPO_ROOT, "runs/pairwise_mcnemar_iid.json")
    with open(iid_out, "w") as f:
        json.dump({"systems": list(iid_systems.keys()),
                   "n_per_system": {k: len(v) for k, v in iid_systems.items()},
                   "accept_rate": {k: sum(v.values()) / len(v) for k, v in iid_systems.items()},
                   "pairs": iid_results}, f, indent=2)
    print(f"\nwrote {iid_out}")

    # OOD systems
    ood_files: List[Tuple[str, str]] = [
        ("regex_v2", "runs/ood/alfworld_regex_v2_unseen.json"),
        ("gpt3.5",   "runs/ood/alfworld_cloud_gpt35.json"),
        ("qwen7b",   "runs/ood/alfworld_cloud_together_qwen7b.json"),
        ("mini",     "runs/ood/alfworld_cloud_openai_mini.json"),
        ("haiku",    "runs/ood/alfworld_cloud_claude_haiku.json"),
        ("gpt4o",    "runs/ood/alfworld_cloud_gpt4o.json"),
        ("sonnet",   "runs/ood/alfworld_cloud_claude_sonnet.json"),
    ]
    ood_systems: Dict[str, Dict[str, int]] = {}
    for name, rel in ood_files:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            print(f"[OOD] skip missing: {rel}")
            continue
        ood_systems[name] = load_accepts(path)
        print(f"[OOD] {name}: n={len(ood_systems[name])} accept_rate={sum(ood_systems[name].values())/len(ood_systems[name]):.3f}")

    ood_results = pairwise(ood_systems)
    ood_out = os.path.join(REPO_ROOT, "runs/pairwise_mcnemar_ood.json")
    with open(ood_out, "w") as f:
        json.dump({"systems": list(ood_systems.keys()),
                   "n_per_system": {k: len(v) for k, v in ood_systems.items()},
                   "accept_rate": {k: sum(v.values()) / len(v) for k, v in ood_systems.items()},
                   "pairs": ood_results}, f, indent=2)
    print(f"wrote {ood_out}")

    print("\n% ====== IID LaTeX table ======")
    print(latex_table(iid_systems, iid_results,
                      caption="ALFWorld IID (seen, $N{=}251$): pairwise McNemar and Cohen's $h$.",
                      label="tab:alfworld_pairwise_iid"))
    print("\n% ====== OOD LaTeX table ======")
    print(latex_table(ood_systems, ood_results,
                      caption="ALFWorld OOD (unseen, $N{=}251$): pairwise McNemar and Cohen's $h$.",
                      label="tab:alfworld_pairwise_ood"))

    def adjacent(systems_dict: Dict[str, Dict[str, int]],
                 results: Dict[str, dict],
                 label: str) -> None:
        names = list(systems_dict.keys())
        print(f"\n% ====== Adjacent-rung separation ({label}) ======")
        for a, b in zip(names, names[1:]):
            key = f"{a}__vs__{b}"
            if key not in results:
                key = f"{b}__vs__{a}"
            r = results[key]
            star = r["sig"] or "ns"
            print(f"  {a:>10s}  vs  {b:<10s}  n={r['n_common']:3d}  "
                  f"b={r['b_only_a']:3d}  c={r['c_only_b']:3d}  "
                  f"p={r['p_value']:.4g} [{star:>3s}]  h={r['cohens_h']:+.3f} ({r['effect_class']})")

    adjacent(iid_systems, iid_results, "IID")
    adjacent(ood_systems, ood_results, "OOD")

    def extremes(results: Dict[str, dict], label: str) -> None:
        pairs = list(results.values())
        min_p = min(pairs, key=lambda v: v["p_value"])
        max_h = max(pairs, key=lambda v: abs(v["cohens_h"]))
        print(f"\n[{label}] smallest p: {min_p['system_a']} vs {min_p['system_b']} "
              f"p={min_p['p_value']:.4g} h={min_p['cohens_h']:+.3f}")
        print(f"[{label}] largest |h|: {max_h['system_a']} vs {max_h['system_b']} "
              f"h={max_h['cohens_h']:+.3f} p={max_h['p_value']:.4g}")

    extremes(iid_results, "IID")
    extremes(ood_results, "OOD")


if __name__ == "__main__":
    main()
