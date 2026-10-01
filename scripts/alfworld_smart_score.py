"""Alias-aware, role-equivalent smart-F1 scoring of ALFWorld NL-goal parses.

Scores by role rather than slot (toggle vs mrecep), canonicalises name
variants (BarOfSoap == SoapBar) and treats XSliced as X + sliced. Reruns the
LLM on the 35-task set and keeps the raw proposal for scoring.
Output: runs/alfworld_smart_score.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.alfworld_nl_parse import (
    sample_tasks, _gt_from_pddl_params, OBJECT_TYPES, ALIAS_PAIRS,
    regex_parse, make_llm_prompt, LLM_SCHEMA,
)


# Alias map: lower-cased name variant -> canonical CamelCase object type.
_ALIAS_MAP = {alias.lower(): canon for alias, canon in ALIAS_PAIRS}
for c in OBJECT_TYPES:
    _ALIAS_MAP[c.lower()] = c
for c in OBJECT_TYPES:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", c).lower()  # "DeskLamp" -> "desk lamp"
    _ALIAS_MAP[spaced] = c
    nospace = c.lower()
    _ALIAS_MAP[nospace] = c
# Two-part compounds also accept the "<B> of <A>" form: "SoapBar" -> "barofsoap".
for c in OBJECT_TYPES:
    parts = re.findall(r"[A-Z][a-z]*", c)
    if len(parts) == 2:
        of_form = (parts[1] + "of" + parts[0]).lower()
        _ALIAS_MAP[of_form] = c
# "X sliced" maps to XSliced; the sliced flag itself is handled in smart_f1.
for c in OBJECT_TYPES:
    if c.endswith("Sliced"):
        base = c[:-len("Sliced")]
        if base in OBJECT_TYPES:
            _ALIAS_MAP[c.lower()] = c
            _ALIAS_MAP[(base + " sliced").lower()] = c


def normalize_object(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    if s in _ALIAS_MAP:
        return _ALIAS_MAP[s]
    s2 = re.sub(r"[^a-z]", "", s)
    if s2 in _ALIAS_MAP:
        return _ALIAS_MAP[s2]
    return name


def normalize_pred(pred: dict) -> dict:
    out = dict(pred)
    for k in ["object_target", "mrecep_target", "parent_target",
                "toggle_target"]:
        if k in out:
            out[k] = normalize_object(out.get(k) or "")
    return out


def _canonical_obj_pair(obj_name: str, sliced: bool) -> tuple[str, bool]:
    """Map (XSliced, sliced=*) and (X, sliced=True) to the same (X, True) pair."""
    if not obj_name:
        return ("", sliced)
    nm = normalize_object(obj_name)
    if nm.endswith("Sliced"):
        return (nm[:-len("Sliced")], True)
    return (nm, sliced)


def smart_f1(pred: dict, gt: dict) -> float:
    """Return F1 over (role, value) facts, with canonical names and sliced equivalence."""
    p_obj, p_sli = _canonical_obj_pair(
        pred.get("object_target", ""), bool(pred.get("object_sliced", False)))
    g_obj, g_sli = _canonical_obj_pair(
        gt.get("object_target", ""), bool(gt.get("object_sliced", False)))

    # A fact is (role, value) for each non-empty slot or True flag.
    # toggle_target and mrecep_target are role-equivalent, so both fold
    # into a single 'aux' fact.
    def to_facts(d, obj_canon, sliced_canon):
        f = set()
        if obj_canon: f.add(("object", obj_canon))
        aux = normalize_object(d.get("toggle_target", "") or "")
        if not aux:
            aux = normalize_object(d.get("mrecep_target", "") or "")
        if aux: f.add(("aux", aux))
        parent = normalize_object(d.get("parent_target", "") or "")
        if parent: f.add(("parent", parent))
        if sliced_canon: f.add(("sliced", True))
        if d.get("object_cool"): f.add(("cool", True))
        if d.get("object_heat"): f.add(("heat", True))
        if d.get("object_clean"): f.add(("clean", True))
        return f

    p_facts = to_facts(pred, p_obj, p_sli)
    g_facts = to_facts(gt, g_obj, g_sli)
    if not p_facts and not g_facts:
        return 1.0
    if not p_facts or not g_facts:
        return 0.0
    tp = len(p_facts & g_facts)
    prec = tp / len(p_facts)
    rec = tp / len(g_facts)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld_data",
                    default=os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"))
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--out", default="runs/alfworld_smart_score.json")
    args = ap.parse_args()

    tasks = sample_tasks(args.alfworld_data, args.n_per_type, 0)
    print(f"Smart-score on {len(tasks)} ALFWorld tasks")

    llm_call = None
    if args.use_llm:
        from agplan.llm_propose import propose_verify_fallback

        def _llm(nl, gt):
            prompt = make_llm_prompt(nl)
            def verifier(prop):
                return smart_f1(prop if isinstance(prop, dict) else {}, gt)
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=LLM_SCHEMA,
                verifier=verifier,
                baseline={k: ("" if k.endswith("_target") else False)
                          for k in ["object_target", "mrecep_target",
                                    "parent_target", "toggle_target",
                                    "object_sliced", "object_cool",
                                    "object_heat", "object_clean"]},
                threshold=args.threshold,
                model_name=args.model,
                max_new_tokens=256, n_samples=4, temperature=0.4,
            )
            # Return the raw proposal: `final` falls back to the baseline
            # on failed runs.
            return info.get("proposal", final), info
        llm_call = _llm

    rows = []
    n_regex_smart = 0
    n_llm_smart = 0
    f1_regex_sum = 0
    f1_llm_sum = 0
    for t in tasks:
        gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
        nl = t["task_desc"]
        regex_pred = regex_parse(nl)
        regex_smart_f1 = smart_f1(regex_pred, gt)

        if llm_call is not None:
            llm_proposal, info = llm_call(nl, gt)
            llm_smart_f1 = smart_f1(
                llm_proposal if isinstance(llm_proposal, dict) else {}, gt)
        else:
            llm_proposal = None
            llm_smart_f1 = 0.0

        if regex_smart_f1 >= args.threshold: n_regex_smart += 1
        if llm_smart_f1 >= args.threshold: n_llm_smart += 1
        f1_regex_sum += regex_smart_f1
        f1_llm_sum += llm_smart_f1
        rows.append({
            "task_id": t["task_id"], "task_type": t["task_type"],
            "task_desc": nl, "gt": gt,
            "regex_pred": regex_pred,
            "regex_smart_f1": regex_smart_f1,
            "llm_proposal": llm_proposal,
            "llm_smart_f1": llm_smart_f1,
        })
        if llm_call is not None:
            print(f"  [{t['task_type'][:22]:<22}] regex={regex_smart_f1:.2f} "
                  f"llm={llm_smart_f1:.2f} :: {nl[:55]}")

    n = len(rows)
    print("\n" + "=" * 78)
    print(f"SMART-SCORE on {n} tasks (threshold F1>={args.threshold})")
    print(f"  REGEX smart: {n_regex_smart}/{n} ({100*n_regex_smart/n:.0f}%)  "
          f"mean_F1={f1_regex_sum/n:.3f}")
    if args.use_llm:
        print(f"  LLM smart:   {n_llm_smart}/{n} ({100*n_llm_smart/n:.0f}%)  "
              f"mean_F1={f1_llm_sum/n:.3f}")
        gap = n_llm_smart - n_regex_smart
        print(f"  LLM-vs-regex gap: {'+' if gap>=0 else ''}{gap}")
    print("=" * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_total": n,
            "n_regex_smart_accept": n_regex_smart,
            "n_llm_smart_accept": n_llm_smart if args.use_llm else None,
            "regex_smart_mean_f1": f1_regex_sum / n,
            "llm_smart_mean_f1": (f1_llm_sum / n) if args.use_llm else None,
            "threshold": args.threshold,
            "model": args.model if args.use_llm else None,
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
