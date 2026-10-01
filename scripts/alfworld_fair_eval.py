"""Single-shot ALFWorld NL-goal parse with no candidate re-ranking.

One greedy decode per task with the enum-constrained prompt and schema,
scored against GT with extended smart F1; regex also gets one shot.
Output: runs/alfworld_fair_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.alfworld_nl_parse import (
    sample_tasks, _gt_from_pddl_params, regex_parse,
)
from scripts.alfworld_enum_constrained import (
    make_enum_prompt, build_enum_schema, smart_f1_extended,
)


def call_greedy(prompt: str, schema: str, model_name: str) -> dict:
    """Run one greedy constrained decode and parse its JSON."""
    from agplan.decoding.xgrammar_wrapper import (
        DecodeConfig, XGrammarConstrainedDecoder,
    )
    dec = XGrammarConstrainedDecoder(model_name=model_name, schema=schema)
    cfg = DecodeConfig(do_sample=False, max_new_tokens=256)
    raw = dec.generate(
        user_prompt=prompt,
        system_prompt="You are a careful, precise assistant. Return only valid JSON matching the schema.",
        cfg=cfg)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld_data",
                    default=os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"))
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/alfworld_fair_eval.json")
    args = ap.parse_args()

    tasks = sample_tasks(args.alfworld_data, args.n_per_type, 0)
    schema = build_enum_schema()
    print(f"FAIR EVAL: {len(tasks)} tasks, greedy-only, no rerank")

    rows = []
    n_regex = 0
    n_llm = 0
    f_regex = 0
    f_llm = 0
    for t in tasks:
        gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
        nl = t["task_desc"]
        regex_pred = regex_parse(nl)
        rs = smart_f1_extended(regex_pred, gt)
        if rs >= args.threshold: n_regex += 1
        f_regex += rs

        prompt = make_enum_prompt(nl)
        llm_pred = call_greedy(prompt, schema, args.model)
        ls = smart_f1_extended(llm_pred, gt)
        if ls >= args.threshold: n_llm += 1
        f_llm += ls

        rows.append({
            "task_id": t["task_id"], "task_type": t["task_type"],
            "task_desc": nl, "gt": gt,
            "regex_pred": regex_pred, "regex_f1": rs,
            "llm_pred": llm_pred, "llm_f1": ls,
        })
        print(f"  [{t['task_type'][:22]:<22}] r={rs:.2f} l={ls:.2f} :: {nl[:55]}")

    n = len(rows)
    print(f"\n=== FAIR EVAL (greedy, no rerank, smart F1) ===")
    print(f"  REGEX: {n_regex}/{n} ({100*n_regex/n:.0f}%) mean_F1={f_regex/n:.3f}")
    print(f"  LLM:   {n_llm}/{n} ({100*n_llm/n:.0f}%) mean_F1={f_llm/n:.3f}")
    print(f"  Gap: {n_llm - n_regex:+d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "method": "greedy-only-no-rerank",
            "n_total": n,
            "n_regex_accept": n_regex,
            "n_llm_accept": n_llm,
            "regex_mean_f1": f_regex / n,
            "llm_mean_f1": f_llm / n,
            "threshold": args.threshold,
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
