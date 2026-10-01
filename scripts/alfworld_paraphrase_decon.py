"""Paraphrase probe for the ALFWorld OOD scale curve (goal-string contamination control).

Each valid_unseen task description (runs/ood/alfworld_cloud_gpt4o.json) is
rewritten by Llama-3.3-70B under two conditions (struct: new syntax, same
words; lex: synonyms allowed), then re-parsed by the same regex and LLM
systems and scored with smart F1. Outputs: runs/alfworld_paraphrase_decon_*.json.

Usage: python3 scripts/alfworld_paraphrase_decon.py --stage gen|eval --systems regex gpt4o claude_sonnet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.alfworld_enum_constrained import (  # noqa: E402
    make_enum_prompt, smart_f1_extended,
)
from scripts.alfworld_cloud_llm import (  # noqa: E402
    call_openai, call_anthropic, call_together,
)
from scripts.alfworld_regex_v2 import regex_parse_v2  # noqa: E402

PARAPHRASER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
PARAPHRASE_TEMPERATURE = 0.7
SOURCE_JSON = "runs/ood/alfworld_cloud_gpt4o.json"
PARAS_JSON = "runs/alfworld_paraphrase_decon_paras.json"
OUT_TMPL = "runs/alfworld_paraphrase_decon_{cond}_{sys}.json"

GEN_PROMPTS = {
    "struct": (
        "Rewrite the following household-task instruction using a "
        "clearly different sentence structure (change the voice, clause "
        "order, or sentence form). Keep every object and location "
        "mentioned with exactly the same words. Do not add, remove, or "
        "reinterpret any information. Reply with ONLY the rewritten "
        "instruction.\n\nInstruction: {nl}"
    ),
    "lex": (
        "Rewrite the following household-task instruction in your own "
        "words: change the sentence structure AND use natural everyday "
        "synonyms where they fit, while keeping the meaning identical "
        "(the same objects, places, and required actions). Do not add "
        "or remove information. Reply with ONLY the rewritten "
        "instruction.\n\nInstruction: {nl}"
    ),
}

EVAL_SYSTEMS = {
    # name -> (kind, model)
    "regex": ("regex", None),
    "gpt4o": ("openai", "gpt-4o"),
    "claude_sonnet": ("anthropic", "claude-sonnet-4-6"),
    "together_qwen7b": ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
}


def _call_paraphraser(nl: str, cond: str) -> str:
    from together import Together
    client = Together()
    rsp = client.chat.completions.create(
        model=PARAPHRASER_MODEL,
        messages=[{"role": "user",
                   "content": GEN_PROMPTS[cond].format(nl=nl)}],
        max_tokens=200, temperature=PARAPHRASE_TEMPERATURE,
    )
    text = (rsp.choices[0].message.content or "").strip()
    # Strip quotes/fences the model sometimes adds; keep the first line only.
    text = text.strip('"').strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
    text = text.splitlines()[0].strip() if text else ""
    return text


def _guard_ok(orig: str, para: str) -> bool:
    if not para:
        return False
    if para.casefold().strip(". ") == orig.casefold().strip(". "):
        return False
    if not (0.4 * len(orig) <= len(para) <= 4.0 * len(orig)):
        return False
    return True


def stage_gen(args) -> None:
    src = json.load(open(SOURCE_JSON))["rows"]
    if args.n_limit:
        src = src[: args.n_limit]
    rows = []
    t0 = time.time()
    for i, r in enumerate(src):
        row = {
            "task_id": r["task_id"], "task_type": r["task_type"],
            "task_desc": r["task_desc"], "gt": r["gt"],
        }
        for cond in ("struct", "lex"):
            para, usable = "", False
            for _attempt in range(2):
                try:
                    para = _call_paraphraser(r["task_desc"], cond)
                except Exception as e:  # noqa: BLE001
                    para = ""
                    print(f"  gen error {r['task_id']}: {e}")
                if _guard_ok(r["task_desc"], para):
                    usable = True
                    break
            row[f"para_{cond}"] = para
            row[f"usable_{cond}"] = usable
        rows.append(row)
        if (i + 1) % 25 == 0 or i == len(src) - 1:
            n_ok_s = sum(1 for x in rows if x["usable_struct"])
            n_ok_l = sum(1 for x in rows if x["usable_lex"])
            print(f"  gen {i+1}/{len(src)}  usable struct={n_ok_s} "
                  f"lex={n_ok_l}  ({time.time()-t0:.0f}s)")
    out = {
        "paraphraser_model": PARAPHRASER_MODEL,
        "paraphrase_temperature": PARAPHRASE_TEMPERATURE,
        "gen_prompts": GEN_PROMPTS,
        "source_json": SOURCE_JSON,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_total": len(rows),
        "n_usable_struct": sum(1 for x in rows if x["usable_struct"]),
        "n_usable_lex": sum(1 for x in rows if x["usable_lex"]),
        "rows": rows,
    }
    Path(PARAS_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(PARAS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote: {PARAS_JSON}  "
          f"(struct {out['n_usable_struct']}/{out['n_total']}, "
          f"lex {out['n_usable_lex']}/{out['n_total']})")


def _eval_one(system: str, nl: str) -> dict:
    kind, model = EVAL_SYSTEMS[system]
    if kind == "regex":
        return regex_parse_v2(nl)
    prompt = make_enum_prompt(nl)
    if kind == "openai":
        return call_openai(prompt, model, None)
    if kind == "anthropic":
        pred = call_anthropic(prompt, model, None)
        time.sleep(1.25)  # stay under the 50 req/min rate limit
        return pred
    if kind == "together":
        return call_together(prompt, model, None)
    raise ValueError(system)


def stage_eval(args) -> None:
    paras = json.load(open(PARAS_JSON))["rows"]
    if args.n_limit:
        paras = paras[: args.n_limit]
    for cond in args.conditions:
        usable = [r for r in paras if r[f"usable_{cond}"]]
        for system in args.systems:
            print(f"=== eval {cond} / {system}  (N={len(usable)}) ===")
            rows, n_accept, f1_sum = [], 0, 0.0
            t0 = time.time()
            for i, r in enumerate(usable):
                nl = r[f"para_{cond}"]
                pred = _eval_one(system, nl)
                sf1 = smart_f1_extended(
                    pred if isinstance(pred, dict) else {}, r["gt"])
                if sf1 >= args.threshold:
                    n_accept += 1
                f1_sum += sf1
                rows.append({
                    "task_id": r["task_id"], "task_type": r["task_type"],
                    "task_desc_orig": r["task_desc"],
                    "task_desc_para": nl,
                    "gt": r["gt"], "pred": pred, "smart_f1": sf1,
                })
                if (i + 1) % 25 == 0 or i == len(usable) - 1:
                    print(f"  {i+1}/{len(usable)}  cum {n_accept}/{i+1} "
                          f"({time.time()-t0:.0f}s)")
            out_path = OUT_TMPL.format(cond=cond, sys=system)
            with open(out_path, "w") as f:
                json.dump({
                    "condition": cond, "system": system,
                    "model": EVAL_SYSTEMS[system][1],
                    "paraphraser_model": PARAPHRASER_MODEL,
                    "threshold": args.threshold,
                    "n_total": len(usable), "n_accept": n_accept,
                    "mean_f1": f1_sum / max(1, len(usable)),
                    "wall_seconds": time.time() - t0,
                    "rows": rows,
                }, f, indent=2, default=str)
            print(f"  {cond}/{system}: {n_accept}/{len(usable)} = "
                  f"{100*n_accept/max(1,len(usable)):.1f}%  "
                  f"mean_F1={f1_sum/max(1,len(usable)):.3f}")
            print(f"  wrote: {out_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "eval", "all"],
                    default="all")
    ap.add_argument("--conditions", nargs="+",
                    default=["struct", "lex"],
                    choices=["struct", "lex"])
    ap.add_argument("--systems", nargs="+",
                    default=["regex", "gpt4o", "claude_sonnet"],
                    choices=list(EVAL_SYSTEMS))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--n_limit", type=int, default=0,
                    help="cap rows for smoke tests")
    args = ap.parse_args()
    if args.stage in ("gen", "all"):
        stage_gen(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)


if __name__ == "__main__":
    main()
