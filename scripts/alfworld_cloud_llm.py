"""Parse the 35-task ALFWorld NL-goal set with hosted LLMs (OpenAI, Anthropic, HF, Together).

Same tasks (5 per task type x 7 types) and alias-normalized smart F1 as
alfworld_smart_score.py; each provider needs its API-key env var.
Output: runs/alfworld_cloud_<provider>.json.

Usage: python -m scripts.alfworld_cloud_llm --providers openai_mini claude_haiku gpt4o
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.alfworld_nl_parse import (
    sample_tasks, _gt_from_pddl_params, regex_parse,
)
from scripts.alfworld_enum_constrained import (
    make_enum_prompt, smart_f1_extended,
)


PROVIDERS = {
    "openai_mini": {"vendor": "openai", "model": "gpt-4o-mini"},
    "gpt4o":       {"vendor": "openai", "model": "gpt-4o"},
    "gpt35":       {"vendor": "openai", "model": "gpt-3.5-turbo"},
    "claude_haiku":{"vendor": "anthropic", "model": "claude-haiku-4-5"},
    "claude_sonnet":{"vendor": "anthropic", "model": "claude-sonnet-4-6"},
    "claude_opus": {"vendor": "anthropic", "model": "claude-opus-4-7"},
    "hf_qwen7b":   {"vendor": "hf", "model": "Qwen/Qwen2.5-7B-Instruct"},
    "hf_llama8b":  {"vendor": "hf", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    "together_qwen7b":  {"vendor": "together", "model": "Qwen/Qwen2.5-7B-Instruct-Turbo"},
    "together_llama70b":{"vendor": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
}


def call_openai(prompt: str, model: str, schema: dict) -> dict:
    from openai import OpenAI
    client = OpenAI()
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You parse ALFWorld task descriptions to "
                         "structured JSON. Output ONLY valid JSON, "
                         "no commentary."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3, max_tokens=400,
    )
    try:
        return json.loads(rsp.choices[0].message.content)
    except Exception:
        return {}


def call_anthropic(prompt: str, model: str, schema: dict) -> dict:
    import anthropic
    import time as _time
    client = anthropic.Anthropic()
    # Exponential backoff on rate limits.
    for attempt in range(6):
        try:
            rsp = client.messages.create(
                model=model, max_tokens=400, temperature=0.3,
                system="You parse ALFWorld task descriptions to structured "
                       "JSON. Output ONLY valid JSON, no commentary, no "
                       "markdown fences.",
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.RateLimitError:
            _time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16, 32 sec
        except Exception:
            return {}
    else:
        return {}
    text = rsp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"): text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        return {}


def call_hf(prompt: str, model: str, schema: dict) -> dict:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ.get("HF_TOKEN"))
    resp = client.chat_completion(
        model=model,
        messages=[
            {"role": "system",
             "content": "You parse ALFWorld task descriptions to "
                         "structured JSON. Output ONLY valid JSON, "
                         "no commentary, no markdown fences."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400, temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"): text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        return {}


def call_cloud(provider_key: str, prompt: str) -> dict:
    p = PROVIDERS[provider_key]
    if p["vendor"] == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return {"_error": "no OPENAI_API_KEY"}
        return call_openai(prompt, p["model"], None)
    if p["vendor"] == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {"_error": "no ANTHROPIC_API_KEY"}
        return call_anthropic(prompt, p["model"], None)
    if p["vendor"] == "hf":
        if not os.environ.get("HF_TOKEN"):
            return {"_error": "no HF_TOKEN"}
        return call_hf(prompt, p["model"], None)
    if p["vendor"] == "together":
        if not os.environ.get("TOGETHER_API_KEY"):
            return {"_error": "no TOGETHER_API_KEY"}
        return call_together(prompt, p["model"], None)
    return {"_error": f"unknown vendor {p['vendor']}"}


def call_together(prompt: str, model: str, schema: dict) -> dict:
    from together import Together
    client = Together()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You parse ALFWorld task descriptions to "
                         "structured JSON. Output ONLY valid JSON, "
                         "no commentary, no markdown fences."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400, temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"): text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld_data",
                    default=os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"))
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--providers", nargs="+",
                    default=["openai_mini", "claude_haiku"])
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    tasks = sample_tasks(args.alfworld_data, args.n_per_type, 0)
    print(f"Running {len(tasks)} ALFWorld tasks across "
          f"{len(args.providers)} cloud LLMs")

    # Regex baseline for reference
    regex_acc = 0
    regex_sum = 0
    for t in tasks:
        gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
        rs = smart_f1_extended(regex_parse(t["task_desc"]), gt)
        if rs >= args.threshold: regex_acc += 1
        regex_sum += rs
    print(f"REGEX baseline: {regex_acc}/{len(tasks)} "
          f"({100*regex_acc/len(tasks):.0f}%)  mean_F1={regex_sum/len(tasks):.3f}\n")

    for prov in args.providers:
        if prov not in PROVIDERS:
            print(f"Unknown provider: {prov} — skipping")
            continue
        print(f"=== {prov} ({PROVIDERS[prov]['model']}) ===")
        rows = []
        n_accept = 0
        f1_sum = 0
        t0 = time.time()
        for i, t in enumerate(tasks):
            gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
            nl = t["task_desc"]
            prompt = make_enum_prompt(nl)
            llm_pred = call_cloud(prov, prompt)
            sf1 = smart_f1_extended(
                llm_pred if isinstance(llm_pred, dict) else {}, gt)
            if sf1 >= args.threshold: n_accept += 1
            f1_sum += sf1
            rows.append({
                "task_id": t["task_id"], "task_type": t["task_type"],
                "task_desc": nl, "gt": gt,
                "llm_pred": llm_pred, "smart_f1": sf1,
            })
            if (i+1) % 5 == 0 or i == len(tasks) - 1:
                print(f"  {i+1}/{len(tasks)}  "
                      f"cumulative {n_accept}/{i+1}  "
                      f"({(time.time()-t0):.0f}s elapsed)")

        dt = time.time() - t0
        print(f"  {prov}: {n_accept}/{len(tasks)} = "
              f"{100*n_accept/len(tasks):.0f}%  "
              f"mean_F1={f1_sum/len(tasks):.3f}  ({dt:.0f}s)")
        print(f"  vs regex 9/35 = 26%; gap: {n_accept - 9:+d}\n")

        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{args.out_dir}/alfworld_cloud_{prov}.json", "w") as f:
            json.dump({
                "provider": prov,
                "model": PROVIDERS[prov]["model"],
                "n_total": len(tasks),
                "n_accept": n_accept,
                "mean_f1": f1_sum / len(tasks),
                "wall_seconds": dt,
                "rows": rows,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
