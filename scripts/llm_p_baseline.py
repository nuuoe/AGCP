"""LLM+P baseline (Liu et al., 2023) on Depots.

The LLM is given the domain and the problem's objects, init and goal and
must emit a PDDL problem file, which pyperplan (A* with hAdd) then
solves. Reports solve rate, plan length and timings. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import tempfile
from pathlib import Path


def extract_problem_nl(problem_path: str) -> dict:
    """Read a Depots PDDL problem and return its objects, init and goal blocks as text."""
    text = open(problem_path).read()
    obj_m = re.search(r"\(:objects\s+(.*?)\)\s*\(", text, re.DOTALL)
    objects = obj_m.group(1).strip() if obj_m else ""
    init_m = re.search(r"\(:init\s+(.*?)\)\s*\(", text, re.DOTALL)
    init_block = init_m.group(1).strip() if init_m else ""
    goal_m = re.search(r"\(:goal\s+\((?:and\s+)?(.*?)\)\s*\)\s*\)",
                       text, re.DOTALL)
    goal_block = goal_m.group(1).strip() if goal_m else ""
    return {
        "objects": objects,
        "init": init_block,
        "goal": goal_block,
    }


def build_llm_p_prompt(problem_nl: dict, depots_domain_text: str) -> str:
    """Construct the LLM+P-style prompt."""
    return f"""You are given a Depots planning problem in PDDL.
The Depots domain (provided as reference) defines the action set.

=== DEPOTS DOMAIN ===
{depots_domain_text}

=== PROBLEM TO SOLVE ===
Objects:
{problem_nl['objects']}

Initial state predicates:
{problem_nl['init']}

Goal predicates (must all be true at the end):
{problem_nl['goal']}

Emit ONLY the PDDL problem file that pyperplan should solve.
Use the format:
(define (problem <name>) (:domain depots)
  (:objects ...)
  (:init ...)
  (:goal (and ...))
)
"""


def call_llm(prompt: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct",
              max_new_tokens: int = 1024) -> str:
    """Call a local Qwen model and return the generated text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hasattr(call_llm, "_cache"):
        call_llm._cache = {}
    if model not in call_llm._cache:
        tok = AutoTokenizer.from_pretrained(model)
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=(torch.float16
                                      if torch.cuda.is_available()
                                      else torch.float32),
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
        except Exception:
            m = AutoModelForCausalLM.from_pretrained(model)
        call_llm._cache[model] = (tok, m)
    tok, m = call_llm._cache[model]
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new_tokens,
                       do_sample=False)
    gen = out[0][len(inp.input_ids[0]):]
    return tok.decode(gen, skip_special_tokens=True)


def extract_pddl(text: str) -> str:
    """Extract the first balanced (define ...) block from LLM output."""
    start = text.find("(define")
    if start < 0:
        return ""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "(": depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
        i += 1
    return text[start:]


def try_solve(domain_path: str, problem_pddl_text: str,
                time_limit: float = 30.0) -> dict:
    """Write the problem PDDL to a temp file and solve it with pyperplan."""
    from pyperplan.search import astar_search
    from pyperplan.heuristics.relaxation import hAddHeuristic
    from pyperplan.pddl.parser import Parser
    from pyperplan.grounding import ground

    with tempfile.NamedTemporaryFile(suffix=".pddl", delete=False,
                                       mode="w") as f:
        f.write(problem_pddl_text)
        prob_path = f.name
    try:
        parser = Parser(domain_path, prob_path)
        dom = parser.parse_domain()
        prob = parser.parse_problem(dom)
        task = ground(prob, remove_irrelevant_operators=False)
        t0 = time.time()
        try:
            sol = astar_search(task, heuristic=hAddHeuristic(task))
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - t0}
        elapsed = time.time() - t0
        if sol is None:
            return {"success": False, "error": "no plan", "time": elapsed}
        return {"success": True, "plan_length": len(sol), "time": elapsed}
    except Exception as e:
        return {"success": False, "error": f"parse: {e}", "time": 0}
    finally:
        os.unlink(prob_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    help="Depots GT domain PDDL")
    ap.add_argument("--problems", required=True,
                    help="Glob for Depots problem files")
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print(f"LLM+P BASELINE on Depots, model={args.model}")
    print("=" * 60)

    domain_text = open(args.domain).read()
    test_paths = sorted(glob.glob(args.problems))[:args.n_test]

    results = []
    for pp in test_paths:
        print(f"\n  {Path(pp).name}:")
        nl = extract_problem_nl(pp)
        prompt = build_llm_p_prompt(nl, domain_text)
        t0 = time.time()
        try:
            gen = call_llm(prompt, model=args.model)
        except Exception as e:
            print(f"    LLM call error: {e}")
            results.append({"problem": Path(pp).name,
                             "llm_success": False, "error": str(e)})
            continue
        gen_time = time.time() - t0
        problem_pddl = extract_pddl(gen)
        if not problem_pddl:
            print(f"    LLM emitted no PDDL block")
            results.append({"problem": Path(pp).name,
                             "llm_success": False,
                             "error": "no pddl",
                             "llm_time": gen_time})
            continue
        solve_res = try_solve(args.domain, problem_pddl)
        success = solve_res.get("success", False)
        results.append({
            "problem": Path(pp).name,
            "llm_success": success,
            "llm_time": gen_time,
            "plan_length": solve_res.get("plan_length"),
            "solve_time": solve_res.get("time"),
            "error": solve_res.get("error"),
        })
        flag = "OK" if success else f"FAIL: {solve_res.get('error','?')}"
        print(f"    LLM gen {gen_time:.1f}s; {flag}; "
              f"plan_len={solve_res.get('plan_length')}")

    n_solved = sum(1 for r in results if r.get("llm_success"))
    print("\n" + "=" * 60)
    print(f"LLM+P: {n_solved}/{len(results)} = "
          f"{100*n_solved/max(len(results),1):.1f}% solved")
    print("=" * 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "n_solved": n_solved,
            "n_total": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
