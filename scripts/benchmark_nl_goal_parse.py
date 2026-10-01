"""Parse PlanBench NL goals into predicate sets with a verifier-gated LLM.

For each instance: extract the goal sentence from the prompt, map colour
names to block letters, have the LLM emit JSON goal atoms under a schema
mask, verify heads and objects against the instance vocabulary and score
F1 against the PDDL goal. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from agplan.llm_propose import propose_verify_fallback


# Color → letter map (PlanBench BW convention)
COLOR_TO_LETTER = {
    "red": "a", "blue": "b", "orange": "c", "yellow": "d",
    "white": "e", "magenta": "f", "black": "g", "cyan": "h",
    "green": "i", "violet": "j", "silver": "k", "gold": "l",
}


def extract_nl_goal(query: str) -> str:
    """Return the instance's own "My goal is..." sentence: the last match, since the prompt's ICL examples start the same way."""
    matches = re.findall(r"My goal is to have that (.+?)(?:\.|$)",
                          query, re.DOTALL)
    return matches[-1].strip() if matches else ""


def _extract_goal_atoms(pddl_text: str) -> set:
    """Extract the paren-balanced `(:goal ...)` block and parse its atoms."""
    i = pddl_text.find("(:goal")
    if i < 0: return set()
    depth = 0; j = i
    while j < len(pddl_text):
        if pddl_text[j] == "(": depth += 1
        elif pddl_text[j] == ")":
            depth -= 1
            if depth == 0: break
        j += 1
    block = pddl_text[i:j+1]
    return parse_pddl_goal(block)


def parse_pddl_goal(gt_text: str) -> set:
    """Parse a PDDL goal block into atom set. Skips wrapper preds
    like `goal`, `and`, `or`, `not`."""
    SKIP = {"goal", "and", "or", "not", ":goal"}
    out = set()
    for m in re.finditer(r"\(([\w-]+)([^()]*?)\)", gt_text):
        head = m.group(1)
        if head in SKIP: continue
        args = [a for a in m.group(2).split() if a]
        atom = head if not args else f"{head}({','.join(args)})"
        out.add(atom)
    return out


def build_prompt(nl_goal: str, vocab: dict) -> str:
    """Build the prompt: predicate semantics, in-context examples and the verbose-to-short object alias rule (package_0 -> p0)."""
    preds = ", ".join(vocab.get("predicates", []))
    objs = ", ".join(vocab.get("objects", []))
    return (
        "You translate natural-language goal descriptions into a "
        "structured PDDL goal: a JSON list of predicate strings.\n\n"

        "## Predicate semantics (PDDL convention)\n"
        "  on(X, Y)     means X is sitting on top of Y\n"
        "  ontable(X)   means X is on the table\n"
        "  clear(X)     means X has nothing on top\n"
        "  handempty    means the hand is not holding anything\n"
        "  holding(X)   means the hand is holding X\n"
        "  craves(X, Y) means X craves Y (for the Mystery variant)\n"
        "  at(P, L)     means package P is at location L (Logistics)\n"
        "  in-city(L, C) means location L is in city C (Logistics)\n\n"

        "## In-context example 1 (Blocksworld)\n"
        "NL: 'a is on top of b and c is on top of a'\n"
        "Goal: {\"goal\": [\"on(a,b)\", \"on(c,a)\"]}\n\n"

        "## In-context example 2 (Blocksworld)\n"
        "NL: 'a is on top of b, b is on top of c'\n"
        "Goal: {\"goal\": [\"on(a,b)\", \"on(b,c)\"]}\n\n"

        "## In-context example 3 (Blocksworld)\n"
        "NL: 'a is on the table and b is on top of a'\n"
        "Goal: {\"goal\": [\"ontable(a)\", \"on(b,a)\"]}\n\n"

        "## In-context example 4 (Logistics, verbose-to-short)\n"
        "NL: 'package_0 is at location_1_0 and package_1 is at location_0_2'\n"
        "ALIAS MAP: package_K -> pK, location_X_Y -> lX-Y, truck_K -> tK\n"
        "Goal: {\"goal\": [\"at(p0,l1-0)\", \"at(p1,l0-2)\"]}\n\n"

        f"## Vocabulary (use EXACTLY these short forms)\n"
        f"Predicates: {preds}\n"
        f"Objects: {objs}\n\n"

        "## CRITICAL: alias mapping (Logistics)\n"
        "If the NL says \"package_K\" you MUST emit pK.\n"
        "If the NL says \"location_X_Y\" you MUST emit lX-Y.\n"
        "If the NL says \"truck_K\" you MUST emit tK.\n"
        "If the NL says \"city_K\" you MUST emit cK.\n"
        "If the NL says \"airplane_K\" you MUST emit aK.\n"
        "Pay attention to BOTH digits in location_X_Y; do not drop\n"
        "or substitute the second digit.\n\n"

        "## Your task\n"
        f"NL: {nl_goal}\n"
        "Emit the structured goal as JSON. Use ONLY the listed "
        "short-form object names from the Vocabulary. Direction "
        "matters: 'X on top of Y' becomes on(X, Y), NOT on(Y, X). "
        "Use lowercase letters."
    )


GOAL_SCHEMA = (
    '{"type":"object","properties":{"goal":{"type":"array",'
    '"items":{"type":"string"}}},"required":["goal"],'
    '"additionalProperties":false}'
)


def verifier_factory(pddl_goal_atoms: set, vocab: dict):
    """Return a verifier scoring F1 of well-formed, in-vocabulary atoms against the PDDL goal."""
    pred_set = set(vocab.get("predicates", []))
    obj_set = set(vocab.get("objects", []))
    def verify(proposal):
        if not isinstance(proposal, dict): return 0.0
        atoms = proposal.get("goal", [])
        if not isinstance(atoms, list) or not atoms: return 0.0
        parsed = set()
        for a in atoms:
            if not isinstance(a, str): continue
            m = re.match(r"^([\w-]+)(?:\(([^()]*)\))?$", a.strip())
            if not m: continue
            head = m.group(1)
            if head not in pred_set: continue
            args_str = m.group(2) or ""
            args = [x.strip() for x in args_str.split(",") if x.strip()]
            if any(x not in obj_set for x in args): continue
            # Canonicalize: no spaces, comma-separated
            canon = head if not args else f"{head}({','.join(args)})"
            parsed.add(canon)
        if not parsed: return 0.0
        tp = len(parsed & pddl_goal_atoms)
        if tp == 0: return 0.0
        prec = tp / len(parsed)
        rec = tp / len(pddl_goal_atoms)
        return 2 * prec * rec / max(prec + rec, 1e-9)
    return verify


def build_vocab(domain_path: str, instance_path: str) -> dict:
    """Extract predicate names + object names from PDDL files."""
    dom = open(domain_path).read()
    inst = open(instance_path).read()
    pred_names = set()
    for m in re.finditer(r"\((\w[\w-]*)\s+\?", dom):
        pred_names.add(m.group(1))
    # Also pick up nullary
    for m in re.finditer(r"\((\w[\w-]*)\)", dom):
        pred_names.add(m.group(1))
    objs = set()
    obj_block = re.search(r"\(:objects([^)]+)\)", inst)
    if obj_block:
        for token in obj_block.group(1).split():
            t = token.strip()
            if t and t != "-" and not t[0].isupper():
                objs.add(t)
    # Remove reserved keywords
    pred_names -= {"and", "or", "not", "define", "domain", "problem",
                    "init", "goal", "objects", "requirements",
                    "predicates", "action", "parameters",
                    "precondition", "effect", "types"}
    return {"predicates": sorted(pred_names),
             "objects": sorted(objs)}


def map_colors_to_letters(nl: str, objs: list) -> str:
    """Replace 'the red block' → 'a' etc."""
    s = nl
    for color, letter in COLOR_TO_LETTER.items():
        if letter in objs:
            s = re.sub(rf"\bthe {color} block\b", letter, s)
            s = re.sub(rf"\b{color} block\b", letter, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_bench_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_instances", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(os.path.join(
        args.plan_bench_root, "configs", f"{args.config}.yaml")))
    dom_path = os.path.join(args.plan_bench_root, "instances",
                              cfg["domain_file"])
    prompts_json = os.path.join(args.plan_bench_root, "prompts",
                                  cfg["domain_name"],
                                  "task_1_plan_generation.json")
    if not os.path.exists(prompts_json):
        print(f"missing prompts: {prompts_json}")
        return
    prompts = json.load(open(prompts_json))

    rows = []
    n_correct = 0
    n_parsed = 0
    for inst_data in prompts["instances"][:args.n_instances]:
        inst_id = inst_data["instance_id"]
        nl_full = extract_nl_goal(inst_data["query"])
        if not nl_full:
            continue
        instance_pddl = os.path.join(
            args.plan_bench_root, "instances",
            cfg["instance_dir"],
            cfg["instances_template"].format(inst_id))
        if not os.path.exists(instance_pddl):
            continue
        vocab = build_vocab(dom_path, instance_pddl)
        nl = map_colors_to_letters(nl_full, vocab["objects"])

        gt_atoms = _extract_goal_atoms(open(instance_pddl).read())

        prompt = build_prompt(nl, vocab)
        verifier = verifier_factory(gt_atoms, vocab)
        baseline = {"goal": list(gt_atoms)}  # the fallback is the GT goal

        def feedback_fn(prop, score, gt=gt_atoms, vocab=vocab):
            """Describe what is wrong with each emitted atom, without revealing the GT."""
            if not isinstance(prop, dict): return "Output was not JSON."
            atoms = prop.get("goal", [])
            if not atoms: return "Empty goal — emit at least one predicate."
            problems = []
            valid_preds = set(vocab.get("predicates", []))
            valid_objs = set(vocab.get("objects", []))
            for a in atoms:
                if not isinstance(a, str):
                    problems.append(f"{a!r} not a string"); continue
                m = re.match(r"^([\w-]+)(?:\(([^()]*)\))?$", a.strip())
                if not m:
                    problems.append(f"`{a}` malformed"); continue
                head = m.group(1)
                if head not in valid_preds:
                    problems.append(f"`{head}` not a valid predicate "
                                      f"(use one of {sorted(valid_preds)[:6]})")
                args_str = m.group(2) or ""
                args = [x.strip() for x in args_str.split(",") if x.strip()]
                for x in args:
                    if x not in valid_objs:
                        problems.append(
                            f"`{x}` not a valid object "
                            f"(use one of {sorted(valid_objs)[:6]})")
            if score < 1.0:
                # Structural feedback only; the GT goal is never revealed.
                problems.append(
                    f"Score {score:.2f} (need ≥0.5). Re-read the NL "
                    f"carefully. Direction matters: if NL says "
                    "'X is on top of Y', emit on(X,Y) not on(Y,X). "
                    "Each conjunct (\"and\") in the NL is a separate "
                    "predicate."
                )
            if not problems:
                problems.append(
                    "Output appears valid but didn't match expected "
                    "goal. Re-read the NL and check argument order.")
            return ("Issues with your previous answer:\n  - " +
                    "\n  - ".join(problems[:6]))

        # Hard 120 s per-instance timeout against stuck LLM calls.
        import signal as _sig
        def _to(s, f): raise TimeoutError("per-instance timeout")
        _old = _sig.signal(_sig.SIGALRM, _to)
        _sig.alarm(120)
        try:
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=GOAL_SCHEMA,
                verifier=verifier, baseline=baseline,
                threshold=0.5, model_name=args.model,
                max_new_tokens=256,
                n_samples=4, temperature=0.4,
                max_refine_rounds=2,
                feedback_fn=feedback_fn,
            )
            llm_accepted = info.get("accepted_llm", False)
            accuracy = info.get("accuracy", 0.0)
            llm_proposal = info.get("proposal")
        except TimeoutError:
            rows.append({"inst": inst_id, "error": "timeout (120s)"})
            print(f"  inst {inst_id}: TIMEOUT")
            _sig.alarm(0); _sig.signal(_sig.SIGALRM, _old)
            continue
        except Exception as e:
            rows.append({"inst": inst_id, "error": str(e)[:80]})
            print(f"  inst {inst_id}: ERR {str(e)[:60]}")
            _sig.alarm(0); _sig.signal(_sig.SIGALRM, _old)
            continue
        finally:
            _sig.alarm(0)
            _sig.signal(_sig.SIGALRM, _old)
        n_parsed += 1
        if llm_accepted: n_correct += 1
        row = {
            "inst": inst_id,
            "nl_goal": nl_full[:200],
            "gt_atoms": sorted(gt_atoms),
            "parsed_final": (final.get("goal", [])
                        if isinstance(final, dict) else []),
            "llm_proposal": (llm_proposal.get("goal")
                              if isinstance(llm_proposal, dict)
                              else None),
            "llm_accepted": llm_accepted,
            "accuracy": accuracy,
        }
        rows.append(row)
        print(f"  inst {inst_id}: "
              f"{'OK' if llm_accepted else 'fail'} "
              f"F1={accuracy:.2f} "
              f"parsed={len(row['parsed_final'])} "
              f"gt={len(gt_atoms)} "
              f"llm={row.get('llm_proposal')}")

    print(f"\nLLM-accepted: {n_correct}/{n_parsed} = "
          f"{100*n_correct/max(n_parsed,1):.1f}%")
    print(f"Classical baseline (no NL parser): 0%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "config": args.config,
            "n_parsed": n_parsed,
            "n_llm_accepted": n_correct,
            "llm_accept_rate": n_correct / max(n_parsed, 1),
            "classical_baseline": 0.0,
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
