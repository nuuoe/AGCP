"""LLM-planner variant of alfworld_closed_loop_e6.py.

Same parsed goal atoms, TextWorld env and `won` flag, but each step prompts
Llama-3.3-70B (Together.ai) with the goal, recent history and admissible
actions; inadmissible outputs are snapped to the closest admissible one.
Output: runs/alfworld_closed_loop_e6_llm_planner.json.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path


# The core-5 task types; pick_two_obj_and_place is deliberately excluded.
SUPPORTED_TASK_TYPES = {
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
}


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def find_game_file(task_id: str) -> str | None:
    base = os.path.expanduser(
        f"~/.cache/alfworld/json_2.1.1/valid_seen/{task_id}")
    matches = sorted(glob.glob(os.path.join(base, "trial_*/game.tw-pddl")))
    return matches[0] if matches else None


def atoms_to_goal_nl(atoms: dict, task_type: str) -> str:
    """Render parsed goal atoms as a one-line NL goal; the free-form task_desc is never shown to the planner."""
    obj = (atoms.get("object_target") or "").lower()
    parent = (atoms.get("parent_target") or "").lower()
    toggle = (atoms.get("toggle_target") or "").lower()
    cool = bool(atoms.get("object_cool"))
    heat = bool(atoms.get("object_heat"))
    clean = bool(atoms.get("object_clean"))

    if task_type == "look_at_obj_in_light":
        return f"Examine the {obj} using the {toggle}."
    if task_type == "pick_and_place_simple":
        return f"Put a {obj} on/in a {parent}."
    if task_type == "pick_clean_then_place_in_recep":
        return f"Clean a {obj} (use the sinkbasin), then put it on/in a {parent}."
    if task_type == "pick_heat_then_place_in_recep":
        return f"Heat a {obj} (use the microwave), then put it on/in a {parent}."
    if task_type == "pick_cool_then_place_in_recep":
        return f"Cool a {obj} (use the fridge), then put it on/in a {parent}."
    if task_type == "pick_two_obj_and_place":
        return f"Put two {obj}s on/in a {parent}."
    # Generic fallback
    parts = [f"Get a {obj}"]
    if cool: parts.append("cool it")
    if heat: parts.append("heat it")
    if clean: parts.append("clean it")
    if parent: parts.append(f"put it on/in a {parent}")
    if toggle: parts.append(f"using the {toggle}")
    return ". ".join(parts) + "."


def call_together_action(
    client, model: str, system_prompt: str, user_prompt: str,
    max_tokens: int = 64, temperature: float = 0.0,
    retry: int = 3,
) -> tuple[str, str]:
    """Request one action; return (cleaned_action, raw_text)."""
    last_err = None
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            # Strip "Action:"-style preambles and quoting.
            cleaned = text
            cleaned = re.sub(r"^\s*(?:action|next action|>)\s*:\s*",
                             "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip().strip("`").strip('"').strip("'")
            for line in cleaned.splitlines():
                line = line.strip().strip("`").strip('"').strip("'")
                if line:
                    cleaned = line
                    break
            return cleaned, text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Together API failed after {retry} retries: {last_err}")


SYSTEM_PROMPT = (
    "You are a textworld agent in ALFRED, a household environment. Each "
    "step you receive a goal, the current observation, and a list of "
    "admissible actions. You must output EXACTLY ONE action, copied "
    "verbatim from the admissible list, with no explanation, no quotes, "
    "no preface.\n"
    "\n"
    "Navigation: you can only `go to <receptacle>` (e.g. `go to desk 1`, "
    "`go to sidetable 2`). You CANNOT `go to <object>` — small objects "
    "like desklamps, knives, books are on/in receptacles. To find a "
    "small object, visit candidate receptacles one by one until you see "
    "it in the observation.\n"
    "\n"
    "Containers: `open <container>` (drawer/fridge/microwave/cabinet/etc) "
    "before taking objects from inside. Take with `take <obj> from <recep>`. "
    "Place with `put <obj> in/on <recep>`.\n"
    "\n"
    "Recipes:\n"
    "  clean X: take X, go to sinkbasin 1, `clean X with sinkbasin 1`\n"
    "  heat X:  take X, go to microwave 1, `heat X with microwave 1`\n"
    "  cool X:  take X, go to fridge 1, `cool X with fridge 1`\n"
    "  examine X with desklamp: take X, find the desklamp (it sits on "
    "    a desk or sidetable — visit them to find it), then `use "
    "    desklamp 1` while holding X.\n"
    "\n"
    "Be systematic. If the target object isn't visible, visit "
    "receptacles you haven't checked yet. Avoid pointless loops."
)


def render_user_prompt(
    goal_nl: str,
    history: list[tuple[str, str]],
    current_obs: str,
    admissible: list[str],
    visited: list[str] | None = None,
    inventory_text: str | None = None,
) -> str:
    """Build the per-step user prompt.

    Args:
        history: (obs, action) pairs for the last K steps, newest last.
        visited: receptacles already visited, newest last.
    """
    parts = [f"Goal: {goal_nl}"]
    if visited:
        seen_set = []
        for v in visited:
            if v not in seen_set:
                seen_set.append(v)
        parts.append(f"\nReceptacles you have already visited: "
                     f"{', '.join(seen_set)}")
    if history:
        parts.append("\nRecent history (oldest first):")
        for i, (o, a) in enumerate(history):
            o_short = o.strip().replace("\n", " ")
            if len(o_short) > 200:
                o_short = o_short[:197] + "..."
            parts.append(f"  obs: {o_short}")
            parts.append(f"  action: {a}")
    obs_short = current_obs.strip()
    parts.append(f"\nCurrent observation: {obs_short}")
    if inventory_text:
        parts.append(f"Inventory: {inventory_text}")
    parts.append("\nAdmissible actions (you must pick one verbatim):")
    for a in admissible:
        parts.append(f"  - {a}")
    parts.append(
        "\nOutput ONLY the next action, copied verbatim from the list above. "
        "No quotes, no prefix, no explanation.")
    return "\n".join(parts)


def snap_to_admissible(
    proposed: str, admissible: list[str]
) -> tuple[str, bool]:
    """Return (best_match, was_fallback); a case-insensitive admissible match is returned unchanged."""
    norm_admissible = {a.strip().lower(): a for a in admissible}
    p = proposed.strip().strip(".").strip().lower()
    if p in norm_admissible:
        return norm_admissible[p], False
    # Closest difflib match, with progressively looser cutoffs.
    for cutoff in (0.7, 0.5, 0.3):
        m = difflib.get_close_matches(p, list(norm_admissible.keys()),
                                       n=1, cutoff=cutoff)
        if m:
            return norm_admissible[m[0]], True
    # No match: fall back to the first admissible action.
    return admissible[0], True


def run_one_task(
    task_id: str, task_type: str, parsed_atoms: dict, gt_atoms: dict,
    together_client, model: str,
    max_steps: int = 30, history_k: int = 4,
    verbose: bool = False,
) -> dict:
    """Run the LLM-planner closed loop on one ALFWorld task."""
    import textworld
    import textworld.gym  # noqa: F401  (registers .gym.make)
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredInfos,
    )

    t0 = time.time()
    game_file = find_game_file(task_id)
    if game_file is None:
        return {"task_id": task_id, "stage_failed": "game_lookup",
                 "error": "no game.tw-pddl found",
                 "execution_success": False, "plan_length": 0,
                 "llm_calls": 0, "fallbacks": 0,
                 "seconds": time.time() - t0}

    if task_type not in SUPPORTED_TASK_TYPES:
        return {"task_id": task_id, "stage_failed": "unsupported_task_type",
                 "error": f"{task_type} not in core 5",
                 "execution_success": False, "plan_length": 0,
                 "llm_calls": 0, "fallbacks": 0,
                 "seconds": time.time() - t0}

    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, facts=True,
        extras=["gamefile"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    env_id = textworld.gym.register_games(
        [game_file], request_infos, batch_size=1, asynchronous=False,
        max_episode_steps=max_steps, wrappers=wrappers,
        name=f"agcp_llm_{abs(hash(task_id))}")
    env = textworld.gym.make(env_id)
    try:
        obs, info = env.reset()
    except Exception as e:
        return {"task_id": task_id, "stage_failed": "env_reset",
                 "error": str(e), "execution_success": False,
                 "plan_length": 0, "llm_calls": 0, "fallbacks": 0,
                 "seconds": time.time() - t0}
    if info["won"][0]:
        env.close()
        return {"task_id": task_id, "stage_failed": None,
                 "error": "won_at_reset", "execution_success": True,
                 "plan_length": 0, "llm_calls": 0, "fallbacks": 0,
                 "seconds": time.time() - t0,
                 "task_type": task_type, "parsed_atoms": parsed_atoms}

    goal_nl = atoms_to_goal_nl(parsed_atoms, task_type)
    if verbose:
        print(f"  goal_nl: {goal_nl}")

    history: list[tuple[str, str]] = []
    actions_taken: list[str] = []
    visited: list[str] = []
    fallback_steps: list[int] = []
    inadmissible_count = 0
    won = False
    api_error = None
    last_action = ""
    n_llm_calls = 0

    current_obs = obs[0]
    current_admissible = info["admissible_commands"][0]

    for step in range(max_steps):
        prompt = render_user_prompt(
            goal_nl=goal_nl,
            history=history[-history_k:],
            current_obs=current_obs,
            admissible=current_admissible,
            visited=visited,
        )
        try:
            proposed, raw = call_together_action(
                together_client, model, SYSTEM_PROMPT, prompt,
                max_tokens=48, temperature=0.0)
            n_llm_calls += 1
        except Exception as e:  # noqa: BLE001
            api_error = f"api: {e}"
            break

        action, was_fallback = snap_to_admissible(proposed,
                                                   current_admissible)
        if was_fallback:
            inadmissible_count += 1
            fallback_steps.append(step)
            if verbose:
                print(f"  [step {step}] LLM proposed {proposed!r} -> "
                       f"fallback {action!r}")

        try:
            obs, _, dones, info = env.step([action])
        except Exception as e:  # noqa: BLE001
            api_error = f"env.step: {e}"
            break
        actions_taken.append(action)
        history.append((current_obs, action))
        current_obs = obs[0]
        current_admissible = info["admissible_commands"][0]
        last_action = action

        if info["won"][0]:
            won = True
            break
        if dones[0]:
            break
        if verbose:
            print(f"  step {step+1}: {action}  "
                   f"-> obs: {current_obs[:80]}")
    env.close()

    stage_failed = None
    if not won:
        if api_error:
            stage_failed = "api_error"
        elif step + 1 >= max_steps:
            stage_failed = "max_steps"
        else:
            stage_failed = "goal_check"

    return {
        "task_id": task_id,
        "task_type": task_type,
        "parsed_atoms": parsed_atoms,
        "gt_atoms": gt_atoms,
        "goal_nl": goal_nl,
        "execution_success": bool(won),
        "plan_length": len(actions_taken),
        "llm_calls": n_llm_calls,
        "fallbacks": inadmissible_count,
        "fallback_steps": fallback_steps,
        "actions_head": actions_taken[:8],
        "actions_tail": actions_taken[-8:] if len(actions_taken) > 8 else [],
        "stage_failed": stage_failed,
        "error": api_error,
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_json",
                    default="runs/alfworld_cloud_claude_sonnet.json",
                    help="E4 NL-parse results to pull parsed_atoms from")
    ap.add_argument("--out_json",
                    default="runs/alfworld_closed_loop_e6_llm_planner.json")
    ap.add_argument("--min_f1", type=float, default=0.0,
                    help="Min smart_f1 threshold (0.0 = unfiltered, matches "
                         "_full_unfiltered baseline of HandCoded variant)")
    ap.add_argument("--max_n", type=int, default=10_000,
                    help="Cap on N tasks attempted (debug)")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--history_k", type=int, default=4)
    ap.add_argument("--model",
                    default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="If out_json exists, skip tasks already in it.")
    args = ap.parse_args()

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)

    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not set. source .env first.")
        sys.exit(1)

    from together import Together
    together_client = Together()
    print(f"Together model: {args.model}")

    e4 = json.load(open(args.parsed_json))
    rows = e4["rows"]
    high = [r for r in rows if r.get("smart_f1", 0.0) >= args.min_f1]
    cand = []
    for r in high:
        if r["task_type"] not in SUPPORTED_TASK_TYPES:
            continue
        if find_game_file(r["task_id"]) is None:
            continue
        cand.append(r)
    print(f"E4 candidates with f1>={args.min_f1}: {len(high)} "
           f"(supported_core5+game_file: {len(cand)})")
    cand = cand[: args.max_n]
    print(f"Running LLM-planner closed-loop on N={len(cand)} tasks "
           f"(max_steps={args.max_steps}, history_k={args.history_k})")

    existing: dict[str, dict] = {}
    if args.resume and os.path.exists(args.out_json):
        try:
            prev = json.load(open(args.out_json))
            for r in prev.get("results", []):
                if r.get("execution_success") is not None and r.get("stage_failed") != "api_error":
                    existing[r["task_id"]] = r
            print(f"Resume mode: skipping {len(existing)} previously-completed tasks")
        except Exception as e:
            print(f"Resume failed to read prev: {e}")

    results = list(existing.values())
    n_exec_ok = sum(1 for r in results if r.get("execution_success"))
    t_start = time.time()

    for i, r in enumerate(cand, 1):
        task_id = r["task_id"]
        if task_id in existing:
            continue
        task_type = r["task_type"]
        parsed = r["llm_pred"]
        gt = r["gt"]
        print(f"\n[{i}/{len(cand)}] {task_id}")
        print(f"  task_type={task_type}")
        print(f"  parsed (E4): {parsed}")
        try:
            res = run_one_task(
                task_id, task_type, parsed, gt,
                together_client, args.model,
                max_steps=args.max_steps,
                history_k=args.history_k,
                verbose=args.verbose,
            )
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            res = {"task_id": task_id, "task_type": task_type,
                    "stage_failed": "uncaught", "error": str(e),
                    "execution_success": False, "plan_length": 0,
                    "llm_calls": 0, "fallbacks": 0,
                    "parsed_atoms": parsed, "gt_atoms": gt,
                    "seconds": 0.0}
        results.append(res)
        if res.get("execution_success"):
            n_exec_ok += 1
        print(f"  -> won={res.get('execution_success')} "
               f"plan_len={res.get('plan_length')} "
               f"calls={res.get('llm_calls')} "
               f"fallbacks={res.get('fallbacks')} "
               f"stage_failed={res.get('stage_failed')}")
        # Write after every task so partial runs are recoverable.
        n = len(results)
        ci_e = wilson_ci(n_exec_ok, n)
        from collections import Counter
        ttype_totals = Counter(rr.get("task_type") for rr in results)
        ttype_succ = Counter(rr.get("task_type") for rr in results
                              if rr.get("execution_success"))
        per_type = {
            t: {"n": ttype_totals[t], "won": ttype_succ.get(t, 0)}
            for t in ttype_totals
        }
        out = {
            "parsed_json": args.parsed_json,
            "model": args.model,
            "min_f1": args.min_f1,
            "max_steps": args.max_steps,
            "history_k": args.history_k,
            "n_total": n,
            "n_execution_success": n_exec_ok,
            "execution_success_rate": n_exec_ok / n,
            "execution_success_ci95": list(ci_e),
            "per_task_type": per_type,
            "wall_seconds": time.time() - t_start,
            "results": results,
        }
        json.dump(out, open(args.out_json, "w"), indent=2, default=str)

    n = len(results)
    if n == 0:
        print("\nNo tasks attempted — aborting.")
        return
    ci_e = wilson_ci(n_exec_ok, n)
    print("\n" + "=" * 60)
    print("AGCP closed-loop E6 (LLM-PLANNER) on ALFWorld TextWorld")
    print(f"  Parsed source: {args.parsed_json}")
    print(f"  Model: {args.model}")
    print(f"  Min E4 smart_f1: {args.min_f1}")
    print(f"  N tasks (post-filter): {n}")
    print(f"  Env 'won' (executed): {n_exec_ok}/{n} "
           f"= {100*n_exec_ok/n:.1f}% "
           f"(95% CI {100*ci_e[0]:.1f}-{100*ci_e[1]:.1f}%)")
    print(f"  Wall time: {time.time()-t_start:.1f}s")
    print(f"  Output: {args.out_json}")
    print("=" * 60)

    # Per-task-type breakdown
    from collections import Counter
    ttype_totals = Counter(r.get("task_type") for r in results)
    ttype_succ = Counter(r.get("task_type") for r in results
                          if r.get("execution_success"))
    print("Per-task-type breakdown:")
    for t in sorted(ttype_totals):
        nt = ttype_totals[t]
        kt = ttype_succ.get(t, 0)
        ci = wilson_ci(kt, nt)
        print(f"  {t}: {kt}/{nt} = {100*kt/nt:.1f}% "
               f"[{100*ci[0]:.0f}-{100*ci[1]:.0f}]")

    # Failure breakdown
    fail_stages = Counter(r.get("stage_failed") for r in results
                          if not r.get("execution_success"))
    print("Failure stage breakdown:")
    for stage, c in fail_stages.most_common():
        print(f"  {stage}: {c}")


if __name__ == "__main__":
    main()
