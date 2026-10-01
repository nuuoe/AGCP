"""Iterative ALFWorld goal parsing with cross-task failure hints and parse priors.

Tasks run in a fixed order. After each failure the LLM proposes a one-line
parse-only rule that is cached per task_type and prepended to later prompts,
alongside slot-value frequencies from successful parses. Inputs as in
alfworld_iterative_e4.py. Output: runs/alfworld_iterative_e4_with_refinement.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from scripts.alfworld_cloud_llm import call_anthropic, call_openai

# Parse/rule LLM dispatch; overridden by --provider in main().
_PROVIDER = ("anthropic", "claude-sonnet-4-6")


def _call_parse_llm(prompt: str) -> dict:
    vendor, model = _PROVIDER
    if vendor == "anthropic":
        return call_anthropic(prompt, model, None)
    if vendor == "openai":
        return call_openai(prompt, model, None)
    raise ValueError(vendor)
from scripts.alfworld_enum_constrained import make_enum_prompt
from scripts.alfworld_closed_loop_e6 import (
    SUPPORTED_TASK_TYPES, find_game_file, wilson_ci,
)
from scripts.alfworld_iterative_e4 import (
    CORE_5, extract_env_affordances, verify_atoms,
)


def build_lessons(relevant_failures: list[dict]) -> str:
    """Render the cached rule_hints as a bullet block; return "" if there are none."""
    if not relevant_failures:
        return ""
    lines = ["", "", "## LESSONS FROM PAST FAILURES on similar task_type"]
    lines.append("These hints come from prior failures of this same "
                 "task_type. Apply them when relevant.")
    for f in relevant_failures:
        hint = f.get("rule_hint", "").strip()
        if hint:
            lines.append(f"  - {hint}")
    return "\n".join(lines) + "\n"


def make_enum_prompt_with_lessons(nl: str, lessons: str) -> str:
    """Insert the lessons block immediately before the 'Your task' section of the base prompt."""
    base = make_enum_prompt(nl)
    if not lessons:
        return base
    if "## Your task" in base:
        head, tail = base.split("## Your task", 1)
        return head + lessons + "\n## Your task" + tail
    return base + lessons


def build_feedback_prompt_with_lessons(nl: str, prev_atoms: dict,
                                          errors: list[str],
                                          env_aff: dict, task_type: str,
                                          iter_num: int,
                                          lessons: str) -> str:
    """Build the iteration-k prompt with both verifier feedback and lessons."""
    base = make_enum_prompt_with_lessons(nl, lessons)
    feedback = f"""

## ITERATION {iter_num+1} — verifier feedback

Your previous parse was:
{json.dumps(prev_atoms, indent=2)}

It failed verification against this room's affordances:
"""
    for e in errors:
        feedback += f"  - {e}\n"

    feedback += f"""

For reference, the room contains:
  Objects (canonical types): {sorted(env_aff['object_types'])[:30]}
  Receptacles (canonical types): {sorted(env_aff['receptacle_types'])}
  Toggleables (canonical types): {sorted(env_aff['toggleable_types'])}
  has_fridge={env_aff['has_fridge']}, has_microwave={env_aff['has_microwave']}, \
has_sink={env_aff['has_sink']}, has_stove={env_aff['has_stove']}

The task is task_type={task_type}.

Please re-parse the task accordingly, paying attention to the
constraints above AND any lessons from past failures listed above.
Output ONLY valid JSON, no commentary.

Task to re-parse: "{nl}"
"""
    return base + feedback


def derive_failure_reason(atoms: dict, env_aff: dict, task_type: str,
                            verifier_errors: list[str],
                            planner_error: str | None,
                            planner_success: bool,
                            execution_success: bool,
                            verified: bool) -> str:
    """Build a failure-reason string without consulting GT.

    Uses, in order: env-fact verifier errors, the planner error message,
    or a generic "env did not signal won" note that lists the alternative
    slot values in the room.
    """
    if execution_success:
        return ""
    if verifier_errors:
        return ("verifier errors: " +
                "; ".join(verifier_errors[:3]))
    if not planner_success and planner_error:
        return f"planner error: {planner_error[:300]}"
    if verified and planner_success and not execution_success:
        # Parse verified and planner finished but no win: most likely a
        # valid but wrong slot value among ambiguous candidates.
        recs = sorted(env_aff.get("receptacle_types", []))
        toggs = sorted(env_aff.get("toggleable_types", []))
        chosen_parent = (atoms or {}).get("parent_target", "")
        chosen_toggle = (atoms or {}).get("toggle_target", "")
        msg = ("Plan executed to completion but env never signalled "
               "'won' — most likely a slot value was ambiguous. ")
        if task_type == "look_at_obj_in_light":
            msg += (f"Chose toggle_target='{chosen_toggle}'. Other "
                    f"toggleables in env: "
                    f"{[t for t in toggs if t != chosen_toggle]}.")
        else:
            msg += (f"Chose parent_target='{chosen_parent}'. Other "
                    f"receptacles in env: "
                    f"{[r for r in recs if r != chosen_parent][:8]}.")
        return msg
    return "unknown failure (no verifier errors, no planner error, env did not signal won)"


def llm_propose_rule(nl: str, atoms: dict, env_aff: dict,
                       task_type: str, failure_reason: str,
                       model: str = "claude-sonnet-4-6") -> str:
    """Ask the LLM for one short parse-only rule describing what went wrong.

    The model sees only the task NL, its own parsed atoms, the env-fact
    view of the room and the failure reason; no GT.
    """
    env_view = {
        "objects": sorted(env_aff.get("object_types", []))[:30],
        "receptacles": sorted(env_aff.get("receptacle_types", [])),
        "toggleables": sorted(env_aff.get("toggleable_types", [])),
        "has_fridge": env_aff.get("has_fridge"),
        "has_microwave": env_aff.get("has_microwave"),
        "has_sink": env_aff.get("has_sink"),
        "has_stove": env_aff.get("has_stove"),
    }
    prompt = f"""An ALFWorld GOAL-PARSING task may have produced a
WRONG ATOM. Help us learn ONLY about the PARSING decision.

Task description (NL): "{nl}"
Task type: {task_type}
Parsed atoms (the goal we extracted from the NL):
{json.dumps(atoms, indent=2)}
Available env-facts in this room (NO GROUND TRUTH provided):
{json.dumps(env_view, indent=2)}
Failure reason: {failure_reason}

THE ONLY ACCEPTABLE OUTPUT is a SLOT-VALUE-CORRECTION rule about
the parsed atoms (object_target, mrecep_target, parent_target,
toggle_target, object_sliced/cool/heat/clean).

DO NOT propose action-execution advice ("face the lamp", "explore
drawers", "verify location", "after toggling"). The planner is
FIXED and cannot follow such advice. Only the PARSING step can use
lessons.

If the failure is clearly not a parse error (e.g., planner timed out
while reaching a valid goal), output the EXACT WORD: NO_PARSE_LESSON

Otherwise output ONE line in EXACTLY this format:
  For {task_type} tasks where NL says '<phrase>', <slot> should be <value> not <wrong_value>

OR for boolean modifiers:
  For {task_type} tasks where NL says '<phrase>', set <flag>=true

Output ONLY the single rule line, no quotes, no markdown.
"""
    # Plain-text call (call_anthropic parses JSON), dispatched on the
    # --provider vendor so the rule LLM matches the parse LLM.
    import anthropic
    import time as _time
    vendor, prov_model = _PROVIDER
    model = prov_model
    _sys = ("You are a debugging assistant. Output ONE short "
            "lesson sentence and nothing else. No JSON, no "
            "markdown, no quotes.")
    client = anthropic.Anthropic() if vendor == "anthropic" else None
    for attempt in range(6):
        try:
            if vendor == "anthropic":
                rsp = client.messages.create(
                    model=model, max_tokens=120, temperature=0.3,
                    system=_sys,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = rsp.content[0].text.strip()
            else:
                from openai import OpenAI
                rsp = OpenAI().chat.completions.create(
                    model=model, max_tokens=120, temperature=0.3,
                    messages=[{"role": "system", "content": _sys},
                              {"role": "user", "content": prompt}],
                )
                text = (rsp.choices[0].message.content or "").strip()
            text = text.lstrip("-* ").strip().strip('"').strip("'")
            text = text.split("\n")[0].strip()
            if len(text) > 240:
                text = text[:240]
            # Reject rules that are not about the parse.
            if "NO_PARSE_LESSON" in text:
                return ""
            exec_keywords = [
                "after toggling", "face the", "examine ", "look at the",
                "explore ", "verify the location", "navigate to",
                "before placing", "actually openable", "retry with",
                "while holding", "trigger the win",
            ]
            if any(k in text.lower() for k in exec_keywords):
                return ""
            # Require the canonical rule shape.
            if "should be" not in text.lower() and "set " not in text.lower():
                return ""
            return text
        except anthropic.RateLimitError:
            _time.sleep(2 ** attempt)
        except Exception:
            return ""
    return ""


def run_one_task_iterative_with_refinement(
    task_id: str, task_type: str, nl: str,
    gt_for_logging: dict, lessons: str,
    max_steps: int = 80, max_iters: int = 3,
    verbose: bool = False,
) -> dict:
    """Run the iterative parse and hand-coded TW execution with lessons injected into every prompt."""
    import textworld
    import textworld.gym  # noqa: F401
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredInfos,
    )
    from alfworld.agents.expert import HandCodedTWAgent
    import alfworld.gen.constants as constants

    t0 = time.time()
    game_file = find_game_file(task_id)
    if game_file is None:
        return {"task_id": task_id, "task_type": task_type,
                 "stage_failed": "game_lookup",
                 "error": "no game.tw-pddl found",
                 "iterations_used": 0, "atoms_per_iteration": [],
                 "errors_per_iteration": [], "final_atoms": None,
                 "verified": False, "planner_success": False,
                 "execution_success": False, "plan_length": 0,
                 "seconds": time.time() - t0}

    if task_type not in SUPPORTED_TASK_TYPES:
        return {"task_id": task_id, "task_type": task_type,
                 "stage_failed": "unsupported_task_type",
                 "error": f"{task_type} not in handcoded_expert_tw",
                 "iterations_used": 0, "atoms_per_iteration": [],
                 "errors_per_iteration": [], "final_atoms": None,
                 "verified": False, "planner_success": False,
                 "execution_success": False, "plan_length": 0,
                 "seconds": time.time() - t0}

    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, facts=True,
        extras=["gamefile"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    env_id = textworld.gym.register_games(
        [game_file], request_infos, batch_size=1, asynchronous=False,
        max_episode_steps=max_steps, wrappers=wrappers,
        name=f"agcp_iter_refine_{abs(hash(task_id))}")
    env = textworld.gym.make(env_id)
    try:
        obs, info = env.reset()
    except Exception as e:
        return {"task_id": task_id, "task_type": task_type,
                 "stage_failed": "env_reset", "error": str(e),
                 "iterations_used": 0, "atoms_per_iteration": [],
                 "errors_per_iteration": [], "final_atoms": None,
                 "verified": False, "planner_success": False,
                 "execution_success": False, "plan_length": 0,
                 "seconds": time.time() - t0}

    env_aff = extract_env_affordances(info["facts"][0])

    atoms_per_iter: list[dict] = []
    errors_per_iter: list[list[str]] = []
    last_atoms: dict = {}
    verified = False
    llm_calls = 0
    for k in range(max_iters):
        if k == 0:
            prompt = make_enum_prompt_with_lessons(nl, lessons)
        else:
            prompt = build_feedback_prompt_with_lessons(
                nl, last_atoms, errors_per_iter[-1],
                env_aff, task_type, k, lessons)
        atoms_k = _call_parse_llm(prompt)
        llm_calls += 1
        if not isinstance(atoms_k, dict) or "_error" in atoms_k:
            atoms_k = {}
            errs = ["LLM did not return valid JSON"]
        else:
            errs = verify_atoms(atoms_k, env_aff, task_type)
        atoms_per_iter.append(atoms_k)
        errors_per_iter.append(errs)
        last_atoms = atoms_k
        if not errs:
            verified = True
            if verbose:
                print(f"    iter {k+1}: VERIFIED")
            break
        if verbose:
            print(f"    iter {k+1}: {len(errs)} errors -- {errs[0]}")

    final_atoms = last_atoms
    iterations_used = len(atoms_per_iter)

    parsed_params = final_atoms
    try:
        agent = HandCodedTWAgent(max_steps=max_steps)
        task_params = {**parsed_params, "task_type": task_type}
        task_params = {
            k: (v.lower() if isinstance(v, str) and v in constants.OBJECTS
                  else v)
            for k, v in task_params.items()
        }
        policy_class = agent.get_task_policy(task_params)
        agent.policy = policy_class(task_params, max_steps=max_steps)
    except Exception as e:
        env.close()
        return {"task_id": task_id, "task_type": task_type,
                 "stage_failed": "planner_init",
                 "error": str(e),
                 "iterations_used": iterations_used,
                 "atoms_per_iteration": atoms_per_iter,
                 "errors_per_iteration": errors_per_iter,
                 "final_atoms": final_atoms,
                 "gt_for_logging": gt_for_logging,
                 "verified": verified,
                 "llm_calls": llm_calls,
                 "planner_success": False, "planner_error": str(e),
                 "execution_success": False, "plan_length": 0,
                 "seconds": time.time() - t0,
                 "env_affordances": {
                     "object_types": sorted(env_aff["object_types"]),
                     "receptacle_types": sorted(env_aff["receptacle_types"]),
                     "toggleable_types": sorted(env_aff["toggleable_types"]),
                     "has_fridge": env_aff["has_fridge"],
                     "has_microwave": env_aff["has_microwave"],
                     "has_sink": env_aff["has_sink"],
                     "has_stove": env_aff["has_stove"],
                 }}

    last_action = ""
    won = False
    trajectory: list[str] = []
    planner_failed = False
    planner_error = None
    for step in range(max_steps):
        try:
            game_state = {
                "feedback": obs[0],
                "admissible_commands": info["admissible_commands"][0],
                "facts": info["facts"][0] if "facts" in info else [],
            }
            action = agent.policy.act(game_state, last_action)
        except Exception as e:
            planner_failed = True
            planner_error = f"{type(e).__name__}: {e}"
            break
        if not isinstance(action, str) or not action:
            planner_failed = True
            planner_error = f"empty action: {action!r}"
            break
        try:
            obs, _, dones, info = env.step([action])
        except Exception as e:
            planner_failed = True
            planner_error = f"env.step error: {e}"
            break
        agent.policy.observe(obs[0])
        last_action = action
        trajectory.append(action)
        if info["won"][0]:
            won = True
            break
        if dones[0]:
            break
    env.close()

    return {
        "task_id": task_id,
        "task_type": task_type,
        "iterations_used": iterations_used,
        "atoms_per_iteration": atoms_per_iter,
        "errors_per_iteration": errors_per_iter,
        "final_atoms": final_atoms,
        "gt_for_logging": gt_for_logging,
        "verified": verified,
        "llm_calls": llm_calls,
        "planner_success": (not planner_failed),
        "planner_error": planner_error,
        "execution_success": bool(won),
        "plan_length": len(trajectory),
        "trajectory_head": trajectory[:5],
        "trajectory_tail": trajectory[-5:] if trajectory else [],
        "stage_failed": (None if won else
                          ("planner" if planner_failed else "goal_check")),
        "seconds": time.time() - t0,
        "env_affordances": {
            "object_types": sorted(env_aff["object_types"]),
            "receptacle_types": sorted(env_aff["receptacle_types"]),
            "toggleable_types": sorted(env_aff["toggleable_types"]),
            "has_fridge": env_aff["has_fridge"],
            "has_microwave": env_aff["has_microwave"],
            "has_sink": env_aff["has_sink"],
            "has_stove": env_aff["has_stove"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source_json",
        default="runs/alfworld_closed_loop_e6_full_unfiltered.json")
    ap.add_argument(
        "--e4_cache_json",
        default="runs/alfworld_cloud_claude_sonnet.json")
    ap.add_argument("--out_json",
                    default="runs/alfworld_iterative_e4_with_refinement.json")
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--provider", default="claude_sonnet",
                    choices=["claude_sonnet", "gpt4o"],
                    help="parse+rule LLM (cross-model replication)")
    ap.add_argument("--cache_window", type=int, default=5,
                    help="Use last N lessons per task_type as context.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global _PROVIDER
    _PROVIDER = {
        "claude_sonnet": ("anthropic", "claude-sonnet-4-6"),
        "gpt4o": ("openai", "gpt-4o"),
    }[args.provider]

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)

    _need_key = ("ANTHROPIC_API_KEY" if _PROVIDER[0] == "anthropic"
                 else "OPENAI_API_KEY")
    if not os.environ.get(_need_key):
        sys.exit(f"{_need_key} not set; source .env first.")

    src = json.load(open(args.source_json))
    e4 = json.load(open(args.e4_cache_json))
    e4_by_id = {r["task_id"]: r for r in e4["rows"]}

    src_results = src["results"]
    core5 = [r for r in src_results if r["task_type"] in CORE_5]
    candidates: list[dict] = []
    for r in core5:
        if r["task_type"] not in SUPPORTED_TASK_TYPES:
            continue
        if find_game_file(r["task_id"]) is None:
            continue
        e4_row = e4_by_id.get(r["task_id"])
        if e4_row is None:
            continue
        candidates.append({
            "task_id": r["task_id"],
            "task_type": r["task_type"],
            "task_desc": e4_row["task_desc"],
            "gt": e4_row["gt"],
            "one_shot_won": r.get("execution_success"),
        })

    print(f"Loaded {len(core5)} core-5 results; {len(candidates)} have "
          f"all required inputs.")
    candidates = candidates[args.start_idx: args.start_idx + args.max_n]
    print(f"Running iterative E4 + refinement on N={len(candidates)} "
          f"tasks (max_iters={args.max_iters}, max_steps={args.max_steps}, "
          f"cache_window={args.cache_window})")

    # Per-task_type cache of parse-only rule hints from failures.
    failure_cache: dict[str, list[dict]] = defaultdict(list)
    # Slot-value counts from successful parses: parse_priors[task_type][slot][value].
    parse_priors: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))

    def build_priors_text(task_type: str) -> str:
        """Render the top-3 values per slot for this task_type as a prompt block."""
        slots = parse_priors.get(task_type, {})
        if not any(slots.values()):
            return ""
        lines = ["", "## PARSE PRIORS from successful past tasks"]
        lines.append("Across past SUCCESSFUL tasks of this same task_type, "
                     "the parsed slot values were:")
        for slot in ("object_target", "mrecep_target", "parent_target",
                       "toggle_target"):
            counts = slots.get(slot, {})
            if not counts:
                continue
            tot = sum(counts.values())
            top = sorted(counts.items(), key=lambda x: -x[1])[:3]
            parts = [f"{v or '<empty>'}={100*c/tot:.0f}%" for v, c in top]
            lines.append(f"  - {slot}: " + ", ".join(parts))
        for slot in ("object_sliced", "object_cool",
                       "object_heat", "object_clean"):
            counts = slots.get(slot, {})
            if not counts:
                continue
            tot = sum(counts.values())
            true_pct = 100 * counts.get("True", 0) / tot
            lines.append(f"  - {slot}: True={true_pct:.0f}%")
        return "\n".join(lines) + "\n"

    results = []
    n_planner_ok = 0
    n_exec_ok = 0
    iters_distrib: dict[int, int] = {}
    n_verified = 0
    total_llm_calls = 0  # all calls including rule proposals
    total_e4_calls = 0    # only E4 iteration calls
    total_rule_calls = 0  # only rule-proposal calls
    t_start = time.time()
    for i, c in enumerate(candidates, 1):
        task_id = c["task_id"]
        task_type = c["task_type"]
        nl = c["task_desc"]
        gt = c["gt"]

        relevant = failure_cache.get(task_type, [])
        recent = relevant[-args.cache_window:]
        priors_text = build_priors_text(task_type)
        lessons = build_lessons(recent) + priors_text

        print(f"\n[{i}/{len(candidates)}] {task_id}")
        print(f"  task_type={task_type}")
        print(f"  nl={nl[:80]!r}")
        if recent:
            print(f"  lessons in context: {len(recent)} "
                  f"(of {len(relevant)} total for type)")
        try:
            res = run_one_task_iterative_with_refinement(
                task_id, task_type, nl, gt,
                lessons=lessons,
                max_steps=args.max_steps,
                max_iters=args.max_iters,
                verbose=args.verbose)
        except Exception as e:
            traceback.print_exc()
            res = {"task_id": task_id, "task_type": task_type,
                    "stage_failed": "uncaught", "error": str(e),
                    "iterations_used": 0, "atoms_per_iteration": [],
                    "errors_per_iteration": [], "final_atoms": None,
                    "verified": False, "llm_calls": 0,
                    "planner_success": False, "execution_success": False,
                    "plan_length": 0, "seconds": 0.0,
                    "gt_for_logging": gt}

        res["lessons_provided"] = recent
        res["lessons_text"] = lessons
        res["order_index"] = i - 1

        iters_distrib[res["iterations_used"]] = (
            iters_distrib.get(res["iterations_used"], 0) + 1)
        if res.get("verified"):
            n_verified += 1
        e4_calls = res.get("llm_calls", 0)
        total_e4_calls += e4_calls
        total_llm_calls += e4_calls
        if res.get("planner_success"):
            n_planner_ok += 1
        if res.get("execution_success"):
            n_exec_ok += 1
            final_atoms = res.get("final_atoms") or {}
            for slot in ("object_target", "mrecep_target", "parent_target",
                           "toggle_target"):
                v = final_atoms.get(slot, "")
                parse_priors[task_type][slot][str(v)] += 1
            for slot in ("object_sliced", "object_cool",
                           "object_heat", "object_clean"):
                v = final_atoms.get(slot, False)
                parse_priors[task_type][slot][str(v)] += 1

        # On failure, propose a parse-only rule for the failure cache.
        rule_added = False
        if not res.get("execution_success"):
            failure_reason = derive_failure_reason(
                atoms=res.get("final_atoms") or {},
                env_aff={
                    "object_types": set(res.get("env_affordances", {}).get("object_types", [])),
                    "receptacle_types": set(res.get("env_affordances", {}).get("receptacle_types", [])),
                    "toggleable_types": set(res.get("env_affordances", {}).get("toggleable_types", [])),
                    "has_fridge": res.get("env_affordances", {}).get("has_fridge"),
                    "has_microwave": res.get("env_affordances", {}).get("has_microwave"),
                    "has_sink": res.get("env_affordances", {}).get("has_sink"),
                    "has_stove": res.get("env_affordances", {}).get("has_stove"),
                },
                task_type=task_type,
                verifier_errors=(res.get("errors_per_iteration") or [[]])[-1],
                planner_error=res.get("planner_error"),
                planner_success=res.get("planner_success", False),
                execution_success=res.get("execution_success", False),
                verified=res.get("verified", False),
            )
            env_aff_for_rule = {
                "object_types": res.get("env_affordances", {}).get("object_types", []),
                "receptacle_types": res.get("env_affordances", {}).get("receptacle_types", []),
                "toggleable_types": res.get("env_affordances", {}).get("toggleable_types", []),
                "has_fridge": res.get("env_affordances", {}).get("has_fridge"),
                "has_microwave": res.get("env_affordances", {}).get("has_microwave"),
                "has_sink": res.get("env_affordances", {}).get("has_sink"),
                "has_stove": res.get("env_affordances", {}).get("has_stove"),
            }
            try:
                rule_hint = llm_propose_rule(
                    nl=nl, atoms=res.get("final_atoms") or {},
                    env_aff=env_aff_for_rule,
                    task_type=task_type,
                    failure_reason=failure_reason)
                total_rule_calls += 1
                total_llm_calls += 1
            except Exception as e:
                rule_hint = ""
                print(f"  rule-proposal call failed: {e}")
            if rule_hint:
                failure_cache[task_type].append({
                    "task_id": task_id,
                    "nl": nl,
                    "failed_atoms": res.get("final_atoms"),
                    "failure_reason": failure_reason,
                    "rule_hint": rule_hint,
                    "order_index": i - 1,
                })
                rule_added = True
                res["new_rule_hint"] = rule_hint
                res["failure_reason"] = failure_reason

        print(f"  -> iters={res.get('iterations_used')} "
              f"verified={res.get('verified')} "
              f"planner_ok={res.get('planner_success')} "
              f"exec_ok={res.get('execution_success')} "
              f"plan_len={res.get('plan_length')}"
              + (f"  RULE_ADDED: {res.get('new_rule_hint')[:80]!r}"
                  if rule_added else ""))

        results.append(res)

        # Persist after every task
        n = len(results)
        ci_e = wilson_ci(n_exec_ok, n)
        fc_serializable = {k: v for k, v in failure_cache.items()}
        out = {
            "source_json": args.source_json,
            "e4_cache_json": args.e4_cache_json,
            "max_iters": args.max_iters,
            "max_steps": args.max_steps,
            "cache_window": args.cache_window,
            "n_total": n,
            "n_verified_within_max_iters": n_verified,
            "n_planner_success": n_planner_ok,
            "n_execution_success": n_exec_ok,
            "execution_success_rate": n_exec_ok / n,
            "execution_success_ci95": list(ci_e),
            "iterations_distribution": iters_distrib,
            "total_llm_calls": total_llm_calls,
            "total_e4_calls": total_e4_calls,
            "total_rule_calls": total_rule_calls,
            "wall_seconds": time.time() - t_start,
            "failure_cache_final": fc_serializable,
            "results": results,
        }
        json.dump(out, open(args.out_json, "w"), indent=2, default=str)

    n = len(results)
    if n == 0:
        print("\nNo tasks attempted -- aborting.")
        return
    ci_e = wilson_ci(n_exec_ok, n)
    ci_p = wilson_ci(n_planner_ok, n)
    print("\n" + "=" * 70)
    print("ITERATIVE E4 + ONLINE REFINEMENT on ALFWorld")
    print(f"  N tasks: {n}")
    print(f"  Iters distribution: {iters_distrib}")
    print(f"  Verified within max_iters: {n_verified}/{n} "
          f"= {100*n_verified/n:.1f}%")
    print(f"  Planner reached goal: {n_planner_ok}/{n} = "
          f"{100*n_planner_ok/n:.1f}% (95% CI "
          f"{100*ci_p[0]:.1f}-{100*ci_p[1]:.1f}%)")
    print(f"  Env 'won' (executed): {n_exec_ok}/{n} = "
          f"{100*n_exec_ok/n:.1f}% (95% CI "
          f"{100*ci_e[0]:.1f}-{100*ci_e[1]:.1f}%)")
    print(f"  Total LLM calls: {total_llm_calls} "
          f"(E4: {total_e4_calls}, rules: {total_rule_calls})")
    print(f"  Wall time: {time.time()-t_start:.1f}s")
    print(f"  Output: {args.out_json}")
    print("=" * 70)

    # Per-task-type breakdown
    print("\nPer-task-type breakdown:")
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r)
    for tt in sorted(by_type):
        rs = by_type[tt]
        n_t = len(rs)
        won = sum(1 for r in rs if r.get("execution_success"))
        ci = wilson_ci(won, n_t)
        print(f"  {tt}: {won}/{n_t} = {100*won/n_t:.0f}% "
              f"(95% CI {100*ci[0]:.0f}-{100*ci[1]:.0f}%)")

    # Early-vs-late split.
    half = n // 2
    early = results[:half]
    late = results[half:]
    e_won = sum(1 for r in early if r.get("execution_success"))
    l_won = sum(1 for r in late if r.get("execution_success"))
    e_ci = wilson_ci(e_won, len(early))
    l_ci = wilson_ci(l_won, len(late))
    print("\nEARLY-vs-LATE accuracy (self-improvement signal):")
    print(f"  First half  ({len(early)}): {e_won}/{len(early)} = "
          f"{100*e_won/len(early):.1f}% "
          f"(CI {100*e_ci[0]:.1f}-{100*e_ci[1]:.1f}%)")
    print(f"  Second half ({len(late)}): {l_won}/{len(late)} = "
          f"{100*l_won/len(late):.1f}% "
          f"(CI {100*l_ci[0]:.1f}-{100*l_ci[1]:.1f}%)")
    delta = (l_won / len(late) if len(late) else 0) - (
        e_won / len(early) if len(early) else 0)
    print(f"  Delta (late - early): {100*delta:+.1f}pp "
          f"{'(POSITIVE -- self-improvement signal)' if delta > 0 else '(NULL/NEGATIVE)'}")

    # Per-task-type: first-occurrence vs last-occurrence
    print("\nPer-task-type: first-occurrence vs last-occurrence:")
    for tt in sorted(by_type):
        rs = by_type[tt]
        if len(rs) < 4:
            print(f"  {tt}: only {len(rs)} tasks, skipping")
            continue
        q = max(1, len(rs) // 4)
        early_t = rs[:q]
        late_t = rs[-q:]
        e_w = sum(1 for r in early_t if r.get("execution_success"))
        l_w = sum(1 for r in late_t if r.get("execution_success"))
        print(f"  {tt}: first{q}: {e_w}/{len(early_t)} = "
              f"{100*e_w/len(early_t):.0f}%; last{q}: {l_w}/{len(late_t)} = "
              f"{100*l_w/len(late_t):.0f}%")

    # Final failure_cache size per task_type
    print("\nFinal failure_cache size per task_type:")
    for tt in sorted(failure_cache):
        print(f"  {tt}: {len(failure_cache[tt])} rules")


if __name__ == "__main__":
    main()
