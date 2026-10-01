"""Iterative ALFWorld goal parsing with environment-fact feedback.

The LLM parse is checked against affordances extracted from the env's
facts (never against ground truth) and re-prompted with the errors for up
to --max_iters rounds; the final atoms drive the hand-coded TW planner.
Inputs: --source_json, --e4_cache_json. Output: runs/alfworld_iterative_e4.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, ".")

from scripts.alfworld_cloud_llm import call_anthropic, call_openai

# Parse-LLM dispatch; overridden by --provider in main().
_PROVIDER = ("anthropic", "claude-sonnet-4-6")


def _call_parse_llm(prompt: str) -> dict:
    vendor, model = _PROVIDER
    if vendor == "anthropic":
        return call_anthropic(prompt, model, None)
    if vendor == "openai":
        return call_openai(prompt, model, None)
    raise ValueError(vendor)
from scripts.alfworld_enum_constrained import (
    make_enum_prompt, normalize_pred_extended, OBJECT_TYPES,
)
from scripts.alfworld_closed_loop_e6 import (
    SUPPORTED_TASK_TYPES, find_game_file, wilson_ci,
)


CORE_5 = {
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
}


def extract_env_affordances(facts) -> dict:
    """Extract the affordances needed to verify a parsed atoms dict from info['facts'].

    Returns:
        object_types, receptacle_types, toggleable_types: sets of canonical
            class names (e.g. "Bowl", "DeskLamp");
        has_fridge / has_microwave / has_sink / has_stove: bool;
        raw_objects / raw_receptacles / raw_toggleables: sorted instance
            ids such as "bowl 1".
    """
    canon = {ot.lower(): ot for ot in OBJECT_TYPES}

    obj_types: set[str] = set()       # canonical, e.g. "Bowl"
    raw_objects: set[str] = set()      # "bowl 1"
    recep_types: set[str] = set()
    raw_receptacles: set[str] = set()
    toggleable_types: set[str] = set()
    raw_toggleables: set[str] = set()
    type_names_seen: set[str] = set()  # raw lowercased class names

    for f in facts:
        if f.name == "objecttype":
            inst_var, type_var = f.arguments
            inst = inst_var.name.strip()
            type_lower = type_var.name.replace("type", "").strip()
            type_names_seen.add(type_lower)
            raw_objects.add(inst)
            if type_lower in canon:
                obj_types.add(canon[type_lower])
        elif f.name == "receptacletype":
            inst_var, type_var = f.arguments
            inst = inst_var.name.strip()
            type_lower = type_var.name.replace("type", "").strip()
            type_names_seen.add(type_lower)
            raw_receptacles.add(inst)
            if type_lower in canon:
                recep_types.add(canon[type_lower])
        elif f.name == "toggleable":
            (inst_var,) = f.arguments
            inst = inst_var.name.strip()
            raw_toggleables.add(inst)
            # Toggleables carry no type predicate; the class name is the
            # instance-id prefix (e.g. "desklamp 1").
            class_part = inst.rsplit(" ", 1)[0] if " " in inst else inst
            if class_part in canon:
                toggleable_types.add(canon[class_part])
            else:
                if "lamp" in class_part or "light" in class_part:
                    for c in OBJECT_TYPES:
                        if c.lower() == class_part:
                            toggleable_types.add(c)

    has_fridge = "fridge" in type_names_seen
    has_microwave = "microwave" in type_names_seen
    has_sink = ("sink" in type_names_seen) or ("sinkbasin" in type_names_seen)
    has_stove = ("stoveburner" in type_names_seen) or ("stoveknob" in type_names_seen)

    return {
        "object_types": obj_types,
        "receptacle_types": recep_types,
        "toggleable_types": toggleable_types,
        "has_fridge": has_fridge,
        "has_microwave": has_microwave,
        "has_sink": has_sink,
        "has_stove": has_stove,
        "raw_objects": sorted(raw_objects),
        "raw_receptacles": sorted(raw_receptacles),
        "raw_toggleables": sorted(raw_toggleables),
    }


def verify_atoms(atoms: dict, env_aff: dict, task_type: str) -> list[str]:
    """Check parsed atoms against env affordances; return error strings (empty when valid)."""
    errors: list[str] = []
    if not isinstance(atoms, dict):
        return ["parse did not produce a JSON object"]

    norm_atoms = normalize_pred_extended(atoms)
    obj_t = (norm_atoms.get("object_target") or "").strip()
    mrecep_t = (norm_atoms.get("mrecep_target") or "").strip()
    parent_t = (norm_atoms.get("parent_target") or "").strip()
    toggle_t = (norm_atoms.get("toggle_target") or "").strip()

    if not obj_t:
        errors.append("object_target is empty; what is the task picking up?")
    elif obj_t not in env_aff["object_types"]:
        errors.append(
            f'object_target "{obj_t}" not in env. '
            f'Available objects in this room: '
            f'{sorted(env_aff["object_types"])}'
        )

    if task_type != "look_at_obj_in_light":
        if not parent_t:
            errors.append(
                "parent_target is empty; "
                "this task type requires a destination receptacle."
            )
        elif parent_t not in env_aff["receptacle_types"]:
            errors.append(
                f'parent_target "{parent_t}" not a receptacle in env. '
                f'Available receptacles: {sorted(env_aff["receptacle_types"])}'
            )

    if task_type == "look_at_obj_in_light":
        if not toggle_t:
            errors.append(
                "toggle_target is empty; "
                "look_at_obj_in_light requires a light source."
            )
        elif toggle_t not in env_aff["toggleable_types"]:
            errors.append(
                f'toggle_target "{toggle_t}" not toggleable in env. '
                f'Available toggleables: {sorted(env_aff["toggleable_types"])}'
            )
    elif toggle_t:
        # A stray toggle_target on a non-light task is harmless; not an error.
        pass

    if norm_atoms.get("object_cool") and not env_aff["has_fridge"]:
        errors.append(
            "object_cool=true requires a fridge in env, but none present."
        )
    if norm_atoms.get("object_heat") and not env_aff["has_microwave"]:
        errors.append(
            "object_heat=true requires a microwave in env, but none present."
        )
    if norm_atoms.get("object_clean") and not env_aff["has_sink"]:
        errors.append(
            "object_clean=true requires a sink/sinkbasin in env, but none present."
        )

    if mrecep_t and mrecep_t not in env_aff["receptacle_types"] \
            and mrecep_t not in env_aff["object_types"]:
        # A movable receptacle (Bowl, Cup) is an object type, so both sets count.
        errors.append(
            f'mrecep_target "{mrecep_t}" not found in env '
            f'(neither as object nor receptacle).'
        )

    return errors


def build_feedback_prompt(nl: str, prev_atoms: dict, errors: list[str],
                            env_aff: dict, task_type: str, iter_num: int) -> str:
    """Build the re-parse prompt for iteration iter_num >= 1 with the verifier's errors."""
    base = make_enum_prompt(nl)

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
constraints above. Output ONLY valid JSON, no commentary.

Task to re-parse: "{nl}"
"""
    return base + feedback


def run_one_task_iterative(task_id: str, task_type: str, nl: str,
                             gt_for_logging: dict, max_steps: int = 80,
                             max_iters: int = 3, verbose: bool = False) -> dict:
    """Run the iterative parse and hand-coded TW execution on one ALFWorld task."""
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
        name=f"agcp_iter_{abs(hash(task_id))}")
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
            prompt = make_enum_prompt(nl)
        else:
            prompt = build_feedback_prompt(
                nl, last_atoms, errors_per_iter[-1],
                env_aff, task_type, k)
        atoms_k = _call_parse_llm(prompt)
        llm_calls += 1
        if not isinstance(atoms_k, dict) or "_error" in atoms_k:
            # Non-JSON output counts as a failed iteration.
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
            print(f"    iter {k+1}: {len(errs)} errors — {errs[0]}")

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
                 "seconds": time.time() - t0}

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
        default="runs/alfworld_closed_loop_e6_full_unfiltered.json",
        help="E6 unfiltered results: provides task_id+task_type list of N=119")
    ap.add_argument(
        "--e4_cache_json",
        default="runs/alfworld_cloud_claude_sonnet.json",
        help="Sonnet E4 cache: provides task_desc and gt-for-logging")
    ap.add_argument("--out_json",
                    default="runs/alfworld_iterative_e4.json")
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--provider", default="claude_sonnet",
                    choices=["claude_sonnet", "gpt4o", "openai_mini",
                             "claude_haiku"],
                    help="parse-LLM for the iterative loop "
                         "(cross-model replication)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global _PROVIDER
    _PROVIDER = {
        "claude_sonnet": ("anthropic", "claude-sonnet-4-6"),
        "claude_haiku": ("anthropic", "claude-haiku-4-5"),
        "gpt4o": ("openai", "gpt-4o"),
        "openai_mini": ("openai", "gpt-4o-mini"),
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

    print(f"Loaded {len(core5)} core-5 results; {len(candidates)} have all "
          f"required inputs.")
    candidates = candidates[args.start_idx: args.start_idx + args.max_n]
    print(f"Running iterative E4 on N={len(candidates)} tasks "
          f"(max_iters={args.max_iters}, max_steps={args.max_steps})")

    results = []
    n_planner_ok = 0
    n_exec_ok = 0
    iters_distrib: dict[int, int] = {}
    n_verified = 0
    total_llm_calls = 0
    t_start = time.time()
    for i, c in enumerate(candidates, 1):
        task_id = c["task_id"]
        task_type = c["task_type"]
        nl = c["task_desc"]
        gt = c["gt"]
        print(f"\n[{i}/{len(candidates)}] {task_id}")
        print(f"  task_type={task_type}")
        print(f"  nl={nl[:80]!r}")
        try:
            res = run_one_task_iterative(
                task_id, task_type, nl, gt,
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
        results.append(res)
        iters_distrib[res["iterations_used"]] = (
            iters_distrib.get(res["iterations_used"], 0) + 1)
        if res.get("verified"):
            n_verified += 1
        total_llm_calls += res.get("llm_calls", 0)
        if res.get("planner_success"):
            n_planner_ok += 1
        if res.get("execution_success"):
            n_exec_ok += 1
        print(f"  -> iters={res.get('iterations_used')} "
              f"verified={res.get('verified')} "
              f"planner_ok={res.get('planner_success')} "
              f"exec_ok={res.get('execution_success')} "
              f"plan_len={res.get('plan_length')}")

        # Write after every task so partial runs are recoverable.
        n = len(results)
        ci_e = wilson_ci(n_exec_ok, n)
        out = {
            "source_json": args.source_json,
            "e4_cache_json": args.e4_cache_json,
            "max_iters": args.max_iters,
            "max_steps": args.max_steps,
            "n_total": n,
            "n_verified_within_max_iters": n_verified,
            "n_planner_success": n_planner_ok,
            "n_execution_success": n_exec_ok,
            "execution_success_rate": n_exec_ok / n,
            "execution_success_ci95": list(ci_e),
            "iterations_distribution": iters_distrib,
            "total_llm_calls": total_llm_calls,
            "wall_seconds": time.time() - t_start,
            "results": results,
        }
        json.dump(out, open(args.out_json, "w"), indent=2, default=str)

    n = len(results)
    if n == 0:
        print("\nNo tasks attempted — aborting.")
        return
    ci_e = wilson_ci(n_exec_ok, n)
    ci_p = wilson_ci(n_planner_ok, n)
    print("\n" + "=" * 70)
    print("ITERATIVE E4 on ALFWorld TextWorld (HandCoded TW)")
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
          f"(avg {total_llm_calls/n:.2f} per task)")
    print(f"  Wall time: {time.time()-t_start:.1f}s")
    print(f"  Output: {args.out_json}")
    print("=" * 70)

    # Per-task-type breakdown
    print("\nPer-task-type breakdown:")
    from collections import defaultdict
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


if __name__ == "__main__":
    main()
