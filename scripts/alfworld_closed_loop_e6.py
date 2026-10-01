"""Closed-loop execution of LLM-parsed ALFWorld goals in the TextWorld backend.

For each task whose NL parse reached smart_f1 >= --min_f1, the parsed
pddl_params (never the ground-truth ones) parameterise ALFWorld's hand-coded
TextWorld policy; the plan is executed step by step and success is read from
the env's `won` flag. Output: runs/alfworld_closed_loop_e6.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import traceback
from pathlib import Path


SUPPORTED_TASK_TYPES = {
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_two_obj_and_place",
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


def run_one_task(task_id: str, task_type: str, parsed_params: dict,
                  gt_params: dict, max_steps: int = 80,
                  verbose: bool = False) -> dict:
    """Plan and execute one ALFWorld task from its parsed params."""
    import textworld
    import textworld.gym  # noqa: F401  (registers .gym.make)
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredInfos,
    )
    from alfworld.agents.expert import HandCodedTWAgent
    import alfworld.gen.constants as constants

    t0 = time.time()
    game_file = find_game_file(task_id)
    if game_file is None:
        return {"task_id": task_id, "stage_failed": "game_lookup",
                 "error": "no game.tw-pddl found",
                 "planner_success": False, "execution_success": False,
                 "plan_length": 0, "seconds": time.time() - t0}

    if task_type not in SUPPORTED_TASK_TYPES:
        return {"task_id": task_id, "stage_failed": "unsupported_task_type",
                 "error": f"{task_type} not in handcoded_expert_tw",
                 "planner_success": False, "execution_success": False,
                 "plan_length": 0, "seconds": time.time() - t0}

    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, facts=True, extras=["gamefile"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    env_id = textworld.gym.register_games(
        [game_file], request_infos, batch_size=1, asynchronous=False,
        max_episode_steps=max_steps, wrappers=wrappers,
        name=f"agcp_{abs(hash(task_id))}")
    env = textworld.gym.make(env_id)
    try:
        obs, info = env.reset()
    except Exception as e:
        return {"task_id": task_id, "stage_failed": "env_reset",
                 "error": str(e), "planner_success": False,
                 "execution_success": False, "plan_length": 0,
                 "seconds": time.time() - t0}
    if info["won"][0]:
        env.close()
        return {"task_id": task_id, "stage_failed": None,
                 "error": "won_at_reset", "planner_success": True,
                 "execution_success": True, "plan_length": 0,
                 "seconds": time.time() - t0}

    # The policy is parameterised by the parsed params, not the GT ones.
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
        return {"task_id": task_id, "stage_failed": "planner_init",
                 "error": str(e), "planner_success": False,
                 "execution_success": False, "plan_length": 0,
                 "seconds": time.time() - t0,
                 "parsed_params": parsed_params, "gt_params": gt_params,
                 "task_type": task_type}

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
        if verbose and (step + 1) % 10 == 0:
            print(f"    step {step+1}: {action}")
    env.close()

    return {
        "task_id": task_id,
        "task_type": task_type,
        "parsed_params": parsed_params,
        "gt_params": gt_params,
        "planner_success": (not planner_failed),
        "planner_error": planner_error,
        "execution_success": bool(won),
        "plan_length": len(trajectory),
        "trajectory_head": trajectory[:5],
        "trajectory_tail": trajectory[-5:] if trajectory else [],
        "stage_failed": (None if won else
                          ("planner" if planner_failed else "goal_check")),
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_json",
                    default="runs/alfworld_cloud_claude_sonnet.json",
                    help="E4 NL-parse results to pull parsed_params from")
    ap.add_argument("--out_json",
                    default="runs/alfworld_closed_loop_e6.json")
    ap.add_argument("--min_f1", type=float, default=0.95)
    ap.add_argument("--max_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)

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
    print(f"E4 high-F1 candidates: {len(high)} "
          f"(supported+game_file: {len(cand)})")
    cand = cand[: args.max_n]
    print(f"Running closed-loop on N={len(cand)} tasks "
          f"(max_steps={args.max_steps})")

    results = []
    n_planner_ok = 0
    n_exec_ok = 0
    t_start = time.time()
    for i, r in enumerate(cand, 1):
        task_id = r["task_id"]
        task_type = r["task_type"]
        # Use the LLM-parsed params; gt is carried along for the log only.
        parsed = r["llm_pred"]
        gt = r["gt"]
        print(f"\n[{i}/{len(cand)}] {task_id}")
        print(f"  task_type={task_type}")
        print(f"  parsed (E4): {parsed}")
        try:
            res = run_one_task(task_id, task_type, parsed, gt,
                                  max_steps=args.max_steps,
                                  verbose=args.verbose)
        except Exception as e:
            traceback.print_exc()
            res = {"task_id": task_id, "task_type": task_type,
                    "stage_failed": "uncaught", "error": str(e),
                    "planner_success": False,
                    "execution_success": False, "plan_length": 0,
                    "parsed_params": parsed, "gt_params": gt,
                    "seconds": 0.0}
        results.append(res)
        if res.get("planner_success"):
            n_planner_ok += 1
        if res.get("execution_success"):
            n_exec_ok += 1
        print(f"  -> planner_ok={res.get('planner_success')} "
              f"exec_ok={res.get('execution_success')} "
              f"plan_len={res.get('plan_length')} "
              f"stage_failed={res.get('stage_failed')}")
        # Write after every task so partial runs are recoverable.
        n = len(results)
        ci_p = wilson_ci(n_planner_ok, n)
        ci_e = wilson_ci(n_exec_ok, n)
        out = {
            "parsed_json": args.parsed_json,
            "min_f1": args.min_f1,
            "n_total": n,
            "n_planner_success": n_planner_ok,
            "n_execution_success": n_exec_ok,
            "planner_success_rate": n_planner_ok / n,
            "planner_success_ci95": list(ci_p),
            "execution_success_rate": n_exec_ok / n,
            "execution_success_ci95": list(ci_e),
            "wall_seconds": time.time() - t_start,
            "results": results,
        }
        json.dump(out, open(args.out_json, "w"), indent=2, default=str)

    n = len(results)
    if n == 0:
        print("\nNo tasks attempted — aborting.")
        return
    ci_p = wilson_ci(n_planner_ok, n)
    ci_e = wilson_ci(n_exec_ok, n)
    print("\n" + "=" * 60)
    print("AGCP closed-loop E6 on ALFWorld TextWorld")
    print(f"  Parsed source: {args.parsed_json}")
    print(f"  Min E4 smart_f1: {args.min_f1}")
    print(f"  N tasks (post-filter): {n}")
    print(f"  Planner reached goal: {n_planner_ok}/{n} "
          f"= {100*n_planner_ok/n:.1f}% "
          f"(95% CI {100*ci_p[0]:.1f}-{100*ci_p[1]:.1f}%)")
    print(f"  Env 'won' (executed): {n_exec_ok}/{n} "
          f"= {100*n_exec_ok/n:.1f}% "
          f"(95% CI {100*ci_e[0]:.1f}-{100*ci_e[1]:.1f}%)")
    print(f"  Wall time: {time.time()-t_start:.1f}s")
    print(f"  Output: {args.out_json}")
    print("=" * 60)

    # Failure breakdown
    from collections import Counter
    fail_stages = Counter(r.get("stage_failed") for r in results
                            if not r.get("execution_success"))
    print("Failure stage breakdown:")
    for stage, c in fail_stages.most_common():
        print(f"  {stage}: {c}")


if __name__ == "__main__":
    main()
