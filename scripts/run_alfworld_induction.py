"""Induce lifted action schemas from random walks in ALFWorld.

AlfredTWEnv (TextWorld backend) exposes the PDDL-form state in
`info["facts"]`. Facts are normalised to `pred(arg_id,...)`, admissible
commands are parsed into (action, args), and the (sb, action, args, sa)
transitions feed induce_lifted_models_positional. Writes models to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

from scripts.induce_positional import induce_lifted_models_positional


def normalize_fact(f: str) -> str:
    """Convert an ALFWorld fact `pred(arg1 : type1, ...)` to `pred(arg1_id,...)`,
    dropping the type annotations and replacing spaces in names."""
    s = str(f).strip()
    m = re.match(r"^(\w+)\((.*)\)$", s)
    if not m:
        return s
    pred = m.group(1)
    inner = m.group(2)
    parts = [p.strip() for p in inner.split(",")]
    arg_names = []
    for p in parts:
        name = re.sub(r"\s*:\s*\w+", "", p).strip()
        # `loc 19` -> `loc_19`
        name = name.replace(" ", "_")
        arg_names.append(name)
    if not arg_names:
        return pred
    return f"{pred}({','.join(arg_names)})"


def normalize_state(facts) -> frozenset[str]:
    return frozenset(normalize_fact(f) for f in facts)


def parse_admissible_command(cmd: str) -> tuple[str, tuple[str, ...]]:
    """Parse 'go to X', 'open X', 'take Y from X', 'put Y in/on X'
    etc. into (action_name, args)."""
    cmd = cmd.strip().lower()
    m = re.match(r"^go to (.+)$", cmd)
    if m: return ("goto", (m.group(1).replace(" ", "_"),))
    m = re.match(r"^open (.+)$", cmd)
    if m: return ("open", (m.group(1).replace(" ", "_"),))
    m = re.match(r"^close (.+)$", cmd)
    if m: return ("close", (m.group(1).replace(" ", "_"),))
    m = re.match(r"^take (.+) from (.+)$", cmd)
    if m: return ("take", (m.group(1).replace(" ", "_"),
                              m.group(2).replace(" ", "_")))
    m = re.match(r"^put (.+) (?:in|on) (.+)$", cmd)
    if m: return ("put", (m.group(1).replace(" ", "_"),
                            m.group(2).replace(" ", "_")))
    # ALFWorld's admissible list phrases put as "move X to Y"
    m = re.match(r"^move (.+) to (.+)$", cmd)
    if m: return ("put", (m.group(1).replace(" ", "_"),
                            m.group(2).replace(" ", "_")))
    m = re.match(r"^use (.+)$", cmd)
    if m: return ("use", (m.group(1).replace(" ", "_"),))
    m = re.match(r"^examine (.+)$", cmd)
    if m: return ("examine", (m.group(1).replace(" ", "_"),))
    m = re.match(r"^clean (.+) with (.+)$", cmd)
    if m: return ("clean_with", (m.group(1).replace(" ", "_"),
                                    m.group(2).replace(" ", "_")))
    m = re.match(r"^heat (.+) with (.+)$", cmd)
    if m: return ("heat_with", (m.group(1).replace(" ", "_"),
                                   m.group(2).replace(" ", "_")))
    m = re.match(r"^cool (.+) with (.+)$", cmd)
    if m: return ("cool_with", (m.group(1).replace(" ", "_"),
                                   m.group(2).replace(" ", "_")))
    m = re.match(r"^slice (.+) with (.+)$", cmd)
    if m: return ("slice_with", (m.group(1).replace(" ", "_"),
                                    m.group(2).replace(" ", "_")))
    # Fallback: first word = action, rest = single arg
    parts = cmd.split(maxsplit=1)
    if len(parts) == 1:
        return (parts[0], ())
    return (parts[0], (parts[1].replace(" ", "_"),))


def collect_rollouts(env_handle, n_episodes: int = 5,
                       n_steps_per_episode: int = 50,
                       seed: int = 0) -> list:
    """Collect transitions via random walks on alfworld env."""
    rng = random.Random(seed)
    transitions = []
    for ep in range(n_episodes):
        obs, info = env_handle.reset()
        sb = normalize_state(info["facts"][0])
        admissible = info["admissible_commands"][0]
        for _ in range(n_steps_per_episode):
            if not admissible: break
            cmd = rng.choice(admissible)
            action, args = parse_admissible_command(cmd)
            obs, scores, dones, info = env_handle.step([cmd])
            sa = normalize_state(info["facts"][0])
            transitions.append((sb, action, args, sa))
            sb = sa
            admissible = info["admissible_commands"][0]
            if dones[0]: break
    return transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=5)
    ap.add_argument("--n_episodes_per_game", type=int, default=3)
    ap.add_argument("--n_steps_per_episode", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = {
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "hybrid": {"start_eps": 100000, "thor_prob": 0.5,
                        "eval_mode": "tw"},
            "thor": {"screen_width": 300, "screen_height": 300},
        },
        "dataset": {
            "data_path": os.path.expanduser(
                "~/.cache/alfworld/json_2.1.1/train"),
            "eval_id_data_path": os.path.expanduser(
                "~/.cache/alfworld/json_2.1.1/valid_seen"),
            "eval_ood_data_path": os.path.expanduser(
                "~/.cache/alfworld/json_2.1.1/valid_unseen"),
            "num_train_games": args.n_games,
            "num_eval_games": 1,
        },
        "logic": {
            "domain": os.path.expanduser(
                "~/.cache/alfworld/logic/alfred.pddl"),
            "grammar": os.path.expanduser(
                "~/.cache/alfworld/logic/alfred.twl2"),
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }
    print("=" * 60)
    print("ALFWORLD INDUCTION")
    print("=" * 60)
    env = AlfredTWEnv(config, train_eval="train")
    tw = env.init_env(batch_size=1)

    transitions = collect_rollouts(
        tw, n_episodes=args.n_games * args.n_episodes_per_game,
        n_steps_per_episode=args.n_steps_per_episode,
    )
    trs_ok = [t for t in transitions if t[0] != t[-1]]
    print(f"\n[1] Collected {len(transitions)} transitions "
          f"({len(trs_ok)} non-noop)")

    by_action = defaultdict(int)
    for _sb, action, _args, _sa in trs_ok:
        by_action[action] += 1
    print(f"[2] Action distribution:")
    for a, c in sorted(by_action.items(), key=lambda x: -x[1]):
        print(f"    {a}: {c}")

    print(f"\n[3] Inducing lifted action schemas...")
    models = induce_lifted_models_positional(trs_ok)
    for action, m in sorted(models.items()):
        print(f"\n  {action}  (n={m['n_examples']})")
        for k in ("pre_pos", "eff_add", "eff_del"):
            atoms = sorted(m[k])
            if not atoms:
                print(f"    {k}: (empty)")
                continue
            print(f"    {k}: {atoms[:6]}{'...' if len(atoms) > 6 else ''}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_transitions": len(trs_ok),
            "action_counts": dict(by_action),
            "models": {a: {k: sorted(v) if isinstance(v, (set, frozenset))
                              else v
                              for k, v in m.items()}
                          for a, m in models.items()},
        }, f, indent=2, default=str)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
