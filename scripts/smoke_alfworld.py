"""Smoke test: instantiate AlfredTWEnv, reset, and take one step."""
import os
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


def main():
    config = {
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "hybrid": {
                "start_eps": 100000,
                "thor_prob": 0.5,
                "eval_mode": "tw",
            },
            "thor": {
                "screen_width": 300,
                "screen_height": 300,
            },
        },
        "dataset": {
            "data_path": os.path.expanduser(
                "~/.cache/alfworld/json_2.1.1/train"),
            "eval_id_data_path": os.path.expanduser(
                "~/.cache/alfworld/json_2.1.1/valid_seen"),
            "eval_ood_data_path": os.path.expanduser(
                "~/.cache/alfworld/json_2.1.1/valid_unseen"),
            "num_train_games": 5,
            "num_eval_games": 1,
        },
        "logic": {
            "domain": os.path.expanduser(
                "~/.cache/alfworld/logic/alfred.pddl"),
            "grammar": os.path.expanduser(
                "~/.cache/alfworld/logic/alfred.twl2"),
        },
        "general": {
            "training_method": "dagger",
        },
        "dagger": {
            "training": {"max_nb_steps_per_episode": 50},
        },
    }

    env = AlfredTWEnv(config, train_eval="train")
    tw = env.init_env(batch_size=1)
    obs, info = tw.reset()
    print("=== info keys ===", list(info.keys()))
    print("\n=== facts (if present) ===")
    if "facts" in info:
        facts = info["facts"][0]
        print(f"n_facts: {len(facts)}")
        for f in facts[:15]:
            print(f"  {f}")
    print("\n=== admissible_commands ===")
    print(info["admissible_commands"][0][:8])

    action = info["admissible_commands"][0][0]
    print(f"\n=== STEP: {action} ===")
    obs2, scores, dones, info2 = tw.step([action])
    if "facts" in info2:
        new_facts = info2["facts"][0]
        added = set(map(str, new_facts)) - set(map(str, info["facts"][0]))
        removed = set(map(str, info["facts"][0])) - set(map(str, new_facts))
        print(f"  added facts: {sorted(added)[:5]}")
        print(f"  removed facts: {sorted(removed)[:5]}")

if __name__ == "__main__":
    main()
