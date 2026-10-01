"""ALFWorld closed loop (E6) with a planner over AGCP-induced schemas.

Replaces the HandCoded TextWorld expert of alfworld_closed_loop_e6.py with
greedy best-first search over schemas induced from random rollouts (cached
in runs/alfworld_induction_v2.json). Goals come from the LLM-parsed params,
never ground truth; success is the TextWorld `won` flag. Writes --out_json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.induce_positional import induce_lifted_models_positional  # noqa: E402
from scripts.run_alfworld_induction import parse_admissible_command  # noqa: E402

INDUCTION_CACHE = "runs/alfworld_induction_v2.json"

DYNAMIC_PREDS = {"at_recep", "holds", "holdsany", "inreceptacle",
                 "opened", "checked", "iscool", "ishot", "isclean",
                 "issliced", "toggled", "istoggled"}

# Actions the exploration loop must cover before induction stops
# (all appear in core-5 plans); coverage measured in examples.
TARGET_ACTIONS = {"goto": 5, "take": 5, "put": 5, "open": 5,
                  "use": 5, "clean_with": 5, "heat_with": 5,
                  "cool_with": 5}


# Fact normalization
def _strip_types(f: str):
    s = str(f).strip()
    m = re.match(r"^(\w+)\((.*)\)$", s)
    if not m:
        return s, ()
    pred = m.group(1)
    args = []
    for p in m.group(2).split(","):
        name = re.sub(r"\s*:\s*\w+", "", p).strip().replace(" ", "_")
        args.append(name)
    return pred, tuple(args)


def normalize_state_v2(facts) -> frozenset[str]:
    raw = [_strip_types(f) for f in facts]
    loc2receps: dict[str, list[str]] = {}
    for pred, args in raw:
        if pred == "receptacleatlocation" and len(args) == 2:
            loc2receps.setdefault(args[1], []).append(args[0])
    out = set()
    for pred, args in raw:
        if pred.startswith("not_"):
            continue
        if pred == "objectatlocation":
            continue
        if pred == "atlocation" and len(args) == 2:
            for r in loc2receps.get(args[1], []):
                out.add(f"at_recep({args[0]},{r})")
            continue
        if pred == "receptacleatlocation":
            continue  # folded into at_recep rewriting
        out.add(f"{pred}({','.join(args)})" if args else pred)
    return frozenset(out)


# Induction (cached)
def _make_config(n_games: int, task_types):
    return {
        "env": {"type": "AlfredTWEnv", "regen_game_files": False,
                "domain_randomization": False,
                "task_types": list(task_types),
                "expert_timeout_steps": 150,
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
                "hybrid": {"start_eps": 100000, "thor_prob": 0.5,
                           "eval_mode": "tw"},
                "thor": {"screen_width": 300, "screen_height": 300}},
        "dataset": {"data_path": os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/train"),
                    "eval_id_data_path": os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"),
                    "eval_ood_data_path": os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_unseen"),
                    "num_train_games": n_games, "num_eval_games": 1},
        "logic": {"domain": os.path.expanduser(
                      "~/.cache/alfworld/logic/alfred.pddl"),
                  "grammar": os.path.expanduser(
                      "~/.cache/alfworld/logic/alfred.twl2")},
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }
def _run_batch(config, n_episodes: int, n_steps: int, rng,
               action_counts: dict, noop_types: set,
               transitions: list) -> None:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    env = AlfredTWEnv(config, train_eval="train")
    tw = env.init_env(batch_size=1)
    for ep in range(n_episodes):
        obs, info = tw.reset()
        sb = normalize_state_v2(info["facts"][0])
        admissible = info["admissible_commands"][0]
        for _ in range(n_steps):
            if not admissible:
                break
            # Stratified exploration over state-changing action types: pick
            # the least-sampled type among admissible commands; types seen
            # to have empty effects (look, inventory, help) get only an
            # epsilon share. Uses global counts and observed
            # effect-emptiness only, no task or goal knowledge.
            typed = [(parse_admissible_command(c)[0], c)
                     for c in admissible]
            eff = [(t, c) for t, c in typed if t not in noop_types]
            # While an object is held, demote `put` with p=0.6 so transport
            # chains (goto appliance, open, interact) can form; the signal
            # is the observed holds(agent1, .) atom only.
            holding = any(a.startswith("holds(agent1,") for a in sb)
            if holding and rng.random() < 0.6:
                eff_np = [(t, c) for t, c in eff if t != "put"]
                if eff_np:
                    eff = eff_np
            if eff and rng.random() > 0.05:
                min_ct = min(action_counts.get(t, 0) for t, _ in eff)
                pool = [c for t, c in eff
                        if action_counts.get(t, 0) == min_ct]
            else:
                pool = [c for _, c in typed]
            cmd = rng.choice(pool)
            action, args = parse_admissible_command(cmd)
            action_counts[action] = action_counts.get(action, 0) + 1
            obs, sc, dones, info = tw.step([cmd])
            sa = normalize_state_v2(info["facts"][0])
            transitions.append((sb, action, args, sa))
            sb = sa
            admissible = info["admissible_commands"][0]
            if dones[0]:
                break


def collect_and_induce(n_games: int, n_episodes_per_game: int,
                       n_steps: int, seed: int = 0,
                       max_batches: int = 8) -> dict:
    """Run exploration batches, rotating the train-split task types so
    appliance- and lamp-rich scenes appear, until every TARGET_ACTION has
    enough examples or max_batches is reached; induce and cache the models."""
    rng = random.Random(seed)
    transitions: list = []
    action_counts: dict[str, int] = {}
    noop_types: set = set()
    rotations = [(1, 2, 3, 4, 5, 6), (4, 5), (2,), (3, 4, 5), (2, 6),
                 (4, 5), (2,), (1, 2, 3, 4, 5, 6)]
    for b in range(max_batches):
        need = {a for a, k in TARGET_ACTIONS.items()
                if sum(1 for t in transitions
                       if t[1] == a and t[0] != t[-1]) < k}
        if b > 0 and not need:
            break
        cfg = _make_config(n_games, rotations[b % len(rotations)])
        print(f"  [batch {b}] task_types={rotations[b % len(rotations)]} "
              f"still need: {sorted(need) if b else 'ALL'}", flush=True)
        _run_batch(cfg, n_games * n_episodes_per_game, n_steps, rng,
                   action_counts, noop_types, transitions)
        # update no-op set from observed effects so far
        by_a: dict[str, int] = {}
        for sb, a, g, sa in transitions:
            if sb != sa:
                by_a[a] = by_a.get(a, 0) + 1
        noop_types = {a for a in action_counts
                      if action_counts[a] >= 10 and by_a.get(a, 0) == 0}
        print(f"  [batch {b}] counts={dict(sorted(action_counts.items(), key=lambda x: -x[1]))} "
              f"noops={sorted(noop_types)}", flush=True)
    trs_ok = [t for t in transitions if t[0] != t[-1]]
    models = induce_lifted_models_positional(trs_ok)
    out = {"n_transitions": len(trs_ok),
           "action_counts": action_counts,
           "models": {a: {k: sorted(v) if isinstance(v, (set, frozenset))
                          else v for k, v in m.items()}
                      for a, m in models.items()}}
    Path(INDUCTION_CACHE).parent.mkdir(parents=True, exist_ok=True)
    with open(INDUCTION_CACHE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# Planner over induced models
def _ground(atom_tmpl: str, args: tuple[str, ...]) -> str:
    s = atom_tmpl
    for i, a in enumerate(args, start=1):
        s = s.replace("{a%d}" % i, a)
    return s


def _instances_of(state: frozenset[str], canon_type: str,
                  kind: str) -> list[str]:
    """Scene instances matching a parsed canonical type, via
    objecttype/receptacletype facts. canon_type e.g. 'Mug',
    instance ids e.g. 'mug_1' with facts objecttype(mug_1,mugtype)."""
    t = canon_type.strip().lower()
    if not t:
        return []
    pred = "objecttype" if kind == "obj" else "receptacletype"
    out = []
    for a in state:
        m = re.match(rf"^{pred}\((\w+),(\w+)\)$", a)
        if m and m.group(2) == f"{t}type":
            out.append(m.group(1))
    return sorted(out)


class InducedModelPlanner:
    def __init__(self, models: dict, verbose: bool = False,
                 min_examples: int = 3):
        # keep only actions with usable schemas: non-empty effects AND
        # enough transition examples that the intersection is not a
        # single-example overfit
        self.models = {a: m for a, m in models.items()
                       if (m.get("eff_add") or m.get("eff_del"))
                       and m.get("n_examples", 0) >= min_examples}
        self.verbose = verbose

    def _arity(self, m: dict) -> int:
        mx = 0
        for k in ("pre_pos", "eff_add", "eff_del"):
            for t in m.get(k, []):
                for i in (1, 2, 3):
                    if "{a%d}" % i in t:
                        mx = max(mx, i)
        return mx

    def _candidate_args(self, state, relevant_objs, relevant_receps):
        receps = sorted({m.group(1) for a in state
                         for m in [re.match(r"^receptacletype\((\w+),", a)]
                         if m})
        return relevant_objs, sorted(set(relevant_receps) | set(receps))

    def plan(self, state0: frozenset[str], goal_sets: list[set[str]],
             relevant_objs: list[str], relevant_receps: list[str],
             max_depth: int = 14, max_nodes: int = 300000):
        """Greedy best-first search (h = min unsatisfied goal atoms
        over goal candidates) to any state satisfying ANY goal_set.
        Grounding is goal-regressed: object args range over
        relevant_objs; put/interaction receptacle args over receps
        that appear in a goal candidate OR are named by the static
        preconditions of an operator whose add-effect predicate is
        needed by the goal (derived from the induced schemas, not a
        hand list); goto ranges over all scene receptacles."""
        statics = frozenset(a for a in state0
                            if a.split("(")[0] not in DYNAMIC_PREDS)
        dyn0 = frozenset(a for a in state0
                         if a.split("(")[0] in DYNAMIC_PREDS)
        objs, receps = self._candidate_args(state0, relevant_objs,
                                            relevant_receps)

        # goal-regression receptacle relevance
        needed_preds = {g.split("(")[0] for gs in goal_sets for g in gs}
        goal_receps = set()
        for gs in goal_sets:
            for g in gs:
                m2 = re.match(r"^inreceptacle\(\w+,(\w+)\)$", g)
                if m2:
                    goal_receps.add(m2.group(1))
        support_receps = set()
        for action, m in self.models.items():
            adds = {re.sub(r"\{a\d\}", "", t).split("(")[0]
                    for t in m.get("eff_add", [])}
            if not (adds & needed_preds):
                continue
            for t in m.get("pre_pos", []):
                m3 = re.match(r"^receptacletype\(\{a\d\},(\w+)\)$", t)
                if m3:
                    for a in state0:
                        m4 = re.match(rf"^receptacletype\((\w+),{m3.group(1)}\)$", a)
                        if m4:
                            support_receps.add(m4.group(1))
        target_receps = sorted(goal_receps | support_receps)

        ground_ops = []
        for action, m in self.models.items():
            ar = self._arity(m)
            pools = []
            if action == "goto":
                pools = [[receps]] if ar == 1 else []
            elif action in ("open", "close", "examine"):
                pools = [[receps + objs]] if ar == 1 else []
            elif action == "use":
                pools = [[objs + receps]] if ar == 1 else []
            elif action == "take":
                pools = [[objs, receps]] if ar == 2 else []
            elif action == "put":
                pools = [[objs, target_receps or receps]] if ar == 2 else []
            elif action in ("clean_with", "heat_with", "cool_with",
                            "slice_with"):
                pools = [[objs, target_receps or (receps + objs)]] \
                    if ar == 2 else []
            else:
                continue
            if not pools:
                continue
            pool = pools[0]
            def rec(prefix, k):
                if k == len(pool):
                    ground_ops.append((action, tuple(prefix)))
                    return
                for v in pool[k]:
                    rec(prefix + [v], k + 1)
            rec([], 0)

        # pre-ground schema atom templates per op, filter by statics
        ops = []
        for action, args in ground_ops:
            m = self.models[action]
            pre = [_ground(t, args) for t in m.get("pre_pos", [])]
            # static preconditions must hold in s0 (they never change)
            ok = True
            for p in pre:
                if p.split("(")[0] not in DYNAMIC_PREDS and \
                        p not in statics:
                    ok = False
                    break
            if not ok:
                continue
            add = [_ground(t, args) for t in m.get("eff_add", [])]
            dele = [_ground(t, args) for t in m.get("eff_del", [])]
            dyn_pre = [p for p in pre
                       if p.split("(")[0] in DYNAMIC_PREDS]
            ops.append((action, args, frozenset(dyn_pre),
                        frozenset(a for a in add
                                  if a.split("(")[0] in DYNAMIC_PREDS),
                        frozenset(d for d in dele
                                  if d.split("(")[0] in DYNAMIC_PREDS)))
        if self.verbose:
            print(f"    grounded ops: {len(ops)}")

        def goal_hit(dyn):
            return any(g <= dyn for g in goal_sets)

        def h(dyn):
            return min(sum(1 for a in g if a not in dyn)
                       for g in goal_sets)

        if goal_hit(dyn0):
            return []
        import heapq
        visited = {dyn0}
        ctr = 0
        heap = [(h(dyn0), 0, ctr, dyn0, [])]
        n_nodes = 0
        while heap:
            _hv, depth, _c, dyn, path = heapq.heappop(heap)
            n_nodes += 1
            if n_nodes > max_nodes or depth >= max_depth:
                continue
            for action, args, pre, add, dele in ops:
                if not pre <= dyn:
                    continue
                nd = set(dyn)
                # data-verified singleton families: replace-on-add
                if action == "goto":
                    nd = {a for a in nd
                          if not a.startswith("at_recep(agent1,")}
                nd -= dele
                nd |= add
                ndf = frozenset(nd)
                if ndf in visited:
                    continue
                visited.add(ndf)
                npath = path + [(action, args)]
                if goal_hit(ndf):
                    return npath
                ctr += 1
                heapq.heappush(heap, (h(ndf), depth + 1, ctr, ndf,
                                      npath))
        return None


# Goal templates from parsed params (type-level, existential)
def goal_candidates(task_type: str, params: dict,
                    state0: frozenset[str]) -> list[set[str]]:
    obj_t = (params.get("object_target") or "").strip()
    par_t = (params.get("parent_target") or "").strip()
    tog_t = (params.get("toggle_target") or "").strip()
    objs = _instances_of(state0, obj_t, "obj")
    pars = _instances_of(state0, par_t, "recep")
    togs = (_instances_of(state0, tog_t, "obj")
            + _instances_of(state0, tog_t, "recep"))
    mods = []
    if params.get("object_cool"):
        mods.append("iscool")
    if params.get("object_heat"):
        mods.append("ishot")
    if params.get("object_clean"):
        mods.append("isclean")
    if params.get("object_sliced"):
        mods.append("issliced")

    goals: list[set[str]] = []
    if task_type == "look_at_obj_in_light":
        for o in objs:
            for t in togs:
                goals.append({f"holds(agent1,{o})", f"istoggled({t})"})
    else:
        for o in objs:
            for p in pars:
                g = {f"inreceptacle({o},{p})"}
                for mflag in mods:
                    g.add(f"{mflag}({o})")
                goals.append(g)
    return goals


# Execution
def render_and_execute(env, info, plan, max_steps: int = 80,
                       verbose: bool = False):
    """Match each planned (action, args) against admissible commands;
    one goto-retry for interaction targets. Returns (won, n_steps,
    fail_reason)."""
    n = 0
    for action, args in plan:
        adm = info["admissible_commands"][0]
        cmd = None
        for c in adm:
            a2, g2 = parse_admissible_command(c)
            if a2 == action and tuple(g2) == tuple(args):
                cmd = c
                break
        if cmd is None and action in ("take", "put", "open", "close",
                                      "use", "clean_with", "heat_with",
                                      "cool_with", "slice_with"):
            # Generic recovery primitives, driven only by the env's
            # admissible-commands feedback (no task knowledge):
            # (1) goto the action's target receptacle and retry;
            # (2) if still blocked and the env offers `open <target>`,
            #     open it and retry once (plain intersection cannot
            #     learn openability-conditional preconditions).
            target = (args[-1] if action in ("take", "put")
                      else args[-1] if action in ("clean_with",
                                                  "heat_with",
                                                  "cool_with")
                      else args[0])
            goto_target = target
            goto_cmds = [c for c in adm
                         if parse_admissible_command(c)[0] == "goto"]
            if not any(parse_admissible_command(c)[1][:1] == (target,)
                       for c in goto_cmds):
                # target is an object (e.g. a lamp): goto its
                # CONTAINING receptacle, looked up from current facts
                for f in info["facts"][0]:
                    fs = str(f)
                    m5 = re.match(
                        rf"^inreceptacle\({re.escape(target.replace('_', ' '))}\s*:[^,]*,\s*(.+?)\s*:", fs)
                    if m5:
                        goto_target = m5.group(1).replace(" ", "_")
                        break
            for c in adm:
                a2, g2 = parse_admissible_command(c)
                if a2 == "goto" and g2 and g2[0] == goto_target:
                    obs, sc, done, info = env.step([c])
                    n += 1
                    adm = info["admissible_commands"][0]
                    break
            for c in adm:
                a2, g2 = parse_admissible_command(c)
                if a2 == action and tuple(g2) == tuple(args):
                    cmd = c
                    break
            if cmd is None:
                for c in adm:
                    a2, g2 = parse_admissible_command(c)
                    if a2 == "open" and g2 and g2[0] == target:
                        obs, sc, done, info = env.step([c])
                        n += 1
                        adm = info["admissible_commands"][0]
                        break
                for c in adm:
                    a2, g2 = parse_admissible_command(c)
                    if a2 == action and tuple(g2) == tuple(args):
                        cmd = c
                        break
        if cmd is None and action == "goto":
            # A goto that TextWorld does not list is (in every case we
            # observed) a goto to the agent's CURRENT location group —
            # TW removes go-to-here from admissible. Verify from the
            # facts and skip the redundant step.
            cur = normalize_state_v2(info["facts"][0])
            if args and f"at_recep(agent1,{args[0]})" in cur:
                continue
        if cmd is None:
            return False, n, f"not admissible: {action}{args}", info
        obs, sc, done, info = env.step([cmd])
        n += 1
        if verbose:
            print(f"      exec: {cmd}")
        if info["won"][0]:
            return True, n, "", info
        if done[0] if isinstance(done, (list, tuple)) else done:
            return bool(info["won"][0]), n, "episode done", info
        if n >= max_steps:
            return False, n, "max_steps", info
    return bool(info["won"][0]), n, "plan exhausted", info


def run_one(task_id: str, task_type: str, parsed: dict, models: dict,
            max_depth: int, verbose: bool = False) -> dict:
    import textworld
    import textworld.gym  # noqa: F401
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredInfos,
    )
    t0 = time.time()
    base = os.path.expanduser(
        f"~/.cache/alfworld/json_2.1.1/valid_seen/{task_id}")
    matches = sorted(glob.glob(os.path.join(base, "trial_*/game.tw-pddl")))
    if not matches:
        return {"task_id": task_id, "stage_failed": "game_lookup",
                "execution_success": False}
    ri = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
    eid = textworld.gym.register_games(
        [matches[0]], ri, batch_size=1, asynchronous=False,
        max_episode_steps=200,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        name=f"agcpip_{abs(hash(task_id))}")
    env = textworld.gym.make(eid)
    try:
        obs, info = env.reset()
    except Exception as e:  # noqa: BLE001
        return {"task_id": task_id, "stage_failed": "env_reset",
                "error": str(e)[:200], "execution_success": False}
    s0 = normalize_state_v2(info["facts"][0])
    goals = goal_candidates(task_type, parsed, s0)
    if not goals:
        env.close()
        return {"task_id": task_id, "task_type": task_type,
                "stage_failed": "no_goal_candidates",
                "execution_success": False,
                "seconds": round(time.time() - t0, 1)}
    rel_objs = sorted({a for g in goals for atom in g
                       for a in re.findall(r"\((?:agent1,)?(\w+)", atom)
                       if not a.startswith("agent")})
    rel_receps = []
    planner = InducedModelPlanner(models, verbose=verbose)
    plan = planner.plan(s0, goals, rel_objs, rel_receps,
                        max_depth=max_depth)
    if plan is None:
        env.close()
        return {"task_id": task_id, "task_type": task_type,
                "stage_failed": "planner_no_plan",
                "n_goal_candidates": len(goals),
                "execution_success": False,
                "seconds": round(time.time() - t0, 1)}
    won, n_steps, reason, final_info = render_and_execute(
        env, info, plan, verbose=verbose)
    row = {"task_id": task_id, "task_type": task_type,
           "plan": [f"{a}({','.join(g)})" for a, g in plan],
           "plan_length": len(plan), "n_exec_steps": n_steps,
           "execution_success": bool(won),
           "fail_reason": reason if not won else "",
           "n_goal_candidates": len(goals),
           "seconds": round(time.time() - t0, 1)}
    if not won and final_info is not None:
        final = normalize_state_v2(final_info["facts"][0])
        best = min(goals, key=lambda g: len(g - final))
        row["goal_atoms_missing"] = sorted(best - final)
    env.close()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_json",
                    default="runs/alfworld_cloud_claude_sonnet.json")
    ap.add_argument("--source_json",
                    default="runs/alfworld_closed_loop_e6_full_unfiltered.json")
    ap.add_argument("--reinduce", action="store_true")
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--n_episodes_per_game", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=60)
    ap.add_argument("--max_depth", type=int, default=14)
    ap.add_argument("--max_n", type=int, default=1000)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--task_types", nargs="+", default=None)
    ap.add_argument("--out_json",
                    default="runs/alfworld_e6_induced_planner.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.reinduce or not os.path.exists(INDUCTION_CACHE):
        print("Inducing schemas from fresh rollouts (v2 normalization)...")
        ind = collect_and_induce(args.n_games, args.n_episodes_per_game,
                                 args.n_steps)
    else:
        ind = json.load(open(INDUCTION_CACHE))
    models = ind["models"]
    print(f"induced models: { {a: len(m.get('eff_add', [])) for a, m in models.items()} }")

    core5 = {"look_at_obj_in_light", "pick_and_place_simple",
             "pick_clean_then_place_in_recep",
             "pick_cool_then_place_in_recep",
             "pick_heat_then_place_in_recep"}
    src = json.load(open(args.source_json))["results"]
    e4 = {r["task_id"]: r for r in
          json.load(open(args.parsed_json))["rows"]}
    tasks = []
    seen = set()
    for r in src:
        if r["task_type"] not in core5 or r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        if r["task_id"] not in e4:
            continue
        if args.task_types and r["task_type"] not in args.task_types:
            continue
        tasks.append((r["task_id"], r["task_type"],
                      e4[r["task_id"]]["llm_pred"]))
    # round-robin by task type so --max_n pilots are balanced
    by_type: dict[str, list] = {}
    for t in tasks:
        by_type.setdefault(t[1], []).append(t)
    rr, idx = [], 0
    while any(by_type.values()):
        for tt in sorted(by_type):
            if by_type[tt]:
                rr.append(by_type[tt].pop(0))
    tasks = rr[args.start_idx: args.start_idx + args.max_n]
    print(f"running induced-planner E6 on N={len(tasks)} tasks")

    rows, n_ok = [], 0
    rows_path = args.out_json + ".rows.jsonl"
    rows_f = open(rows_path, "a")
    for i, (tid, tt, parsed) in enumerate(tasks):
        r = run_one(tid, tt, parsed if isinstance(parsed, dict) else {},
                    models, args.max_depth, verbose=args.verbose)
        rows.append(r)
        rows_f.write(json.dumps(r, default=str) + "\n")
        rows_f.flush()
        n_ok += int(r.get("execution_success", False))
        print(f"[{i+1}/{len(tasks)}] {tt[:28]:<28} "
              f"{'WON' if r.get('execution_success') else 'fail'} "
              f"({r.get('stage_failed') or r.get('fail_reason','')})"[:90],
              flush=True)

    rows_f.close()
    out = {"parsed_json": args.parsed_json,
           "induction_cache": INDUCTION_CACHE,
           "n_total": len(rows), "n_execution_success": n_ok,
           "execution_success_rate": n_ok / max(1, len(rows)),
           "results": rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nInduced-planner E6: {n_ok}/{len(rows)} = "
          f"{100*n_ok/max(1,len(rows)):.1f}%")
    print(f"wrote: {args.out_json}")


if __name__ == "__main__":
    main()
