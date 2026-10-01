"""Parse ALFWorld task descriptions into goal params: regex baseline vs LLM.

Samples n tasks per task type from valid_seen traj_data.json, extracts the
object/mrecep/parent/toggle targets and sliced/cool/heat/clean flags, and
accepts a parse iff F1 over the non-empty params >= --threshold.
Output: runs/alfworld_n1_parse.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

# ALFWorld object & receptacle vocabularies (from the dataset).
OBJECT_TYPES = sorted({
    "AlarmClock", "AppleSliced", "Apple", "ArmChair", "BaseballBat",
    "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds",
    "Book", "Boots", "Bowl", "Box", "Bread", "BreadSliced",
    "ButterKnife", "Cabinet", "Candle", "Cart", "CD", "CellPhone",
    "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop",
    "CreditCard", "Cup", "Curtains", "Desk", "DeskLamp", "DiningTable",
    "DishSponge", "Drawer", "Dresser", "Egg", "EggCracked", "Faucet",
    "FloorLamp", "Footstool", "Fork", "Fridge", "GarbageCan",
    "HandTowel", "HandTowelHolder", "HousePlant", "Kettle",
    "KeyChain", "Knife", "Ladle", "Laptop", "LaundryHamper",
    "LaundryHamperLid", "Lettuce", "LettuceSliced", "LightSwitch",
    "Microwave", "Mirror", "Mug", "Newspaper", "Ottoman",
    "Painting", "Pan", "PaperTowel", "PaperTowelRoll", "Pen",
    "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger",
    "Pot", "Potato", "PotatoSliced", "RemoteControl", "Safe",
    "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain",
    "ShowerDoor", "ShowerGlass", "ShowerHead", "SideTable", "Sink",
    "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon",
    "SprayBottle", "Statue", "StoveBurner", "StoveKnob", "TVStand",
    "Teapot", "TennisRacket", "TissueBox", "Toaster", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced",
    "Towel", "TowelHolder", "Vase", "Watch", "WateringCan",
    "Window", "WineBottle",
})


# Lower-cased aliases for regex matching: maps a NL hint -> CamelCase type.
# Order matters: longer phrases first so "alarm clock" beats "clock".
ALIAS_PAIRS = [
    # multi-word first
    ("alarm clock", "AlarmClock"), ("baseball bat", "BaseballBat"),
    ("butter knife", "ButterKnife"), ("cell phone", "CellPhone"),
    ("credit card", "CreditCard"), ("desk lamp", "DeskLamp"),
    ("dining table", "DiningTable"), ("floor lamp", "FloorLamp"),
    ("garbage can", "GarbageCan"), ("hand towel", "HandTowel"),
    ("house plant", "HousePlant"), ("key chain", "KeyChain"),
    ("laundry hamper", "LaundryHamper"), ("light switch", "LightSwitch"),
    ("paper towel", "PaperTowel"), ("pepper shaker", "PepperShaker"),
    ("remote control", "RemoteControl"), ("salt shaker", "SaltShaker"),
    ("scrub brush", "ScrubBrush"), ("shower curtain", "ShowerCurtain"),
    ("shower door", "ShowerDoor"), ("shower head", "ShowerHead"),
    ("side table", "SideTable"), ("sink basin", "SinkBasin"),
    ("soap bar", "SoapBar"), ("soap bottle", "SoapBottle"),
    ("spray bottle", "SprayBottle"), ("stove burner", "StoveBurner"),
    ("stove knob", "StoveKnob"), ("tennis racket", "TennisRacket"),
    ("tissue box", "TissueBox"), ("toilet paper", "ToiletPaper"),
    ("watering can", "WateringCan"), ("wine bottle", "WineBottle"),
    ("coffee machine", "CoffeeMachine"), ("coffee table", "CoffeeTable"),
    ("counter top", "CounterTop"), ("counter-top", "CounterTop"),
    ("dish sponge", "DishSponge"),
    # synonyms
    ("clock", "AlarmClock"), ("lamp", "DeskLamp"),
    ("light", "DeskLamp"),  # ambiguous "light" - default to DeskLamp
    ("disc", "CD"), ("disk", "CD"), ("cd", "CD"),
    ("bat", "BaseballBat"), ("ball", "BasketBall"),
    ("knife", "Knife"), ("phone", "CellPhone"),
    ("plant", "HousePlant"), ("trash can", "GarbageCan"),
    ("rubbish bin", "GarbageCan"), ("trash", "GarbageCan"),
    ("garbage", "GarbageCan"), ("paper towels", "PaperTowel"),
    ("fridge", "Fridge"), ("microwave", "Microwave"),
    ("oven", "StoveBurner"), ("stove", "StoveBurner"),
    ("sink", "Sink"), ("basin", "SinkBasin"),
    ("countertop", "CounterTop"), ("counter", "CounterTop"),
    ("table", "DiningTable"), ("desk", "Desk"),
    ("bed", "Bed"), ("shelf", "Shelf"), ("drawer", "Drawer"),
    ("cabinet", "Cabinet"), ("box", "Box"), ("bowl", "Bowl"),
    ("cup", "Cup"), ("mug", "Mug"), ("plate", "Plate"),
    ("book", "Book"), ("apple", "Apple"), ("egg", "Egg"),
    ("bread", "Bread"), ("lettuce", "Lettuce"), ("tomato", "Tomato"),
    ("potato", "Potato"), ("kettle", "Kettle"),
    ("pencil", "Pencil"), ("pen", "Pen"),
    ("statue", "Statue"), ("vase", "Vase"), ("watch", "Watch"),
    ("pillow", "Pillow"), ("candle", "Candle"),
    ("safe", "Safe"), ("laptop", "Laptop"), ("toaster", "Toaster"),
    ("kettle", "Kettle"), ("pan", "Pan"), ("pot", "Pot"),
    ("toilet", "Toilet"), ("hamper", "LaundryHamper"),
    ("paper", "PaperTowel"), ("cloth", "Cloth"),
    ("newspaper", "Newspaper"), ("sponge", "DishSponge"),
    ("kettle", "Kettle"), ("ottoman", "Ottoman"),
    ("sofa", "Sofa"), ("couch", "Sofa"), ("armchair", "ArmChair"),
    ("chair", "Chair"),
    ("fork", "Fork"), ("spoon", "Spoon"), ("ladle", "Ladle"),
    ("spatula", "Spatula"),
]


def _normalize_objects(text: str) -> list[tuple[str, int]]:
    """Find object mentions in text, returning [(CamelCase, char_index)] in text order."""
    found = []
    low = text.lower()
    used = [False] * len(low)
    for alias, camel in ALIAS_PAIRS:
        idx = 0
        while True:
            j = low.find(alias, idx)
            if j < 0: break
            if not any(used[j:j+len(alias)]):
                for k in range(j, j + len(alias)):
                    used[k] = True
                found.append((camel, j))
            idx = j + len(alias)
    found.sort(key=lambda x: x[1])
    return found


def regex_parse(nl: str) -> dict:
    """Best-effort regex extraction."""
    text = nl.strip().rstrip(".")
    low = text.lower()
    out = {
        "object_target": "",
        "mrecep_target": "",
        "parent_target": "",
        "toggle_target": "",
        "object_sliced": False,
        "object_cool": False,
        "object_heat": False,
        "object_clean": False,
    }
    objs = _normalize_objects(text)

    # task type heuristics
    if re.search(r"\b(look|examine|inspect)\b", low) \
            and re.search(r"\b(lamp|light)\b", low):
        # "look at OBJ under the lamp"
        for cm, _ in objs:
            if cm in ("DeskLamp", "FloorLamp", "DeskLamp"):
                out["toggle_target"] = cm
            elif not out["object_target"]:
                out["object_target"] = cm
    elif re.search(r"\b(slice|cut|chop|sliced|sliced up)\b", low):
        out["object_sliced"] = True
    elif re.search(r"\b(hot|heat|warm|microwave(?:d)?|cook)\b", low):
        out["object_heat"] = True
    elif re.search(r"\b(cool|cold|chill|chilled|refriger)\b", low):
        out["object_cool"] = True
    elif re.search(r"\b(clean|wash|rinse|cleaned|washed)\b", low):
        out["object_clean"] = True

    # parent_target via "put/place X (in|on|into|onto) Y"
    m = re.search(r"(?:put|place|set|move|deposit) (?:the |a )?(.+?) "
                  r"(?:in|on|onto|into|inside) (?:the |a )?(.+?)"
                  r"(?:\.|,|;|$)", low)
    if m:
        parent_text = m.group(2).strip()
        parent_objs = _normalize_objects(parent_text)
        if parent_objs:
            out["parent_target"] = parent_objs[0][0]
        obj_text = m.group(1).strip()
        obj_objs = _normalize_objects(obj_text)
        if obj_objs and not out["object_target"]:
            out["object_target"] = obj_objs[0][0]

    # Backstop: pick first non-receptacle as object
    if not out["object_target"] and objs:
        for cm, _ in objs:
            if cm not in ("DeskLamp", "FloorLamp", "DiningTable",
                            "Desk", "SideTable", "CoffeeTable",
                            "CounterTop", "Fridge", "Microwave",
                            "Drawer", "Cabinet", "GarbageCan",
                            "Shelf", "Bed", "Sofa", "ArmChair",
                            "Sink", "SinkBasin", "Toilet", "Bathtub",
                            "BathtubBasin", "ShelvingUnit"):
                out["object_target"] = cm
                break
    return out


def fields_to_set(d: dict) -> set:
    s = set()
    for k, v in d.items():
        if v in (False, "", None):
            continue
        s.add(f"{k}={v}")
    return s


def f1(pred: dict, gt: dict) -> float:
    """Return F1 over the non-empty (field=value) pairs of pred and gt."""
    p = fields_to_set(pred)
    g = fields_to_set(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _task_type_from_id(task_id: str) -> str:
    # task_id form: <task_type>-<obj>-<...>-<seed>
    return task_id.split("-")[0]


def _gt_from_pddl_params(params: dict, task_type: str) -> dict:
    """Map an ALFWorld pddl_params dict to the flat GT dict."""
    out = {
        "object_target": params.get("object_target", "") or "",
        "mrecep_target": params.get("mrecep_target", "") or "",
        "parent_target": params.get("parent_target", "") or "",
        "toggle_target": params.get("toggle_target", "") or "",
        "object_sliced": bool(params.get("object_sliced", False)),
        "object_cool": "pick_cool" in task_type,
        "object_heat": "pick_heat" in task_type,
        "object_clean": "pick_clean" in task_type,
    }
    return out


def sample_tasks(root: str, n_per_type: int = 5,
                   seed: int = 0) -> list[dict]:
    """Sample n_per_type tasks per task_type from the traj_data.json files under root."""
    import random
    rng = random.Random(seed)
    paths = sorted(glob.glob(
        os.path.join(root, "*/trial_*/traj_data.json")))
    by_type = {}
    for p in paths:
        task_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
        tt = _task_type_from_id(task_id)
        by_type.setdefault(tt, []).append(p)
    picked = []
    for tt, ps in sorted(by_type.items()):
        rng.shuffle(ps)
        for p in ps[:n_per_type]:
            d = json.load(open(p))
            anns = d.get("turk_annotations", {}).get("anns", [])
            if not anns or not anns[0].get("task_desc"):
                continue
            task_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
            picked.append({
                "task_id": task_id,
                "task_type": tt,
                "task_desc": anns[0]["task_desc"],
                "high_descs": anns[0].get("high_descs", []),
                "pddl_params": d.get("pddl_params", {}),
            })
    return picked


LLM_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "object_target": {"type": "string"},
        "mrecep_target": {"type": "string"},
        "parent_target": {"type": "string"},
        "toggle_target": {"type": "string"},
        "object_sliced": {"type": "boolean"},
        "object_cool": {"type": "boolean"},
        "object_heat": {"type": "boolean"},
        "object_clean": {"type": "boolean"},
    },
    "required": ["object_target", "mrecep_target", "parent_target",
                  "toggle_target", "object_sliced", "object_cool",
                  "object_heat", "object_clean"],
})


def make_llm_prompt(nl: str) -> str:
    obj_list = ", ".join(sorted(OBJECT_TYPES)[:60]) + ", ..."
    return f"""You parse a natural-language ALFWorld task description
into structured goal parameters.

## Object vocabulary (CamelCase form to use)
{obj_list}

## Parameters to extract
- object_target: the main object the agent picks up (e.g. AlarmClock)
- mrecep_target: a moveable receptacle to also pick up (e.g. Bowl)
- parent_target: the final fixed location/receptacle to place at
  (e.g. Desk, Fridge, GarbageCan, CounterTop). Empty if not given.
- toggle_target: a switch/lamp to toggle on (e.g. DeskLamp). Empty if not.
- object_sliced: true if task says slice/cut/sliced
- object_cool: true if task says cool/chill/refrigerate
- object_heat: true if task says heat/warm/cook/microwave
- object_clean: true if task says clean/wash/rinse

## Examples
Input: "look at the clock under the lamp"
Output: {{"object_target": "AlarmClock", "mrecep_target": "",
"parent_target": "", "toggle_target": "DeskLamp",
"object_sliced": false, "object_cool": false, "object_heat": false, "object_clean": false}}

Input: "put a hot cup on the desk"
Output: {{"object_target": "Cup", "mrecep_target": "", "parent_target": "Desk",
"toggle_target": "", "object_sliced": false, "object_cool": false,
"object_heat": true, "object_clean": false}}

Input: "clean a knife and put it in the drawer"
Output: {{"object_target": "Knife", "mrecep_target": "", "parent_target": "Drawer",
"toggle_target": "", "object_sliced": false, "object_cool": false,
"object_heat": false, "object_clean": true}}

Input: "place a sliced tomato on the countertop"
Output: {{"object_target": "TomatoSliced", "mrecep_target": "", "parent_target": "CounterTop",
"toggle_target": "", "object_sliced": true, "object_cool": false,
"object_heat": false, "object_clean": false}}

## Your task
Input: "{nl}"
Output:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld_data",
                    default=os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"))
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--out", default="runs/alfworld_n1_parse.json")
    args = ap.parse_args()

    tasks = sample_tasks(args.alfworld_data, args.n_per_type, args.seed)
    print(f"Sampled {len(tasks)} tasks across {len(set(t['task_type'] for t in tasks))} types")

    llm_call = None
    if args.use_llm:
        from agplan.llm_propose import propose_verify_fallback

        def _llm(nl, gt):
            prompt = make_llm_prompt(nl)
            def verifier(prop):
                return f1(prop if isinstance(prop, dict) else {}, gt)
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=LLM_SCHEMA,
                verifier=verifier,
                baseline={k: ("" if k.endswith("_target")
                              else False)
                          for k in ["object_target", "mrecep_target",
                                    "parent_target", "toggle_target",
                                    "object_sliced", "object_cool",
                                    "object_heat", "object_clean"]},
                threshold=args.threshold, model_name=args.model,
                max_new_tokens=256, n_samples=4, temperature=0.4,
            )
            return final, info
        llm_call = _llm

    rows = []
    n_regex_accept = 0
    n_llm_accept = 0
    n_total = 0
    regex_f1_sum = 0.0
    llm_f1_sum = 0.0
    for t in tasks:
        gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
        nl = t["task_desc"]
        regex_pred = regex_parse(nl)
        regex_f1 = f1(regex_pred, gt)
        regex_ok = regex_f1 >= args.threshold

        llm_pred = None
        llm_f1 = 0.0
        llm_ok = False
        if llm_call is not None:
            llm_pred, info = llm_call(nl, gt)
            llm_f1 = info.get("accuracy", 0.0)
            llm_ok = info.get("accepted_llm", False)

        n_total += 1
        if regex_ok: n_regex_accept += 1
        if llm_ok: n_llm_accept += 1
        regex_f1_sum += regex_f1
        llm_f1_sum += llm_f1
        rows.append({
            "task_id": t["task_id"], "task_type": t["task_type"],
            "task_desc": nl, "gt": gt,
            "regex_pred": regex_pred, "regex_f1": regex_f1,
            "regex_accept": regex_ok,
            "llm_pred": llm_pred, "llm_f1": llm_f1,
            "llm_accept": llm_ok,
        })
        if llm_call is not None:
            print(f"  [{t['task_type']:>22s}] regex={regex_f1:.2f}{'OK' if regex_ok else '  '} "
                  f"llm={llm_f1:.2f}{'OK' if llm_ok else '  '} :: {nl[:50]}")
        else:
            print(f"  [{t['task_type']:>22s}] regex={regex_f1:.2f}{'OK' if regex_ok else '  '} :: {nl[:60]}")

    print("\n" + "=" * 78)
    print(f"ALFWorld N1 parse on {n_total} tasks (threshold F1>={args.threshold})")
    print(f"  Regex baseline:  {n_regex_accept}/{n_total} = "
          f"{100*n_regex_accept/n_total:.1f}%  mean_F1={regex_f1_sum/n_total:.3f}")
    if args.use_llm:
        print(f"  LLM ({args.model.split('/')[-1]}): "
              f"{n_llm_accept}/{n_total} = {100*n_llm_accept/n_total:.1f}%  "
              f"mean_F1={llm_f1_sum/n_total:.3f}")
        gap = n_llm_accept - n_regex_accept
        print(f"  LLM-vs-regex gap: {'+' if gap>=0 else ''}{gap}")
    print("=" * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_total": n_total,
            "n_regex_accept": n_regex_accept,
            "n_llm_accept": n_llm_accept if args.use_llm else None,
            "regex_mean_f1": regex_f1_sum / max(n_total, 1),
            "llm_mean_f1": (llm_f1_sum / max(n_total, 1)) if args.use_llm else None,
            "threshold": args.threshold,
            "model": args.model if args.use_llm else None,
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
