"""ALFWorld NL-goal parse with an enum-constrained JSON schema and extended aliases.

The schema restricts every *_target slot to ALFWorld's canonical object
types, so a small model cannot invent types such as "EggHeated" or use
generic names ("Table" for "DiningTable"). Scoring is smart F1 with the
extra aliases below. Output: runs/alfworld_enum_constrained.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, ".")
from scripts.alfworld_nl_parse import (
    sample_tasks, _gt_from_pddl_params, OBJECT_TYPES,
    regex_parse,
)
from scripts.alfworld_smart_score import (
    smart_f1, normalize_object as _norm_obj_base,
)


# Extra aliases for name variants seen in model output.
EXTRA_ALIASES = {
    "table": "DiningTable",
    "sink": "SinkBasin",
    "stove": "StoveBurner",
    "stove top": "StoveBurner",
    "stovetop": "StoveBurner",
    "kitchen island": "CounterTop",
    "counter": "CounterTop",
    "fridge": "Fridge",
    "refrigerator": "Fridge",
    "garbage bin": "GarbageCan",
    "garbage can": "GarbageCan",
    "trash bin": "GarbageCan",
    "trash can": "GarbageCan",
    "ottoman": "Ottoman",
    "drawer": "Drawer",
    "cabinet": "Cabinet",
    "shelf": "Shelf",
    "desk": "Desk",
    "bed": "Bed",
    "sofa": "Sofa",
    "couch": "Sofa",
    "arm chair": "ArmChair",
    "armchair": "ArmChair",
    "lamp": "DeskLamp",
    "floor lamp": "FloorLamp",
    "desk lamp": "DeskLamp",
    "card": "CreditCard",
    "credit card": "CreditCard",
    "computer": "Laptop",
    "laptop": "Laptop",
    "tv stand": "TVStand",
    "television stand": "TVStand",
    "frying pan": "Pan",
    "pan": "Pan",
    "pans": "Pan",
    "remote": "RemoteControl",
    "remotes": "RemoteControl",
    "remote control": "RemoteControl",
    "books": "Book",
    "book": "Book",
    "dresser": "Dresser",
    "toilet": "Toilet",
    "toilet paper": "ToiletPaper",
    "back of the toilet": "Toilet",
    "tub": "Bathtub",
    "bathtub": "Bathtub",
    "shelf above the table": "Shelf",
    "shelf": "Shelf",
    "shelves": "Shelf",
    "shelving unit": "ShelvingUnit",
    "soap bar": "SoapBar",
    "bar of soap": "SoapBar",
    "spray bottle": "SprayBottle",
    "cd": "CD",
    "disc": "CD",
    "disk": "CD",
    "egg": "Egg",
    "tomato": "Tomato",
    "apple": "Apple",
    "apple slice": "AppleSliced",
    "sliced apple": "AppleSliced",
    "bread": "Bread",
    "slice of bread": "BreadSliced",
    "lettuce": "Lettuce",
    "potato": "Potato",
    "plate": "Plate",
    "white plate": "Plate",
    "knife": "Knife",
    "butter knife": "ButterKnife",
    "fork": "Fork",
    "spoon": "Spoon",
    "bowl": "Bowl",
    "mug": "Mug",
    "cup": "Cup",
    "glass": "Cup",
    "kettle": "Kettle",
    "pot": "Pot",
    "watch": "Watch",
    "cloth": "Cloth",
    "rinsed knife": "Knife",  # rinsed = clean modifier
    "alarm clock": "AlarmClock",
    "clock": "AlarmClock",
    "phone": "CellPhone",
    "cell phone": "CellPhone",
    "spatula": "Spatula",
    "pencil": "Pencil",
    "pencils": "Pencil",
    "key chain": "KeyChain",
    "house plant": "HousePlant",
    "plant": "HousePlant",
    "vase": "Vase",
    "candle": "Candle",
    "statue": "Statue",
    "bench": "Sofa",  # ALFWorld has no Bench type
    "kitchen drawer": "Drawer",
    "night stand": "SideTable",
    "nightstand": "SideTable",
    "side table": "SideTable",
    "coffee table": "CoffeeTable",
    "dining table": "DiningTable",
    "metal pan": "Pan",
    "large metal pan": "Pan",
}


def normalize_extended(name: str) -> str:
    """Like alfworld_smart_score.normalize_object but with extra aliases."""
    if not name:
        return ""
    s = name.strip().lower()
    if s in EXTRA_ALIASES:
        return EXTRA_ALIASES[s]
    base = _norm_obj_base(name)
    if base in OBJECT_TYPES:
        return base
    s2 = re.sub(r"[^a-z]", "", s)
    if s2 in {k.replace(" ", ""): v for k, v in EXTRA_ALIASES.items()}.values():
        return s2  # already canonical
    for alias, canon in EXTRA_ALIASES.items():
        if alias.replace(" ", "") == s2:
            return canon
    return name


def normalize_pred_extended(pred: dict) -> dict:
    out = dict(pred)
    for k in ["object_target", "mrecep_target", "parent_target",
                "toggle_target"]:
        if k in out:
            out[k] = normalize_extended(out.get(k, "") or "")
    return out


def smart_f1_extended(pred: dict, gt: dict) -> float:
    """smart_f1 with extended aliasing applied first."""
    return smart_f1(normalize_pred_extended(pred), normalize_pred_extended(gt))


# Every slot enum lists all canonical types plus "" for unfilled slots.
ALL_OBJ_TYPES = sorted(OBJECT_TYPES) + [""]


def build_enum_schema() -> str:
    schema = {
        "type": "object",
        "properties": {
            "object_target": {"type": "string", "enum": ALL_OBJ_TYPES},
            "mrecep_target": {"type": "string", "enum": ALL_OBJ_TYPES},
            "parent_target": {"type": "string", "enum": ALL_OBJ_TYPES},
            "toggle_target": {"type": "string", "enum": ALL_OBJ_TYPES},
            "object_sliced": {"type": "boolean"},
            "object_cool": {"type": "boolean"},
            "object_heat": {"type": "boolean"},
            "object_clean": {"type": "boolean"},
        },
        "required": ["object_target", "mrecep_target", "parent_target",
                      "toggle_target", "object_sliced", "object_cool",
                      "object_heat", "object_clean"],
    }
    return json.dumps(schema)


def make_enum_prompt(nl: str) -> str:
    valid = ", ".join(sorted(OBJECT_TYPES)[:60]) + ", ..."
    return f"""Parse this ALFWorld task description into structured
parameters. The object types MUST be chosen from ALFWorld's canonical
vocabulary; the schema enforces this. Common types include:
{valid}

Important conventions:
- For 'slice of X' or 'sliced X': use object_target=X (NOT XSliced)
  AND set object_sliced=true
- For 'heated X' or 'warm X' or 'cooked X': use object_target=X
  AND set object_heat=true (NOT XHeated)
- For 'chilled X' or 'cold X' or 'cool X': use object_target=X
  AND set object_cool=true (NOT XChilled)
- For 'clean X' or 'rinsed X' or 'washed X': use object_target=X
  AND set object_clean=true
- For lamps: toggle_target=DeskLamp or FloorLamp (NOT mrecep_target)
- For 'place X in Y' / 'put X on Y': X is object_target, Y is parent_target
- For 'pencil' use Pencil, for 'desk' use Desk, for 'fridge' use Fridge,
  for 'counter' use CounterTop, for 'sink' use SinkBasin, for 'stove'
  use StoveBurner, for 'table' use DiningTable, for 'cards' use CreditCard

## Examples
Input: "look at the clock under the lamp"
Output: {{"object_target": "AlarmClock", "mrecep_target": "",
"parent_target": "", "toggle_target": "DeskLamp",
"object_sliced": false, "object_cool": false, "object_heat": false, "object_clean": false}}

Input: "Place clean lettuce in the fridge"
Output: {{"object_target": "Lettuce", "mrecep_target": "",
"parent_target": "Fridge", "toggle_target": "",
"object_sliced": false, "object_cool": false, "object_heat": false, "object_clean": true}}

Input: "Put a slice of bread into a microwave"
Output: {{"object_target": "Bread", "mrecep_target": "",
"parent_target": "Microwave", "toggle_target": "",
"object_sliced": true, "object_cool": false, "object_heat": false, "object_clean": false}}

Input: "Put a heated white plate in a cabinet"
Output: {{"object_target": "Plate", "mrecep_target": "",
"parent_target": "Cabinet", "toggle_target": "",
"object_sliced": false, "object_cool": false, "object_heat": true, "object_clean": false}}

## Your task
Input: "{nl}"
Output:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld_data",
                    default=os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"))
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--out",
                    default="runs/alfworld_enum_constrained.json")
    args = ap.parse_args()

    tasks = sample_tasks(args.alfworld_data, args.n_per_type, 0)
    print(f"ENUM-constrained smart-score on {len(tasks)} ALFWorld tasks")
    schema = build_enum_schema()

    llm_call = None
    if args.use_llm:
        from agplan.llm_propose import propose_verify_fallback

        def _llm(nl, gt):
            prompt = make_enum_prompt(nl)
            def verifier(prop):
                return smart_f1_extended(
                    prop if isinstance(prop, dict) else {}, gt)
            final, info = propose_verify_fallback(
                prompt=prompt, schema_json=schema,
                verifier=verifier,
                baseline={k: ("" if k.endswith("_target") else False)
                          for k in ["object_target", "mrecep_target",
                                    "parent_target", "toggle_target",
                                    "object_sliced", "object_cool",
                                    "object_heat", "object_clean"]},
                threshold=args.threshold,
                model_name=args.model,
                max_new_tokens=256, n_samples=4, temperature=0.4,
            )
            return info.get("proposal", final), info
        llm_call = _llm

    rows = []
    n_regex_smart = 0
    n_llm_smart = 0
    f1_regex_sum = 0
    f1_llm_sum = 0
    for t in tasks:
        gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
        nl = t["task_desc"]
        regex_pred = regex_parse(nl)
        rsf1 = smart_f1_extended(regex_pred, gt)

        if llm_call is not None:
            llm_proposal, info = llm_call(nl, gt)
            llmsf1 = smart_f1_extended(
                llm_proposal if isinstance(llm_proposal, dict) else {}, gt)
        else:
            llm_proposal = None
            llmsf1 = 0.0

        if rsf1 >= args.threshold: n_regex_smart += 1
        if llmsf1 >= args.threshold: n_llm_smart += 1
        f1_regex_sum += rsf1
        f1_llm_sum += llmsf1
        rows.append({
            "task_id": t["task_id"], "task_type": t["task_type"],
            "task_desc": nl, "gt": gt,
            "regex_pred": regex_pred, "regex_smart_f1_ext": rsf1,
            "llm_proposal": llm_proposal, "llm_smart_f1_ext": llmsf1,
        })
        if llm_call is not None:
            print(f"  [{t['task_type'][:22]:<22}] r={rsf1:.2f} "
                  f"l={llmsf1:.2f} :: {nl[:60]}")

    n = len(rows)
    print("\n" + "=" * 78)
    print(f"ENUM-CONSTRAINED smart-score (n={n}, thr={args.threshold})")
    print(f"  REGEX (extended-aliased smart F1): "
          f"{n_regex_smart}/{n} ({100*n_regex_smart/n:.0f}%)  "
          f"mean_F1={f1_regex_sum/n:.3f}")
    if args.use_llm:
        print(f"  LLM (enum-schema + extended-aliased smart F1): "
              f"{n_llm_smart}/{n} ({100*n_llm_smart/n:.0f}%)  "
              f"mean_F1={f1_llm_sum/n:.3f}")
        gap = n_llm_smart - n_regex_smart
        print(f"  LLM-vs-regex gap: {'+' if gap>=0 else ''}{gap}")
    print("=" * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_total": n,
            "n_regex_accept": n_regex_smart,
            "n_llm_accept": n_llm_smart if args.use_llm else None,
            "regex_mean_f1": f1_regex_sum / n,
            "llm_mean_f1": (f1_llm_sum / n) if args.use_llm else None,
            "threshold": args.threshold,
            "model": args.model if args.use_llm else None,
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
