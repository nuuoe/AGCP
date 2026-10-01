"""Regex baseline with the EXTRA_ALIASES vocabulary merged into extraction.

regex_parse_v2 matches NL against ALIAS_PAIRS plus EXTRA_ALIASES (longest
phrase first), so the regex sees the same vocabulary the LLM prompt lists.
Output: runs/alfworld_regex_v2.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.alfworld_nl_parse import (
    sample_tasks, _gt_from_pddl_params, ALIAS_PAIRS,
)
from scripts.alfworld_enum_constrained import EXTRA_ALIASES, smart_f1_extended


# EXTRA_ALIASES first (longest phrase first), then ALIAS_PAIRS.
MERGED_ALIASES: list[tuple[str, str]] = []
seen = set()
for alias, canon in sorted(EXTRA_ALIASES.items(),
                              key=lambda x: -len(x[0])):
    if alias not in seen:
        MERGED_ALIASES.append((alias, canon))
        seen.add(alias)
for alias, canon in ALIAS_PAIRS:
    if alias.lower() not in seen:
        MERGED_ALIASES.append((alias.lower(), canon))
        seen.add(alias.lower())


def _normalize_objects_v2(text: str) -> list[tuple[str, int]]:
    found = []
    low = text.lower()
    used = [False] * len(low)
    for alias, camel in MERGED_ALIASES:
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


def regex_parse_v2(nl: str) -> dict:
    """regex_parse with the merged alias map and slightly wider verb/preposition patterns."""
    text = nl.strip().rstrip(".")
    low = text.lower()
    out = {
        "object_target": "", "mrecep_target": "",
        "parent_target": "", "toggle_target": "",
        "object_sliced": False, "object_cool": False,
        "object_heat": False, "object_clean": False,
    }
    objs = _normalize_objects_v2(text)

    if re.search(r"\b(look|examine|inspect)\b", low) \
            and re.search(r"\b(lamp|light)\b", low):
        for cm, _ in objs:
            if cm in ("DeskLamp", "FloorLamp"):
                out["toggle_target"] = cm
            elif not out["object_target"]:
                out["object_target"] = cm
    elif re.search(r"\b(slice|cut|chop|sliced|sliced up)\b", low):
        out["object_sliced"] = True
    elif re.search(r"\b(hot|heat|warm|microwave(?:d)?|cook|cooked|heated)\b", low):
        out["object_heat"] = True
    elif re.search(r"\b(cool|cold|chill|chilled|refriger)\b", low):
        out["object_cool"] = True
    elif re.search(r"\b(clean|wash|rinse|cleaned|washed|rinsed)\b", low):
        out["object_clean"] = True

    # parent_target via place/put/move patterns
    m = re.search(r"(?:put|place|set|move|deposit|take) "
                  r"(?:the |a )?(.+?) "
                  r"(?:in|on|onto|into|inside|to) (?:the |a )?(.+?)"
                  r"(?:\.|,|;|$)", low)
    if m:
        parent_text = m.group(2).strip()
        p_objs = _normalize_objects_v2(parent_text)
        if p_objs: out["parent_target"] = p_objs[0][0]
        obj_text = m.group(1).strip()
        o_objs = _normalize_objects_v2(obj_text)
        if o_objs and not out["object_target"]:
            out["object_target"] = o_objs[0][0]

    if not out["object_target"] and objs:
        for cm, _ in objs:
            if cm not in ("DeskLamp", "FloorLamp", "DiningTable",
                            "Desk", "SideTable", "CoffeeTable",
                            "CounterTop", "Fridge", "Microwave",
                            "Drawer", "Cabinet", "GarbageCan",
                            "Shelf", "Bed", "Sofa", "ArmChair",
                            "Sink", "SinkBasin", "Toilet", "Bathtub",
                            "BathtubBasin", "ShelvingUnit", "TVStand",
                            "Dresser", "Ottoman"):
                out["object_target"] = cm
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld_data",
                    default=os.path.expanduser(
                        "~/.cache/alfworld/json_2.1.1/valid_seen"))
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", default="runs/alfworld_regex_v2.json")
    args = ap.parse_args()

    tasks = sample_tasks(args.alfworld_data, args.n_per_type, 0)
    print(f"Regex v2 (MERGED aliases: {len(MERGED_ALIASES)}) on {len(tasks)} tasks")

    rows = []
    n_ok = 0
    f_sum = 0
    for t in tasks:
        gt = _gt_from_pddl_params(t["pddl_params"], t["task_type"])
        nl = t["task_desc"]
        pred = regex_parse_v2(nl)
        f = smart_f1_extended(pred, gt)
        if f >= args.threshold: n_ok += 1
        f_sum += f
        rows.append({
            "task_id": t["task_id"], "task_type": t["task_type"],
            "task_desc": nl, "gt": gt, "regex_v2_pred": pred,
            "regex_v2_f1": f,
        })
        print(f"  [{t['task_type'][:22]:<22}] f={f:.2f} :: {nl[:55]}")

    n = len(rows)
    print(f"\nRegex v2 (smart F1): {n_ok}/{n} = {100*n_ok/n:.0f}%  mean_F1={f_sum/n:.3f}")
    print(f"(original regex was 9/35 = 26%, mean_F1=0.671)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_total": n, "n_accept": n_ok,
            "mean_f1": f_sum / n,
            "threshold": args.threshold,
            "n_merged_aliases": len(MERGED_ALIASES),
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
