"""Rebuild runs/sci_reduce_coverage_curve.json (Fig. 2, Table 4) from the
per-seed leave-out files and check it against the stored aggregate.

Each runs/scireduce_leaveout/leaveout_real_mystery*_s<seed>.jsonl is one run
of scripts/run_leave_out_n.py, one row per held-out trajectory with a
"coverage" flag. The held-out size is the row count and the training size is
50 minus it (total_trajectories_available in the stored file). Per training
size the first four seeds (n_seeds_per_setting) give "values", the coverage
in whole percent, with the mean and the sample standard deviation to one
decimal. Every entry of the stored file is compared with the rebuilt one; the
exit code is non-zero if an entry mismatches or cannot be rebuilt. --out
writes the rebuilt aggregate in the stored format.

The curve is the sweep over training sizes 10, 15, 20, 25, 30, 35 and 40
(held-out 40 down to 10) with seeds 0 to 3. The untagged
leaveout_real_mystery_s<seed>.jsonl files are the holdout-10 (training size
40) runs, of which s4 is an extra seed the stored curve does not use; the
leaveout_real_mystery_n<train>_s<seed>.jsonl files hold the other sizes.
The runner is deterministic given the seed, and the whole sweep regenerates
in a few seconds with

    for n in 10 15 20 25 30 35 40; do for s in 0 1 2 3; do
      PYTHONPATH=. python3 scripts/run_leave_out_n.py \\
        --mystery_jsonl runs/scireduce_input/mystery.jsonl --domain mystery \\
        --n_holdout $((50 - n)) --seed $s \\
        --out runs/scireduce_leaveout/leaveout_real_mystery_n${n}_s${s}.jsonl
    done; done
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORED = "runs/sci_reduce_coverage_curve.json"
LEAVEOUT_DIR = "runs/scireduce_leaveout"
FILE_PATTERN = "leaveout_real_mystery*_s*.jsonl"


def load_runs(leaveout_dir, total):
    runs = {}
    for path in sorted(glob.glob(os.path.join(leaveout_dir, FILE_PATTERN))):
        m = re.search(r"_s(\d+)\.jsonl$", path)
        if not m:
            continue
        seed = int(m.group(1))
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        held_out = len(rows)
        covered = sum(1 for r in rows if r["coverage"])
        key = (total - held_out, seed)
        if key in runs and runs[key][:2] != (covered, held_out):
            raise ValueError(f"conflicting runs for train size {key[0]} seed {seed}: "
                             f"{runs[key][2]} and {path}")
        runs.setdefault(key, (covered, held_out, path))
    return runs


def build_entry(train_size, seeds, runs):
    pcts = [100 * runs[(train_size, s)][0] / runs[(train_size, s)][1] for s in seeds]
    return {
        "train_size": train_size,
        "held_out": runs[(train_size, seeds[0])][1],
        "mean_coverage_pct": round(sum(pcts) / len(pcts), 1),
        "stdev": round(statistics.stdev(pcts), 1) if len(pcts) > 1 else 0.0,
        "values": [round(p) for p in pcts],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--leaveout_dir", default=os.path.join(ROOT, LEAVEOUT_DIR))
    ap.add_argument("--stored", default=os.path.join(ROOT, STORED))
    ap.add_argument("--out", default=None, help="write the rebuilt aggregate to this path")
    args = ap.parse_args()

    with open(args.stored) as f:
        stored = json.load(f)
    total = stored["total_trajectories_available"]
    n_seeds = stored["n_seeds_per_setting"]
    runs = load_runs(args.leaveout_dir, total)
    stored_by_size = {e["train_size"]: e for e in stored["results"]}

    rebuilt = []
    problems = 0
    print(f"{'train':>5} {'held':>4} {'seeds':<12} {'values':<18} {'mean':>5} {'stdev':>5}  status")
    for train_size in sorted(set(k[0] for k in runs) | set(stored_by_size)):
        seeds = sorted(s for (t, s) in runs if t == train_size)
        used, extra = seeds[:n_seeds], seeds[n_seeds:]
        stored_entry = stored_by_size.get(train_size)
        if not used:
            problems += 1
            print(f"{train_size:>5} {stored_entry['held_out']:>4} {'-':<12} {'-':<18} {'-':>5} {'-':>5}  "
                  f"missing (stored {stored_entry['values']})")
            continue
        entry = build_entry(train_size, used, runs)
        rebuilt.append(entry)
        if len(used) < n_seeds:
            status = f"partial, {len(used)} of {n_seeds} seeds"
            problems += 1
        elif stored_entry is None:
            status = "not in stored file"
        elif entry == stored_entry:
            status = "matches stored"
        else:
            status = (f"MISMATCH, stored values {stored_entry['values']} "
                      f"mean {stored_entry['mean_coverage_pct']} stdev {stored_entry['stdev']}")
            problems += 1
        if extra:
            status += ", unused seeds " + ", ".join(
                f"s{s}={100 * runs[(train_size, s)][0] / runs[(train_size, s)][1]:.0f}%"
                for s in extra)
        print(f"{train_size:>5} {entry['held_out']:>4} {str(used):<12} {str(entry['values']):<18} "
              f"{entry['mean_coverage_pct']:>5} {entry['stdev']:>5}  {status}")

    if args.out:
        out = {k: v for k, v in stored.items() if k != "results"}
        out["results"] = rebuilt
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote: {args.out}")

    print("stored aggregate reproduced" if not problems
          else f"{problems} stored entr{'y' if problems == 1 else 'ies'} not reproduced")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
