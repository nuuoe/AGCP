"""Cross-domain schema reuse with LLM hints vs the signature-subset baseline.

For each target action in the IPC-2023 target domains, the baseline accepts
any source schema whose predicate names fit the target vocabulary and
verifies it on held-out rollouts; the LLM variant instead picks one
(source_domain, source_action), verified the same way. Output: --out JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agplan.llm_propose import propose_verify_fallback


def induce_models_for_domain(base_dir: str, domain: str,
                                n_problems: int = 5,
                                steps_per_problem: int = 100) -> dict:
    from scripts.induce_pddl_generic import (
        load_task, random_rollout_pddl,
    )
    from scripts.induce_positional import (
        induce_lifted_models_positional,
    )
    from scripts.strip_negative_preconditions import strip_negative_pre
    import tempfile

    dom_dir = os.path.join(base_dir, domain)
    dom_path = os.path.join(dom_dir, "domain.pddl")
    if not os.path.isfile(dom_path):
        return {}

    # Strip ADL if needed
    if domain in {"childsnack", "ferry", "satellite"}:
        tmpdir = tempfile.mkdtemp()
        stripped = os.path.join(tmpdir, f"{domain}_domain.pddl")
        with open(stripped, "w") as f:
            f.write(strip_negative_pre(open(dom_path).read()))
        dom_path = stripped

    import glob
    probs = []
    for sub in ["training/easy", "testing/easy"]:
        probs.extend(sorted(glob.glob(
            os.path.join(dom_dir, sub, "p*.pddl"))))
    if not probs:
        return {}
    probs = probs[:n_problems]

    trs = []
    for i, p in enumerate(probs):
        try:
            _, task = load_task(dom_path, p)
        except Exception:
            continue
        for s in range(3):
            trs.extend(random_rollout_pddl(
                task, n_steps=steps_per_problem, seed=i*100+s))
    trs_ok = [t for t in trs if t[0] != t[-1]]
    return induce_lifted_models_positional(trs_ok)


def verify_schema_on_target(schema: dict, target_trs: list) -> float:
    """Apply lifted schema to target rollouts; return accuracy."""
    from scripts.grounding import ground_lifted
    if not target_trs: return 0.0
    add_t = set(schema.get("eff_add", []))
    del_t = set(schema.get("eff_del", []))
    ok = 0
    for sb, args, sa in target_trs:
        add_g = ground_lifted(add_t, args)
        del_g = ground_lifted(del_t, args)
        predicted = (sb - del_g) | add_g
        if predicted == sa: ok += 1
    return ok / max(len(target_trs), 1)


def signature_subset(source_schema: dict, target_vocab: set) -> bool:
    """Return True if every predicate name in the source schema is in the target vocabulary."""
    import re as _re
    def names(model):
        out = set()
        for k in ("pre_pos", "eff_add", "eff_del"):
            for p in model.get(k, set()):
                m = _re.match(r"^([\w-]+)", p)
                if m: out.add(m.group(1))
        return out
    return names(source_schema) <= target_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="external/ipc2023-learning")
    ap.add_argument("--source_domains",
                    default="blocksworld,ferry,floortile,miconic,satellite")
    ap.add_argument("--target_domains",
                    default="rovers,transport,childsnack")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_problems", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources = args.source_domains.split(",")
    targets = args.target_domains.split(",")

    print(f"Inducing source-domain schemas ({len(sources)} domains)...")
    source_models = {}
    for d in sources:
        m = induce_models_for_domain(args.base_dir, d,
                                        n_problems=args.n_problems)
        if m:
            source_models[d] = m
            print(f"  {d}: {len(m)} schemas")

    print(f"\nGathering target-domain rollouts ({len(targets)} domains)...")
    target_data = {}
    for t in targets:
        from scripts.induce_pddl_generic import (
            load_task, random_rollout_pddl,
        )
        import glob
        dom_dir = os.path.join(args.base_dir, t)
        dom_path = os.path.join(dom_dir, "domain.pddl")
        if t in {"childsnack", "ferry", "satellite"}:
            import tempfile
            from scripts.strip_negative_preconditions import strip_negative_pre
            tmpdir = tempfile.mkdtemp()
            stripped = os.path.join(tmpdir, f"{t}_domain.pddl")
            with open(stripped, "w") as f:
                f.write(strip_negative_pre(open(dom_path).read()))
            dom_path = stripped
        probs = []
        for sub in ["training/easy", "testing/easy"]:
            probs.extend(sorted(glob.glob(
                os.path.join(dom_dir, sub, "p*.pddl"))))
        probs = probs[:args.n_problems]
        target_trs_by_action = {}
        target_vocab = set()
        for i, p in enumerate(probs):
            try:
                _, task = load_task(dom_path, p)
            except Exception:
                continue
            for s in range(3):
                for sb, action, ag, sa in random_rollout_pddl(
                        task, n_steps=100, seed=i*100+s):
                    if sb == sa: continue
                    target_trs_by_action.setdefault(action, []).append(
                        (sb, ag, sa))
                    for atom in sb | sa:
                        nm = atom.split("(")[0]
                        target_vocab.add(nm)
        target_data[t] = {
            "trs_by_action": target_trs_by_action,
            "vocab": target_vocab,
        }
        print(f"  {t}: {len(target_trs_by_action)} actions, "
              f"vocab size {len(target_vocab)}")

    # Per target action: signature-subset baseline, then the LLM hint,
    # both verified on held-out rollouts.
    print("\nRunning baseline vs LLM-hint reuse...")
    sig_reuses = []
    llm_reuses = []
    rows = []

    for tgt, td in target_data.items():
        for tgt_action, trs in td["trs_by_action"].items():
            if len(trs) < 5: continue
            held_out = trs[:30]

            # --- Signature baseline ---
            sig_match = None
            for sd_name, sd_models in source_models.items():
                for sd_action, sd_schema in sd_models.items():
                    if signature_subset(sd_schema, td["vocab"]):
                        score = verify_schema_on_target(sd_schema,
                                                          held_out)
                        if score >= 0.9:
                            sig_match = (sd_name, sd_action, score)
                            break
                if sig_match: break

            # --- LLM-hint ---
            sources_descr = []
            for sd_name, sd_models in source_models.items():
                acts = sorted(sd_models.keys())[:6]
                sources_descr.append(f"  {sd_name}: {acts}")
            prompt = (
                "You match a target-domain action to a source-"
                "domain action with the SAME SHAPE (same pattern of "
                "preconditions, adds, deletes; under predicate "
                "renaming).\n\n"
                "## Step-by-step procedure\n"
                "1. Identify what the target action DOES "
                "(move-agent? acquire-object? toggle-state? assemble?).\n"
                "2. Find the source action with the SAME semantic role.\n"
                "3. Verify arity + pre/add/del pattern matches.\n\n"
                "## In-context examples\n"
                "Ex1: target 'navigate' in Rovers (move-agent role) "
                "→ source 'logistics.drive-truck' (move-agent). Both: "
                "pre{at(agent,loc1)}, add{at(agent,loc2)}, "
                "del{at(agent,loc1)}.\n"
                "Ex2: target 'drive' in Transport (move-vehicle) → "
                "source 'rovers.navigate' (move-agent). Same shape.\n"
                "Ex3: target 'pick-up' in Blocksworld (acquire-object) "
                "→ source 'gripper.pick' (acquire-object). Same shape.\n"
                "Ex4: target 'stack' in Blocksworld (assemble) → "
                "no direct gripper match; pick the closest source "
                "semantically (e.g., gripper.drop is the closest, "
                "though imperfect).\n\n"
                f"## Target action to match\n"
                f"  name: {tgt_action}\n"
                f"  target-domain predicate vocabulary: "
                f"{sorted(td['vocab'])[:20]}\n\n"
                f"## Source domains + actions to choose from\n"
                + "\n".join(sources_descr) + "\n\n"
                "Reason about semantic role FIRST. Pick the BEST "
                "(source_domain, source_action) match.\n"
                "Output ONLY this JSON (no commentary):\n"
                "{\"source_domain\": \"...\", \"source_action\": \"...\"}"
            )
            schema_json = (
                '{"type":"object","properties":'
                '{"source_domain":{"type":"string"},'
                '"source_action":{"type":"string"}},'
                '"required":["source_domain","source_action"]}'
            )

            def llm_verify(prop, sd_models=source_models,
                            held_out=held_out):
                if not isinstance(prop, dict): return 0.0
                sd_name = prop.get("source_domain")
                sd_act = prop.get("source_action")
                if sd_name not in sd_models: return 0.0
                if sd_act not in sd_models[sd_name]: return 0.0
                schema = sd_models[sd_name][sd_act]
                return verify_schema_on_target(schema, held_out)

            try:
                final, info = propose_verify_fallback(
                    prompt=prompt, schema_json=schema_json,
                    verifier=llm_verify,
                    baseline={"source_domain": "", "source_action": ""},
                    threshold=0.9, model_name=args.model,
                    max_new_tokens=256,
                )
                llm_match = None
                if info.get("accepted_llm"):
                    llm_match = (final.get("source_domain"),
                                  final.get("source_action"),
                                  info.get("accuracy", 0))
            except Exception as e:
                llm_match = None
                info = {"error": str(e)[:60]}

            row = {
                "target": tgt, "target_action": tgt_action,
                "sig_match": sig_match,
                "llm_match": llm_match,
            }
            rows.append(row)
            if sig_match: sig_reuses.append(row)
            if llm_match: llm_reuses.append(row)
            print(f"  {tgt}.{tgt_action}: "
                  f"sig={sig_match}, llm={llm_match}")

    print("\n" + "=" * 60)
    print(f"Cross-domain reuse comparison")
    print(f"  Signature-subset baseline: {len(sig_reuses)} pairs")
    print(f"  LLM-hint:                  {len(llm_reuses)} pairs")
    print(f"  LLM-only finds (not in sig): "
          f"{len([r for r in llm_reuses if not r['sig_match']])}")
    print("=" * 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "sig_reuses": len(sig_reuses),
            "llm_reuses": len(llm_reuses),
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
