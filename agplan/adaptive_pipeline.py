"""Adaptive pipeline: LLM proposal with verification, model escalation and memory.

The loop skips the LLM when the baseline verifier already passes the
threshold, tries the smallest configured model first and escalates on
verification failure, keeps accepted (prompt, proposal) pairs as
in-context examples, updates the per-action transition store
incrementally, and prunes predicates that no action uses.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agplan.llm_propose import propose_verify_fallback


@dataclass
class PipelineMemory:
    """Few-shot memory of accepted proposals across a pipeline run."""
    accepted: list[dict] = field(default_factory=list)
    # accepted entry: {"stage": str, "prompt": str, "proposal": dict,
    #                  "accuracy": float, "context_tag": str}

    def add(self, stage: str, prompt: str, proposal: dict,
             accuracy: float, context_tag: str = ""):
        self.accepted.append({
            "stage": stage, "prompt": prompt, "proposal": proposal,
            "accuracy": accuracy, "context_tag": context_tag,
        })

    def few_shot_for(self, stage: str, context_tag: str = "",
                      k: int = 3) -> list[dict]:
        """Return up to k most recent accepted proposals matching
        stage+tag, for in-context use."""
        matches = [e for e in reversed(self.accepted)
                   if e["stage"] == stage
                   and (not context_tag or e["context_tag"] == context_tag)]
        return matches[:k]


@dataclass
class AdaptiveStage:
    """One stage of the pipeline (e.g. predicate-discovery,
    action-model-synthesis)."""
    name: str
    build_prompt: Callable[..., str]
    schema_json: str
    verifier: Callable[[Any], float]
    baseline: Any
    threshold: float = 0.9
    # ordered list of (model_id, max_new_tokens) to try in escalation order
    model_ladder: list[tuple[str, int]] = field(default_factory=lambda: [
        ("Qwen/Qwen2.5-0.5B-Instruct", 256),
        ("Qwen/Qwen2.5-1.5B-Instruct", 384),
        ("Qwen/Qwen2.5-7B-Instruct", 512),
    ])
    context_tag: str = ""
    # If baseline verifier score >= skip_threshold, skip LLM
    skip_threshold: float = 0.95


def run_adaptive_stage(
    stage: AdaptiveStage,
    memory: PipelineMemory,
    use_mock: bool = False,
    mock_value: Any = None,
    **prompt_kwargs,
) -> dict:
    """Run one stage adaptively.

    Returns dict with: final, accepted_llm, accuracy, model_used,
    skipped_llm, fallback_used.
    """
    # (A) ADAPTIVE INVOCATION: check baseline first
    baseline_acc = stage.verifier(stage.baseline)
    if baseline_acc >= stage.skip_threshold:
        return {
            "final": stage.baseline,
            "accepted_llm": False,
            "accuracy": baseline_acc,
            "model_used": None,
            "skipped_llm": True,
            "fallback_used": False,
        }

    # (C) FEW-SHOT MEMORY: prepend matching prior accepted proposals
    few_shot = memory.few_shot_for(stage.name, stage.context_tag, k=3)
    fs_text = ""
    if few_shot:
        fs_text = "\n\nPrior successful proposals (use as guides):\n"
        for ex in few_shot:
            fs_text += f"- {json.dumps(ex['proposal'])[:200]}\n"

    prompt = stage.build_prompt(**prompt_kwargs) + fs_text

    # (B) CAPACITY ESCALATION: try smallest model first
    result = None
    for model_id, max_tokens in stage.model_ladder:
        final, info = propose_verify_fallback(
            prompt=prompt,
            schema_json=stage.schema_json,
            verifier=stage.verifier,
            baseline=stage.baseline,
            threshold=stage.threshold,
            model_name=model_id,
            max_new_tokens=max_tokens,
            use_mock=use_mock,
            mock_value=mock_value,
        )
        result = (final, info, model_id)
        if info["accepted_llm"]:
            break

    final, info, model_used = result
    if info["accepted_llm"]:
        memory.add(stage.name, prompt, info["proposal"],
                    info["accuracy"], stage.context_tag)
    return {
        "final": final,
        "accepted_llm": info["accepted_llm"],
        "accuracy": info["accuracy"],
        "model_used": model_used if info["accepted_llm"] else None,
        "skipped_llm": False,
        "fallback_used": info.get("fallback_used", False),
    }


@dataclass
class IncrementalInductor:
    """Maintains running per-action transition stats; updates
    incrementally rather than re-running intersection-stats from
    scratch each round."""
    # action -> list of (sb, args, sa) tuples
    transitions: dict = field(default_factory=lambda: defaultdict(list))
    # action -> running predicate counts
    pre_inter: dict = field(default_factory=lambda: defaultdict(set))

    def add_transitions(self, new_trs: list):
        """Append new transitions; update running statistics."""
        for sb, action, args, sa in new_trs:
            if sb == sa:  # skip no-ops
                continue
            self.transitions[action].append((sb, args, sa))

    def induce(self) -> dict:
        """Recompute models from the current store via the shared
        intersection-stats induction.
        """
        # Reuse the shared induction (already handles causal effects)
        # by constructing the flat transition list it expects.
        all_trs = []
        for action, items in self.transitions.items():
            for sb, args, sa in items:
                all_trs.append((sb, action, args, sa))
        from scripts.induce_pddl_generic import induce_lifted_models
        return induce_lifted_models(all_trs)


def prune_vocabulary(models: dict,
                      vocab: set[str]) -> set[str]:
    """(E) Remove from vocab any predicate NAME that does not
    appear in any action's pre/eff_add/eff_del. Compares predicate
    NAMES (the bit before the parenthesis) so that templated
    predicates in the model can be matched against ground
    predicates in the vocab.
    """
    import re
    def _name(p: str) -> str:
        m = re.match(r"^([a-zA-Z_][\w-]*)", p)
        return m.group(1) if m else p

    used_names: set[str] = set()
    for m in models.values():
        for k in ("pre_pos", "eff_add", "eff_del"):
            for p in m.get(k, []):
                used_names.add(_name(p))
    return {v for v in vocab if _name(v) in used_names}


@dataclass
class SchemaMemory:
    """Cross-domain store of induced lifted schemas, keyed by (action name, signature hash).

    A stored schema is a reuse candidate for a new action when its predicate
    signature is a subset of the new domain's observed vocabulary; reuse is
    tried before fresh induction.
    """
    schemas: dict = field(default_factory=dict)
    # (action_name, sig_hash) -> {"pre_pos": set, "eff_add": set,
    #                              "eff_del": set, "arity": int,
    #                              "provenance_domain": str,
    #                              "n_verified": int}

    def _signature(self, model: dict) -> str:
        import re
        names = set()
        for k in ("pre_pos", "eff_add", "eff_del"):
            for p in model.get(k, []):
                m = re.match(r"^([a-zA-Z_][\w-]*)", p)
                if m: names.add(m.group(1))
        return ",".join(sorted(names))

    def add(self, action_name: str, model: dict,
             provenance: str = ""):
        key = (action_name, self._signature(model))
        existing = self.schemas.get(key)
        if existing:
            existing["n_verified"] = existing.get("n_verified", 0) + 1
        else:
            self.schemas[key] = {
                **{k: set(model.get(k, [])) for k in
                    ("pre_pos", "eff_add", "eff_del")},
                "arity": model.get("arity", 0),
                "provenance_domain": provenance,
                "n_verified": 1,
            }

    def candidates_for(self, action_name: str,
                        observed_predicate_names: set) -> list[dict]:
        """Return memory entries whose schema's signature is a
        subset of the observed predicate vocabulary. These are the
        candidates to TRY before fresh induction."""
        out = []
        for (a_name, sig), entry in self.schemas.items():
            if a_name != action_name:
                continue
            sig_names = set(sig.split(","))
            if sig_names <= observed_predicate_names:
                out.append(entry)
        # Sort by verification count (more-verified first)
        return sorted(out, key=lambda e: -e.get("n_verified", 0))

    def structural_candidates(self, observed_predicate_names: set
                               ) -> list[dict]:
        """Return ALL memory entries (regardless of original action
        name) whose predicate-name signature is a subset of the
        observed vocabulary. Used for cross-domain compositional
        reuse where action names differ between source and target
        domain (e.g., Logistics's `drive-truck` vs Depots's `Drive`,
        Blocksworld's `stack` vs Depots's `Drop`).

        Caller verifies each candidate against held-out transitions
        and accepts the first that exceeds the threshold."""
        out = []
        for (a_name, sig), entry in self.schemas.items():
            sig_names = set(sig.split(","))
            if sig_names <= observed_predicate_names:
                out.append(entry)
        return sorted(out, key=lambda e: -e.get("n_verified", 0))


def try_schema_reuse(
    memory: SchemaMemory,
    action_name: str,
    held_out_transitions: list,
    verifier: Callable[[dict, list], float],
    accept_threshold: float = 0.9,
) -> Optional[dict]:
    """For each candidate schema for `action_name`, verify on the
    domain's held-out transitions. Return the first that passes,
    else None (caller falls back to fresh induction).
    """
    # Collect observed predicate names from held-out
    import re
    obs_names = set()
    for t in held_out_transitions:
        sb = t[0] if len(t) >= 1 else set()
        sa = t[-1] if len(t) >= 1 else set()
        for p in (sb | sa):
            m = re.match(r"^([a-zA-Z_][\w-]*)", p)
            if m: obs_names.add(m.group(1))

    for cand in memory.candidates_for(action_name, obs_names):
        score = verifier(cand, held_out_transitions)
        if score >= accept_threshold:
            return cand
    return None


__all__ = [
    "PipelineMemory", "AdaptiveStage", "run_adaptive_stage",
    "IncrementalInductor", "prune_vocabulary",
    "SchemaMemory", "try_schema_reuse",
]
