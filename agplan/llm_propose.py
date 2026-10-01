"""Propose with an LLM, verify against held-out evidence, fall back to a baseline.

Every LLM stage (predicate discovery, family selection, goal parsing,
action-model synthesis) follows the same pattern: format a prompt, decode
JSON under an XGrammar schema mask, parse the proposal, score it with a
verifier and accept it only above a threshold.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional


def propose_verify_fallback(
    prompt: str, schema_json: str,
    verifier: Callable[[Any], float],
    baseline: Any,
    threshold: float = 0.9,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 384,
    use_mock: bool = False,
    mock_value: Any = None,
    n_samples: int = 4,
    temperature: float = 0.5,
    system_prompt: str = "You are a careful, precise assistant. Return only valid JSON matching the schema.",
    max_refine_rounds: int = 0,
    feedback_fn: Optional[Callable[[Any, float], str]] = None,
) -> tuple[Any, dict]:
    """Decode a JSON proposal, verify it, and return it or the baseline.

    verifier maps a parsed proposal to an accuracy in [0, 1]. The best of
    the sampled candidates (and a greedy decode when n_samples > 1) is
    accepted at or above threshold; otherwise baseline is returned. With
    use_mock, mock_value stands in for the model output. Returns (value,
    info) where info has accepted_llm, proposal, accuracy, fallback_used,
    n_candidates_tried, refine_rounds and refine_history.
    """
    if use_mock:
        proposal = mock_value if mock_value is not None else baseline
        proposals_tried = [proposal]
    else:
        try:
            from agplan.decoding.xgrammar_wrapper import (
                DecodeConfig, XGrammarConstrainedDecoder,
            )
            decoder = XGrammarConstrainedDecoder(
                model_name=model_name, schema=schema_json,
            )
            # Small models under a tight schema mask often stall under
            # greedy decoding; sampled candidates reach valid proposals
            # more often.
            cfg = DecodeConfig(do_sample=(n_samples > 1),
                                temperature=temperature,
                                top_p=0.95,
                                max_new_tokens=max_new_tokens)
            raws = decoder.generate_n(user_prompt=prompt,
                                         system_prompt=system_prompt,
                                         n=n_samples, cfg=cfg)
            proposals_tried = []
            for r in raws:
                try:
                    proposals_tried.append(json.loads(r))
                except Exception:
                    pass
            # Greedy decode as one more candidate.
            if n_samples > 1:
                try:
                    greedy_cfg = DecodeConfig(do_sample=False,
                                                max_new_tokens=max_new_tokens)
                    g = decoder.generate(user_prompt=prompt,
                                            system_prompt=system_prompt,
                                            cfg=greedy_cfg)
                    proposals_tried.append(json.loads(g))
                except Exception:
                    pass
            proposal = (proposals_tried[0] if proposals_tried
                         else {"error": "no valid JSON"})
        except Exception as e:
            proposal = {"error": f"{type(e).__name__}: {e}"}
            proposals_tried = [proposal]

    # A verifier exception is reported once per call and scores the
    # candidate 0.0, so batch runs keep going.
    best_acc = 0.0
    best_proposal = proposal
    _verifier_err_logged = False
    for p in proposals_tried:
        try:
            acc_p = verifier(p)
        except Exception as _ve:
            if not _verifier_err_logged:
                import sys, traceback
                print(f"[llm_propose] WARNING: verifier raised "
                      f"{type(_ve).__name__}: {_ve}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                _verifier_err_logged = True
            acc_p = 0.0
        if acc_p > best_acc:
            best_acc = acc_p
            best_proposal = p
    acc = best_acc
    proposal = best_proposal

    # Refinement: below threshold, append verifier feedback (feedback_fn)
    # to the prompt and resample, up to max_refine_rounds times.
    refine_history = []
    if not use_mock and max_refine_rounds > 0 and acc < threshold:
        try:
            from agplan.decoding.xgrammar_wrapper import (
                DecodeConfig, XGrammarConstrainedDecoder,
            )
            decoder = XGrammarConstrainedDecoder(
                model_name=model_name, schema=schema_json,
            )
        except Exception:
            decoder = None

        round_idx = 0
        while (decoder is not None and round_idx < max_refine_rounds
                and acc < threshold):
            round_idx += 1
            if feedback_fn is not None:
                try:
                    fb = feedback_fn(proposal, acc)
                except Exception:
                    fb = f"Previous proposal scored {acc:.2f} (threshold {threshold}). Try again, more carefully."
            else:
                fb = (f"Previous proposal: {json.dumps(proposal)[:200]}\n"
                       f"That scored {acc:.2f} (need {threshold}). "
                       "Revise your answer.")
            refined_prompt = (
                prompt + "\n\n## Previous attempt feedback (round "
                f"{round_idx}):\n{fb}\n\n"
                "Please revise. Output a corrected JSON."
            )
            try:
                cfg = DecodeConfig(do_sample=True,
                                    temperature=temperature,
                                    top_p=0.95,
                                    max_new_tokens=max_new_tokens)
                raws = decoder.generate_n(user_prompt=refined_prompt,
                                             system_prompt=system_prompt,
                                             n=n_samples, cfg=cfg)
                round_candidates = []
                for r in raws:
                    try: round_candidates.append(json.loads(r))
                    except Exception: pass
                round_best_acc = acc
                round_best = proposal
                for p in round_candidates:
                    try:
                        a = verifier(p)
                    except Exception as _ve2:
                        if not _verifier_err_logged:
                            import sys, traceback
                            print(f"[llm_propose:refine] WARNING: "
                                  f"verifier raised "
                                  f"{type(_ve2).__name__}: {_ve2}",
                                  file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                            _verifier_err_logged = True
                        a = 0.0
                    if a > round_best_acc:
                        round_best_acc = a
                        round_best = p
                refine_history.append({
                    "round": round_idx, "best_acc": round_best_acc,
                    "n_candidates": len(round_candidates),
                })
                acc = round_best_acc
                proposal = round_best
            except Exception:
                break

    accepted = acc >= threshold
    final = proposal if accepted else baseline
    return final, {
        "accepted_llm": accepted,
        "proposal": proposal,
        "accuracy": acc,
        "fallback_used": not accepted,
        "n_candidates_tried": len(proposals_tried),
        "refine_rounds": len(refine_history),
        "refine_history": refine_history,
    }


__all__ = ["propose_verify_fallback"]
