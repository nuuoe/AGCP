# AGCP: Abstraction-based Grammar-Constrained Planning

Code, saved experiment outputs and verification scripts for

> **Where Does Plan Validity Live in Grammar-Constrained LLM Planning? An Applicability Mask Induced from Environment Rollouts.** Nijesh Upreti. Findings of AACL-IJCNLP 2026.

AGCP induces lifted action schemas from environment rollouts, compiles the applicable and goal-reaching action sequences into a context-free grammar, and enforces that grammar as a token mask during LLM decoding (XGrammar). Natural-language goals are parsed into predicate atoms under a JSON schema and corrected with verifier feedback. The paper evaluates on PlanBench, the IPC 2023 learning track, AutoPlanBench and ALFWorld.

## Contents

```
agplan/            package: induction, grammar compilation, SCI-ReDuce, XGrammar wrapper
scripts/           experiment, analysis and verification scripts (run from this directory)
scripts/figures/   figure scripts for Fig. 2 and Fig. 3
runs/              saved outputs behind every reported number (JSON/JSONL)
domains/           PDDL domains and sample instances (PlanBench, IPC 2023)
```

Every number in the paper can be checked against `runs/` without a GPU or an API key:

```bash
python scripts/verify_numbers.py     # headline numbers of Secs. 3.3, 5.1-5.4 and App. L (85 checks)
python scripts/verify_tables.py      # Table 2 cells, E6 variants, N4 demonstration, regex pooled rate
python scripts/verify_all.py         # claim harness, including the evaluation-pattern triggers of Table 8
python scripts/paper_statistics.py   # appendix statistics: clustered bootstrap, McNemar p-values, failure tally
```

## Installation

Python 3.12. The versions in `requirements.txt` are the ones used for the reported runs.

```bash
pip install -r requirements.txt
pip install -e .
```

Optional, only for re-running experiments:

- `bash scripts/setup_external.sh` clones PlanBench (LLMs-Planning), the IPC 2023 learning-track benchmarks and AutoPlanBench into `external/`. The AutoPlanBench scripts need `external/autoplanbench` on `PYTHONPATH` and that project's own requirements.
- ALFWorld: `pip install alfworld && alfworld-download` (the scripts default to `~/.cache/alfworld/json_2.1.1/{valid_seen,valid_unseen}`).
- VAL (`validate` on `PATH`) for `scripts/val_audit.py`.
- Cloud models read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `TOGETHER_API_KEY` from the environment.

Scripts import `agplan` and each other as `scripts.<name>`, so run them from the repository root. Two smoke tests check the environment: `scripts/smoke_test_xgrammar.py` (constrained decoding with a small local model) and `scripts/smoke_alfworld.py` (the ALFWorld environment).

## Reproducing the paper

| Paper location | Script | Saved output in `runs/` |
|---|---|---|
| Table 1, No-LLM column | `llm_removal_ablation.py` | `llm_removal_*.json` |
| Table 1, AGCP column; App. K T1 to T8 | `run_agcp_on_planbench.py`, then `score_agcp_planbench.py`; `val_audit.py` | `planbench_t1_final/`, `agcp_depots_n30_score.json`, `score_t_*.json`, `val_audit_planbench_t1.json`; raw decodes in `planbench_responses/` |
| Table 1, LLM+P column; Table 8 pattern 2 | `llm_p_baseline.py`, `llm_p_depots_matched.py`, `rescore_llm_p.py` | `llm_p_depots_lexsorted.json`, `llm_p_depots_matched_n30.json`, `llm_p_rescored_nonempty.json` |
| Sec. 5.1 mask isolation (0/160 vs 160/160) | `run_genv_planbench.py --mask {syntax,genv}` | `mask_isolation_*.jsonl` |
| Sec. 5.1 and Ethics, jailbreak prompts | `run_adversarial_robustness.py` | `adversarial_q7/adversarial.jsonl` |
| Sec. 5.1 IPC 2023 component F1; App. O reuse | `run_ipc2023_breadth.py`, `refine_preconditions.py`, `cross_domain_reuse_sweep.py` | `ipc2023_breadth_v4.json`, `ipc2023_breadth_full50.json`, `*_refined.json`, `cross_domain_reuse_sweep.json` |
| Sec. 3.3 compile provenance | `compile_provenance_stats.py --suite {planbench,apb} [--fix_format]` | `compile_provenance_*.json` |
| Table 2, Fig. 3, Tables 5 and 6, Tables 9 to 11 | `alfworld_regex_v2.py`, `alfworld_fair_eval.py`, `alfworld_cloud_llm.py`; `per_task_type_analysis.py`, `pairwise_mcnemar_cohen.py` | `alfworld_regex_v2_full.json`, `alfworld_fair_eval_full.json`, `alfworld_cloud_*.json`, `ood/`, `seed{1,2,3}/`, `per_task_type_stratified.json`, `pairwise_mcnemar_*.json` |
| Sec. 5.2 paraphrase control; App. I | `alfworld_paraphrase_decon.py`; `paper_statistics.py` | `alfworld_paraphrase_decon_*.json` |
| Sec. 5.3 parse correction (RQ3); App. H, App. U | `alfworld_iterative_e4.py`, `alfworld_iterative_e4_with_refinement.py`; `analyze_rq3_parse_level.py`, `analyze_iterative_e4_validation.py`, `analyze_iterative_e4_refinement.py` | `alfworld_iterative_e4*.json`, `rq3_parse_level.json` |
| Table 3, Sec. 5.4, App. R, App. T | `alfworld_closed_loop_e6.py`, `alfworld_closed_loop_e6_llm_planner.py`, `alfworld_closed_loop_e6_induced_planner.py`, `run_alfworld_induction.py` | `alfworld_closed_loop_e6*.json`, `alfworld_e6_induced_planner.json`, `alfworld_induction*.json` |
| Sec. 5.4 and App. L, SCI-ReDuce end to end | `scireduce_e2e.py --variant {raw,budget}` | `scireduce_e2e_{raw,budget}.json`, input pool `scireduce_input/mystery.jsonl` |
| Fig. 2 and Table 4, SCI-ReDuce coverage | `run_leave_out_n.py` per split; `scireduce_coverage_aggregate.py`; `figures/make_scireduce_curve.py` | `scireduce_leaveout/*.jsonl`, `sci_reduce_coverage_curve.json` |
| Table 7 and App. G, PlanBench goal parsing | `benchmark_nl_goal_parse.py`, `regex_nl_goal_baseline.py`, `n1_canonicalize_rescore.py`, `categorize_n1_failures.py`, `n1_paraphrase_robustness.py`, `n1_distractor_robustness.py` | `n1_parse_*.json`, `regex_baseline_*.json`, `n1_alias_prompted_*.json`, `n1_failure_modes.json`, `n1_paraphrase.json`, `n1_distractor.json` |
| Table 8 patterns 1, 3, 4, 5 | `demo_closed_loop.py`, `demo_closed_loop_no_gt_fallback.py`, `alfworld_enum_constrained.py`, `alfworld_nl_parse.py`, `alfworld_smart_score.py` | `n4_*.json`, `alfworld_enum_constrained*.json`, `alfworld_n1_parse*.json`, `alfworld_smart_score*.json` |
| App. K predicate discovery, reuse probe, ablation ladder, AutoPlanBench breadth | `llm_predicate_discovery_strict.py`, `benchmark_n2_predicate_discovery.py`, `benchmark_n3_llm_reuse.py`, `run_ablation_ladder.py`, `run_agcp_on_apb_simple.py` | `n2_strict_N15.json`, `n2_loophole_verifier.json`, `n3_improved.json`, `ablation_ladder.json`, `final_apb_*.json` |

The four GPU experiments were run with these invocations:

```bash
python scripts/run_adversarial_robustness.py --instances_dir <planbench blocksworld instances> \
    --n_instances 5 --k 4 --model Qwen/Qwen2.5-7B-Instruct --max_new_tokens 384 --out runs/adversarial_q7/adversarial.jsonl
python scripts/run_genv_planbench.py --instances_dir <mystery blocksworld instances> --domain mystery \
    --model Qwen/Qwen2.5-1.5B-Instruct --n_instances 50 --k 16 --max_extra 4 --max_new_tokens 384 --out runs/scireduce_input/mystery.jsonl
python scripts/run_ipc2023_breadth.py --base_dir external/ipc2023-learning --n_problems 8 \
    --n_rollouts_per_problem 5 --steps_per_rollout 200 --out runs/ipc2023_breadth_v4.json
python scripts/run_agcp_on_apb_simple.py --apb_root external/autoplanbench/autoplanbench_dataset/apb2.0_dataset \
    --domain <domain> --model Qwen/Qwen2.5-1.5B-Instruct --n_rollouts 10 --out runs/final_apb_<domain>.json
```

## Notes on the saved outputs

- The `seed1/`, `seed2/` and `seed3/` directories under `runs/` are independent re-runs of the cloud models at temperature 0.3; the provider APIs expose no seed, as the paper's Limitations section states.
- `mask_isolation_syntax_mystery.jsonl` was produced with the syntax-only grammar, which does not depend on the Mystery obfuscation map, so its candidates coincide with the Blocksworld syntax arm.
- `run_agcp_on_planbench.py` counts completed instances only on success and `score_agcp_planbench.py` skips instances with an empty response. All reported PlanBench cells use pinned instance sets (`--specific_instances`; N=10 and the matched N=30 set), and `val_audit.py` cross-checks them with VAL.
- `runs/planbench_responses/` holds the raw PlanBench-format decodes behind Table 1 and App. K; `score_agcp_planbench.py` reads such directories through `--plan_bench_root`.

## License

Copyright 2026 Nijesh Upreti. The code is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`). Benchmark files under `domains/` and `runs/planbench_responses/` derive from PlanBench (MIT License) and the IPC 2023 learning-track benchmarks and remain under their own terms.

## Citation

```bibtex
@inproceedings{upreti2026agcp,
  title     = {Where Does Plan Validity Live in Grammar-Constrained {LLM} Planning? An Applicability Mask Induced from Environment Rollouts},
  author    = {Upreti, Nijesh},
  booktitle = {Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026},
  year      = {2026}
}
```
