"""Recount the paper's headline claims from the run outputs under runs/.

Each check recounts a stated number from a run file, replays the classical
pipeline on a bundled Blocksworld instance with a mocked LLM, or reproduces
a documented evaluation-inflation pattern, which is reported as EXPECTED.
Needs no GPU or API. Run from the repository root: python scripts/verify_all.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
SCRIPTS = REPO_ROOT / "scripts"
DOMAINS = REPO_ROOT / "domains"
EXTERNAL = REPO_ROOT / "external"

OK = "OK"
EXPECTED = "EXPECTED"
FAIL = "FAIL"

Row = tuple  # (claim, stated, actual, status)


def load(path: Path):
    if not path.exists():
        print(f"  missing: {path.relative_to(REPO_ROOT)}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  parse error on {path.name}: {e}", file=sys.stderr)
        return None


def load_jsonl(path: Path):
    if not path.exists():
        print(f"  missing: {path.relative_to(REPO_ROOT)}")
        return None
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def status(cond: bool) -> str:
    return OK if cond else FAIL


def first_int(s) -> int | None:
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


# C1: Table 1, Depots at N=30.
def check_c1_depots() -> list[Row]:
    a = load(RUNS / "agcp_depots_n30_score.json")
    p = load(RUNS / "llm_p_depots_matched_n30.json")
    lex = load(RUNS / "llm_p_depots_lexsorted.json")
    resc = load(RUNS / "llm_p_rescored_nonempty.json")
    if None in (a, p, lex, resc):
        return [("C1 Table 1 Depots N=30", "inputs present", "missing input", FAIL)]

    agcp_ids = {first_int(r.get("instance_id")) for r in a["per_instance"]}
    llmp_ids = {first_int(r.get("problem")) for r in p["results"]}
    shared = agcp_ids & llmp_ids
    print(f"  C1: AGCP instances {min(agcp_ids)}..{max(agcp_ids)} (n={len(agcp_ids)}), "
          f"LLM+P instances {min(llmp_ids)}..{max(llmp_ids)} (n={len(llmp_ids)}), "
          f"shared={len(shared)}")
    rows = [("C1a Table 1 Depots: AGCP and LLM+P scored on the same instances",
             "30 shared", f"{len(shared)} shared",
             status(len(shared) == len(agcp_ids) == len(llmp_ids) == 30))]

    agcp_solved = sum(1 for r in a["per_instance"] if r.get("success"))
    rows.append(("C1b Table 1 Depots AGCP", "30/30",
                 f"{agcp_solved}/{len(a['per_instance'])}",
                 status(agcp_solved == len(a["per_instance"]) == 30)))

    def recount(run):
        raw = sum(1 for r in run["results"] if r.get("llm_success"))
        strict = sum(1 for r in run["results"]
                     if r.get("llm_success") and (r.get("plan_length") or 0) > 0)
        return raw, strict, len(run["results"])

    def rescore_agrees(raw, strict, n):
        return any(e.get("n_orig_success") == raw and e.get("n_strict_success") == strict
                   and e.get("n_total") == n for e in resc.get("summary", []))

    raw, strict, n = recount(p)
    agree = rescore_agrees(raw, strict, n)
    rows.append(("C1c Table 1 Depots LLM+P under the non-empty-plan criterion",
                 "0/30 (10/30 before the criterion)",
                 f"{strict}/{n} ({raw}/{n} before; rescore file agrees={agree})",
                 status(strict == 0 and raw == 10 and n == 30 and agree)))

    raw, strict, n = recount(lex)
    agree = rescore_agrees(raw, strict, n)
    rows.append(("C1d Table 8 LLM+P empty-plan counting error (lexicographic run)",
                 "12/30 reported, 0/30 non-empty",
                 f"{raw}/{n} reported, {strict}/{n} non-empty (rescore file agrees={agree})",
                 status(raw == 12 and strict == 0 and n == 30 and agree)))
    return rows


# C2: schema library reuse (Sec. 5.1, App. O).
def check_c2_reuse() -> Row:
    claim = "C2 Sec. 5.1 schemas reused from the library (App. O)"
    d = load(RUNS / "ipc2023_breadth_v4.json")
    if d is None:
        return (claim, "2 reused, 44 induced", "missing input", FAIL)
    results = d.get("results", {})
    pairs = [(dom, act, info.get("reuse_source", "?"))
             for dom, body in results.items()
             for act, info in body.get("per_action", {}).items() if info.get("reused")]
    n_actions = sum(len(b.get("per_action", {})) for b in results.values())
    for dom, act, src in pairs:
        print(f"  C2: {dom}.{act} reused from {src}")
    n_summary = d.get("summary", {}).get("n_reused")
    ok = len(pairs) == 2 and n_actions == 46 and n_summary == 2
    return (claim, "2 reused, 44 induced",
            f"{len(pairs)} reused, {n_actions - len(pairs)} induced", status(ok))


# C3: Table 1, AGCP column.
def check_c3_table1_agcp() -> Row:
    cells = {
        "Blocksworld": (RUNS / "planbench_t1_final" / "blocksworld_3_score.json", 10),
        "Mystery": (RUNS / "planbench_t1_final" / "mystery_blocksworld_3_score.json", 10),
        "Logistics": (RUNS / "planbench_t1_final" / "logistics_score.json", 10),
        "Depots": (RUNS / "agcp_depots_n30_score.json", 30),
    }
    parts, solved, total, ok = [], 0, 0, True
    for name, (path, n_expected) in cells.items():
        d = load(path)
        if d is None:
            ok = False
            parts.append(f"{name} missing")
            continue
        ns, nt = int(d.get("n_solved", 0)), int(d.get("n_total", 0))
        parts.append(f"{name} {ns}/{nt}")
        solved += ns
        total += nt
        ok = ok and ns == nt == n_expected
    print("  C3: " + ", ".join(parts))
    return ("C3 Table 1 AGCP column (PlanBench T1, matched instance sets)",
            "10/10, 10/10, 10/10, 30/30 = 60/60", f"{solved}/{total}",
            status(ok and solved == total == 60))


# C4: runner and scorer accounting. Neither script counts an instance that
# produced no response, so a score file's denominator can be smaller than
# the number of instances attempted. C4c checks the released score files.
def check_c4_accounting() -> list[Row]:
    rows = []
    runner = SCRIPTS / "run_agcp_on_planbench.py"
    scorer = SCRIPTS / "score_agcp_planbench.py"

    claim = "C4a runner counts planned instances only toward --max_instances"
    if runner.exists():
        lines = runner.read_text().splitlines()
        success_only = any("n_done += 1" in l
                           and "else:" in "\n".join(lines[max(0, i - 6):i])
                           for i, l in enumerate(lines))
        guard = any("n_done >= args.max_instances" in l for l in lines)
        present = success_only and guard
        rows.append((claim, "accounting pattern; guarded by C4c",
                     "present (n_done += 1 on success only; stop at n_done >= max_instances)"
                     if present else "not present",
                     EXPECTED if present else OK))
    else:
        rows.append((claim, "script present", "missing script", FAIL))

    claim = "C4b scorer skips instances with an empty llm_raw_response"
    if scorer.exists():
        present = 'not inst.get("llm_raw_response")' in scorer.read_text()
        rows.append((claim, "accounting pattern; guarded by C4c",
                     "present (empty responses leave the denominator)"
                     if present else "not present",
                     EXPECTED if present else OK))
    else:
        rows.append((claim, "script present", "missing script", FAIL))

    files = [RUNS / "planbench_t1_final" / f for f in (
        "blocksworld_3_score.json", "mystery_blocksworld_3_score.json",
        "logistics_score.json", "depots_score.json")]
    files += sorted(RUNS.glob("agcp_*.json"))
    gaps = []
    for f in files:
        d = load(f)
        if d is None:
            gaps.append(f"{f.name}: missing")
            continue
        ids = [first_int(r.get("instance_id")) for r in d.get("per_instance", [])]
        contiguous = (ids and None not in ids
                      and max(ids) - min(ids) + 1 == len(set(ids)) == len(ids))
        if len(ids) != d.get("n_total") or not contiguous:
            gaps.append(f"{f.name}: per_instance={len(ids)} n_total={d.get('n_total')}")
    for g in gaps:
        print(f"  C4c: {g}")
    rows.append(("C4c released score files: n_total matches per_instance, instance range has no gaps",
                 "0 files with gaps", f"{len(gaps)} files with gaps ({len(files)} checked)",
                 status(not gaps)))
    return rows


# C5: Table 7, PlanBench NL parse (Sec. 5.1).
def check_c5_table7() -> list[Row]:
    doms = ["blocksworld_3", "mystery_blocksworld_3", "logistics", "depots"]
    columns = [
        ("C5a Table 7 +alias column (Sec. 5.1: all 120 NL goals parsed)",
         "n1_alias_prompted_{}.json", [30, 30, 30, 30]),
        ("C5b Table 7 LLM+v. column (corrected extraction)",
         "n1_parse_corrected_{}.json", [30, 30, 17, 30]),
        ("C5c Table 7 note: uncorrected extraction (Table 8: 30% pooled)",
         "n1_parse_uncorrected_{}.json", [14, 12, 9, 1]),
    ]
    rows = []
    for claim, pattern, expected in columns:
        got = []
        for dom in doms:
            d = load(RUNS / pattern.format(dom))
            got.append(None if d is None else (d.get("n_llm_accepted"), d.get("n_parsed")))
        ok = all(g is not None and g == (e, 30) for g, e in zip(got, expected))
        k = sum(g[0] for g in got if g)
        n = sum(g[1] for g in got if g)
        actual = "/".join(str(g[0]) if g else "?" for g in got) + f" of 30 each = {k}/{n}"
        stated = "/".join(map(str, expected)) + f" = {sum(expected)}/120"
        rows.append((claim, stated, actual, status(ok)))

    d = load(RUNS / "n1_parse_corrected_canon_logistics.json")
    ok = (d is not None and d.get("n_pre") == 17 and d.get("n_post") == 22
          and d.get("n_total") == 30 and d.get("new_accepts") == 5)
    actual = ("missing input" if d is None else
              f"{d.get('n_pre')} -> {d.get('n_post')} of {d.get('n_total')} (+{d.get('new_accepts')})")
    rows.append(("C5d Table 7 +canon on Logistics", "17 -> 22 of 30 (+5)", actual, status(ok)))
    return rows


# C6: predicate discovery (App. K).
def loophole_score() -> float:
    """Inline copy of the superseded verify_predicate_set: a proposal passes
    when at least 60% of its predicates change across some transition, so a
    single varying predicate scores 1.0."""
    def holds(spec, state):
        return state.get(spec["field"]) == spec["value"]

    def verify(specs, pairs):
        if not specs:
            return 0.0
        n_relevant = sum(1 for spec in specs
                         if any(holds(spec, sb) != holds(spec, sa) for sb, sa in pairs[:30]))
        return n_relevant / len(specs)

    spec = [{"name": "handempty", "kind": "field_eq_value", "field": "held", "value": None}]
    return verify(spec, [({"held": None}, {"held": "a"})])


def check_c6_predicates() -> list[Row]:
    rows = []
    claim = "C6a App. K leakage-controlled predicate discovery, Qwen-1.5B"
    d = load(RUNS / "n2_strict_N15.json")
    if d is None:
        rows.append((claim, "11/15", "missing input", FAIL))
    else:
        v = d.get("verifier", {})
        ok = (d.get("n_seeds") == 15 and d.get("n_llm_accepted") == 11
              and v.get("min_distinct") == 3 and v.get("rel_threshold") == 0.6)
        rows.append((claim, "11/15 (>=3 predicates, >=60% relevant)",
                     f"{d.get('n_llm_accepted')}/{d.get('n_seeds')} "
                     f"(min_distinct={v.get('min_distinct')}, rel_threshold={v.get('rel_threshold')})",
                     status(ok)))

    score = loophole_score()
    loop = load(RUNS / "n2_loophole_verifier.json")
    accepted = f"{loop.get('n_llm_accepted')}/{loop.get('n_seeds')}" if loop else "?"
    print(f"  C6b: 1-predicate proposal scores {score} under the superseded verifier")
    rows.append(("C6b App. K verifier failure mode: a 1-predicate proposal scores 1.0",
                 "pattern reproduced (separate from Table 8)",
                 f"score={score} passes the 0.6 threshold; that verifier accepted "
                 f"{accepted} in runs/n2_loophole_verifier.json",
                 EXPECTED if score >= 0.6 else FAIL))
    return rows


# C7: Table 8, unreported ground-truth fallback on Mystery 2-11.
def check_c7_closed_loop() -> list[Row]:
    rejected = achieved = n = 0
    goals = set()
    for i in range(2, 12):
        d = load(RUNS / f"n4_closed_loop_mystery_inst{i}.json")
        if d is None:
            continue
        n += 1
        s3 = d.get("trace", {}).get("step3_nl_parse", {})
        rejected += not s3.get("llm_accepted")
        achieved += bool(d.get("goal_achieved"))
        goals.add(s3.get("nl_goal", ""))
    print(f"  C7: {n} traces, parse rejected on {rejected}, goal_achieved on {achieved}, "
          f"{len(goals)} distinct goals")
    rows = [("C7a Table 8 ground-truth fallback: parse rejected yet success reported",
             "4/10 rejected, 10/10 reported",
             f"{rejected}/{n} rejected, {achieved}/{n} reported",
             status(n == 10 and rejected == 4 and achieved == 10))]

    ok_e4 = n2 = 0
    for i in range(2, 12):
        d = load(RUNS / f"n4_no_gt_fallback_mystery_blocksworld_3_inst{i}.json")
        if d is None:
            continue
        n2 += 1
        ok_e4 += bool(d.get("llm_e4_succeeded") and d.get("success_against_gt"))
    rows.append(("C7b Table 8 corrected run without fallback: LLM-driven success, Mystery 2-11",
                 "10/10", f"{ok_e4}/{n2}", status(n2 == 10 and ok_e4 == 10)))
    rows.append(("C7c closed-loop traces carry distinct instance goals",
                 ">= 2 distinct", f"{len(goals)} distinct over {n}",
                 status(n == 10 and len(goals) >= 2)))
    return rows


# C8: IPC-2023 component F1 (Sec. 5.1).
def check_c8_ipc() -> Row:
    claim = "C8 Sec. 5.1 IPC-2023 schemas at component-F1 >= 0.9"
    d = load(RUNS / "ipc2023_breadth_v4.json")
    if d is None:
        return (claim, "46/46 over 10 domains", "missing input", FAIL)
    results = d.get("results", {})
    n_act = n_pass = 0
    for body in results.values():
        for info in body.get("per_action", {}).values():
            n_act += 1
            if min(info.get("pre_f1", 0), info.get("add_f1", 0), info.get("del_f1", 0)) >= 0.9:
                n_pass += 1
    print(f"  C8: domains={len(results)}, schemas={n_act}, at F1>=0.9: {n_pass}")
    return (claim, "46/46 over 10 domains", f"{n_pass}/{n_act} over {len(results)} domains",
            status(len(results) == 10 and n_act == n_pass == 46))


# C9: Depots schema replay; the file is released but the paper does not quote it.
def check_c9_depots_replay() -> Row:
    claim = "C9 Depots schema replay (runs/depots_replay_50.json, not quoted in the paper)"
    d = load(RUNS / "depots_replay_50.json")
    if d is None:
        return (claim, "50/50", "missing input", FAIL)
    s = d.get("summary", {})
    nr, ng = s.get("n_replayable"), s.get("n_gt_planned")
    return (claim, "50 replayable, 50 planned", f"{nr} replayable, {ng} planned",
            status(nr == 50 and ng == 50))


# C10: jailbreak prompts (Sec. 5.1, Ethical Considerations).
def check_c10_adversarial() -> list[Row]:
    rows = load_jsonl(RUNS / "adversarial_q7" / "adversarial.jsonl")
    if rows is None:
        return [("C10 jailbreak prompts", "0/100 coarse, 100/100 G_env", "missing input", FAIL)]
    sound, k = Counter(), Counter()
    for r in rows:
        sound[r["mask"]] += r["n_sound"]
        k[r["mask"]] += r["k"]
    prompts = {r["adversarial"] for r in rows}
    print(f"  C10: {len(rows)} rows, prompts={sorted(prompts)}, "
          f"coarse {sound['coarse']}/{k['coarse']}, genv {sound['genv']}/{k['genv']}")
    return [
        ("C10a coarse-syntax grammar admits no sound plan under 5 jailbreak prompts",
         "0/100 over 5 prompts", f"{sound['coarse']}/{k['coarse']} over {len(prompts)} prompts",
         status(sound["coarse"] == 0 and k["coarse"] == 100 and len(prompts) == 5)),
        ("C10b G_env admits every sample under the same prompts",
         "100/100", f"{sound['genv']}/{k['genv']}",
         status(sound["genv"] == k["genv"] == 100)),
    ]


# C11: Table 8, regex extraction confounded by the ICL example.
def check_b12_icl_regex() -> Row:
    pattern = re.compile(r"My goal is to have that (.+?)(?:\.|$)", re.DOTALL)
    prompt_file = (EXTERNAL / "LLMs-Planning" / "plan-bench" / "prompts"
                   / "blocksworld_3" / "task_1_plan_generation.json")
    query = None
    if prompt_file.exists():
        d = load(prompt_file)
        if d and d.get("instances"):
            query = d["instances"][0].get("query")
    source = "PlanBench prompt" if query else "synthetic two-statement prompt"
    query = query or (
        "[STATEMENT]\nAs initial conditions I have that ...\n"
        "My goal is to have that the example goal block A on block B.\n"
        "[PLAN] ... [PLAN END]\n"
        "[STATEMENT]\nAs initial conditions I have that ...\n"
        "My goal is to have that the instance goal block C on block D.\n"
    )
    hits = [h.strip() for h in pattern.findall(query)]
    first = hits[0] if hits else ""
    print(f"  C11: {len(hits)} matches in the {source}; first = \"{first[:60]}\"")
    reproduced = len(hits) >= 2 and first != hits[-1]
    return ("C11 Table 8: regex extraction confounded by the ICL example",
            "pattern reproduced",
            f"first match is the example goal, not the instance goal ({source})"
            if reproduced else f"{len(hits)} match(es); pattern not reproduced",
            EXPECTED if reproduced else FAIL)


# C12: SCI-ReDuce class-indexed productions (App. P).
def check_b13_reduce_s() -> Row:
    claim = "C12 App. P: SCI-ReDuce emits class-indexed productions N_c -> M N_c'"
    f = REPO_ROOT / "agplan" / "grammar_learning" / "reduce_s.py"
    if not f.exists():
        return (claim, "compile_state_aware_ebnf builds them", "missing module", FAIL)
    txt = f.read_text()
    has_compile = "def compile_state_aware_ebnf" in txt
    has_rule = 'ws "," ws {_class_name(dst)}' in txt
    print(f"  C12: compile_state_aware_ebnf={has_compile}, class-indexed rule builder={has_rule}")
    return (claim, "compile_state_aware_ebnf builds them",
            "present" if has_compile and has_rule else "not found",
            status(has_compile and has_rule))


# C13: pipeline trace on a bundled Blocksworld instance with a mocked LLM.
# E1 induces schemas from random rollouts, E2 compiles G_env from them, E3 and
# E4 stand in for the LLM with the BFS plan and the instance goal, E5 stores
# and retrieves the schemas, E6 executes the plan and verifies it against the
# parsed goal.
def tier4_pipeline_trace() -> list[Row]:
    out: list[Row] = []
    t0 = time.time()

    bw = DOMAINS / "planbench" / "blocksworld"
    dom_path = bw / "domain.pddl"
    problems = sorted((bw / "problems").glob("instance-*.pddl"),
                      key=lambda p: first_int(p.stem) or 0)
    if not dom_path.exists() or not problems:
        ext = EXTERNAL / "LLMs-Planning" / "plan-bench" / "instances" / "blocksworld"
        dom_path = ext / "generated_domain.pddl"
        problems = sorted((ext / "generated_basic_3").glob("instance-*.pddl"),
                          key=lambda p: first_int(p.stem) or 0)
    if not dom_path.exists() or not problems:
        return [("C13 Blocksworld inputs", "domains/planbench/blocksworld present",
                 "missing", FAIL)]
    inst_path = problems[0]
    print(f"  C13: {inst_path.relative_to(REPO_ROOT)}")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.induce_pddl_generic import (
            load_task, random_rollout_pddl, state_to_predicates, parse_op_name,
        )
        from scripts.induce_positional import induce_lifted_models_positional
        from scripts.run_agcp_on_apb_simple import _llm_json_to_lisp, parse_lisp_plan
        from agplan.grammars.generic_grammar import compile_from_action_models
        from agplan.adaptive_pipeline import SchemaMemory
        from agplan.llm_propose import propose_verify_fallback
    except Exception as e:
        return [("C13 imports", "pyperplan and agplan importable",
                 f"{type(e).__name__}: {e}", FAIL)]

    # E1: rollouts -> induced schemas
    try:
        _, task = load_task(str(dom_path), str(inst_path))
        trs = []
        for seed in range(10):
            trs.extend(random_rollout_pddl(task, n_steps=80, seed=seed))
        trs = [t for t in trs if t[0] != t[-1]]
        models = induce_lifted_models_positional(trs)
        e1_ok = bool(models) and all("pre_pos" in m and "eff_add" in m for m in models.values())
        out.append(("C13 E1 trajectories -> induced schemas", "schemas induced",
                    f"{len(models)} schemas from {len(trs)} transitions: {sorted(models)}",
                    status(e1_ok)))
    except Exception as e:
        out.append(("C13 E1 trajectories -> induced schemas", "schemas induced",
                    f"{type(e).__name__}: {e}", FAIL))
        return out

    # E2: schemas -> G_env; the parsed goal is what a correct E4 parse yields.
    try:
        init_state = state_to_predicates(task.initial_state)
        parsed_goal = set(state_to_predicates(task.goals))
        ground_ops: dict = {}
        for op in task.operators:
            name, args = parse_op_name(op.name)
            ground_ops.setdefault(name, []).append(args)
        preds = {"on", "ontable", "clear", "handempty", "holding"}
        objects = sorted({a for atom in (init_state | parsed_goal)
                          for a in re.findall(r"[a-z][\w-]*", atom) if a not in preds})
        cfg = compile_from_action_models(models, init_state, parsed_goal, objects=objects,
                                         max_extra=4, ground_op_filter=ground_ops)
        e2_ok = bool(cfg)
        out.append(("C13 E2 induced schemas -> G_env", "grammar compiled",
                    f"{len(cfg)} chars" if e2_ok else "no grammar (BFS exhausted)",
                    status(e2_ok)))
        if not e2_ok:
            return out
    except Exception as e:
        out.append(("C13 E2 induced schemas -> G_env", "grammar compiled",
                    f"{type(e).__name__}: {e}", FAIL))
        return out

    # E3: decoding under the mask, mocked with the shortest plan found by BFS
    # over the ground operators.
    try:
        start = task.initial_state
        parent = {start: None}
        queue = deque([start])
        goal_state = None
        while queue:
            s = queue.popleft()
            if task.goals <= s:
                goal_state = s
                break
            for op in task.operators:
                if op.applicable(s):
                    ns = op.apply(s)
                    if ns not in parent:
                        parent[ns] = (s, op.name)
                        queue.append(ns)
        names = []
        s = goal_state
        while s is not None and parent[s] is not None:
            s, name = parent[s]
            names.append(name)
        names.reverse()
        mock_plan = {"plan": [f"{n.strip('()').split()[0]}({','.join(n.strip('()').split()[1:])})"
                              for n in names]}
        plan_obj, info = propose_verify_fallback(
            prompt="(mocked)", schema_json='{"type":"object"}',
            verifier=lambda prop: 1.0 if isinstance(prop, dict) and prop.get("plan") else 0.0,
            baseline={"plan": []}, threshold=0.5, use_mock=True, mock_value=mock_plan,
        )
        e3_ok = (info.get("accepted_llm") is True and not info.get("fallback_used")
                 and plan_obj.get("plan") == mock_plan["plan"] and goal_state is not None)
        out.append(("C13 E3 decoding under the mask (mocked)", "mock accepted, no fallback",
                    f"{len(mock_plan['plan'])}-step plan accepted (fallback={info.get('fallback_used')})",
                    status(e3_ok)))
    except Exception as e:
        out.append(("C13 E3 decoding under the mask (mocked)", "mock accepted, no fallback",
                    f"{type(e).__name__}: {e}", FAIL))
        return out

    # E4: NL-goal parse, mocked. A sound parse is accepted; a rejected parse
    # falls back to the supplied baseline and reports it in fallback_used.
    try:
        def goal_verifier(prop):
            goal = prop.get("goal") if isinstance(prop, dict) else None
            return 1.0 if goal is not None and set(goal) == parsed_goal else 0.0

        baseline = {"goal": sorted(parsed_goal)}
        good, info_good = propose_verify_fallback(
            prompt="(mocked)", schema_json='{"type":"object"}', verifier=goal_verifier,
            baseline=baseline, threshold=0.5, use_mock=True,
            mock_value={"goal": sorted(parsed_goal)},
        )
        bad, info_bad = propose_verify_fallback(
            prompt="(mocked)", schema_json='{"type":"object"}', verifier=goal_verifier,
            baseline=baseline, threshold=0.5, use_mock=True, mock_value={"goal": ["junk(x)"]},
        )
        accepted = info_good.get("accepted_llm") is True and set(good.get("goal", [])) == parsed_goal
        reported = info_bad.get("fallback_used") is True and set(bad.get("goal", [])) == parsed_goal
        out.append(("C13 E4 NL-goal parse (mocked)", "sound parse accepted, fallback reported",
                    f"sound parse accepted={accepted}; rejected parse falls back and reports it={reported}",
                    status(accepted and reported)))
    except Exception as e:
        out.append(("C13 E4 NL-goal parse (mocked)", "sound parse accepted, fallback reported",
                    f"{type(e).__name__}: {e}", FAIL))
        return out

    # E5: schema library store and lookup
    try:
        mem = SchemaMemory()
        for name, model in models.items():
            mem.add(name, model, provenance=inst_path.stem)
        observed = set()
        for m in models.values():
            for key in ("pre_pos", "eff_add", "eff_del"):
                for atom in m.get(key, []):
                    mm = re.match(r"^([a-zA-Z_][\w-]*)", atom)
                    if mm:
                        observed.add(mm.group(1))
        probe = next(iter(models))
        cands = mem.candidates_for(probe, observed)
        e5_ok = len(mem.schemas) == len(models) and len(cands) >= 1
        out.append(("C13 E5 schema library store and lookup", "stored and retrieved",
                    f"stored {len(mem.schemas)}, lookup for {probe} found {len(cands)}",
                    status(e5_ok)))
    except Exception as e:
        out.append(("C13 E5 schema library store and lookup", "stored and retrieved",
                    f"{type(e).__name__}: {e}", FAIL))
        return out

    # E6: execute the plan and verify against the parsed goal
    try:
        steps = parse_lisp_plan(_llm_json_to_lisp(json.dumps(plan_obj)))
        op_by_name = {op.name: op for op in task.operators}
        state = task.initial_state
        n_applied = 0
        for action, args in steps:
            op = op_by_name.get(f"({action} {' '.join(args)})")
            if op is None or not op.applicable(state):
                break
            state = op.apply(state)
            n_applied += 1
        reached = parsed_goal <= set(state_to_predicates(state))
        e6_ok = n_applied == len(steps) and reached
        out.append(("C13 E6 execute and verify against the parsed goal", "plan applies, goal reached",
                    f"applied {n_applied}/{len(steps)} actions; parsed goal reached={reached}",
                    status(e6_ok)))
    except Exception as e:
        out.append(("C13 E6 execute and verify against the parsed goal", "plan applies, goal reached",
                    f"{type(e).__name__}: {e}", FAIL))
        return out

    statuses = [r[3] for r in out]
    chain = " -> ".join(f"E{i + 1} {s}" for i, s in enumerate(statuses))
    out.append(("C13 pipeline trace", "all six edges OK",
                f"{chain} ({time.time() - t0:.1f}s)", status(all(s == OK for s in statuses))))
    return out


SECTIONS = [
    ("C1 Table 1 Depots N=30", check_c1_depots),
    ("C2 schema library reuse", check_c2_reuse),
    ("C3 Table 1 AGCP column", check_c3_table1_agcp),
    ("C4 runner and scorer accounting", check_c4_accounting),
    ("C5 Table 7 PlanBench NL parse", check_c5_table7),
    ("C6 predicate discovery", check_c6_predicates),
    ("C7 closed loop on Mystery 2-11", check_c7_closed_loop),
    ("C8 IPC-2023 component F1", check_c8_ipc),
    ("C9 Depots schema replay", check_c9_depots_replay),
    ("C10 jailbreak prompts", check_c10_adversarial),
    ("C11 ICL-example regex confound", check_b12_icl_regex),
    ("C12 SCI-ReDuce productions", check_b13_reduce_s),
    ("C13 pipeline trace", tier4_pipeline_trace),
]


def main() -> int:
    print("Headline-claim check")
    print(f"repository: {REPO_ROOT}")
    rows: list[Row] = []
    for title, fn in SECTIONS:
        print(f"\n--- {title} ---")
        result = fn()
        rows.extend(result if isinstance(result, list) else [result])

    header = ("CLAIM", "STATED", "ACTUAL", "STATUS")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(4)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*header))
    print("-" * (sum(widths) + 9))
    for r in rows:
        print(fmt.format(*r))

    counts = Counter(r[3] for r in rows)
    print(f"\nsummary: OK={counts[OK]} EXPECTED={counts[EXPECTED]} FAIL={counts[FAIL]} "
          f"of {len(rows)} checks")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
