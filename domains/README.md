# Benchmark domains

Domain files and a representative sample of problem instances for each
benchmark used in the paper, so that the smoke tests and the scripts run
without cloning the upstream repositories. The full instance sets are in
the upstream repositories listed below.

## Layout

```
domains/
├── planbench/                  # Valmeekam et al., PlanBench (LLMs-Planning)
│   ├── blocksworld/            # domain.pddl + problems/instance-{1..10}.pddl
│   ├── logistics/
│   ├── depots/
│   └── mystery_blocksworld/    # obfuscated Blocksworld
└── ipc2023_learning/           # IPC 2023 learning track, 10 domains
    └── <domain>/               # domain.pddl + training/easy, testing/{easy,medium,hard}
                                #   with 5 problems per sub-track
```

## Sources

| Local path            | Upstream                                              |
|-----------------------|-------------------------------------------------------|
| `planbench/*`         | https://github.com/karthikv792/LLMs-Planning (`plan-bench/instances`) |
| `ipc2023_learning/*`  | https://github.com/ipc2023-learning/benchmarks        |

Some `domain.pddl` files carry a `;; source:` comment with the exact upstream
path. To obtain the full benchmarks:

```bash
git clone https://github.com/karthikv792/LLMs-Planning.git external/LLMs-Planning
git clone https://github.com/ipc2023-learning/benchmarks.git external/ipc2023-learning
```

ALFWorld is not included here; install the `alfworld` package and run
`alfworld-download` as described in the top-level README.

## Negative preconditions

The childsnack, ferry and satellite domains use `:negative-preconditions`,
which pyperplan does not accept. `scripts/run_ipc2023_breadth.py` regenerates
STRIPS-only variants at run time through
`scripts/strip_negative_preconditions.py`; no pre-converted copies are stored.
