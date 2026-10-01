#!/usr/bin/env bash
# Clone the third-party benchmark repositories that are not on PyPI into
# external/. The repository ships domain files and sample instances under
# domains/, so this is only needed for the full benchmark sets and for the
# AutoPlanBench experiments. Idempotent.
#
# Usage: bash scripts/setup_external.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="${REPO_ROOT}/external"
mkdir -p "${EXT}"

clone() {  # clone <url> <dir>
  if [[ -d "${EXT}/$2" ]]; then
    echo "[setup_external] $2 already present"
  else
    git clone --depth 1 "$1" "${EXT}/$2"
  fi
}

# PlanBench instances and generators (Valmeekam et al.)
clone https://github.com/karthikv792/LLMs-Planning.git LLMs-Planning
# IPC 2023 learning-track benchmarks
clone https://github.com/ipc2023-learning/benchmarks.git ipc2023-learning
# AutoPlanBench (Stein et al.); scripts/run_agcp_on_autoplanbench.py imports
# its llm_planning package from this directory
clone https://github.com/minecraft-saar/autoplanbench.git autoplanbench

# ALFWorld games are installed separately:
#   pip install alfworld && alfworld-download

echo "[setup_external] done"
