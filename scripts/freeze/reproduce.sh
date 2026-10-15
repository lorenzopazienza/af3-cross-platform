#!/bin/bash
# Data freeze manifest of the pilot2, main-study, memory-limit and follow-up sessions (read-only bucket listing). Run from the repository root:
#   bash scripts/freeze/reproduce.sh
set -eu
analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://af2-tpu-benchmark-af3-results \
  --out results/freeze 2>&1 | tee results/freeze/run.log
