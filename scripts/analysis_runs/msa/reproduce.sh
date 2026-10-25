#!/bin/bash
# Follow-up 6, AlphaFold3's own data pipeline (notes/analysis_plan.md section 10, row of 2026-10-19; label msa).
# Descriptive part (MSA depth and Neff, template audit, pipeline timings, determinism), after
# scripts/nas/ingest_official.py has frozen the three deliveries; FETCH_RCSB=1 fills the RCSB cache of the template
# audit (read-only queries to data.rcsb.org), later runs read the cache:
#   DATE=YYYYMMDD DESCRIPTIVE=1 [FETCH_RCSB=1] bash scripts/analysis_runs/msa/reproduce.sh
# Everything, once every arm has finished or been reported incomplete; scripts/analysis_runs/msa/sessions.txt
# names the msa sessions the section 10 rows list:
#   DATE=YYYYMMDD [FETCH_RCSB=1] bash scripts/analysis_runs/msa/reproduce.sh
set -eu
O=results/analysis/msa_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is msa_DATE)}
mkdir -p "$O"
RCSB=""
[ -z "${FETCH_RCSB:-}" ] || RCSB="--fetch_rcsb"
{
if [ -n "${DESCRIPTIVE:-}" ]; then
  analysis/.venv/bin/python analysis/msa.py --descriptive_only --out "$O" $RCSB
else
  S=scripts/analysis_runs/msa/sessions.txt
  [ -f "$S" ] || { echo "!! $S does not exist: list the msa sessions of the section 10 rows there first" >&2; exit 2; }
  analysis/.venv/bin/python analysis/collect.py --labels msa --sessions "$S" --out "$O" --jobs 8
  analysis/.venv/bin/python analysis/msa.py --samples "$O/samples.csv" --out "$O" $RCSB \
    --main results/analysis/main_20261014/primary_v2 --replicates 10000 --seed 2026
fi
} 2>&1 | tee "$O/run.log"
