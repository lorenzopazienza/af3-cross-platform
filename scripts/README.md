# scripts/

Scripts and tests that the study's results depend on, kept here under version control; their outputs live under
the gitignored `results/`. Moved out of `results/` on 2026-10-15; `MOVED.tsv` lists every old and new path. Run
everything from the repository root.

- `freeze/`: the data freeze (`make_freeze_manifest.py`, `reproduce*.sh`) and its index (`freeze_index.py`); output `results/freeze/`.
- `release/`: the Zenodo release (`release.py`, its tests, `RUNBOOK.md`, the record README drafts in `records/`);
  outputs `results/release/`.
- `monitor/`: checks run while launches were in progress (main study 2026-10-13, follow-ups 2026-10-15); outputs
  `results/analysis/main_monitor_20261013/`, `results/analysis/followups_20261015/`.
- `analysis_runs/`: one folder per analysis run, each with the `reproduce.sh` that regenerates its outputs under
  `results/analysis/<same name>/` (and `paper_assets/` for `results/paper_assets/`);
  `check_byte_identity.sh` reruns them and checks that their outputs come back byte for byte.
