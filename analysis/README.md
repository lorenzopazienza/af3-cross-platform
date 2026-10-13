# Analysis

Implements `notes/analysis_plan.md` (pre-specified). Reads only local files; creates no cloud resources.

```bash
uv venv --python 3.12 analysis/.venv
uv pip install --python analysis/.venv/bin/python -r analysis/requirements.txt
source analysis/.venv/bin/activate

python3 analysis/collect.py --labels pilot2 --out results/analysis/pilot2   # tidy table + M1 (cached)
python3 analysis/contrasts.py --samples results/analysis/pilot2/samples.csv --labels pilot2
python3 analysis/cost.py --samples results/analysis/pilot2/samples.csv --labels pilot2
python3 -m pytest analysis -q
```

| Script | Plan | Output (under `results/analysis/<name>/`) |
|---|---|---|
| `score_ligand.py` | M1 | one prediction's ligand RMSD, PoseBusters checks, success; `--self_test` scores the crystal against itself |
| `collect.py` | sections 4, 8, 11 | `samples.csv`: one row per (platform, attention, libtpu, target, seed, repetition, sample); runs with `JAX_DEFAULT_MATMUL_PRECISION=highest` are the section 11 conditions V6h and Gh; config `l4_xla_det` is Gd (section 10, 2026-10-11); `--labels` selects sessions; a part with more than one fully exit-0 session keeps the first completed and lists the others in `excluded_sessions.csv` (section 10, 2026-10-13); `ranking_score` is AlphaFold3's full-precision value from `<target>_ranking_scores.csv`, the rounded summary value is `ranking_score_summary` (section 10, 2026-10-12) |
| `contrasts.py` | M2 to M5, sections 5, 6, 8, 11 | `contrasts/`: flip rates, M4/M5 medians and p90, bootstrap intervals, HW minus RC, Q3, pilot checks (check 2 as 2a, 2b, then bit-identity of all outputs except the mmCIF timestamp line), the section 11 contrasts PR and HWh, the 2026-10-11 contrasts RC(Gd), DET, HW(V6,Gd) and HW(Gd,C), the M2 cross-check against AlphaFold3's top-level model, a `role` column (only section 5 primary hardware pairs are "primary pair"), `report.md` |
| `cutoff.py` | section 10, 2026-10-13 | release-date table of the 60 targets (`release_dates.csv`); the training-cutoff sensitivity analysis on the 39 targets released after 2021-09-30, written by `contrasts.py` after its other outputs (`sensitivity_post_cutoff.csv`, last section of `report.md`) |
| `cost.py` | M6, M7 | `cost/`: cost per structure and per sample at each price, amortised compilation, memory per bucket, `report.md` |

M1, as implemented: the pocket is every crystal residue (data/rcsb/) with a heavy atom within 10 Å of a
heavy atom of the reference ligand (the PoseBusters SDF); the prediction is superposed on the crystal by
the pocket C-alpha atoms (Kabsch), residues matched by `label_seq_id` (the AlphaFold3 input is the RCSB
entity sequence; identity checked at 95% or more); the ligand RMSD is RDKit's symmetry-corrected
heavy-atom RMSD (`rdMolAlign.CalcRMS`, no realignment); PoseBusters runs with its `dock` configuration
(physical and chemical checks, no RMSD test) on the superposed ligand with the superposed predicted
protein (`--pb_protein crystal` uses the crystal protein instead). Success: RMSD below 2 Å and every
check passed. A prediction that cannot be read or matched is not a success; the reason is in `error`.

Everything produced from official-weight outputs stays under `results/` (gitignored). A published copy of
AlphaFold3 outputs keeps AlphaFold3's `TERMS_OF_USE.md`; the outputs are never used to train a model.
