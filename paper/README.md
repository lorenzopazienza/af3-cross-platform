# Paper assets

This folder holds the tables and figures of the paper "Same Seed, Different Top Prediction: Cross-Platform
Reproducibility of AlphaFold 3 Inference on TPUs, GPUs, and CPUs", the scripts that copy and relabel them, and the
list of the numbers the paper prints with their sources. The text of the paper and of its supplement is on arXiv:
[arXiv ID]. It is not in this repository.

## Contents

| Path | What |
|---|---|
| `assets/` | every table and figure the paper and the supplement include, as copied by `build_assets.sh`; never edited by hand. `assets/SHA256SUMS` holds the SHA-256 of every copied asset; each source folder's `sources.tsv` lists the inputs of its generator with their SHA-256; `assets/README.md` and `assets/isc/README.md` give the role of each asset |
| `NUMBERS.md` | every number in the text of the paper and the supplement, with the source its author cited and a literal or rounded check against that source (see below) |
| `build_assets.sh` | copies the generated assets from `results/` into `assets/`, after hashing again every input listed in their `sources.tsv`, and runs `derive_tables.py`, `center_guides.py` and `relabel_tables.py` |
| `tools/derive_tables.py` | main-text tables: row and column subsets of the generated CSVs, cells copied verbatim (`assets/derived/`) |
| `tools/relabel_tables.py` | copies of the generated tables with a fixed list of label substitutions (the paper's short platform names), numbers unchanged, each substitution listed in the file header (`assets/relabeled/`) |
| `tools/relabel_figures.py` | the same relabeling for the supplement figures, by rerunning their generators (`assets/relabeled_figures/`) |
| `tools/center_guides.py` | the table of the AlphaFold 3 user guides of 28 computing centers and the counts the paper cites, from `research/center_guides.csv` |

The generators are in `analysis/`: `paper_assets.py` (main study and follow-ups 1 to 5), `paper_assets_fu678.py`,
`paper_assets_fu9.py`, `paper_assets_fu10_11_12.py` and `paper_assets_isc.py` (the paper's own figures, tables,
captions and campaign counts). Their tests: `analysis/tests/test_paper_assets_isc.py`.

## Regenerating and checking the assets

The generators read the analysis outputs under `results/`, which are not in git: they are in the Zenodo records cited
in the paper's Data and Code Availability section (each record's README gives its layout). With `results/` in place
and the figure environment installed (`analysis/requirements-figures.txt`, in `analysis/figures/.venv`), run from the
repository root:

```bash
bash scripts/analysis_runs/paper_assets/reproduce_followups34.sh           # results/paper_assets_followups34
analysis/figures/.venv/bin/python analysis/paper_assets_fu678.py           # results/paper_assets_fu678
analysis/figures/.venv/bin/python analysis/paper_assets_fu9.py             # results/paper_assets_fu9
analysis/figures/.venv/bin/python analysis/paper_assets_fu10_11_12.py      # results/paper_assets_fu10_11_12
analysis/figures/.venv/bin/python analysis/paper_assets_isc.py --reserve   # results/paper_assets_isc
bash paper/build_assets.sh
analysis/figures/.venv/bin/python paper/tools/relabel_figures.py --work <empty scratch folder> --check
analysis/figures/.venv/bin/python -m pytest analysis/tests/test_paper_assets_isc.py
```

What shows that nothing moved:

- `build_assets.sh` stops if an input differs from its `sources.tsv`, and prints
  "0 new or changed since the last copy (SHA256SUMS)" when every copied asset has the SHA-256 recorded in
  `assets/SHA256SUMS`; any other asset is listed by name;
- `relabel_figures.py --check` reruns the figure generators without the relabeling and compares each PDF with the
  copied one byte for byte (without `--check` it rewrites `assets/relabeled_figures/`);
- `git status` shows no change under `paper/assets/`.

Run `paper_assets_isc.py` with `--reserve`: it also writes the reserve cost and size figures, and without it
`captions.tex` lacks their captions. The figures embed Arial, read from `/System/Library/Fonts/Supplemental/Arial.ttf`
(macOS); with another copy of the font the PDFs differ in bytes, and `sources.tsv` records the font file's SHA-256.

The checkout that holds `results/` defaults to the repository's main working tree (the parent of
`git rev-parse --git-common-dir`): the repository root in a clone, the main checkout from a linked worktree. Override
it with `--main` (generators and `relabel_figures.py`), `MAIN_CHECKOUT` (`build_assets.sh`) or `AF3_MAIN` (the tests).

## NUMBERS.md

`NUMBERS.md` was written from the source comments of the paper's LaTeX files, which are not in this repository: the
"where" column (`sections/<file>.tex:<lines>`) points into them, and the numbers can be found in the arXiv text. Most
sources are analysis outputs under `results/`, repository files (`notes/`, `analysis/`, `scripts/`, ...) or pages
read on a stated date. Some sources are the author's working notes, which are not public; the rows that cite them were
checked against those notes when `NUMBERS.md` was written, and most of them also name an analysis output or record.

## Terms of use of the per-target tables

The per-target tables in `assets/` derive from AlphaFold 3 outputs (structures, confidences and ranking scores), so
they fall under the AlphaFold 3 Output Terms of Use:
https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.
