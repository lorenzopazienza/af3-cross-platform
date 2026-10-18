"""The frozen-input manifest of follow-up 5 (notes/analysis_plan.md section 10, row of 2026-10-18; label feat): the
60 PoseBusters targets of inputs/manifest.csv and the 30 follow-up 1 targets of inputs/ppi/manifest.csv in one file,
so one plan (one part per host) covers all 90 with the launcher's single plan manifest.

    python3 inputs/feat/build_feat_manifest.py [--check]

Each row is the source row's columns common to both manifests, unchanged (input path and SHA-256 included), plus
source_manifest. No input file is copied or changed. --check exits 1 if inputs/feat/manifest.csv differs from what
this script writes. Standard library only.
"""
import argparse
import csv
import io
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCES = ['inputs/manifest.csv', 'inputs/ppi/manifest.csv']
OUT = REPO / 'inputs' / 'feat' / 'manifest.csv'


def build():
  tables = [list(csv.DictReader(open(REPO / s))) for s in SOURCES]
  first = list(tables[0][0])
  common = [c for c in first if all(c in t[0] for t in tables)]
  ids = [r['pdb_id'] for t in tables for r in t]
  if len(ids) != len(set(ids)):
    raise SystemExit('a pdb_id is in both manifests')
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=common + ['source_manifest'], lineterminator='\n')
  w.writeheader()
  for s, t in zip(SOURCES, tables):
    for r in t:
      w.writerow({**{c: r[c] for c in common}, 'source_manifest': s})
  return buf.getvalue()


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  text = build()
  if args.check:
    same = OUT.is_file() and OUT.read_text() == text
    print(f'{OUT.relative_to(REPO)}: {"up to date" if same else "DIFFERS from its sources"}')
    return 0 if same else 1
  OUT.write_text(text)
  print(f'wrote {OUT.relative_to(REPO)}: {text.count(chr(10)) - 1} rows')
  return 0


if __name__ == '__main__':
  sys.exit(main())
