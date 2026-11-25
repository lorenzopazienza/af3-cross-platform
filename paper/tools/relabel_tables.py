"""Relabeled copies of the generated tables for the paper (paper/assets/relabeled/), from paper/assets/tables/.

    python3 paper/tools/relabel_tables.py paper/assets/tables paper/assets/relabeled

The frozen tables are never edited. This script applies a fixed list of label substitutions to the printed cells
(comment lines are copied unchanged), so the paper says "plan, §11" where the analysis code wrote "section 11",
and the M1 failure of the three glycan targets points to its sensitivity table; in the follow-up tables of
Follow-ups 6 and 9, "primary pair" names the follow-up's own headline pair, not a pre-registered primary pair.

Since 2026-11-11 it also applies the naming decisions D1 and D2 of the paper (reader-facing names instead of codes) and
reading notes R20 and R21:
- D1: the arm codes (V5, V6, G, C, A, H and their variants) become the short platform names (v5e, v6e, L4, CPU,
  A100, H100, L4 Triton, L4 deterministic, v6e highest, ...), and the contrast notations HW(), RC(), SD(), DU, DET,
  PR(), HWh(), W(), IN(), SW() become short names ("v5e vs v6e", "Recompile, L4", "Seeds, v6e", "Default attention,
  L4", "Determinism flag, L4", "Precision, v6e", "Warm rerun, v5e official", "Inputs, L4", "Software stack, v6e").
  A code at the start of a cell takes the capitalised form ("Recompile, L4"); a code inside a longer cell takes the
  lower-case form ("v6e vs L4 minus recompile L4"). In the reference column of the HW minus RC tables, RC(P) and
  RC(Q) (the recompilation of the pair's first and second platform) become "first platform" and "second platform".
- D2: "(target, seed)" as the unit becomes "predictions".
- R20 and R21: the warm-rerun check and the masked mmCIF line are described in words (the R21 cell in its short
  form, "all outputs except the line recording when each file was written", approved 2026-11-11).
- R35: in T23_feat_hosts, the arms on the Cascade Lake host read "L4 and A100 arms of the main study and Follow-up 3".
Every substitution made is listed in the header of the output file, and a rule that matches nothing in a table
where it is expected stops the script.

Since 2026-11-21 (style round of 2026-11-20: no table of the supplement prints below 8 pt), the tables of LAYOUT are
laid out for the supplement's 8 pt by tools/table_layout.py: header cells on several lines, paragraph columns, an
interval under its estimate, a spanning header for a repeated qualifier, or a longtable that breaks across pages. The
cells are unchanged; table_layout.py reads them back from the new text and stops if any differs.

The numbers are never touched. The rules of RULES may only replace text that contains no digit other than those of
section numbers and dates. The D1 names contain digits (v5e, L4, A100), so the D1, D2, R20 and R21 substitutions are
checked instead: in every line, the sequence of numbers before and after them, with the platform names and codes
removed on both sides, must be the same, or the script stops.
"""
import pathlib
import re
import sys

import importlib.util

# table_layout.py is loaded by path, not through sys.path: paper/tools/ also holds numbers.py, which on sys.path hides the
# standard library module numbers (numpy needs it) in the processes that relabel_figures.py starts
_spec = importlib.util.spec_from_file_location('table_layout', pathlib.Path(__file__).resolve().parent / 'table_layout.py')
_table_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_table_layout)
layout = _table_layout.layout

# (table or '*', exact old text, new text)
RULES = [
    ('*', 'matched hardware (section 11)', 'matched hardware (plan, \\S11)'),
    ('*', 'precision (section 11)', 'precision (plan, \\S11)'),
    ('*', 'deterministic ops (2026-10-11)', 'deterministic ops (plan, \\S10, 2026-10-11)'),
    ('T03_checks', 'FAIL, targets: 7OZC, 7WDT, 7WJB', 'FAIL: 3 glycan targets, found after the primary results (Table~\\ref{tab:glycans})'),
    ('T25a_msa_contrasts', 'primary pair', 'reading-rule pair'),
    ('T28a_h100_contrasts', 'primary pair', 'headline of the follow-up'),
]
EXPECTED = {'T03_checks': ['FAIL, targets: 7OZC, 7WDT, 7WJB'], 'T04_flip_rates': ['precision (section 11)'],
            'T05_hw_minus_rc': ['matched hardware (section 11)'], 'T25a_msa_contrasts': ['primary pair'],
            'T28a_h100_contrasts': ['primary pair'], 'T23_feat_hosts': ['G, GT, Gd, Gh, A, Ah, AT']}

# ---- D1: arm codes -> short platform names (style guide rule 15) ----
ARMS = {
    'V5': 'v5e', 'V6': 'v6e', 'G': 'L4', 'C': 'CPU', 'A': 'A100', 'H': 'H100',
    'GT': 'L4 Triton', 'Gd': 'L4 deterministic', 'V6h': 'v6e highest', 'Gh': 'L4 highest', 'Ah': 'A100 highest',
    'AT': 'A100 Triton', 'V6f': 'v6e float32', 'Gf': 'L4 float32', 'V5o': 'v5e official', 'V6o': 'v6e official',
    'Go': 'L4 official', 'V6s': 'v6e older stack', 'Gs': 'L4 older stack',
}
ARM = '|'.join(sorted(ARMS, key=len, reverse=True))
# Tables whose cells hold no arm codes but letters or atom names that look like them (T17: atom names such as
# C1(C), P1(P)); D1 leaves them unchanged.
SKIP_D1 = {'T17_glycan_audit'}
# Text that must keep its letters: follow-up 10's variant A, the C-alpha atoms, the reading codes (P, D, R, N).
PROTECT = ['variant A', 'C-alpha']
# (capitalised form at the start of a cell, lower-case form inside a cell); {x} is the platform name
FORMS = {'RC': ('Recompile, {x}', 'recompile {x}'), 'SD': ('Seeds, {x}', 'seeds {x}'),
         'PR': ('Precision, {x}', 'precision {x}'), 'W': ('Warm rerun, {x}', 'warm rerun {x}'),
         'IN': ('Inputs, {x}', 'inputs {x}'), 'SW': ('Software stack, {x}', 'software stack {x}'),
         'DU': ('Default attention, {x}', 'default attention {x}')}
# D2, R20, R21: exact cell text (table or '*', old, new)
TEXT = [
    ('*', '2b warm rerun loaded its source executable',
     '2b warm rerun loaded the model compiled by the first run'),
    ('*', 'all outputs (mmCIF timestamp line ignored)',
     'all outputs except the line recording when each file was written'),
    ('*', '(target, seed) units with 5 samples', 'predictions with 5 samples'),
    ('*', '(target, seed) units', 'predictions'),
    ('*', '(target, seed) with a difference', 'predictions with a difference'),
    ('*', ' (target, seed) & ', ' predictions & '),
    ('*', ' (target, seed) \\\\', ' predictions \\\\'),
    ('T33c_large_by_group', 'HW targets & RC targets', 'pair targets & recompile targets'),
    ('T19d_ppi_per_target', 'RC(V6) flips', 'recompile v6e flips'),
    ('T19d_ppi_per_target', 'RC(G) flips', 'recompile L4 flips'),
    # R35 (2026-11-11): the host table names the arms on the Cascade Lake host as a group, not one by one
    ('T23_feat_hosts', ' & G, GT, Gd, Gh, A, Ah, AT & ', ' & L4 and A100 arms of the main study and Follow-up 3 & '),
]
# Supplement layout at 8 pt (keyword arguments of table_layout.layout). wrap: printed characters per header line;
# spec: column specification (P{w}: ragged-right paragraph column); stack: an interval under its estimate; group: one
# spanning header line per region over the price columns of Follow-up 3 ("USD, spot, us-east5" read back from
# "USD, us-east5" above "spot"); longtable: breaks across pages.
T22D_GROUP = (lambda c: re.sub(r'^USD, [a-z-]+, ', 'USD, ', c) if c.startswith('USD, ') else None,
              lambda c: c.split(', ')[1],
              lambda g, c: g.replace('USD, ', f'USD, {c}, ', 1))
LAYOUT = {
    'T01_platforms': dict(wrap=12),
    'T03_checks': dict(spec='P{146pt}lP{110pt}P{110pt}', longtable=True),
    'T04_flip_rates': dict(wrap=12, spec='lP{80pt}rllll'),
    'T05_hw_minus_rc': dict(wrap=12, spec='lP{62pt}P{62pt}P{62pt}lll'),
    'T06_m4_m5': dict(wrap=14, spec='P{62pt}P{64pt}rlllllll', stack={j: 'bracket' for j in range(3, 10)},
                      longtable=True),
    'T08_glycans': dict(wrap=12, spec='P{44pt}lP{110pt}P{60pt}P{60pt}P{60pt}'),
    'T12_cutoff_sensitivity': dict(wrap=12, spec='lllP{100pt}ll', longtable=True),
    'T13_exploratory_margin': dict(wrap=12),
    'T15_exploratory_perturbation': dict(wrap=12),
    'T18_memlimit': dict(wrap=12, spec='lllP{100pt}llll'),
    'T18b_memlimit_processes': dict(wrap=10),
    'T19b_ppi_contrasts': dict(wrap=12, spec='llP{150pt}ll'),
    'T19c_ppi_hw_minus_rc': dict(wrap=10, spec='lllP{60pt}llll'),
    'T19d_ppi_per_target': dict(wrap=9),
    'T19f_ppi_exploratory_margin': dict(wrap=10),
    'T19g_ppi_exploratory_perturbation': dict(wrap=12),
    'T20a_embed_pairs': dict(wrap=12, spec='P{60pt}rllllllll', stack={j: 'bracket' for j in range(2, 7)}),
    'T20b_embed_per_target': dict(wrap=12, spec='P{55pt}lrlllllll'),
    'T21a_cpu60_contrasts': dict(wrap=12),
    'T22a_a100_checks': dict(spec='P{200pt}lll'),
    'T22b_a100_contrasts': dict(wrap=12, spec='lP{80pt}rllll'),
    'T22d_a100_cost': dict(wrap=12, group=T22D_GROUP),
    'T23_feat_hosts': dict(wrap=10, spec='lP{60pt}lP{50pt}llP{40pt}llll'),
    'T25a_msa_contrasts': dict(wrap=12),
    'T31b_repl_hw_minus_rc': dict(wrap=12),
    'T32b_prec5_hw_minus_rc': dict(wrap=12, spec='llP{110pt}llll'),
    'T33b_large_differences': dict(wrap=12, spec='lP{90pt}lllll'),
    'T33c_large_by_group': dict(wrap=12),
}

STRIP = re.compile(r'AlphaFold 3|v5e|v6e|L4|A100|H100|float32|\\S\d+|section \d+|Follow-up \d+|\b(?:' + ARM + r')\b')


def numbers(text):
  return re.findall(r'\d+(?:\.\d+)?', STRIP.sub(' ', text))


def arm(code):
  return ARMS[code]


def contrast(m, start):
  """One contrast code -> its short name; start: the code opens the cell."""
  kind, args = m.group('kind'), m.group('args')
  if kind in ('HW', 'HWh'):
    a, b = args.split(',')
    if kind == 'HWh':
      return ' vs '.join(arm(x) + ('' if x == 'C' else ' highest') for x in (a, b))
    return f'{arm(a)} vs {arm(b)}'
  if kind == 'RC' and args in ('P', 'Q'):
    return ('first platform' if args == 'P' else 'second platform')
  cap, low = FORMS[kind]
  return (cap if start else low).format(x=arm(args))


CONTRAST = re.compile(r'\b(?P<kind>HWh|HW|RC|SD|PR|W|IN|SW|DU)\((?P<args>(?:' + ARM + r'|P|Q)(?:,(?:' + ARM + r'))?)\)')


def relabel_cell(cell):
  lead = len(cell) - len(cell.lstrip())
  body = cell.strip()
  if not body:
    return cell
  keep = {}
  for i, p in enumerate(PROTECT):
    if p in body:
      keep[f'@@{i}@@'] = p
      body = body.replace(p, f'@@{i}@@')
  # contrasts written as codes
  body = CONTRAST.sub(lambda m: contrast(m, m.start() == 0), body)
  # DU and DET without a platform are the L4's
  body = re.sub(r'^DU\b', 'Default attention, L4', body)
  body = re.sub(r'^DET\b', 'Determinism flag, L4', body)
  body = re.sub(r'\bDU\b', 'default attention L4', body)
  body = re.sub(r'\bDET\b', 'determinism flag L4', body)
  # differences between two arms written as "V5-G": spaced minus between the names
  body = re.sub(r'\b(' + ARM + r')-(' + ARM + r')\b', lambda m: f'{arm(m.group(1))} - {arm(m.group(2))}', body)
  # D2 in figure labels
  body = body.replace('(target, seed) bit-identical', 'predictions bit-identical')
  # figures write "HW(V5,G) - RC(V5)": the same "minus" as the tables
  body = body.replace(' - recompile ', ' minus recompile ')
  # remaining arm codes
  body = re.sub(r'\b(' + ARM + r')\b', lambda m: arm(m.group(1)), body)
  for k, p in keep.items():
    body = body.replace(k, p)
  return cell[:lead] + body + (' ' if cell.endswith(' ') else '')


def relabel_line(line):
  """D1 on every cell of a tabular row; the row end (\\\\) is kept."""
  m = re.match(r'^(.*?)(\s*\\\\.*)?$', line)
  row, end = m.group(1), m.group(2) or ''
  return ' & '.join(relabel_cell(c) for c in row.split(' & ')) + end


def main():
  src, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
  dest.mkdir(parents=True, exist_ok=True)
  for f in sorted(src.glob('*.tex')):
    lines, done = f.read_text().splitlines(), []
    d1 = {}
    for i, line in enumerate(lines):
      if line.startswith('%'):
        continue
      for table, old, new in RULES:
        if table in ('*', f.stem) and old in line:
          n = line.count(old)
          line = line.replace(old, new)
          done.append(f'{n} x "{old}" -> "{new}"')
      before = line
      for table, old, new in TEXT:
        if table in ('*', f.stem) and old in line:
          n = line.count(old)
          line = line.replace(old, new)
          done.append(f'{n} x "{old.strip()}" -> "{new.strip()}"')
      if f.stem not in SKIP_D1 and '&' in line:
        new = relabel_line(line)
        if new != line:
          for o, n in zip(line.split(' & '), new.split(' & ')):
            if o.strip() != n.strip():
              key = (re.sub(r'\s*\\\\.*$', '', o.strip()), re.sub(r'\s*\\\\.*$', '', n.strip()))
              d1[key] = d1.get(key, 0) + 1
          line = new
      if numbers(before) != numbers(line):
        raise SystemExit(f'{f.name}: numbers changed by D1 in line {i + 1}:\n  {before}\n  {line}')
      lines[i] = line
    for old in EXPECTED.get(f.stem, []):
      if not any(old in d for d in done):
        raise SystemExit(f'{f.name}: expected substitution of "{old}" did not apply')
    done += [f'{n} x "{o}" -> "{w}" (D1)' for (o, w), n in d1.items()]
    head = [f'% Relabeled by paper/tools/relabel_tables.py from assets/tables/{f.name}; numbers unchanged.']
    head += [f'% substitution: {d}' for d in done] or ['% substitution: none']
    text = '\n'.join(head + lines) + '\n'
    if f.stem in LAYOUT:
      opts = LAYOUT[f.stem]
      text = layout(text, name=f.stem, **opts)
      text = text.replace('\n', f'\n% layout for the supplement at 8 pt (tools/table_layout.py, cells unchanged): '
                          f'{", ".join(sorted(opts))}\n', 1)
    (dest / f.name).write_text(text)
  print(f'   {len(list(src.glob("*.tex")))} tables relabeled into {dest.name}/')


if __name__ == '__main__':
  main()
