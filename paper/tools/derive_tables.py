"""Main-text tables of the paper: row and column subsets of the frozen paper-asset CSVs (paper/assets/tables/),
written as booktabs tabulars into paper/assets/derived/. Every cell is copied verbatim from its CSV; only the
selection, the order and the header text are chosen here. The full tables are in the appendix.

    python3 paper/tools/derive_tables.py paper/assets/tables paper/assets/derived

Each output file starts with comments naming the source CSV of every row, so paper/NUMBERS.md can point at a
(file, row, column) cell. Standard library only.
"""
import csv
import pathlib
import sys

# output: list of blocks; a block is (source csv, key column, rows in order, [(source column, header)], label column)
# label column: header of the first column (the key value is printed there, or a fixed label given per row).
SPECS = {
    'Tmain_arms': {
        'header': ['arm', 'device', 'machine type', 'config', 'attention', 'matmul', 'processes', 'USD/h'],
        'blocks': [('T01_platforms.csv', 'arm', ['V5', 'V6', 'G', 'GT', 'Gd', 'V6h', 'Gh', 'C'],
                    ['device', 'machine type', 'config', 'attention', 'matmul precision', 'main processes',
                     'on-demand USD/h'])],
    },
    'Tmain_primary': {
        'header': ['contrast', 'targets', 'top-1 flip (\\%)', 'success flip (\\%)', 'flips that matter (\\%)'],
        'blocks': [('T04_flip_rates.csv', 'contrast',
                    ['HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)', 'RC(V5)', 'RC(V6)', 'RC(G)', 'SD(V5)', 'SD(V6)', 'SD(G)'],
                    ['targets', 'top-1 flip (%)', 'success flip (%)', 'flips that matter (%)'])],
    },
    'Tmain_hw_minus_rc': {
        'header': ['hardware pair', 'minus', 'top-1 flip (pp)', 'success flip (pp)', 'flips that matter (pp)'],
        'blocks': [('T05_hw_minus_rc.csv', ('hardware pair', 'recompilation'),
                    [('HW(V5,V6)', 'RC(V5)'), ('HW(V5,V6)', 'RC(V6)'), ('HW(V6,G)', 'RC(V6)'), ('HW(V6,G)', 'RC(G)'),
                     ('HW(G,C)', 'RC(G)')],
                    ['recompilation', 'flip top1 (pp)', 'flip success (pp)', 'flip matters (pp)'])],
    },
    'Tmain_mechanisms': {
        'header': ['contrast', 'targets', 'top-1 flip (\\%)', 'success flip (\\%)'],
        'blocks': [('T04_flip_rates.csv', 'contrast',
                    ['RC(Gd)', 'DET', 'HW(V6,Gd)', 'DU', 'HW(V6,GT)', 'PR(V6)', 'PR(G)', 'HWh(V6,G)'],
                    ['targets', 'top-1 flip (%)', 'success flip (%)'])],
    },
    'Tmain_followups': {
        'header': ['follow-up', 'contrast', 'targets', 'top-1 flip (\\%)', 'success flip (\\%)'],
        'blocks': [
            ('T19b_ppi_contrasts.csv', ('contrast', 'statistic'),
             [(c, s) for c in ('HW(V5,V6)', 'HW(V6,G)', 'RC(V6)', 'RC(G)')
              for s in ('top-1 flip per seed (%)', 'success flip per seed, DockQ 0.23 (%)')],
             'ppi'),
            ('T22b_a100_contrasts.csv', 'contrast', ['RC(A)', 'HW(G,A)', 'HW(V6,A)', 'HW(V5,A)'],
             ['targets', 'top-1 flip (%)', 'success flip (%)'], 'A100'),
            ('T21a_cpu60_contrasts.csv', 'contrast', ['HW(G,C)', 'HWh(G,C)', 'HW(V5,C)', 'HW(V6,C)'],
             ['targets', 'top-1 flip (%)', 'success flip (%)'], 'C, 60 targets'),
            ('T28a_h100_contrasts.csv', 'contrast', ['HW(H,A)', 'RC(H)'],
             ['targets', 'top-1 flip (%)', 'success flip (%)'], 'H100'),
        ],
    },
    'Tmain_cost': {
        'header': ['platform', 'bucket', 'median wall (s)', 'median compile (s)', 'USD, on demand', 'USD, Spot'],
        'blocks': [('T09_cost.csv', ('platform', 'bucket'),
                    [('V5', '1024'), ('V6', '1024'), ('G', '1024'), ('GT', '1024'), ('Gd', '1024'), ('C', '512')],
                    ['bucket', 'median wall (s)', 'median compile (s)', 'USD per structure, on demand',
                     'USD per structure, Spot'])],
    },
}


def price_range(cell):
  """A list of list prices "0.851;0.854;0.897" (one per region) printed as a range "0.85--0.90"; other cells as they are."""
  parts = cell.split(';')
  if len(parts) > 1 and all(p.replace('.', '', 1).isdigit() for p in parts):
    v = [float(p) for p in parts]
    return f'{min(v):.2f}--{max(v):.2f}'
  return cell


def tex(s):
  return s.replace('\\', '\\textbackslash{}').replace('%', '\\%').replace('&', '\\&').replace('_', '\\_') \
      .replace('#', '\\#')


def read(path):
  with open(path, newline='') as f:
    return list(csv.DictReader(f))


def key_of(row, key):
  return tuple(row[k] for k in key) if isinstance(key, tuple) else row[key]


def ppi_rows(rows, wanted):
  """T19b is long (one row per contrast and statistic): one output row per contrast, the two statistics as columns."""
  out = []
  for c in dict.fromkeys(c for c, _ in wanted):
    cells = [next(r for r in rows if (r['contrast'], r['statistic']) == (c, s))['ppi, 30 targets']
             for cc, s in wanted if cc == c]
    out.append((c, ['30'] + cells))
  return out


def build(name, spec, src):
  lines = [f'% {name}: generated by paper/tools/derive_tables.py from paper/assets/tables/ (cells verbatim); '
           'do not edit.']
  body = []
  for block in spec['blocks']:
    file = block[0]
    rows = read(src / file)
    if file == 'T19b_ppi_contrasts.csv':
      _, _, wanted, label = block
      for c, cells in ppi_rows(rows, wanted):
        lines.append(f'% row "{label} {c}": {file}, rows contrast={c}, column "ppi, 30 targets" '
                     '(statistics: top-1 flip per seed; success flip per seed, DockQ 0.23)')
        body.append([label, c] + cells)
      continue
    key, wanted, cols = block[1], block[2], block[3]
    label = block[4] if len(block) > 4 else None
    index = {key_of(r, key): r for r in rows}
    for k in wanted:
      if k not in index:
        raise SystemExit(f'{name}: row {k} not in {file}')
      r = index[k]
      first = k[0] if isinstance(k, tuple) else k
      lines.append(f'% row "{(label + " ") if label else ""}{first}": {file}, row {key}={k}, columns {cols}')
      body.append(([label] if label else []) + [first] + [price_range(r[c]) if c == 'on-demand USD/h' else r[c] for c in cols])
  ncol = len(spec['header'])
  lines += ['\\begin{tabular}{' + 'l' * ncol + '}', '\\toprule', ' & '.join(spec['header']) + ' \\\\', '\\midrule']
  if name == 'Tmain_arms':
    lines.append('% column USD/h: several regional prices (";"-separated in the CSV) printed as min--max, rounded to 2 decimals')
  lines += [' & '.join(c if '--' in c and c.replace('-', '').replace('.', '').isdigit() else tex(c) for c in row) + ' \\\\' for row in body]
  lines += ['\\bottomrule', '\\end{tabular}']
  return '\n'.join(lines) + '\n'


def main():
  src, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
  dest.mkdir(parents=True, exist_ok=True)
  for name, spec in SPECS.items():
    (dest / f'{name}.tex').write_text(build(name, spec, src))
    print(f'   {name}.tex')


if __name__ == '__main__':
  main()
