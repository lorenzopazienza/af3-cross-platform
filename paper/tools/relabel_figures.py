"""Relabeled copies of the frozen supplement figures F1 to F15 (paper/assets/relabeled_figures/), with the short
platform names the paper uses ("v5e vs v6e", "Recompile, L4").

    analysis/figures/.venv/bin/python paper/tools/relabel_figures.py --work <empty scratch folder> [--main CHECKOUT] \
        [--check] [--dest DIR] [--dump] [--without_readable]

--main is the checkout that holds results/ (default: the parent of `git rev-parse --git-common-dir`, the repository
root in a clone, the main checkout from a linked worktree).

The frozen figures (paper/assets/figures/, copied from the main checkout by build_assets.sh) are never edited. This
script reruns the generators that made them, analysis/paper_assets.py (F1 to F14, as
scripts/analysis_runs/paper_assets/reproduce_followups34.sh runs it) and analysis/paper_assets_fu678.py (F15), on the
main checkout's frozen analysis outputs, and writes every output into --work only:
- the generators are copied into <work>/repo/analysis/ (the analysis modules they import are linked from the main
  checkout), and <work>/repo/{results,targets,inputs} are links to the main checkout, so the generators find their
  inputs where they expect them; their --out is a folder under --work;
- every string matplotlib is given (Text.set_text) passes through the label substitutions of
  paper/tools/relabel_tables.py (relabel_cell), so axis labels, tick labels, legends and titles say "v5e vs v6e"
  and "Recompile, L4" where the frozen figures say HW(V5,V6) and RC(G). A substitution that changes the numbers in a
  string stops the script.
Two figures get a layout step at save time (approved 2026-11-11):
- F2: the row labels keep only the pair ("v5e vs v6e"). Every row of F2 is a hw_minus_rc.csv row with reference
  RC(P), the recompilation of the pair's first platform (at default precision for the highest pairs), which the
  supplement caption states; a row of another form stops the script.
- F6: the point annotations become the numbers 1 to 9, in the order of c_perturbation.csv; the key is written to
  <work>/f6_key.txt for the caption. A replaced annotation must hold no number, so the only new digits are the
  point indices.
A third layout step applies to every figure (a general rule of the figures, so that every label reads without turning
the page): no rotated axis label. A rotated y label is written horizontally above its axis, left-aligned with the left
edge of the y tick labels, above the panel title where there is one, as analysis/paper_assets_isc.py does (ylabel_top).
The figure grows at the top by the room the label needs, so the axes keep their size; the script stops if a label meets
another text, a legend or an axes. The text of the label is unchanged.
A fourth layout step makes every figure readable at the width the supplement prints it (style round of 2026-11-20;
READABLE below gives that width, as a fraction of \textwidth, 516 pt, and analysis/tests/test_relabel_figures.py
checks it against sections/supplement_body.tex):
- the two-column figures are set 7.16 in wide (the text width) and print at \textwidth, so they print at natural size;
- a text that would print below 8 pt (tick labels, legends, annotations, titles, mathtext superscripts) is set to the
  figures' base size, 8 pt, or larger where 8 pt would still print below 8 pt; legends are drawn again at that size;
- series that differ only by color get a marker, a dash or a hatch each (GRAY), in order of first appearance of their
  label, the same in every panel, so that every series reads in grayscale;
- the legends of F12 and of F8's left panel, which at 8 pt covered points or texts inside the axes, are drawn under
  the axes (LEGEND_BELOW), and the figure grows by their height;
- the figure is laid out again (tight_layout, as the generators do, with room for a figure legend).
The step changes no drawn value: --dump writes, for every figure, the data of every artist at save time (lines,
collections, patches, texts, tick labels, limits) and the series' styles apart, so that a run with --without_readable
(the relabeled figures as before this step) can be compared with a run with it. The script stops if any drawn text
prints below 8 pt or an axis label is rotated.
--check reruns without the substitutions and without the layout steps and compares each regenerated PDF with the
frozen one byte for byte: it shows that the rerun reproduces the frozen figures, so the relabeled figures differ from
them only in their labels and layout.
Without --check, the relabeled PDFs are copied to paper/assets/relabeled_figures/, or to --dest (the only write
outside --work).
"""
import argparse
import os
import pathlib
import subprocess
import sys

PAPER = pathlib.Path(__file__).resolve().parent.parent
GEN = PAPER.parent / 'analysis'
FROZEN = PAPER / 'assets' / 'figures'
DEST = PAPER / 'assets' / 'relabeled_figures'
FIGS = [f'F{i}' for i in range(1, 16)]

# Style round of 2026-11-20: the width each figure prints at in the supplement, as a fraction of \textwidth (516 pt;
# the \includegraphics width in sections/supplement_body.tex), and the figure width set at save time (None: the
# generator's). Two-column figures are 7.16 in wide and print at natural size; the others keep the generator's size and
# print at a width where their smallest text, a mathtext superscript included, prints at 8 pt or more.
READABLE = {
    'F1_flip_rates': (1.0, 7.16), 'F2_hw_minus_rc': (1.0, 7.16), 'F3_m4_ligand_rmsd': (0.8, None),
    'F4_q3': (0.75, None), 'F5_flip_vs_margin': (0.75, None), 'F6_predicted_vs_observed': (0.6, None),
    'F7_cost': (0.75, None), 'F8_memory': (1.0, 7.16), 'F9_glycans': (1.0, 7.16), 'F10_cross_session': (1.0, 7.16),
    'F11_ppi_flip_rates': (1.0, 7.16), 'F12_ppi_dockq_per_target': (0.75, 5.375),
    'F13_embed_trunk_vs_structure': (0.8, None), 'F14_a100_flip_rates': (1.0, 7.16), 'F15_q3_20_seeds': (0.75, None)}
# Series told apart by color only in the generator: 'dash' (step lines without markers), 'marker' (lines and error bars
# with a marker), 'hatch' (bars). The other figures draw one series, or series with distinct markers already.
GRAY = {'F3_m4_ligand_rmsd': 'dash', 'F4_q3': 'marker', 'F5_flip_vs_margin': 'marker', 'F7_cost': 'marker',
        'F8_memory': 'marker', 'F9_glycans': 'hatch'}
# Legends drawn under the axes, as figure legends (axes index: columns): inside the axes, at 8 pt, F12's covered points
# and F8's left one covered the device-limit labels
LEGEND_BELOW = {'F12_ppi_dockq_per_target': {0: 2}, 'F8_memory': {0: 4}}
# Decisions of 2026-11-21 on the supplement's doubts: F12 is drawn again with the targets on the y axis, so that their
# names print horizontally (relabel_readable.transpose_targets: the same values, x and y swapped); its size in inches
TRANSPOSE = {'F12_ppi_dockq_per_target': (5.375, 5.6)}
# Labels of the frozen figures in the paper's terms (same decisions): the angstrom sign, the 10^-4 of F3's own log
# ticks, "top-ranked sample" for "top-1" as a sample, and "minus" for a hyphen used as a minus sign. Applied after the
# substitutions of relabel_tables.py, as exact strings; each one is listed here with the value it keeps.
FIG_SUBS = [('M4 ligand RMSD (angstrom; 0 drawn at 1e-4)', 'M4 ligand RMSD (\u00c5), 0 drawn at $10^{-4}$'),
            ('M4 C-alpha RMSD, sample median (angstrom)', 'M4 C-alpha RMSD, sample median (\u00c5)'),
            ('success of overall top-1 (%)', 'success of the overall top-ranked sample (%)'),
            ('v5e - L4 success (pp)', 'success, v5e minus L4 (pp)')]
FIG_SUBS_RE = [(r'^(v5e|v6e|L4) (r[12]) overall top-1$', r'\1 \2, overall top-ranked sample'),
               (r'^(\S.* r[12]) - (\S.* r[12])$', r'\1 vs \2')]   # a pair of runs, as "v5e vs v6e"

BOOT = r'''
import importlib.util, pathlib, re, runpy, sys
spec = importlib.util.spec_from_file_location('relabel_tables', {tools!r} + '/relabel_tables.py')
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
spec = importlib.util.spec_from_file_location('relabel_readable', {tools!r} + '/relabel_readable.py')
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)
READABLE, GRAY, BELOW, DUMP = {readable!r}, {gray!r}, {below!r}, {dump!r}
FIG_SUBS, FIG_SUBS_RE, TRANSPOSE = {fig_subs!r}, {fig_subs_re!r}, {transpose!r}
if READABLE:
  rr.record_legends()
import matplotlib.text
RELABEL = {relabel!r}
orig = matplotlib.text.Text.set_text
def set_text(self, s):
  if RELABEL and isinstance(s, str) and s:
    new = '\n'.join(rt.relabel_cell(x) if x.strip() else x for x in s.split('\n'))
    if rt.numbers(s) != rt.numbers(new):
      raise SystemExit(f'numbers changed by a label substitution: {{s!r}} -> {{new!r}}')
    for x, y in FIG_SUBS:
      new = y if new == x else new
    for x, y in FIG_SUBS_RE:
      new = re.sub(x, y, new)
    if new != s:
      LOG.append((s, new))
    s = new
  return orig(self, s)
LOG = []
matplotlib.text.Text.set_text = set_text
import matplotlib.figure
orig_save = matplotlib.figure.Figure.savefig
KEY = []
F2_ROW = re.compile(r'^(?P<pair>.+) minus recompile (?P<rc>.+)$')
def base(name):
  # the highest-precision arms subtract recompilation at default precision (RC(V6), RC(G))
  return name[:-len(' highest')] if name.endswith(' highest') else name
def fix_f2(fig):
  # F2 plots hw_minus_rc.csv rows with reference RC(P): every row subtracts recompilation on the pair's first
  # platform (at default precision for the highest pairs). The rows keep the pair; the caption names what is
  # subtracted. A label not of that form stops the script.
  for ax in fig.axes:
    labs = [t.get_text() for t in ax.get_yticklabels()]
    if not labs:
      continue  # sharey: the inner panels show no labels; setting an empty list would clear the shared ones
    new = []
    for lab in labs:
      m = F2_ROW.match(lab)
      if not m or m.group('rc') != base(m.group('pair').split(' vs ')[0]):
        raise SystemExit('F2 row label not "<pair> minus recompile <first platform>": %r' % lab)
      if rt.numbers(lab) != rt.numbers(m.group('pair')):
        raise SystemExit('F2 row label would lose a number: %r' % lab)
      new.append(m.group('pair'))
    ax.set_yticks(ax.get_yticks(), new)
  fig.tight_layout()
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  boxes = [ax.title.get_window_extent(r) for ax in fig.axes]
  if any(b1.x1 > b2.x0 for b1, b2 in zip(boxes, boxes[1:])):
    for ax in fig.axes:
      ax.title.set_fontsize(ax.title.get_fontsize() * 0.85)
    fig.tight_layout()
  LOG.append(('F2 row labels', 'pair only; caption: minus recompilation on the first platform of each pair'))
def fix_f6(fig):
  # F6: the point annotations become numbers 1 to n in the order of c_perturbation.csv; the key goes to the caption
  # (written to f6_key.txt). The numbers check: a replaced label must hold no number (rt.numbers is empty), so the
  # only new digits in F6 are the point indices.
  ax = fig.axes[0]
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  placed = []
  for i, t in enumerate(ax.texts, 1):
    old = t.get_text()
    if rt.numbers(old):
      raise SystemExit('F6 label holds a number that the point index would drop: %r' % old)
    KEY.append('%d, %s' % (i, old))
    t.set_text(str(i))
    for off in ((3, 2), (3, -8), (-8, 2), (-8, -8), (6, 6), (6, -11)):
      t.xyann = off
      b = t.get_window_extent(r)
      if not any(b.overlaps(q) for q in placed):
        break
    placed.append(t.get_window_extent(r))
  LOG.append(('F6 point labels', 'numbered 1 to %d' % len(ax.texts)))
def top_ylabels(fig, name):
  # note 3 of 18 November 2026: every rotated y label is written horizontally above its axis (layout only)
  from matplotlib.legend import Legend
  from matplotlib.text import Text
  todo = [(ax, ax.yaxis.label.get_text(), ax.yaxis.label.get_fontsize()) for ax in fig.axes
          if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip()
          and round(ax.yaxis.label.get_rotation()) % 360]
  if not todo:
    return
  for ax, _, _ in todo:
    ax.set_ylabel('')
  pad = 3.0
  W, H = fig.get_size_inches()
  H2 = H + (pad + 1.3 * max(fs for _, _, fs in todo)) / 72
  poss = [(ax, ax.get_position()) for ax in fig.axes]
  fig.set_size_inches(W, H2)
  for ax, q in poss:
    ax.set_position([q.x0, q.y0 * H / H2, q.width, q.height * H / H2])
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  placed = []
  for ax, lab, fs in todo:
    yb = ax.yaxis.get_tightbbox(r)
    y0 = max(ax.bbox.y1, yb.y1)
    for tt in (ax.title, ax._left_title, ax._right_title):
      if tt.get_text().strip():
        y0 = max(y0, tt.get_window_extent(r).y1)
    y0 += pad * fig.dpi / 72
    placed.append(fig.text(yb.x0 / fig.bbox.width, y0 / fig.bbox.height, lab, ha='left', va='bottom', fontsize=fs))
  fig.canvas.draw()
  own = set(id(t) for t in placed)
  legs = list(fig.findobj(Legend))
  inleg = set(id(t) for g in legs for t in g.findobj(Text))
  # tick labels count through their axis's drawn extent (labels of ticks outside the view are kept but not drawn)
  ticks = set(id(t) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for t in axis.get_ticklabels(which='both'))
  others = [(o.get_text()[:30], o.get_window_extent(r)) for o in fig.findobj(Text)
            if o.get_visible() and o.get_text().strip() and id(o) not in own and id(o) not in inleg
            and id(o) not in ticks]
  others += [('legend', g.get_window_extent(r)) for g in legs] + [('axes', ax.bbox) for ax in fig.axes]
  for ax in fig.axes:
    for axis in (ax.xaxis, ax.yaxis):
      tb = axis.get_tightbbox(r)
      if tb is not None:
        others.append(('tick labels', tb))
  for t in placed:
    bb = t.get_window_extent(r)
    if bb.x0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or bb.y1 > fig.bbox.height + 0.5:
      raise SystemExit('%s: the y label %r above the axis leaves the figure' % (name, t.get_text()))
    hit = [n for n, ob in others if bb.overlaps(ob)]
    if hit:
      raise SystemExit('%s: the y label %r above the axis meets %r' % (name, t.get_text(), hit[:3]))
  LOG.append(('y labels of ' + name, 'horizontal, above the axis (note 3 of 18 November 2026)'))
def savefig(self, *a, **k):
  name = pathlib.Path(str(a[0])).stem if a else ''
  # style round of 2026-11-20: width, text of at least 8 pt printed, series apart in grayscale (relabel_readable.py)
  if RELABEL and name in TRANSPOSE and READABLE and not getattr(self, '_relabel_transposed', False):
    self._relabel_transposed = True
    rr.transpose_targets(self, name, TRANSPOSE[name], LOG)
  if RELABEL and name in READABLE and not getattr(self, '_relabel_readable', False):
    self._relabel_readable = True
    rr.readable(self, name, READABLE[name], GRAY.get(name), BELOW.get(name), LOG)
  if RELABEL and not getattr(self, '_relabel_named', False):
    self._relabel_named = True
    if name == 'F2_hw_minus_rc':
      fix_f2(self)
    if name == 'F6_predicted_vs_observed':
      fix_f6(self)
  # longer names can make horizontal x tick labels overlap: rotate them and lay the figure out again (layout only)
  if RELABEL and not getattr(self, '_relabel_fixed', False):
    self._relabel_fixed = True
    self.canvas.draw()
    r = self.canvas.get_renderer()
    turned = False
    for ax in self.axes:
      labs = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
      boxes = [t.get_window_extent(r) for t in labs]
      gap = (4 * self.dpi / 72) if READABLE else 0   # decisions of 2026-11-21: at least 4 pt between two labels
      if any(b1.x1 + gap > b2.x0 for b1, b2 in zip(boxes, boxes[1:])):
        # decisions of 2026-11-21: first the names on two lines, horizontal (split at the space nearest the middle);
        # rotated 30 degrees only if they still overlap
        if READABLE:
          def two(w):
            sp = [i for i, ch in enumerate(w) if ch == ' ']
            if not sp or '\n' in w:
              return w
            i = min(sp, key=lambda j: abs(j - len(w) / 2))
            return w[:i] + '\n' + w[i + 1:]
          old = [t.get_text() for t in ax.get_xticklabels()]
          ax.set_xticks(ax.get_xticks(), [two(w) for w in old])   # through the tick formatter, which draws the labels
          self.tight_layout()
          self.canvas.draw()
          labs = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
          boxes = [t.get_window_extent(r) for t in labs]
          if not any(b1.x1 + gap > b2.x0 for b1, b2 in zip(boxes, boxes[1:])):
            LOG.append(('x tick labels of ' + pathlib.Path(str(a[0])).stem, 'on two lines, horizontal (they overlapped)'))
            continue
          ax.set_xticks(ax.get_xticks(), old)
          labs = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        for t in labs:
          t.set_rotation(30)
          t.set_horizontalalignment('right')
        turned = True
        LOG.append(('x tick labels of ' + pathlib.Path(str(a[0])).stem, 'rotated 30 degrees (they overlapped)'))
    if turned:
      self.tight_layout()
  if RELABEL and not getattr(self, '_relabel_ylabels', False):
    self._relabel_ylabels = True
    top_ylabels(self, name)
  if RELABEL and name in READABLE and not getattr(self, '_relabel_checked', False):
    self._relabel_checked = True
    rr.check_printed(self, name, READABLE[name][0])
  if DUMP and a and str(a[0]).endswith('.pdf'):
    rr.dump(self, name, pathlib.Path(DUMP) / (name + '.json'))
  return orig_save(self, *a, **k)
matplotlib.figure.Figure.savefig = savefig
sys.argv = {argv!r}
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
try:
  runpy.run_path(sys.argv[0], run_name='__main__')
finally:
  log = pathlib.Path({log!r})
  log.write_text(''.join(f'{{a!r}} -> {{b!r}}\n' for a, b in sorted(set(LOG))))
  if KEY:
    pathlib.Path({log!r}).with_name('f6_key.txt').write_text('; '.join(KEY) + '\n')
'''


def run(py, work, main, relabel, readable=True, dump=False):
  repo = work / 'repo'
  (repo / 'analysis').mkdir(parents=True, exist_ok=True)
  for d in ('results', 'targets', 'inputs', 'notes'):
    link = repo / d
    if not link.exists():
      link.symlink_to(main / d)
  for f in (main / 'analysis').glob('*.py'):
    link = repo / 'analysis' / f.name
    if not link.exists():
      link.symlink_to(f)
  for name in ('paper_assets.py', 'paper_assets_fu678.py'):
    dst = repo / 'analysis' / name
    if dst.is_symlink():
      dst.unlink()
    dst.write_text((GEN / name).read_text())
  tag = ('relabeled' if readable else 'relabeled_without_readable') if relabel else 'check'
  out34, out678 = work / tag / 'paper_assets_followups34', work / tag / 'paper_assets_fu678'
  jobs = [
      [str(repo / 'analysis' / 'paper_assets.py'), '--out', str(out34), '--followups34',
       '--cpu60_dir', 'results/analysis/cpu60_20261018', '--a100_dir', 'results/analysis/a100_20261018',
       '--feat_dir', 'results/analysis/feat_20261018'],
      [str(repo / 'analysis' / 'paper_assets_fu678.py'), '--main', str(repo), '--out', str(out678)],
  ]
  env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
  for i, argv in enumerate(jobs):
    boot = work / f'boot_{tag}_{i}.py'
    boot.write_text(BOOT.format(tools=str(PAPER / 'tools'), relabel=relabel, argv=argv,
                                log=str(work / f'substitutions_{tag}_{i}.txt'),
                                readable=READABLE if relabel and readable else {}, gray=GRAY, below=LEGEND_BELOW,
                                fig_subs=FIG_SUBS, fig_subs_re=FIG_SUBS_RE, transpose=TRANSPOSE,
                                dump=str(work / f'dump_{tag}') if dump else ''))
    subprocess.run([py, str(boot)], cwd=str(repo), env=env, check=True)
  figs = {}
  for out in (out34, out678):
    for p in (out / 'figures').glob('F*.pdf'):
      figs[p.stem] = p
  return figs


def main_checkout():
  """Default of --main: the parent of `git rev-parse --git-common-dir`; without git, the repository of this file."""
  try:
    common = subprocess.run(['git', 'rev-parse', '--git-common-dir'], cwd=PAPER, capture_output=True, text=True,
                            check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return PAPER.parent
  common = (PAPER / common).resolve()
  return common.parent if common.name == '.git' else PAPER.parent


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--work', required=True)
  ap.add_argument('--main', default=str(main_checkout()))
  ap.add_argument('--check', action='store_true')
  ap.add_argument('--dest', default=str(DEST),
                  help='folder of the relabeled PDFs (default paper/assets/relabeled_figures)')
  ap.add_argument('--dump', action='store_true', help='write the drawn data of every figure to <work>/dump_<run>/')
  ap.add_argument('--without_readable', action='store_true',
                  help='leave out the readability step (the relabeled figures as before the style round of 2026-11-20)')
  args = ap.parse_args()
  work, main = pathlib.Path(args.work).resolve(), pathlib.Path(args.main).resolve()
  if work == main or main in work.parents:
    raise SystemExit('--work must be outside the main checkout')
  work.mkdir(parents=True, exist_ok=True)
  figs = run(sys.executable, work, main, relabel=not args.check, readable=not args.without_readable, dump=args.dump)
  frozen = sorted(FROZEN.glob('F*.pdf'), key=lambda p: int(p.stem.split('_')[0][1:]))
  missing = [p.stem for p in frozen if p.stem not in figs]
  if missing:
    raise SystemExit(f'not regenerated: {missing}')
  if args.check:
    for p in frozen:
      same = p.read_bytes() == figs[p.stem].read_bytes()
      print(f'   {p.stem}: {"identical" if same else "DIFFERS"} to the frozen PDF')
    return
  dest = pathlib.Path(args.dest).resolve()
  dest.mkdir(parents=True, exist_ok=True)
  for p in frozen:
    (dest / p.name).write_bytes(figs[p.stem].read_bytes())
  print(f'   {len(frozen)} figures relabeled into {dest.relative_to(PAPER) if PAPER in dest.parents else dest}/')


if __name__ == '__main__':
  main()
