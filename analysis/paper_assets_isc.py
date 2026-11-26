"""Paper assets of the ISC 2027 paper: Figs. 1 to 5, Tables I to IV, captions, and the tables the text cites instead of a
ledger (campaign counts, log census, exploratory mechanism counts, CPU flips, follow-up 12 post-cutoff targets).

    analysis/figures/.venv/bin/python analysis/paper_assets_isc.py [--main CHECKOUT] [--out FOLDER] [--reserve]

--main is the checkout that holds results/ (default: main_checkout() of analysis/paper_assets_fu678.py, the repository
root in a clone, the main checkout from a linked worktree); --out defaults to <main>/results/paper_assets_isc.
The main study is read only from results/analysis/main_20261014/primary_v2/ (the M4-corrected output; the folder above it
holds superseded C-alpha values). Every figure and table cell is formatted from a CSV cell by the functions below; the
derived numbers (ratios, counts, ranges) are computed here and named in README.md. The campaign counts and the log census
read the run records of the frozen study sessions (results/af3/, final freeze d706dffc of 2026-11-25; the counts cover
its 295 sessions of the main study and follow-ups 1 to 12), each checked against the freeze
manifest. Reserve figures R1 and R2 are written only with --reserve. Writes figures/*.{pdf,png}, tables/*.{csv,tex},
captions.tex, sources.tsv (path and SHA-256 of every input) and README.md (one line per asset with its role).
"""
import argparse
import collections
import functools
import glob
import hashlib
import json
import math
import multiprocessing
import os
import pathlib
import re
import sys
import types
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_assets_fu678 as base  # noqa: E402

REPO = base.REPO
AN = 'results/analysis'
MAINV2 = f'{AN}/main_20261014/primary_v2'
EXPL = f'{AN}/main_20261014/exploratory'
# The final freeze (results/freeze/INDEX.md; section 10 row of 2026-11-25): 607 sessions, cumulative over freeze 68923012
# of 2026-11-08 (295 sessions, the main study and follow-ups 1 to 12) plus the 144 sessions of follow-ups 13 to 16 and of
# the two later measurements (freeze 4ec185c5 of 2026-11-17, 439 sessions) plus the 168 sessions of follow-ups 17 to 21.
# The generator reads this freeze; the campaign counts
# keep the scope of the 295 sessions (LATER_LABELS are left out of them, and the 295 are checked against freeze 68923012).
FREEZE_DIR = 'results/freeze/followups1721_20261125'
FREEZE_ID = 'd706dffc07fec6d86471e28f062178e4f309193d3bfd6a66de47959eb3bfdd8f'
FREEZE_MADE = '20261125T114315Z'      # the final freeze was made at 2026-11-25T11:43:15Z (results/freeze/INDEX.md)
COUNT_FREEZE_DIR = 'results/freeze/followups612_20261108'   # the 295 sessions the campaign counts cover
COUNT_MADE = '20261108T164235Z'       # freeze 68923012 was made at 2026-11-08T16:42:35Z; later sessions are not counted
LATER_LABELS = ('fu13', 'fu14', 'rcx', 'hlo', 'memlimit_ah', 'warm_th', 'dtp', 'tri', 'rcf', 'aut', 'xvm')
CROSS_SESSION = f'{AN}/main_monitor_20261013/cross_session_summary.txt'
CROSS_SESSION_LINE = 'C    (expected identical): 12/12 (target, seed) bit-identical as expected'
PLAN = 'notes/analysis_plan.md'
SOURCES = base.SOURCES          # rel path -> sha256 (shared with base.read)
ASSETS = {}                     # rel name -> (role, what)
FROZEN = {}                     # 'results/af3/<session>/<path>' -> sha256, filled by load_freeze()
NOTES = []                      # lines for README.md: disagreements with the ledgers and checks

# Style
NAVY, ROSE, INK, BAND, GOLD = '#004488', '#BB5566', '#000000', '#D9D9D9', '#DDAA33'
RULE_LIGHT = '#B8B8B8'   # short rules between follow-up groups in Fig. 1's label area (R119, R119b)
RULE_MAIN = '#999999'    # Fig. 1's rule after the main study, a little darker than RULE_LIGHT
LSTAR = {'navy': 29, 'rose': 50, 'band': 87}     # CIE L* of the tokens (section 2 table), tested
FS = 9
WIDTHS = (3.5, 7.16)

ROLE_MAIN = 'main study (pre-registered); primary and secondary rows as labeled'
ROLE_FU = 'follow-ups (outside the pre-registered study; FU10 and FU11 with OSF addenda registered before their runs)'
ROLE_EXPL = 'exploratory, post hoc (specified after the main results were read); not causal'
ROLE_DESIGN = 'design: main study, memory limit and follow-ups 1 to 12'
ROLE_CAMPAIGN = 'campaign scale and records (descriptive; no contrast, no reading rule)'

STUDY_LABELS = ['pilot2', 'main', 'memlimit', 'ppi', 'embed', 'a100', 'cpu60', 'msa', 'sw', 'seeds', 'h100', 'large',
                'repl', 'prec5']         # the 14 inference labels of the final freeze ('feat' ran featurization only)
DEVICES = ['TPU v5 lite', 'TPU v6 lite', 'NVIDIA L4', 'NVIDIA A100-SXM4-80GB', 'NVIDIA H100 80GB HBM3', 'cpu']
DEVICE_SHORT = {'TPU v5 lite': 'V5 (TPU v5e)', 'TPU v6 lite': 'V6 (TPU v6e)', 'NVIDIA L4': 'L4', 'NVIDIA A100-SXM4-80GB':
                'A100 80 GB', 'NVIDIA H100 80GB HBM3': 'H100 80 GB', 'cpu': 'CPU'}
PLATFORM_DEVICE = {'v5e': 'TPU v5 lite', 'v6e': 'TPU v6 lite', 'l4': 'NVIDIA L4', 'a100': 'NVIDIA A100-SXM4-80GB',
                   'h100': 'NVIDIA H100 80GB HBM3', 'cpu': 'cpu'}

# Values of an earlier, independent count (the campaign ledger, working notes not in this repository) that the
# generator must reproduce; a mismatch stops the script.
MUST = {'processes': 13499, 'exit 0': 13495, 'exit 1, OOM': 4, 'samples': 67475, 'distinct targets': 230,
        'frozen sessions': 295, 'frozen local files': 261630, 'frozen local bytes': 51854099086}
# frozen local files and bytes of the 295 sessions in freeze 4ec185c5: freeze 68923012 had 260930 files and
# 50255319052 bytes; the 2026-11-17 freeze added 700 files (1598780034 bytes) to two main-study sessions, changed none


# ---------------------------------------------------------------------------------------------------------------------
# Formatting: rates one decimal; differences signed, bounds signed only when negative.

def pc(x):
  """A fraction as percent or points, one decimal; -0.0 prints as 0.0."""
  s = f'{100 * float(x):.1f}'
  return '0.0' if s in ('-0.0', '0.0') else s


def spc(x):
  """A difference: the estimate carries its sign."""
  s = pc(x)
  return s if s == '0.0' or s.startswith('-') else '+' + s


def rate_ci(r):
  return f'{pc(r.estimate)} [{pc(r.ci_low)}, {pc(r.ci_high)}]'


def diff_ci(r):
  return f'{spc(r.estimate)} [{pc(r.ci_low)}, {pc(r.ci_high)}]'


def sig3(x):
  x = float(x)
  if x == 0:
    return '0'
  s = f'{x:#.3g}'
  return s[:-1] if s.endswith('.') else s


def sig3_ci(r):
  return f'{sig3(r.estimate)} [{sig3(r.ci_low)}, {sig3(r.ci_high)}]' if float(r.estimate) != 0 or float(r.ci_high) != 0 \
      else '0'


def fint(n, tex=False):
  """Integers: no separator up to four digits, a thin space from five digits in LaTeX (none in CSV)."""
  n = int(n)
  if not tex or abs(n) < 10000:
    return str(n)
  return f'{n:,}'.replace(',', '\\,')


def usd3(x):
  return sig3(x)


TEX_CHARS = {'Å': '\\AA{}', '±': '$\\pm$', '→': '$\\rightarrow$', '†': '\\textsuperscript{\\dag}', '≥': '$\\geq$', '×': '$\\times$',
             '≤': '$\\leq$', 'D\\_F': '$D_F$', 'D\\_T': '$D_T$'}


def tex(s):
  """Plain cell text to LaTeX: escapes, real minus signs, thin spaces from five digits."""
  s = str(s)
  s = s.replace('\\', '\\textbackslash{}').replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#')
  s = re.sub(r'(?<![\w\-.])(\d{5,})(?![\d.])', lambda m: fint(m.group(1), True), s)
  s = re.sub(r'(?<![\w\-$])-(?=\d)', '$-$', s)
  s = re.sub(r'"([^"]*)"', r"``\1''", s)
  for k, v in TEX_CHARS.items():
    s = s.replace(k, v)
  return s


def no_em_dash(text, where):
  if '\u2014' in text:
    raise SystemExit(f'em dash in {where}')
  if 'RC(C)' in text:
    raise SystemExit(f'"RC(C)" in {where} (decisions.md D1)')
  return text


# ---------------------------------------------------------------------------------------------------------------------
# Inputs

def sha_file(p):
  h = hashlib.sha256()
  with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def record(main, rel):
  SOURCES[rel] = sha_file(main / rel)
  return main / rel


def read(main, rel, **kw):
  p = record(main, rel)
  return pd.read_csv(p, low_memory=False, **kw)


def text(main, rel):
  return record(main, rel).read_text()


def load_freeze(main):
  """The final freeze's manifest: its SHA-256 must be the freeze identifier of the row of 2026-11-25."""
  p = main / FREEZE_DIR / 'manifest_sha256.tsv'
  h = sha_file(p)
  if h != FREEZE_ID:
    raise SystemExit(f'{p} is not freeze {FREEZE_ID[:12]} (got {h[:12]})')
  SOURCES[f'{FREEZE_DIR}/manifest_sha256.tsv'] = h
  for line in p.read_text().splitlines()[1:]:
    session, path, _, sha, _ = line.split('\t')
    FROZEN[f'results/af3/{session}/{path}'] = sha


def check_freeze(paths):
  """Every source under results/af3/ of a frozen session must be in the freeze with the same SHA-256."""
  bad = [p for p in paths if p.startswith('results/af3/') and not p.startswith('results/af3/.launcher/')
         and FROZEN.get(p) != SOURCES[p]]
  if bad:
    raise SystemExit(f'{len(bad)} source(s) under results/af3/ differ from freeze {FREEZE_ID[:12]} or are not in it: '
                     + ', '.join(bad[:5]))


def pick(df, **kw):
  """The one row matching kw; a key absent from the frame's columns is ignored (as in the guide's prototype)."""
  m = df
  for k, v in kw.items():
    if v is not None and k in m.columns:
      m = m[m[k] == v]
  if len(m) != 1:
    raise SystemExit(f'expected one row for {kw}, found {len(m)}')
  return m.iloc[0]


# ---------------------------------------------------------------------------------------------------------------------
# Figures: shared style and saver

def style():
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import font_manager
  path = font_manager.findfont('Arial', fallback_to_default=False)
  if pathlib.Path(path).name.lower() != 'arial.ttf':
    raise SystemExit(f'Arial not found (got {path})')
  SOURCES['font: ' + str(path)] = sha_file(path)
  import matplotlib.pyplot as plt
  plt.rcdefaults()
  plt.rcParams.update({
      'font.family': 'Arial', 'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS, 'xtick.labelsize': FS,
      'ytick.labelsize': FS, 'legend.fontsize': FS, 'legend.title_fontsize': FS, 'figure.titlesize': FS,
      'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.default': 'regular', 'mathtext.fontset': 'custom',
      'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial', 'mathtext.cal': 'Arial',
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
      'ytick.major.width': 0.6, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5, 'xtick.major.size': 2.5,
      'ytick.major.size': 2.5, 'lines.linewidth': 1.0, 'legend.frameon': False, 'axes.grid': False,
      'svg.hashsalt': 'af3', 'pdf.compression': 6, 'axes.unicode_minus': True})
  return plt


CLIPPED = {}        # figure name -> texts cut at the edges (checked at the end of main)
FIG_TEXTS = {}      # figure name -> [(text, fontsize)] collected before saving (tested: >= 8 pt)
FIG_SCRIPTS = {}    # figure name -> [(text, size of its smallest glyph)] of the drawn texts (tested: >= 8 pt)


YLAB_PAD_PT = 3.0      # gap between the top of an axis (or its panel title) and the y label written above it
SCRIPT_EXEMPT = {'figR1_cost'}    # reserve figure, printed neither in the paper nor in the supplement


def plain_log_ticks(axis):
  """Decimal labels on a log axis (0.0001, 0.01, 1, 10), as fig_fu13_l has them: the default 10^-4 prints its
  exponent as a superscript at 0.7 of the tick size, 6.3 pt at 9 pt (style round of 2026-11-20)."""
  from matplotlib.ticker import FuncFormatter
  axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def glyph_ratio(t):
  """Smallest glyph of a text over its font size: 1 for plain text, 0.7 for a mathtext superscript or subscript."""
  s = t.get_text()
  s2, ismath = t._preprocess_math(s)
  if ismath is not True:
    return 1.0
  from matplotlib.mathtext import MathTextParser
  fp = t.get_fontproperties().copy()
  fp.set_size(10.0)
  g = MathTextParser('path').parse(s2, 72, fp).glyphs
  return min(x[1] for x in g) / 10.0 if g else 1.0


def ylabel_top(fig, ax, label, grow=True, align=None):
  """A general rule of the figures: no rotated axis label, so that every label reads without turning the page. The y
  label is written horizontally above its axis, left-aligned with the left edge of the y tick labels, at the top of the
  panel (above the panel title where there is one); Fig. 2 (fig_fu13) already writes its y label horizontally. With
  grow, the figure grows at the top by the room the label needs, so every axes keeps its size and place in inches;
  without it, the caller has made the room. With align (a list of axes), the label starts at the leftmost tick-label
  edge of those axes, so stacked panels have their labels in one column. Call it last, when the layout is final; save()
  stops if the label meets another text, a legend or an axes, or if any axis label is rotated."""
  fs = ax.yaxis.label.get_fontsize()
  ax.set_ylabel('')

  def taller(add_in):
    W, H = fig.get_size_inches()
    H2 = H + add_in
    poss = [(a, a.get_position()) for a in fig.axes]
    fig.set_size_inches(W, H2)
    for a, q in poss:
      a.set_position([q.x0, q.y0 * H / H2, q.width, q.height * H / H2])

  def place():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    yb = ax.yaxis.get_tightbbox(r)          # ticks and tick labels (the top tick label may stand above the axis)
    x0, y0 = yb.x0, max(ax.bbox.y1, yb.y1)
    if align:
      x0 = min(a.yaxis.get_tightbbox(r).x0 for a in align)
    for tt in (ax.title, ax._left_title, ax._right_title):      # titles set with loc='left' are _left_title
      if tt.get_text().strip():
        y0 = max(y0, tt.get_window_extent(r).y1)
    y0 += YLAB_PAD_PT * fig.dpi / 72
    return fig.text(x0 / fig.bbox.width, y0 / fig.bbox.height, label, ha='left', va='bottom', fontsize=fs), r

  if grow:
    taller((YLAB_PAD_PT + 1.3 * fs) / 72)
  t, r = place()
  # a legend of the figure anchored at its top (Fig. S1) moves up with the top: grow until the label clears it
  for _ in range(3):
    bb = t.get_window_extent(r)
    low = [g.get_window_extent(r).y0 for g in fig.legends if g.get_window_extent(r).overlaps(bb)]
    if not low or not grow:
      break
    t.remove()
    taller((bb.y1 - min(low)) / fig.dpi + YLAB_PAD_PT / 72)
    t, r = place()
  fig._top_ylabels = getattr(fig, '_top_ylabels', []) + [t]
  return t


def figure_saver(out):
  plt = style()
  d = out / 'figures'
  d.mkdir(parents=True, exist_ok=True)

  def save(fig, name, role, what):
    from matplotlib.text import Text
    texts = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text) if t.get_text().strip()]
    small = [t for t in texts if t[1] < 8]
    if small:
      raise SystemExit(f'{name}: text below 8 pt: {small[:3]}')
    w = round(fig.get_size_inches()[0], 2)
    if w not in WIDTHS:
      raise SystemExit(f'{name}: width {w} in is not 3.5 or 7.16')
    for t, _ in texts:
      no_em_dash(t, name)
    FIG_TEXTS[name] = texts
    # No rotated axis label (ylabel_top writes y labels above the axis)
    for ax in fig.axes:
      for lab in (ax.xaxis.label, ax.yaxis.label):
        if lab.get_visible() and lab.get_text().strip() and round(lab.get_rotation()) % 360:
          raise SystemExit(f'{name}: rotated axis label {lab.get_text()!r}; use ylabel_top')
    # text cut at the figure's edges (reading note R89, 2026-11-12): every drawn text must lie inside the page
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    hidden = set()     # tick labels of ticks outside the view limits are kept by matplotlib but not drawn
    for ax in fig.axes:
      for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = min(lim), max(lim)
        for minor in (False, True):
          ticks = axis.get_minor_ticks() if minor else axis.get_major_ticks()
          locs = axis.get_minorticklocs() if minor else axis.get_majorticklocs()
          for k, tk in enumerate(ticks):
            if k >= len(locs) or not (lo - 1e-9 * abs(hi - lo) <= locs[k] <= hi + 1e-9 * abs(hi - lo)):
              hidden.update({id(tk.label1), id(tk.label2)})
    # superscripts and subscripts (mathtext) print at 0.7 of their text's size: they too must reach 8 pt
    scripts = [(t.get_text(), round(t.get_fontsize() * glyph_ratio(t), 2)) for t in fig.findobj(Text)
               if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    small = [t for t in scripts if t[1] < 8]
    if small and name not in SCRIPT_EXEMPT:
      raise SystemExit(f'{name}: superscript or subscript below 8 pt: {small[:3]}')
    FIG_SCRIPTS[name] = scripts
    # a y label written above its axis (ylabel_top) must meet no other text, legend or axes
    if getattr(fig, '_top_ylabels', None):
      from matplotlib.legend import Legend
      own = {id(t) for t in fig._top_ylabels}
      legs = list(fig.findobj(Legend))
      inleg = {id(t) for g in legs for t in g.findobj(Text)}
      others = [(o.get_text()[:30], o.get_window_extent(rend)) for o in fig.findobj(Text)
                if o.get_visible() and o.get_text().strip() and id(o) not in own | hidden | inleg]
      others += [('legend', g.get_window_extent(rend)) for g in legs]
      others += [('axes', a.bbox) for a in fig.axes]
      for t in fig._top_ylabels:
        bb = t.get_window_extent(rend)
        hit = [k for k, ob in others if bb.overlaps(ob)]
        if hit:
          raise SystemExit(f'{name}: the y label {t.get_text()!r} above the axis meets {hit[:3]}')
    cut = []
    for t in fig.findobj(Text):
      if not (t.get_visible() and t.get_text().strip()) or id(t) in hidden:
        continue
      bb = t.get_window_extent(rend)
      if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
        cut.append((t.get_text()[:30], round(bb.x0, 1), round(bb.y0, 1), round(bb.x1 - W, 1), round(bb.y1 - H, 1)))
    # The same check at the PDF backend's text metrics: Agg measured Fig. 1's legend
    # inside the page, while in the PDF, which the paper includes, it overran both edges by about 2 pt. Offset boxes
    # (Fig. 1's legend, whose keys are not Text) and legends are checked whole; their own texts are placed only when
    # drawn, so they are left to the box.
    from matplotlib import cbook
    from matplotlib.backend_bases import _get_renderer
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.legend import Legend
    from matplotlib.offsetbox import OffsetBox
    canvas = fig.canvas
    with cbook._setattr_cm(fig, dpi=72):
      prend = _get_renderer(fig, FigureCanvasPdf(fig).print_pdf)
      PW, PH = fig.bbox.width, fig.bbox.height
      obs = list(fig.findobj(Legend)) + [a for c in [fig, *fig.axes] for a in c.artists if isinstance(a, OffsetBox)]
      inbox = {id(t) for a in obs for t in a.findobj(Text)}
      boxes = [(t.get_text()[:30], t) for t in fig.findobj(Text)
               if t.get_visible() and t.get_text().strip() and id(t) not in hidden and id(t) not in inbox]
      boxes += [(f'{type(a).__name__} box', a) for a in obs]
      for label, a in boxes:
        bb = a.get_window_extent(prend)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > PW + 0.5 or bb.y1 > PH + 0.5:
          cut.append(('PDF: ' + label, round(bb.x0, 1), round(bb.y0, 1), round(bb.x1 - PW, 1), round(bb.y1 - PH, 1)))
    fig.set_canvas(canvas)
    if cut:
      CLIPPED[name] = cut
    fig.savefig(d / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Producer': None, 'Creator': None})
    fig.savefig(d / f'{name}.png', dpi=300, metadata={'Software': None})
    plt.close(fig)
    ASSETS[f'figures/{name}'] = (role, what)
  return plt, save


def write_csv(out, name, df, role, what):
  d = out / 'tables'
  d.mkdir(parents=True, exist_ok=True)
  df = df.reset_index(drop=True)
  body = df.to_csv(index=False, lineterminator='\n')
  no_em_dash(body, name)
  (d / f'{name}.csv').write_text(body)
  ASSETS[f'tables/{name}.csv'] = (role, what)


# Supplementary tables with a short title (reading note R84, 2026-11-11): the rest of the former caption opens the note
FIG_PERT = 'FIGREFPERT'   # placeholder that write_table turns into Fig.~\\ref{fig:s-perturbation} after escaping
NOTE_REFS = {FIG_PERT: 'Fig.~\\ref{fig:s-perturbation}', 'SECREFQTWO': 'Supplementary Section~\\ref{S-sec:s-main-q2}',
             'SECREFFUFOURTEEN': 'Section~\\ref{sec:s-fu14}', 'SECREFFUTABLE': '\\ref{S-tab:followups-isc}', 'SECREFFUTHIRTEEN': 'Section~\\ref{sec:s-fu13}',
             'FIGREFFUTHIRTEENB': 'Fig.~\\ref{fig:fu13}(b)', 'SECREFWINNERS': 'Table~\\ref{tab:s-winners}',
             'TABREFQTWO': 'Table~\\ref{tab:q2}', 'SECREFCONTRASTS': 'Section~\\ref{sec:contrasts}',
             'SUPPTABQTWO': 'Supplementary Table~\\ref{S-tab:q2}', 'TABREFPROV': 'Table~\\ref{tab:s-provisioning}', 'SUPPTABFUSIXTEEN': 'Supplementary Table~\\ref{S-tab:s-fu16}'}


def note_refs(t):
  for k, v in NOTE_REFS.items():
    t = t.replace(k, v)
  return t


def write_note(out, name, notes, role, what):
  """tables/<name>_note.tex alone, for a table whose tabular another float already prints with its own note (the
  appendix of the main paper keeps a short note; the supplement keeps the full one). Same escaping as write_table."""
  nb = f'% {name}, note below the table. Generated by analysis/paper_assets_isc.py; do not edit.\n' + \
      note_refs(tie_pairs(' '.join(tex(n) for n in notes))) + '\n'
  no_em_dash(nb, f'{name}_note')
  (out / 'tables' / f'{name}_note.tex').write_text(nb)
  ASSETS[f'tables/{name}_note.tex'] = (role, what)
SHORT_CAPTIONS = {
    'TS_pert_five': ('Exploratory (c) at five seeds', 'The table gives the perturbation size against the margin and the '
                     'two-sample model for the precision contrasts of Follow-up 12, seeds 1 to 5, as '
                     f'{FIG_PERT} draws them.'),
    'TIV_followups': ('Follow-ups 1 to 21', 'The follow-ups are outside the pre-registered study, and we specified them '
                      'after the main results were read. FU10, FU11, and FU13 to FU21 have OSF addenda registered before '
                      'their runs. The '
                      'table gives what changed, the targets and seeds, the headline with its 95% interval, and the '
                      'reading in the record\'s words, with the paper\'s names for platforms and contrasts. Follow-ups 1 '
                      'to 4 have no reading rule, so their reading is our description.'),
    'TS_fu03_q3': ('Follow-up 3, Q3: success rate over the first n seeds', 'The table gives the success rate of the overall '
                   'top-ranked sample over the first n seeds, in %, per condition, with the A100 and the A100 Triton arm '
                   'added, and the spread between conditions.'),
    'TS_fu03_q3diff': ('Follow-up 3, Q3: differences in success rate', 'The table gives the differences in the success '
                       'rate of the overall top-ranked sample for every pair with the A100 or the A100 Triton arm, in '
                       'percentage points, with 95% intervals.'),
    'TS_fu03_memory': ('Follow-up 3, M7: device memory of the A100 arms', 'The table gives the device memory of the A100, '
                       'A100 Triton, and A100 highest arms per token bucket against the device limit.'),
    'T_rules': ('Reporting rules', 'The table gives how often the reported prediction differs between two platforms, by '
                'rule and number of seeds. It is a follow-up analysis of the frozen outputs and is descriptive.'),
    'TS_rules_seeds': ('Ranking rules: seed consensus', 'The table gives the top-1 flip of the top-ranked sample across n '
                       'consecutive seeds, in disjoint blocks of seeds 1 to 20, and the cost per reported prediction. '
                       'Recompilation on the L4 uses seeds 1 to 5.'),
    'TS_rules_ties': ('Ranking rules: tie tolerance', 'The table gives the top-1 flip of the reported prediction, in %, '
                      'when a tie is declared within t.'),
    'TS_rules_tieshare': ('Ranking rules: share of ties and conflicts', 'Each cell gives the % of units with a tie '
                          'declared on either side, then the % of units whose reported sets are disjoint (conflict).'),
    'TS_rules_rounded': ('Ranking rules: full-precision against rounded score', 'The table gives the top-1 flip, in %, '
                         'when ranking by the full-precision score and by the two-decimal score. Ties go to the lowest '
                         'seed and sample.'),
    'TS_rules_model': ('Exploratory: observed flip rate against two models', 'The table gives the observed top-1 flip '
                       'rate, in %, against the resampling model and the two-sample model of '
                       f'{FIG_PERT}. '
                       'The resampling model adds the five score shifts of another target\'s seed together.'),
    'TS_campaign_changed': ('Campaign simulation: reported samples that change', 'The table gives the reported samples '
                            'that change, in % of jobs, when a share s of jobs runs on the other device.'),
    'TS_campaign_points': ('Campaign simulation: success-rate difference', 'The table gives the success rate of the mixed '
                           'campaign minus that of the campaign on the home device, in percentage points, when a share s '
                           'of jobs runs on the other device.'),
    'TS_campaign': ('The campaign by label and by device', 'The table gives sessions, processes, samples, exit codes, and '
                    'process and VM hours, and the cost at list prices. The cost has recorded items with the dates of '
                    'their plan rows, estimated items, and the total.'),
    'TS_provisioning': ('Provisioning by arm and follow-up', 'The table gives the sessions, processes, machine type, zones '
                        'or regions, and provisioning model of each arm.'),
}


# Supplement-only tables laid out for 8 pt (style round of 2026-11-20: no table of the supplement prints below 8 pt),
# by paper/tools/table_layout.py, which also reads the cells back and stops if any changed. Keyword arguments of its
# layout(): wrap (printed characters per header line), spec (P{w}: ragged-right paragraph column), stack (each item
# of a "a; b" list on its own line), longtable (breaks across pages, header repeated). Tables of the main text are
# never laid out here.
SUPP_LAYOUT = {
    'TS_provisioning': dict(spec='@{}lrrP{3.0cm}P{5.5cm}P{3.2cm}@{}', longtable=True),
    'TS_winners_ext': dict(wrap=10),
    'TS_fu03_memory': dict(wrap=10),
    'TS_fu13_checks': dict(spec='@{}P{150pt}P{280pt}l@{}'),
    'TS_fu13_l': dict(wrap=10),
    'TS_fu15_rc': dict(wrap=10),
    'TS_rules_seeds': dict(wrap=10),
    'TS_campaign_changed': dict(stack={7: 'semicolon'}),
    'TS_campaign_points': dict(stack={7: 'semicolon'}),
    'TS_campaign': dict(wrap=8, longtable=True),
}


@functools.lru_cache(maxsize=None)
def table_layout():
  """paper/tools/table_layout.py of this checkout (the same rules lay out the relabeled tables of the supplement)."""
  import importlib.util
  spec = importlib.util.spec_from_file_location('table_layout', REPO / 'paper' / 'tools' / 'table_layout.py')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def write_table(out, name, df, caption, role, notes=(), colspec=None, tex_df=None, head=None):
  """tables/<name>.csv (plain cells), tables/<name>.tex (the tabular with booktabs only, so a box around it has the
  tabular's natural width) and, with notes, tables/<name>_note.tex (the note text only; the paper inputs it below the
  table with \\isctablenote). head: LaTeX header lines in place of the one header row of column names."""
  write_csv(out, name, df, role, caption)
  if name in SHORT_CAPTIONS:
    caption, moved = SHORT_CAPTIONS[name]
    notes = [moved, *notes]
  t = (tex_df if tex_df is not None else df).reset_index(drop=True)
  cols = list(t.columns)
  spec = colspec or ('l' * len(cols))
  lines = [f'% {name}. Generated by analysis/paper_assets_isc.py; do not edit.', f'% Role: {role}',
           '\\begin{tabular}{@{}' + spec + '@{}}', '\\toprule',
           *(head or [' & '.join(tex(reader_label(c)) for c in cols) + ' \\\\']), '\\midrule']
  for row in t.itertuples(index=False):
    cells = ['' if (isinstance(v, float) and math.isnan(v)) else str(int(v)) if (isinstance(v, float) and v.is_integer())
             else str(v) for v in row]
    if cells[0].startswith('\\GROUP '):
      lines.append(f'\\multicolumn{{{len(cols)}}}{{@{{}}l}}{{\\emph{{{tex(reader_label(cells[0][7:]))}}}}} \\\\')
      continue
    if cells[0].startswith('\\INDENT '):     # a row that qualifies the row above it, indented by a quad
      cells[0] = '\\quad ' + tex(reader_label(cells[0][8:]))
      lines.append(' & '.join([cells[0]] + [tex(reader_label(c)) for c in cells[1:]]) + ' \\\\')
      continue
    lines.append(' & '.join(tex(reader_label(c)) for c in cells) + ' \\\\')
  lines += ['\\bottomrule', '\\end{tabular}']
  body = '\n'.join(lines) + '\n'
  if name in SUPP_LAYOUT:
    body = table_layout().layout(body, name=name, **SUPP_LAYOUT[name])
  no_em_dash(body, name)
  (out / 'tables' / f'{name}.tex').write_text(body)
  ASSETS[f'tables/{name}.tex'] = (role, caption)
  note = out / 'tables' / f'{name}_note.tex'
  if notes:
    nb = f'% {name}, note below the table. Generated by analysis/paper_assets_isc.py; do not edit.\n' + \
        note_refs(tie_pairs(' '.join(tex(n) for n in notes))) + '\n'
    no_em_dash(nb, f'{name}_note')
    note.write_text(nb)
    ASSETS[f'tables/{name}_note.tex'] = (role, f'note below {name}')
  elif note.exists():
    note.unlink()
  CAPTIONS[TABLE_MACROS[name]] = tex(caption)


CAPTIONS = collections.OrderedDict()     # LaTeX macro name (without backslash) -> caption text (LaTeX)
TABLE_MACROS = {'TS_fu03_q3': 'capTabSFuThreeQThree', 'TS_fu03_q3diff': 'capTabSFuThreeQThreeDiff',
                'TS_fu03_memory': 'capTabSFuThreeMemory', 'TS_pert_five': 'capTabSPertFive', 'T_rules': 'capTabRules', 'TS_rules_seeds': 'capTabSRulesSeeds',
                'TS_rules_ties': 'capTabSRulesTies', 'TS_rules_tieshare': 'capTabSRulesTieShare',
                'TS_rules_rounded': 'capTabSRulesRounded', 'TS_rules_model': 'capTabSRulesModel',
                'TS_campaign_changed': 'capTabSCampChanged', 'TS_campaign_points': 'capTabSCampPoints',
                'TI_arms': 'capTabArms', 'TII_headline': 'capTabHeadline', 'TIII_q2': 'capTabQtwo',
                'TIV_followups': 'capTabFollowups', 'TS_campaign': 'capTabCampaign',
                'TS_provisioning': 'capTabProvisioning', 'TS_fu13_checks': 'capTabSFuThirteenChecks',
                'TS_fu13_p': 'capTabSFuThirteenP', 'TS_fu13_eps': 'capTabSFuThirteenEps', 'TS_fu13_l': 'capTabSFuThirteenL',
                'TS_fu13_f': 'capTabSFuThirteenF', 'TS_fu13_df': 'capTabSFuThirteenDF', 'TS_fu13_cost': 'capTabSFuThirteenCost',
                'TS_fu14_checks': 'capTabSFuFourteenChecks', 'TS_fu14_agree': 'capTabSFuFourteenAgree',
                'TS_fu14_df': 'capTabSFuFourteenDF', 'TS_fu14_sec': 'capTabSFuFourteenSec', 'TS_fu14_f': 'capTabSFuFourteenF',
                'TS_fu14_success': 'capTabSFuFourteenSuccess', 'TS_fu14_cost': 'capTabSFuFourteenCost',
                'TS_fu14_memory': 'capTabSFuFourteenMemory', 'TIII_repro': 'capTabRepro', 'TS_e_summary': 'capTabSESummary',
                'TS_returned': 'capTabSReturned', 'TS_rare': 'capTabSRare', 'TS_winners': 'capTabSWinners',
                'TS_winners_ext': 'capTabSWinnersExt', 'TS_pb_flips': 'capTabSPbFlips', 'TS_best25': 'capTabSBestTwentyFive', 'TS_gtat': 'capTabSGtat',
                'TS_fu15_rc': 'capTabSFuFifteenRC', 'TS_fu15_df': 'capTabSFuFifteenDF', 'T_beyond': 'capTabBeyond',
                'TS_fu16': 'capTabSFuSixteen', 'TS_fu17_cells': 'capTabSFuSeventeenCells',
                'TS_fu17_pairs': 'capTabSFuSeventeenPairs', 'TS_fu18': 'capTabSFuEighteen',
                'TS_fu19_rc': 'capTabSFuNineteenRC', 'TS_fu19_df': 'capTabSFuNineteenDF', 'TS_fu20': 'capTabSFuTwenty',
                'TS_fu21': 'capTabSFuTwentyOne'}


# ---------------------------------------------------------------------------------------------------------------------
# Fig. 1, design

COLMAP = {'V5': 'V5', 'V6': 'V6', 'V6h': 'V6', 'G': 'G', 'GT': 'G', 'Gd': 'G', 'Gh': 'G', 'A': 'A', 'AT': 'A', 'Ah': 'A',
          'H': 'H', 'C': 'C'}
FIG1_COLS = ['V5', 'V6', 'G', 'A', 'H', 'C']
# (key, row label, source, runs per column (2: r1 and r2, 1: one run), targets, seeds shown). Checked against the
# source in fig1_runs(); FU8's seeds 6 to 20 and FU12's seeds 3 to 5 extend the main study's seeds 1 to 5 and 1 to 2.
STUDIES = [
    ('main', 'Main study', MAINV2, {'V5': 2, 'V6': 2, 'G': 2, 'C': 1}, 60, 5),
    ('memlimit', 'Memory limit', f'{AN}/memlimit_20261014', {'V5': 1, 'V6': 1, 'G': 1}, 1, 1),
    ('ppi', 'FU1 protein-protein', f'{AN}/ppi_20261015', {'V5': 1, 'V6': 2, 'G': 2}, 30, 5),
    ('embed', 'FU2 trunk outputs', f'{AN}/embed_20261015', {'V5': 1, 'V6': 1, 'G': 2}, 10, 1),
    ('a100', 'FU3 A100', f'{AN}/a100_20261018', {'A': 2}, 60, 5),
    ('cpu60', 'FU4 CPU', f'{AN}/cpu60_20261018', {'C': 1}, 60, 2),
    ('feat', 'FU5 hosts, inputs only', f'{AN}/feat_20261018', {}, 90, 5),
    ('msa', 'FU6 official inputs', f'{AN}/msa_20261102', {'V5': 1, 'V6': 1, 'G': 2}, 60, 5),
    ('sw', 'FU7 older stack', f'{AN}/sw_20261102', {'V6': 1, 'G': 1}, 60, 5),
    ('seeds', 'FU8 seeds', f'{AN}/seeds_20261102', {'V5': 1, 'G': 1}, 60, 20),
    ('h100', 'FU9 H100', f'{AN}/h100_20261104', {'H': 2}, 60, 5),
    ('large', 'FU10 large inputs (OSF)', f'{AN}/large_20261108', {'V5': 2, 'V6': 2, 'G': 2}, 68, 5),
    ('repl', 'FU11 new targets (OSF)', f'{AN}/repl_20261106', {'V5': 2, 'V6': 2, 'G': 2}, 72, 5),
    ('prec5', 'FU12 highest precision', f'{AN}/prec5_20261107', {'V6': 1, 'G': 1}, 60, 5),
]
DEVICE_LABEL = {'TPU v5 lite': 'TPU v5e', 'TPU v6 lite': 'TPU v6e', 'NVIDIA L4': 'NVIDIA L4',
                'NVIDIA A100-SXM4-80GB': 'A100 80 GB', 'NVIDIA H100 80GB HBM3': 'H100 80 GB', 'cpu': 'CPU'}

# Reader-facing names (decision of 2026-11-11, so that a reader needs no key for the codes): the arm codes (V5, G, ...)
# and the contrast notations (HW(V6,G), RC(G), ...) stay the keys of every CSV and lookup; the LaTeX tables, captions,
# notes and figures print these names instead.
ARM_NAME = {'V5': 'v5e', 'V6': 'v6e', 'G': 'L4', 'C': 'CPU', 'A': 'A100', 'H': 'H100', 'GT': 'L4 Triton',
            'Gd': 'L4 deterministic', 'V6h': 'v6e highest', 'Gh': 'L4 highest', 'Ah': 'A100 highest', 'AT': 'A100 Triton',
            'V6f': 'v6e float32', 'Gf': 'L4 float32', 'V5o': 'v5e official', 'V6o': 'v6e official', 'Go': 'L4 official',
            'V6s': 'v6e older stack', 'Gs': 'L4 older stack', 'V5f': 'v5e float32', 'Af': 'A100 float32',
            'Hf': 'H100 float32', 'Ad': 'A100 deterministic', 'Hd': 'H100 deterministic'}
CONTRAST_KIND = {'RC': 'Recompile', 'SD': 'Seeds', 'PR': 'Precision', 'W': 'Warm rerun', 'IN': 'Inputs',
                 'SW': 'Software stack'}
CONTRAST_FIXED = {'DU': 'Default attention, L4', 'DET': 'Determinism flag, L4'}
NOTATION = re.compile(r'HWh?\(\w+,\w+\)(?: minus RC\(\w+\))?|\b(?:RC|SD|PR|W|IN|SW)\(\w+\)|\bDU\b|\bDET\b')
LOWER_START = tuple(CONTRAST_KIND.values()) + tuple(CONTRAST_FIXED.values())


def clab(c):
  """Contrast key to its table and figure label: HW(V6,G) -> 'v6e vs L4', HWh(V6,G) -> 'v6e highest vs L4 highest',
  RC(G) -> 'Recompile, L4', HW(V6,G) minus RC(G) -> 'v6e vs L4 minus recompile L4'."""
  c = str(c).strip()
  m = re.fullmatch(r'(.+?) minus RC\((\w+)\)(.*)', c)
  if m:
    return f'{clab(m.group(1))} minus recompile {ARM_NAME[m.group(2)]}{m.group(3)}'
  m = re.fullmatch(r'HWh\((\w+),(\w+)\)', c)
  if m:
    return f'{ARM_NAME[m.group(1) + "h"]} vs {ARM_NAME[m.group(2) + "h"]}'
  m = re.fullmatch(r'HW\((\w+),(\w+)\)', c)
  if m:
    return f'{ARM_NAME[m.group(1)]} vs {ARM_NAME[m.group(2)]}'
  m = re.fullmatch(r'(RC|SD|PR|W|IN|SW)\((\w+)\)', c)
  if m:
    return f'{CONTRAST_KIND[m.group(1)]}, {ARM_NAME[m.group(2)]}'
  if c in CONTRAST_FIXED:
    return CONTRAST_FIXED[c]
  raise KeyError(f'no reader label for {c!r}')


def reader_label(s):
  """A table cell or header in reader-facing names: a cell that is an arm code becomes the arm's name; every contrast
  notation inside the text becomes its label, capitalized only at the start of the cell."""
  s = str(s)
  if s in ARM_NAME:
    return ARM_NAME[s]

  def rep(m):
    lab = clab(m.group(0))
    return lab if m.start() == 0 or not lab.startswith(LOWER_START) else lab[0].lower() + lab[1:]
  return NOTATION.sub(rep, s)


# Reading note R93 (2026-11-12): a pair "X vs Y" in a caption or a table note never breaks across two lines. Every space
# inside the pair, the arms' own included (v6e highest vs L4 highest), becomes a non-breaking space (~). Arm names are
# tried longest first, so "L4 Triton" binds whole. Labels inside figures are not touched.
PAIR_NAMES = sorted(set(ARM_NAME.values()), key=len, reverse=True)
PAIR = re.compile(r'(?<![\w~])(?:' + '|'.join(re.escape(n) for n in PAIR_NAMES) + r') vs\.? (?:'
                  + '|'.join(re.escape(n) for n in PAIR_NAMES) + r')(?![\w])')


def tie_pairs(s):
  """LaTeX text with every "X vs Y" pair of arm names bound by non-breaking spaces."""
  return PAIR.sub(lambda m: m.group(0).replace(' ', '~'), s)


def pair_name(s):
  """A pair as follow-up 13's outputs write it ('V5-V6', 'G-Gr2' for the L4's second repetition, or a contrast
  notation) in reader-facing names; anything else is returned unchanged."""
  s = str(s)
  if NOTATION.fullmatch(s):
    return clab(s)
  m = re.fullmatch(r'(\w+?)-(\w+?)', s)
  if m and m.group(1) in ARM_NAME and m.group(2) == m.group(1) + 'r2':
    return f'Recompile, {ARM_NAME[m.group(1)]}'
  if m and m.group(1) in ARM_NAME and m.group(2) in ARM_NAME:
    return f'{ARM_NAME[m.group(1)]} vs {ARM_NAME[m.group(2)]}'
  return s


FU_LABEL = {'FU6': 'msa', 'FU7': 'sw', 'FU8': 'seeds', 'FU9': 'h100', 'FU10': 'large', 'FU11': 'repl', 'FU12': 'prec5'}


def recorded_name(item, fu):
  """An item of the recorded spend ('Go r1', 'V6s', 'H r2', 'Gh5', 'smoke tests') with the arm in reader-facing names."""
  tok, _, rest = item.partition(' ')
  try:
    name = prov_arm_name(tok, FU_LABEL.get(fu, ''))
  except KeyError:
    return item
  return f'{name} {rest}'.strip()


def prov_arm_name(arm, label):
  """Arm of the provisioning table: the base arm's name; follow-ups 6 and 7 add their variant, the others are named by
  the group heading (FU8 V5x, FU10 V5l, FU11 V5r, FU12 V6h5 print v5e, v5e, v5e, v6e highest)."""
  suf = ARM_SUFFIX.get(label, '')
  base = arm[:-len(suf)] if suf and arm.endswith(suf) else arm
  name = ARM_NAME[base]
  return {'msa': f'{name} official', 'sw': f'{name} older stack'}.get(label, name)


def runs_from_samples(s):
  """Column -> 2 if any arm in it has an r2 run, else 1 (warm checks 'w' are not runs of the design)."""
  got = {}
  for (p, r), _ in s.groupby(['platform', 'repetition']):
    if r not in ('r1', 'r2'):
      continue
    c = COLMAP[p]
    got[c] = max(got.get(c, 0), 2 if r == 'r2' else 1)
  return got


def fig1_runs(samples, other):
  """The STUDIES constant, checked against samples.csv of each folder (and the other sources); returns its rows."""
  rows = []
  for key, label, src, runs, targets, seeds in STUDIES:
    if key in samples:
      s = samples[key]
      got, t, smax = runs_from_samples(s), s.target.nunique(), int(s.seed.max())
      if key == 'main':
        c = s[s.platform == 'C']
        assert c.target.nunique() == 6 and sorted(c.seed.unique()) == [1, 2], 'main C: 6 targets, seeds 1 and 2'
      if key == 'cpu60':   # the 6 main-study C targets keep their main-study runs (row of 2026-10-16)
        six = set(samples['main'][samples['main'].platform == 'C'].target)
        assert not six & set(s.target)
        t = len(six | set(s.target))
    elif key == 'memlimit':
      p = other['memlimit']
      got = {COLMAP[a]: 1 for a in p.arm.unique()}
      t, smax = 1, 1
      assert set(p.source[p.source.str.startswith('main study')].str.extract(r'\((\w{4}),')[0]) == {'7D5C'}
    elif key == 'embed':
      v = other['embed']
      got = {}
      for (cnd, rep), _ in v.groupby(['condition', 'repetition']):
        got[COLMAP[cnd]] = max(got.get(COLMAP[cnd], 0), 2 if rep == 'r2' else 1)
      t, smax = v.target.nunique(), 1
    elif key == 'feat':
      dec = other['feat_decision']
      assert dec['units'] == targets * seeds == 450 and len(other['feat_hosts']) == 4
      got, t, smax = {}, targets, seeds
    if got != runs or t != targets or smax != seeds:
      raise SystemExit(f'Fig. 1 row {key}: constant {runs, targets, seeds} against source {got, t, smax}')
    rows.append({'study': label, **{c: {2: 'r1 and r2', 1: 'one run'}.get(runs.get(c, 0), '') for c in FIG1_COLS},
                 'targets': targets, 'seeds': seeds, 'source': src})
  return pd.DataFrame(rows)


def table_runs(out, runs, main_samples, con, large_samples):
  dev = main_samples.groupby('platform').agg(device=('device_kind', 'first'), machine=('machine_type', 'first'))
  extra = {k: v for k, v in runs.items() if k in ('A', 'H')}
  nodes = {}
  for p, pos in {'V5': (0.72, 2.05), 'V6': (0.72, 0.95), 'C': (2.85, 2.40), 'G': (2.85, 1.72), 'A': (2.85, 1.04),
                 'H': (2.85, 0.36)}.items():
    d, m = (dev.loc[p, 'device'], dev.loc[p, 'machine']) if p in dev.index else extra[p]
    nodes[p] = (*pos, DEVICE_LABEL[d], m)
  prim = set(con[(con.role == 'primary pair') & (con.kind == 'HW')].contrast)
  if prim != {'HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)'}:
    raise SystemExit(f'primary pairs in contrasts.csv: {sorted(prim)}')
  # Fig. 1 of the earlier drafts (runs per study, drawn) is part (b) of Table I since 2026-11-11 (reading note R58):
  # a tabular with the devices as column heads and targets x seeds in the row label (R59). Panel (a) of the first
  # two-panel figure (devices, machine types, pairs) is part (a), the arms (TI_arms); the nodes stay in fig1_nodes.
  mark = {'r1 and r2': '$\\bullet$', 'one run': '$\\circ$', '': ''}
  lines = ['% TI_runs, runs per study (supplement). Generated by analysis/paper_assets_isc.py; do not edit.',
           f'% Role: {ROLE_DESIGN}',
           '\\begin{tabular}{@{}l' + 'c' * len(FIG1_COLS) + '@{}}', '\\toprule',
           'study (targets $\\times$ seeds) & ' + ' & '.join(ARM_NAME[c] for c in FIG1_COLS) + ' \\\\', '\\midrule']
  for i, r in enumerate(runs['table'].itertuples()):
    lines.append(f'{tex(r.study)}, {r.targets} $\\times$ {r.seeds} & ' + ' & '.join(mark[getattr(r, c)] for c in FIG1_COLS)
                 + ' \\\\')
    if i == 0:      # the main study apart from the follow-ups (R56, R59)
      lines.append('\\midrule')
  lines += ['\\bottomrule', '\\end{tabular}']
  body = '\n'.join(lines) + '\n'
  no_em_dash(body, 'TI_runs')
  (out / 'tables').mkdir(parents=True, exist_ok=True)
  (out / 'tables' / 'TI_runs.tex').write_text(body)
  ASSETS['tables/TI_runs.tex'] = (ROLE_DESIGN, 'supplementary table: runs per study (filled: r1 and r2; open: one run)')
  # Option B of 2026-11-11 (after R58): Table I of the main text is the arms table (TI_arms) with the note TI_main;
  # the runs per study (TI_runs) are a supplementary table with its own title and note
  mc = main_samples[main_samples.platform == 'C']
  ls = large_samples
  r2, nt = ls[(ls.platform == 'V5') & (ls.repetition == 'r2')].target.nunique(), ls.target.nunique()
  assert r2 == ls[(ls.platform == 'V6') & (ls.repetition == 'r2')].target.nunique()
  CAPTIONS['capTabRuns'] = 'Runs per study'
  runs_note = ('A filled marker ($\\bullet$) means two separately compiled runs, which measure recompilation, and an '
               'open marker ($\\circ$) a single run. In the main study, the CPU ran '
               f'{mc.target.nunique()} targets with {mc.seed.nunique()} seeds. In FU10, the second run of the v5e and the '
               f'v6e covers {r2} of {nt} targets. Follow-ups (FU) lie outside the pre-registered study, and we specified '
               'them after we had read the main results. FU10 and FU11 have OSF addenda registered before their runs.')
  # Options round of 2026-11-19 (OPZIONI_DIDASCALIE.md, Table I B): the main paper's note in three sentences; the
  # supplement's TI_arms keeps the full note. An arm with two prices (0.85/0.90, 5.03/5.54) ran at two regional list
  # prices; table_arms() stops if no arm has two prices any more
  main_note = ('Each variant arm differs from its device\'s base arm in one setting. USD/h is the on-demand list price, '
               'and two prices mean two regions (Supplementary Section~\\ref{S-sec:s-main-q2}). ' + MEMORY_NOTE)
  for name, body_, what in (('TI_runs', runs_note, 'note below the supplementary table of runs per study'),
                            ('TI_main', main_note, 'note below Table I of the main text (arms)')):
    no_em_dash(body_, f'{name}_note')
    (out / 'tables' / f'{name}_note.tex').write_text(
        f'% {name}, note. Generated by analysis/paper_assets_isc.py; do not edit.\n' + tie_pairs(body_) + '\n')
    ASSETS[f'tables/{name}_note.tex'] = (ROLE_DESIGN, what)
  write_csv(out, 'fig1_runs', runs['table'], ROLE_DESIGN, 'CSV twin of Fig. 1(b): runs per study and column')
  nd = pd.DataFrame([{'arm': k, 'device': v[2], 'machine_type': v[3]} for k, v in nodes.items()])
  write_csv(out, 'fig1_nodes', nd, ROLE_DESIGN, 'device and machine type per main-study platform (Fig. 1(a) until 2026-11-11)')


# ---------------------------------------------------------------------------------------------------------------------
# Fig. 2, forest

FOLDERS = {'main': f'{MAINV2}/contrasts', 'repl': f'{AN}/repl_20261106', 'msa': f'{AN}/msa_20261102',
           'a100': f'{AN}/a100_20261018', 'h100': f'{AN}/h100_20261104', 'cpu': f'{AN}/cpu60_20261018',
           'prec': f'{AN}/prec5_20261107', 'large': f'{AN}/large_20261108', 'ppi': f'{AN}/ppi_20261015'}
FILTERS = {'main': {}, 'repl': {'targets': 'all 72'}, 'msa': {'targets': 'all 60'}, 'a100': {}, 'h100': {'targets': 'all 60'},
           'cpu': {}, 'prec': {'targets': 'all 60', 'seeds': '1 to 5'}, 'large': {'targets': 'all'}, 'ppi': {}}
# (group name, source, pair, extra filter, arms as (samples folder, platform) for the dagger: one arm has a single run)
# Order: the main study first, then the follow-ups in numerical order
FOREST = [
    ('Main', 'main', 'HW(V5,V6)', {}, [('main', 'V5'), ('main', 'V6')]),
    ('Main', 'main', 'HW(V6,G)', {}, [('main', 'V6'), ('main', 'G')]),
    ('Main', 'main', 'HW(V5,G)', {}, [('main', 'V5'), ('main', 'G')]),
    ('Protein-protein', 'ppi', 'HW(V5,V6)', {}, [('ppi', 'V5'), ('ppi', 'V6')]),
    ('Protein-protein', 'ppi', 'HW(V6,G)', {}, [('ppi', 'V6'), ('ppi', 'G')]),
    ('A100', 'a100', 'HW(G,A)', {}, [('main', 'G'), ('a100', 'A')]),
    ('A100', 'a100', 'HW(V6,A)', {}, [('main', 'V6'), ('a100', 'A')]),
    ('CPU', 'cpu', 'HW(V5,C)', {}, [('main', 'V5'), ('cpu60', 'C')]),
    ('CPU', 'cpu', 'HW(V6,C)', {}, [('main', 'V6'), ('cpu60', 'C')]),
    ('CPU', 'cpu', 'HW(G,C)', {}, [('main', 'G'), ('cpu60', 'C')]),
    ('Official inputs', 'msa', 'HW(V5o,V6o)', {}, [('msa', 'V5'), ('msa', 'V6')]),
    ('Official inputs', 'msa', 'HW(V6o,Go)', {}, [('msa', 'V6'), ('msa', 'G')]),
    ('H100', 'h100', 'HW(H,A)', {}, [('h100', 'H'), ('a100', 'A')]),
    ('Large', 'large', 'HW(V5,V6)', {'group': 'large'}, [('large', 'V5'), ('large', 'V6')]),
    ('Large', 'large', 'HW(V6,G)', {'group': 'large'}, [('large', 'V6'), ('large', 'G')]),
    ('Size control', 'large', 'HW(V5,V6)', {'group': 'control'}, [('large', 'V5'), ('large', 'V6')]),
    ('Size control', 'large', 'HW(V6,G)', {'group': 'control'}, [('large', 'V6'), ('large', 'G')]),
    ('Replication', 'repl', 'HW(V5,V6)', {}, [('repl', 'V5'), ('repl', 'V6')]),
    ('Replication', 'repl', 'HW(V6,G)', {}, [('repl', 'V6'), ('repl', 'G')]),
    ('Precision', 'prec', 'HWh(V6,G)', {}, [('prec5', 'V6h'), ('prec5', 'Gh')]),
]
STATS = [('flip_top1', '(a) Top-1 flip', (0, 40)), ('flip_success', '(b) Success flip', (0, 10.5))]   # x range (R89)
STAT_TICKS = {'flip_top1': range(0, 41, 10), 'flip_success': range(0, 11, 2)}


def rc_row(con, src, rc, stat, extra):
  """An RC estimate: RC(A) from the A100 folder, RC(H) from the H100 folder, otherwise the same folder if it has the
  row, else the main study; checked against HW minus the difference in forest_rows()."""
  if rc == 'RC(A)':
    return pick(con['a100'], contrast='RC(A)', statistic=stat), 'a100'
  if rc == 'RC(H)':
    return pick(con['h100'], contrast='RC(H)', statistic=stat, targets='all 60'), 'h100'
  c = con[src]
  if (c.contrast == rc).any():
    return pick(c, contrast=rc, statistic=stat, **FILTERS[src], **extra), src
  return pick(con['main'], contrast=rc, statistic=stat), 'main'


def forest_rows(con, hwrc, has_r2):
  """Long table of every value drawn in Fig. 2 (fractions, as in the sources), with the display rule: per row and per
  statistic, the listed RC with the larger estimate; no RC drawn if that RC is 0.0 [0.0, 0.0]."""
  rows = []
  for group, src, pair, extra, arms in FOREST:
    dag = '†' if any(not has_r2.get(a, False) for a in arms) else ''
    for stat, _, _ in STATS:
      h = pick(con[src], contrast=pair, statistic=stat, **FILTERS[src], **extra)
      n_t, n = int(h.n_targets), int(h.n)
      base_row = {'group': group, 'targets': n_t, 'seeds': n // n_t, 'pair': pair, 'dagger': dag, 'statistic': stat}
      rows.append({**base_row, 'kind': 'HW', 'reference': '', 'estimate': h.estimate, 'ci_low': h.ci_low,
                   'ci_high': h.ci_high, 'source': FOLDERS[src] + '/contrasts.csv'})
      hw = hwrc[src]
      m = hw[(hw.hardware == pair) & (hw.statistic == stat)]
      for k, v in {**FILTERS[src], **extra}.items():
        if k in m.columns:
          m = m[m[k] == v]
      cands = list(dict.fromkeys(m.recompilation))
      if not cands:
        continue
      got = {rc: rc_row(con, src, rc, stat, extra) for rc in cands}
      best = max(cands, key=lambda k: (float(got[k][0].estimate), -cands.index(k)))
      r, rsrc = got[best]
      if float(r.estimate) == 0 and float(r.ci_low) == 0 and float(r.ci_high) == 0:
        continue
      d = m[m.recompilation == best]
      assert len(d) == 1, (pair, best, stat)
      d = d.iloc[0]
      if abs((float(h.estimate) - float(d.estimate)) - float(r.estimate)) > 1e-9:
        raise SystemExit(f'Fig. 2 {group} {pair} {stat}: HW minus the difference is not {best} of {rsrc}')
      rows.append({**base_row, 'kind': 'RC', 'reference': best, 'estimate': r.estimate, 'ci_low': r.ci_low,
                   'ci_high': r.ci_high, 'source': FOLDERS[rsrc] + '/contrasts.csv'})
      rows.append({**base_row, 'kind': 'HW minus RC', 'reference': best, 'estimate': d.estimate, 'ci_low': d.ci_low,
                   'ci_high': d.ci_high, 'source': FOLDERS[src] + '/hw_minus_rc.csv'})
  return pd.DataFrame(rows)


# group names as drawn (the CSV twin keeps the key), the same study names as Table I(b)
FOREST_GROUP_SHOWN = {'Main': 'Main study', 'Replication': 'FU11 new targets', 'Official inputs': 'FU6 official inputs',
                      'A100': 'FU3 A100', 'H100': 'FU9 H100', 'CPU': 'FU4 CPU', 'Precision': 'FU12 highest precision',
                      'Large': 'FU10 large inputs', 'Size control': 'FU10 size control',
                      'Protein-protein': 'FU1 protein-protein'}


def forest_label(pair):
  """Row label of Fig. 2: the pair in reader-facing names; in the groups that name the variant (official inputs,
  highest precision) the base platforms, so the label fits beside the group name."""
  return clab(re.sub(r'\b(V5|V6|G)o\b', r'\1', pair).replace('HWh(', 'HW('))


LEGEND_GAP_PT = 8.0      # Fig. 1 legend: the same gap between its three entries (R95); 14 until note 14 of 2026-11-17


FOREST_ALIGN_TOL_PT = 0.1


def text_baseline_pt(t, renderer, dpi):
  """Baseline of a drawn one-line Text, in points from the figure's bottom: its extent's bottom plus the descent of its
  line as matplotlib laid it out (at least the descent of "lp", the same for every 8 pt label)."""
  _, info, _ = t._get_layout(renderer)
  if len(info) != 1:
    raise SystemExit(f'Fig. 1: label {t.get_text()!r} is not one line')
  return (t.get_window_extent(renderer).y0 + info[0][1][2]) * 72 / dpi


def forest_check_alignment(fig, ax, study_labels, gspan, ys, keys):
  """R119b: after drawing, each study label's baseline equals the pair label's (one-row group) or lies midway between
  the baselines of the group's first and last pair labels; stops beyond FOREST_ALIGN_TOL_PT. Returns the offsets."""
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  ticks = dict(zip(keys, ax.get_yticklabels()))
  off = {}
  for g, (a, b) in gspan.items():
    first = [k for k in keys if k[0] == g and ys[k] == a][0]
    last = [k for k in keys if k[0] == g and ys[k] == b][0]
    want = (text_baseline_pt(ticks[first], r, fig.dpi) + text_baseline_pt(ticks[last], r, fig.dpi)) / 2
    off[g] = text_baseline_pt(study_labels[g], r, fig.dpi) - want
    if abs(off[g]) > FOREST_ALIGN_TOL_PT:
      raise SystemExit(f'Fig. 1: the label of {g} is {off[g]:+.3f} pt off its rows (tolerance {FOREST_ALIGN_TOL_PT})')
  FOREST_ALIGN.clear()
  FOREST_ALIGN.update(off)
  print('NOTE Fig. 1 label alignment (study label baseline minus its rows, pt): '
        + ', '.join(f'{g} {v:+.3f}' for g, v in off.items()), flush=True)
  return off


FOREST_ALIGN = {}


def fig_forest(out, save, plt, fr):
  """Fig. 2 (reading note R56, 2026-11-11): flip rate per pair (circle, 95% interval) over the larger recompilation
  rate of the pair (line, with its interval as a band); one x scale for both panels; no difference markers (the
  differences stay in the CSV twin, Table II and the supplement); the main study set apart from the follow-ups.
  R119 (2026-11-16): each study label sits on the row of the first pair of its group (it was centred on the group, so
  in groups of two or three rows it fell between two pairs); a light rule (RULE_LIGHT, 0.4 pt) separates every two
  consecutive follow-up groups, at the middle of the gap like the rule after the main study (0.5 pt).
  With bands of 0.84 rows and a gap of 0.30 rows, each rule clears the nearest band by 0.23 rows (about 2 pt).
  R119b (2026-11-16): the light rules run in the label area only, from the figure's left edge to just before the first
  panel (RULE_LIGHT, 0.4 pt), so no rule between follow-up groups crosses the panels; the black rule after the main
  study stays across the label area and both panels. Each study label is again centred on its group, with the y tick
  labels' vertical alignment (va='center_baseline'): in a one-row group its baseline is the pair label's baseline, in
  a group of several rows it lies midway between the baselines of the first and the last row. The generator measures
  both after drawing and stops if either is off by more than 0.1 pt (FOREST_ALIGN_TOL_PT).
  Changes of 2026-11-17: the follow-ups in numerical order after the main study; the rule after the
  main study grey (RULE_MAIN, a little darker than RULE_LIGHT) instead of black; the legend's gap 8 pt instead of 14,
  because at 14 pt the PDF cut its two ends (save() now also measures with the PDF backend)."""
  from matplotlib.patches import Rectangle
  from matplotlib.lines import Line2D
  keys = list(dict.fromkeys(zip(fr.group, fr.pair)))
  groups = list(dict.fromkeys(fr.group))
  if groups[0] != 'Main':
    raise SystemExit('Fig. 2: the main study is not the first group')
  drawn = fr[fr.kind != 'HW minus RC']
  for stat, _, (lo, hi) in STATS:
    x = drawn[drawn.statistic == stat]
    if 100 * x.ci_high.max() > hi or 100 * x.ci_low.min() < lo:
      raise SystemExit(f'Fig. 1: an interval of {stat} leaves the scale {lo} to {hi}')
  y, ys, gap, gspan = 0.0, {}, 0.30, {}
  for g in groups:
    s = y
    for k in keys:
      if k[0] == g:
        ys[k] = y
        y += 1
    gspan[g] = (s, y - 1)
    y += gap
  rowh = 0.13     # 9.4 pt per row for 8 pt row labels (2026-11-12: 0.14 made the named version overflow)
  top_in, bot_in = 0.62, 0.42     # R89: room between the legend and the panel titles; x labels not cut
  H = rowh * (y - gap) + top_in + bot_in
  fig, axes = plt.subplots(1, 2, figsize=(7.16, H), sharey=True)
  fig.subplots_adjust(left=0.375, right=0.985, top=1 - top_in / H, bottom=bot_in / H, wspace=0.10)
  for ax, (stat, title, xl) in zip(axes, STATS):
    f = fr[fr.statistic == stat]
    for k, yy in ys.items():
      r = f[(f.group == k[0]) & (f.pair == k[1])]
      hw = r[r.kind == 'HW'].iloc[0]
      if (r.kind == 'RC').any():
        rc = r[r.kind == 'RC'].iloc[0]
        ax.add_patch(Rectangle((100 * rc.ci_low, yy - 0.42), 100 * (rc.ci_high - rc.ci_low), 0.84, fc=BAND, ec='none',
                               zorder=0))
        ax.plot([100 * rc.estimate] * 2, [yy - 0.42, yy + 0.42], color=INK, lw=0.8, zorder=1)
      ax.plot([100 * hw.ci_low, 100 * hw.ci_high], [yy] * 2, color=NAVY, lw=1.0, zorder=2, solid_capstyle='butt')
      ax.plot(100 * hw.estimate, yy, marker='o', ms=3.8, color=NAVY, mec='white', mew=0.3, ls='none', zorder=3)
    ax.set_xlim(*xl)
    ax.set_xticks(list(STAT_TICKS[stat]))
    ax.set_title(title, loc='left', pad=13, fontstyle='italic')     # one line above the column heads (R89)
    ax.set_xlabel('Predictions with a flip (%)')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
  axes[0].set_ylim(max(ys.values()) + 0.6, -0.6)
  first = fr.drop_duplicates(['group', 'pair'])
  axes[0].set_yticks([ys[k] for k in keys], [f'{forest_label(p)}{d}' for p, d in zip(first.pair, first.dagger)])
  axes[0].tick_params(axis='y', labelsize=8)
  study_labels = {}
  for g, (a, b) in gspan.items():
    r = fr[fr.group == g].iloc[0]
    study_labels[g] = axes[0].annotate(f'{FOREST_GROUP_SHOWN[g]}, {r.targets} × {r.seeds}', xy=(0, (a + b) / 2),
                                       xycoords=('figure fraction', 'data'), xytext=(2, 0), textcoords='offset points',
                                       ha='left', va='center_baseline', fontsize=8)
  # column heads of the label area, level with the panel titles
  axes[0].annotate('Study (targets × seeds)', xy=(0, 1), xycoords=('figure fraction', 'axes fraction'), xytext=(2, 2),
                   textcoords='offset points', ha='left', va='bottom', fontsize=8, fontstyle='italic')     # R94
  axes[0].annotate('Pair', xy=(0, 1), xycoords='axes fraction', xytext=(-4, 2), textcoords='offset points', ha='right',
                   va='bottom', fontsize=8, fontstyle='italic')
  # rules at the middle of each gap between groups (R119b): grey after the main study (RULE_MAIN, note 14b of
  # 2026-11-17; it was black), across the label area and both
  # panels; light between follow-ups, in the label area only; none above the first group or below the last
  for g in groups[:-1]:
    ysep = gspan[g][1] + 0.5 + gap / 2
    yf = fig.transFigure.inverted().transform(axes[0].transData.transform((0, ysep)))[1]
    if g == 'Main':     # this rule crosses the panels, so it must clear the bands on both sides
      if ysep - gspan[g][1] - 0.42 <= 0 or gspan[groups[1]][0] - 0.42 - ysep <= 0:
        raise SystemExit('Fig. 1: the main-study rule touches a band; widen the gap')
      fig.add_artist(Line2D([0.0, 0.985], [yf, yf], transform=fig.transFigure, color=RULE_MAIN, lw=0.5))
    else:
      fig.add_artist(Line2D([0.0, fig.subplotpars.left - 0.012], [yf, yf], transform=fig.transFigure, color=RULE_LIGHT,
                            lw=0.4))
  forest_check_alignment(fig, axes[0], study_labels, gspan, ys, keys)
  # Legend (R95, 2026-11-12): three entries at equal gaps, centred on the figure. The dagger entry has no handle (an
  # empty handle made its gap look larger); the two others draw their key as in the panels. Text 8 pt.
  from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
  hw_, hh_ = 16.0, 8.0                                   # key size in points (matplotlib's default handle at 8 pt)

  def key(draw):
    da = DrawingArea(hw_, hh_, 0, 0)
    for a in draw:
      da.add_artist(a)
    return da
  k_pair = key([Line2D([0, hw_], [hh_ / 2] * 2, color=NAVY, lw=1.0),
                Line2D([hw_ / 2], [hh_ / 2], marker='o', ms=3.8, color=NAVY, mec='white', mew=0.3, ls='none')])
  k_rc = key([Rectangle((0, 0), hw_, hh_, fc=BAND, ec='none'), Line2D([hw_ / 2] * 2, [0, hh_], color=INK, lw=0.8)])
  txt = lambda t: TextArea(t, textprops=dict(fontsize=8))  # noqa: E731
  entries = [HPacker(children=[k_pair, txt('Platform pair (95% interval)')], sep=3.2, align='center'),
             HPacker(children=[k_rc, txt('Recompilation alone on the same platform (line: rate; band: 95% interval)')],
                     sep=3.2, align='center'),
             txt('† At least one arm ran once')]
  legend = AnchoredOffsetbox(loc='upper center', child=HPacker(children=entries, sep=LEGEND_GAP_PT, align='center'),
                             bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, frameon=False, pad=0.0,
                             borderpad=0.1)
  fig.add_artist(legend)
  save(fig, 'fig2_forest', 'main study and follow-ups (rows other than Main are outside the pre-registered study)',
       'Fig. 2: HW flip rates over the larger RC (line and band), every study, top-1 and success')
  out_df = fr.copy()
  write_csv(out, 'fig2_forest', out_df, 'main study and follow-ups', 'CSV twin of Fig. 2 (fractions as in the sources; '
            'the HW minus RC rows are not drawn since 2026-11-11)')
  return H


# ---------------------------------------------------------------------------------------------------------------------
# Fig. 3, ligand RMSD ECDF

# The seed yardstick SD(V6) is dash-dotted: drawn solid, it differed from HW(V5,V6) only by color (style round of
# 2026-11-20: every series reads in grayscale)
ECDF_SERIES = [('RC(G)', INK, ':'), ('HW(V5,V6)', NAVY, '-'), ('HW(V6,G)', ROSE, '--'),
               ('SD(V6)', INK, (0, (6, 1.5, 1, 1.5)))]
ZERO_AT = 1e-4


def ecdf_quantiles(pairs):
  rows = []
  for c, _, _ in ECDF_SERIES:
    x = pairs[pairs.contrast == c].m4_ligand_rmsd.dropna().to_numpy()
    rows.append({'contrast': c, 'n': len(x), 'zeros': int((x == 0).sum()), 'zero_share': float((x == 0).mean()),
                 'median': float(np.median(x)), 'p90': float(np.percentile(x, 90)), 'max': float(x.max()),
                 'at_least_1A': int((x >= 1).sum()), 'at_least_2A': int((x >= 2).sum())})
  return pd.DataFrame(rows)


def fig_rmsd_ecdf(out, save, plt, pairs):
  fig, ax = plt.subplots(figsize=(3.5, 2.6))
  fig.subplots_adjust(left=0.135, right=0.975, top=0.84, bottom=0.17)
  for c, col, ls in ECDF_SERIES:
    x = np.sort(np.maximum(pairs[pairs.contrast == c].m4_ligand_rmsd.dropna().to_numpy(), ZERO_AT))
    ax.step(x, np.arange(1, len(x) + 1) / len(x), color=col, ls=ls, lw=1.1, where='post', label=clab(c))
  ax.set_xscale('log')
  plain_log_ticks(ax.xaxis)
  ax.set_xlim(ZERO_AT, 12)
  ax.set_ylim(0, 1.0)
  ax.set_xlabel('Ligand RMSD between paired samples (Å)')
  ax.set_ylabel('Fraction of sample pairs')
  fig.legend(loc='upper center', ncol=2, handlelength=2.4, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)
  ylabel_top(fig, ax, 'Fraction of sample pairs')
  save(fig, 'fig3_rmsd_ecdf', 'main study (primary pairs, RC(G) and the seed yardstick SD(V6))',
       'Fig. 3: ECDF of the ligand RMSD between paired samples')
  q = ecdf_quantiles(pairs)
  write_csv(out, 'fig3_rmsd_quantiles', q, 'main study', 'CSV twin of Fig. 3: n, zeros, median, p90, max, and the sample '
            'pairs with ligand RMSD of at least 1 and at least 2 Å (at_least_1A, at_least_2A; post hoc counts) per series '
            f'(the ECDF is every value of {MAINV2}/contrasts/pairs_sample.csv m4_ligand_rmsd)')
  return q


# ---------------------------------------------------------------------------------------------------------------------
# Fig. 4, margin (exploratory)

MARGIN_SERIES = [('HW(V6,G)', ROSE, 'D', 1.08), ('HW(V5,V6)', NAVY, 'o', 1.0), ('RC(G)', INK, 's', 0.925)]


def fig_margin(out, save, plt, quint):
  fig, ax = plt.subplots(figsize=(3.5, 2.35))
  fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.19)     # 0.17 cut the x label (R89)
  rows = []
  for c, col, m, f in MARGIN_SERIES:
    d = quint[quint.contrast == c].sort_values('quintile')
    x = d.margin_median * f
    ax.errorbar(x, 100 * d.flip_rate, yerr=[100 * (d.flip_rate - d.ci_low), 100 * (d.ci_high - d.flip_rate)], color=col,
                marker=m, ms=3.6, lw=1, ls=(0, (3, 1.5)) if c == 'HW(V6,G)' else '-',   # grayscale (2026-11-19) elinewidth=0.8, capsize=0, mfc='white' if c == 'RC(G)' else col,
                mec=col if c == 'RC(G)' else 'white', mew=0.8 if c == 'RC(G)' else 0.3, label=clab(c))
    rows += [{'contrast': c, 'quintile': int(r.quintile), 'margin_median': r.margin_median, 'x_drawn': r.margin_median * f,
              'n': int(r.n), 'flips': int(r.flips), 'flip_rate': r.flip_rate, 'ci_low': r.ci_low, 'ci_high': r.ci_high}
             for r in d.itertuples()]
  ax.set_xscale('log')
  plain_log_ticks(ax.xaxis)
  ax.set_xlim(1e-4, 1.5e-2)
  ax.set_ylim(0, 50)
  ax.set_xlabel('Top-1 minus top-2 ranking score (quintile median)')
  ax.set_ylabel('Top-1 flip rate (%)')
  h, lab = ax.get_legend_handles_labels()
  order = [lab.index(clab(c)) for c in ('HW(V5,V6)', 'HW(V6,G)', 'RC(G)')]
  ax.legend([h[i] for i in order], [lab[i] for i in order], loc='upper right', handletextpad=0.4, borderaxespad=0.1,
            title='Exploratory, post hoc')
  ylabel_top(fig, ax, 'Top-1 flip rate (%)')
  save(fig, 'fig4_margin_exploratory', ROLE_EXPL, 'Supplementary figure (Fig. 4 of an earlier draft): per-seed top-1 '
       'flip rate by quintile of the ranking margin')
  write_csv(out, 'fig4_margin', pd.DataFrame(rows), ROLE_EXPL,
            f'CSV twin of the margin figure ({EXPL}/a_margin_quintiles.csv; x_drawn is the offset position, a label only)')


# ---------------------------------------------------------------------------------------------------------------------
# Supplementary figure of the exploratory analysis (c), in the main text until 2026-11-12 (reading notes R96 to R99): flip rate against perturbation over margin (Table S19 / Fig. S10
# promoted, task of 2026-11-10, A1). Every point is a row of c_perturbation.csv; the line is the model of
# analysis/exploratory.py (predicted_flip_rate), which depends on delta / mu only, drawn over the range of the points.

PERT_HW = ['HW(V5,V6)', 'HW(V6,G)', 'HW(V6,GT)', 'HWh(V6,G)']       # platform pairs (HWh: both at highest precision)
PERT_ONE = ['RC(G)', 'DET', 'DU', 'PR(V6)', 'PR(G)']              # arithmetic changes on one platform
# Points are numbered in order of the x value (reading note R70, 2026-11-11: the labels with leader lines read as
# error bars); the key was in the caption until the options round of 2026-11-19 and is now drawn under the x axis
# (PERT_KEY, in the words of the figures' legends). Each number sits above the point's interval ('above') or below it ('below'),
# where a neighbouring interval leaves no room above.
PERT_NUM_POS = {'RC(G)': 'above', 'DET': 'above', 'DU': 'above', 'PR(G)': 'below', 'HW(V5,V6)': 'above',
                'PR(V6)': 'above', 'HWh(V6,G)': 'above', 'HW(V6,G)': 'below', 'HW(V6,GT)': 'above'}
PERT_WORDS = {'RC(G)': 'recompilation on the L4', 'DET': 'the determinism flag on the L4',
              'DU': 'default attention on the L4', 'PR(G)': 'the precision setting on the L4', 'HW(V5,V6)': 'v5e vs v6e',
              'PR(V6)': 'the precision setting on the v6e', 'HWh(V6,G)': 'v6e highest vs L4 highest',
              'HW(V6,G)': 'v6e vs L4', 'HW(V6,GT)': 'v6e vs L4 Triton'}
PERT_KEY = {'RC(G)': 'Recompile, L4', 'DET': 'Determinism flag, L4', 'DU': 'Default attention, L4',
            'PR(G)': 'Precision setting, L4', 'HW(V5,V6)': 'v5e vs v6e', 'PR(V6)': 'Precision setting, v6e',
            'HWh(V6,G)': 'v6e vs L4, both highest', 'HW(V6,G)': 'v6e vs L4', 'HW(V6,GT)': 'v6e vs L4 Triton'}
PERT_KEY_ROW_PT = 10.5    # 8 pt text
PERT_KEY_COL_IN = (0.05, 1.80)     # left edges of the key's two columns (the figure is 3.5 in wide)
PERT_YMAX = 49     # the data end below 30 (numbers included); the legend fills 33 to 49 (R90)
PERT_LEGEND = ['Pair of platforms', 'Pair of platforms at highest precision', 'Change of arithmetic on one platform',
               'Two-sample model']


PREC5_EXPL = f'{AN}/prec5_exploratory_20261110'
PERT_FIVE = ['PR(V6)', 'PR(G)', 'HWh(V6,G)']      # row of 2026-11-10 (paper values, part 1): five seeds replace two


def pert_table(main_dir):
  """The rows of Fig. 4: c_perturbation.csv of the main study (two seeds for the precision contrasts), with PR(V6),
  PR(G) and HWh(V6,G) replaced by follow-up 12's five-seed values (prec5_exploratory_20261110/exploratory_c.csv).
  Each five-seed observed rate and interval must equal follow-up 12's own contrasts.csv (all 60, seeds 1 to 5)."""
  two = read(main_dir, f'{EXPL}/c_perturbation.csv')
  five = read(main_dir, f'{PREC5_EXPL}/exploratory_c.csv')
  if sorted(five.contrast) != sorted(PERT_FIVE) or list(five.columns) != list(two.columns):
    raise SystemExit(f'{PREC5_EXPL}/exploratory_c.csv: contrasts {sorted(five.contrast)} or columns differ')
  con = read(main_dir, f'{AN}/prec5_20261107/contrasts.csv')
  for r in five.itertuples():
    c = pick(con, contrast=r.contrast, statistic='flip_top1', targets='all 60', seeds='1 to 5')
    for a, b in ((r.observed_flip_rate, c.estimate), (r.observed_ci_low, c.ci_low), (r.observed_ci_high, c.ci_high)):
      if not math.isclose(float(a), float(b), rel_tol=1e-9):
        raise SystemExit(f'{r.contrast}: five-seed observed {a} against prec5 contrasts.csv {b}')
  out = pd.concat([two[~two.contrast.isin(PERT_FIVE)].assign(seeds='1 to 5'), five.assign(seeds='1 to 5')],
                  ignore_index=True)
  if two[two.contrast.isin(PERT_FIVE)].n_seed_pairs.max() != 120 or (out.n_seed_pairs != 300).any():
    raise SystemExit('Fig. 4: seed pairs per contrast are not 120 (two seeds) before and 300 after the replacement')
  return out, five



def table_pert_five(out, five):
  rows = []
  for c in PERT_FIVE:
    r = five.set_index('contrast').loc[c]
    rows.append({'Contrast': c, 'Seed pairs': int(r.n_seed_pairs),
                 'Shift / margin': f'{r.delta_over_mu:.2f} [{r.ratio_ci_low:.2f}, {r.ratio_ci_high:.2f}]',
                 'Predicted': f'{pc(r.predicted_flip_rate)} [{pc(r.predicted_ci_low)}, {pc(r.predicted_ci_high)}]',
                 'Observed': f'{pc(r.observed_flip_rate)} [{pc(r.observed_ci_low)}, {pc(r.observed_ci_high)}]'})
  write_table(out, 'TS_pert_five', pd.DataFrame(rows),
              'Exploratory (c) at five seeds: perturbation size against margin and the two-sample model for the '
              'precision contrasts of Follow-up 12, seeds 1 to 5 (as the supplement\'s perturbation figure draws them)', ROLE_EXPL,
              notes=['Analysis (c) as specified on 14 October 2026, applied to Follow-up 12\'s pairs (row of 10 November 2026). '
                     'Flip rates in %, with 95% intervals over targets (10000 replicates, random-number seed 2026).'],
              colspec='lrlll')

def pert_curve(xmax):
  sys.path.insert(0, str(REPO / 'analysis'))
  import exploratory  # noqa: E402  (the model of the row of 2026-10-14, analysis/exploratory.py)
  xs = np.linspace(0.0, xmax, 161)
  return xs, np.array([exploratory.predicted_flip_rate(x, 1.0) for x in xs])


def fig_perturbation(out, save, plt, pert):
  p = pert.set_index('contrast')
  missing = [c for c in PERT_HW + PERT_ONE if c not in p.index]
  if missing or len(p) != len(PERT_HW + PERT_ONE):
    raise SystemExit(f'c_perturbation.csv: contrasts {sorted(p.index)}; expected {PERT_HW + PERT_ONE}')
  xs, ys = pert_curve(0.8)
  for c in p.index:     # the drawn line and the table's prediction agree at every point
    r = p.loc[c]
    _, y = pert_curve(float(r.delta_over_mu))
    if not math.isclose(float(y[-1]), float(r.predicted_flip_rate), rel_tol=1e-9):
      raise SystemExit(f'{c}: model {y[-1]} against table {r.predicted_flip_rate}')
  order = sorted(p.index, key=lambda k: float(p.loc[k, 'delta_over_mu']))
  if set(order) != set(PERT_NUM_POS):
    raise SystemExit(f'Fig. 3: numbered points {sorted(PERT_NUM_POS)} against {order}')
  fig, ax = plt.subplots(figsize=(3.5, 2.45))
  fig.subplots_adjust(left=0.13, right=0.955, top=0.97, bottom=0.18)
  model_line, = ax.plot(xs, 100 * ys, color=INK, lw=0.9, ls='--', zorder=1)
  rows = []
  for group, col, m in ((PERT_ONE, ROSE, 'D'), (PERT_HW, NAVY, 'o')):
    for c in group:
      r = p.loc[c]
      x, y = float(r.delta_over_mu), 100 * float(r.observed_flip_rate)
      ax.errorbar(x, y, yerr=[[y - 100 * r.observed_ci_low], [100 * r.observed_ci_high - y]], color=col, marker=m, ms=3.8,
                  lw=0, elinewidth=0.7, capsize=0, mfc='white' if c == 'HWh(V6,G)' else col,
                  mec=col if c == 'HWh(V6,G)' else 'white', mew=0.8 if c == 'HWh(V6,G)' else 0.3, zorder=3)
      num = order.index(c) + 1
      if PERT_NUM_POS[c] == 'above':
        ax.annotate(str(num), (x, 100 * float(r.observed_ci_high)), xytext=(0, 1.5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
      else:
        ax.annotate(str(num), (x, 100 * float(r.observed_ci_low)), xytext=(0, -1.5), textcoords='offset points',
                    ha='center', va='top', fontsize=8)
      rows.append({'contrast': c, 'number': num, 'group': 'platform pair' if c in PERT_HW else 'one platform',
                   'n_seed_pairs': int(r.n_seed_pairs), 'delta_over_mu': r.delta_over_mu, 'ratio_ci_low': r.ratio_ci_low,
                   'ratio_ci_high': r.ratio_ci_high, 'observed_flip_rate': r.observed_flip_rate,
                   'observed_ci_low': r.observed_ci_low, 'observed_ci_high': r.observed_ci_high,
                   'predicted_flip_rate': r.predicted_flip_rate, 'predicted_ci_low': r.predicted_ci_low,
                   'predicted_ci_high': r.predicted_ci_high})
  hs = [ax.plot([], [], 'o', color=NAVY, ms=3.8, mec='white', mew=0.3)[0],
        ax.plot([], [], 'o', color=NAVY, ms=3.8, mfc='white', mec=NAVY, mew=0.8)[0],
        ax.plot([], [], 'D', color=ROSE, ms=3.8, mec='white', mew=0.3)[0], model_line]
  ax.set_xlim(0, 0.8)
  ax.set_ylim(0, PERT_YMAX)
  ax.set_yticks(range(0, 36, 5))
  # note 3 of 18 November 2026: the y label stands above the axis, so the axis line runs to the top (the ticks still
  # stop at 35, below the legend); it stopped at 35 until then (R90)
  ax.set_xlabel('Median score shift / median top-two margin')
  ax.set_ylabel('Top-1 flip rate (%)')
  # R90 (2026-11-12): the legend sits in the upper left, above every point, number and the model line (the y axis
  # runs to PERT_YMAX, with ticks to 35 only); the entries use the caption's words
  leg = ax.legend(hs, PERT_LEGEND, loc='upper left', fontsize=8, handletextpad=0.4, borderaxespad=0.3, frameon=True,
                  framealpha=1, edgecolor='#BBBBBB', fancybox=False, borderpad=0.4, labelspacing=0.3)
  leg.get_frame().set_linewidth(0.5)
  fig.canvas.draw()
  lb = leg.get_window_extent().transformed(ax.transData.inverted())
  tops = [100 * float(p.loc[c, 'observed_ci_high']) for c in p.index if float(p.loc[c, 'delta_over_mu']) <= lb.x1]
  if lb.y0 <= max(tops) + 4 or lb.y0 <= 100 * float(pert_curve(min(lb.x1, 0.8))[1][-1]):
    raise SystemExit(f'Fig. 2: the legend (from y = {lb.y0:.1f}) meets the points or the model line')
  # Key of the numbered points (options round of 2026-11-19): under the x label, in two columns, numbers in order;
  # the figure grows at the bottom by the key's height, so the axes keep their size and place in inches
  if set(PERT_KEY) != set(order):
    raise SystemExit(f'Fig. 3 key: {sorted(PERT_KEY)} against the points {sorted(order)}')
  rows_k = (len(order) + 1) // 2
  pad_in, row_in = 0.08, PERT_KEY_ROW_PT / 72
  extra = pad_in + rows_k * row_in + 0.02
  W0, H0 = fig.get_size_inches()
  bot_in, top_in = fig.subplotpars.bottom * H0, (1 - fig.subplotpars.top) * H0
  fig.set_size_inches(W0, H0 + extra)
  fig.subplots_adjust(bottom=(bot_in + extra) / (H0 + extra), top=1 - top_in / (H0 + extra))
  key_texts = []
  for k, c in enumerate(order):
    col, row = divmod(k, rows_k)
    key_texts.append(fig.text(PERT_KEY_COL_IN[col], extra - pad_in - (row + 0.5) * row_in, f'{k + 1}  {PERT_KEY[c]}',
                              transform=fig.dpi_scale_trans, ha='left', va='center', fontsize=8))
  fig.canvas.draw()
  rend = fig.canvas.get_renderer()
  xl = ax.xaxis.label.get_window_extent(rend)
  kb = [t.get_window_extent(rend) for t in key_texts]
  if max(b.y1 for b in kb) >= xl.y0 - 2:
    raise SystemExit('Fig. 3: the key meets the x label')
  if max(b.x1 for b in kb[:rows_k]) >= min(b.x0 for b in kb[rows_k:]) - 6:
    raise SystemExit('Fig. 3: the key\'s two columns touch')
  ylabel_top(fig, ax, 'Top-1 flip rate (%)')
  save(fig, 'fig4_perturbation', ROLE_EXPL, 'Supplementary figure (Fig. 2 of an earlier draft, moved for space): top-1 flip rate against the ratio of the median ranking-score '
       'shift to the median top-two margin, nine contrasts, with the two-sample model')
  write_csv(out, 'fig4_perturbation', pd.DataFrame(rows), ROLE_EXPL,
            f'CSV twin of the supplementary perturbation figure ({EXPL}/c_perturbation.csv; PR(V6), PR(G) and HWh(V6,G) from '
            f'{PREC5_EXPL}/exploratory_c.csv, five seeds; the line is analysis/exploratory.py predicted_flip_rate)')
  return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Fig. 5, seed curve (follow-up 8)

def seed_curve(q3):
  d = q3[(q3.statistic == 'success_difference') & (q3.condition == 'V5-G') & (q3.targets == 'all 60')]
  return d.sort_values('n_seeds').reset_index(drop=True)


def fig_seeds(out, save, plt, q3):
  d = seed_curve(q3)
  fig, ax = plt.subplots(figsize=(3.5, 2.25))     # 2.1 in cut the y label at the top (R89)
  fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.18)
  ax.axhspan(-2, 2, color=BAND, lw=0, zorder=0)
  ax.axhline(0, color=INK, lw=0.5, zorder=1)
  two = d.n_seeds == 2
  ax.vlines(d.n_seeds[~two], 100 * d.ci_low[~two], 100 * d.ci_high[~two], color=NAVY, lw=1, zorder=2)
  ax.plot(d.n_seeds[~two], 100 * d.estimate[~two], 'o', ms=3.6, color=NAVY, mec='white', mew=0.3, zorder=3)
  ax.plot(d.n_seeds[two], 100 * d.estimate[two], 'D', ms=4.2, color=ROSE, mec='white', mew=0.3, zorder=4)
  ax.set_xticks([1, 2, 5, 10, 15, 20])
  ax.set_xlim(0.3, 20.7)
  ax.set_yticks(range(-8, 9, 2))
  ax.set_ylim(-9.2, 9.2)
  ax.set_xlabel('First n seeds')
  ax.set_ylabel('v5e minus L4 success (points)')
  ax.text(20.6, 8.6, 'Band: ±2 points', ha='right', va='top')
  r2 = d[two].iloc[0]
  ax.plot([9.0], [-7.6], 'D', ms=4.2, color=ROSE, mec='white', mew=0.3)
  ax.text(9.6, -7.6, f'n = 2: interval [{pc(r2.ci_low)}, {pc(r2.ci_high)}]', va='center')
  ylabel_top(fig, ax, 'v5e minus L4 success (points)')
  save(fig, 'fig5_seed_curve', 'follow-up 8 (outside the pre-registered study)',
       'Fig. 5: v5e minus L4 success of the overall top-1 against the first n seeds, 60 targets')
  write_csv(out, 'fig5_seed_curve', d[['n_seeds', 'condition', 'statistic', 'targets', 'n_targets', 'estimate', 'ci_low',
                                       'ci_high']], 'follow-up 8', 'CSV twin of Fig. 5 (fractions, seeds_20261102/q3.csv)')
  return d


# ---------------------------------------------------------------------------------------------------------------------
# Reserves R1 and R2

def fig_cost(out, save, plt, cost):
  c = cost[cost.cache == 'fresh']
  cols = [('median_usd_per_structure_standard', '(a) On demand'), ('median_usd_per_structure_spot', '(b) Spot'),
          ('median_usd_per_structure_flex_start', '(c) Flex-start')]
  fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1), sharey=True, sharex=True)
  fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.21, wspace=0.16)
  rows = []
  for ax, (col, t) in zip(axes, cols):
    for p, cc, m, ls in (('V5', NAVY, 'o', '-'), ('V6', NAVY, 's', '--'), ('G', ROSE, 'D', '-'), ('GT', ROSE, '^', '--')):
      d = c[c.platform == p].sort_values('bucket')
      if d[col].notna().any():
        ax.plot(d.bucket, d[col], color=cc, marker=m, ls=ls, ms=3.4, label=ARM_NAME[p])
        rows += [{'platform': p, 'bucket': int(r.bucket), 'price_model': col, 'usd_per_structure': getattr(r, col)}
                 for r in d.itertuples()]
    ax.set_yscale('log')
    ax.set_title(t, loc='left', pad=2, fontstyle='italic')
    ax.set_xticks([256, 512, 768, 1024])
    ax.set_xlabel('Token bucket')
  axes[0].set_ylabel('USD per structure')
  axes[0].legend(ncol=2, loc='lower right', handletextpad=0.4, columnspacing=0.8, borderaxespad=0.1)
  ylabel_top(fig, axes[0], 'USD per structure')
  save(fig, 'figR1_cost', 'pre-specified (Q2, M6); reserve', 'Reserve R1: USD per structure by bucket and price model')
  write_csv(out, 'figR1_cost', pd.DataFrame(rows), 'pre-specified (Q2, M6); reserve', 'CSV twin of R1')


def fig_size(out, save, plt, large_hw, rule):
  fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), sharey=True, gridspec_kw={'width_ratios': [1.3, 1]})
  fig.subplots_adjust(left=0.41, right=0.975, top=0.97, bottom=0.30, wspace=0.08)
  rows = []
  for i, (pair, rc) in enumerate((('HW(V5,V6)', 'RC(V5)'), ('HW(V6,G)', 'RC(V6)'))):
    for g, off, fill in (('large', -0.12, True), ('control', 0.12, False)):
      r = pick(large_hw, targets='all', group=g, hardware=pair, recompilation=rc, statistic='flip_top1')
      axes[0].plot([100 * r.ci_low, 100 * r.ci_high], [i + off] * 2, color=NAVY, lw=1)
      axes[0].plot(100 * r.estimate, i + off, 'o', ms=3.8, color=NAVY, mfc=NAVY if fill else 'white', mew=0.9)
      rows.append({'pair': pair, 'reference': rc, 'group': g, 'estimate': r.estimate, 'ci_low': r.ci_low,
                   'ci_high': r.ci_high})
    d = pick(rule, pair=pair, statistic='flip_top1')
    axes[1].plot([100 * d.ci_low, 100 * d.ci_high], [i] * 2, color=ROSE, lw=1)
    axes[1].plot(100 * d.estimate, i, 'D', ms=3.4, color=ROSE, mec='white', mew=0.3)
    rows.append({'pair': pair, 'reference': rc, 'group': 'large minus control', 'estimate': d.estimate,
                 'ci_low': d.ci_low, 'ci_high': d.ci_high})
  axes[1].axvline(0, color=INK, lw=0.6)
  axes[0].set_yticks([0, 1], ['v5e vs v6e\nminus recompile v5e', 'v6e vs L4\nminus recompile v6e'])
  axes[0].set_ylim(1.5, -0.5)
  axes[0].set_xlabel('Points')
  axes[1].set_xlabel('Large minus control', ha='right', x=1.0)
  for ax in axes:
    ax.tick_params(axis='y', length=0)
  save(fig, 'figR2_size', 'follow-up 10 (outside the pre-registered study, OSF addendum); reserve',
       'Reserve R2: platform pair minus recompilation by group (filled large, open control) and large minus control')
  write_csv(out, 'figR2_size', pd.DataFrame(rows), 'follow-up 10; reserve', 'CSV twin of R2')


# ---------------------------------------------------------------------------------------------------------------------
# Tables I to IV

ARM_FOLDER = {'V5': 'main', 'V6': 'main', 'G': 'main', 'GT': 'main', 'Gd': 'main', 'V6h': 'main', 'Gh': 'main', 'C': 'main',
              'A': 'a100', 'AT': 'a100', 'Ah': 'a100', 'H': 'h100'}
PRICE_KEY = {'TPU v5 lite': ('v5e', 'v5litepod-1'), 'TPU v6 lite': ('v6e', 'ct6e-standard-1t'),
             'NVIDIA L4': ('l4', 'g2-standard-8'), 'cpu': ('cpu', 'n2-highmem-16'),
             'NVIDIA A100-SXM4-80GB': ('a100', 'a2-ultragpu-1g'), 'NVIDIA H100 80GB HBM3': ('h100', 'a3-highgpu-1g')}


def table_arms(out, samples, prices):
  std = prices[prices.provisioning == 'standard'].dropna(subset=['usd_per_hour'])
  rows = []
  for arm, folder in ARM_FOLDER.items():
    s = samples[folder]
    s = s[(s.platform == arm) & s.repetition.isin(['r1', 'r2'])]
    dk = s.device_kind.iloc[0]
    diffs = []
    att, mp, cfg = set(s.attention.dropna()), set(s.matmul_precision.dropna()), set(s.config)
    if att == {'triton'}:
      diffs.append('Triton attention kernel')     # labels of 2026-11-17, in words
    elif att != {'xla'}:
      raise SystemExit(f'{arm}: attention {att}')
    if mp == {'highest'}:
      diffs.append('highest matrix-product precision')
    elif mp:
      raise SystemExit(f'{arm}: matmul precision {mp}')
    if any(c.endswith('_det') for c in cfg):
      diffs.append('deterministic-operations flag')
    jax = sorted(set(s.jax.astype(str)))
    lt = sorted(set(s.libtpu.dropna().astype(str)))
    pf, mt = PRICE_KEY[dk]
    p = std[(std.platform == pf) & (std.machine_type == mt)].set_index('region')
    regions = sorted(set(s.zone.str.rsplit('-', n=1).str[0]))
    applied = sorted({float(p.usd_per_hour.get(r, p.usd_per_hour['europe-west4'])) for r in regions})
    usd = f'{applied[0]:.2f}' if len(applied) == 1 or f'{applied[0]:.2f}' == f'{applied[-1]:.2f}' \
        else f'{applied[0]:.2f}/{applied[-1]:.2f}'     # two prices with a slash
    read_on = ', '.join(sorted(set(p.read_on.astype(str))))
    rows.append({'arm': arm, 'device': DEVICE_LABEL[dk], 'differs from base': '; '.join(diffs) or 'base',
                 'jax': ', '.join(jax), 'libtpu': ', '.join(lt), 'on-demand USD/h': usd, 'price read on': read_on})
  df = pd.DataFrame(rows)
  if not df['on-demand USD/h'].str.contains('/').any():
    raise SystemExit('Table I: no arm has two prices; drop the sentence of note 5 from the TI_main note')
  # LaTeX: jax and libtpu are one version each, stated in the note, so the table fits 3.5 in at \footnotesize unscaled
  jx, ltx = set(df.jax), set(df.libtpu) - {''}
  if len(jx) != 1 or len(ltx) != 1 or ', ' in next(iter(jx)) + next(iter(ltx)):
    raise SystemExit(f'Table I: jax {jx}, libtpu {ltx} are no longer one version each')
  tpu = df[df.device.str.startswith('TPU')].arm.tolist()
  assert set(df[df.libtpu != ''].arm) == set(tpu)
  # Style round of 2026-11-19: no device column (the arm's name gives the device; the memory of the A100 and the
  # H100, 80 GB, is in the note), so the table fits the column at \footnotesize without scaling
  tdf = df.drop(columns=['price read on', 'jax', 'libtpu', 'device']).rename(columns={'on-demand USD/h': 'USD/h'})
  big = sorted(set(df[df.device.str.endswith(' 80 GB')].device.str[:-6]))
  if big != ['A100', 'H100'] or set(df.device) - set(DEVICE_LABEL.values()):
    raise SystemExit(f'Table I: devices with 80 GB {big}; the note names the A100 and the H100')
  sw = samples['sw']
  sw_v = sorted(set(sw.jax.astype(str)))
  # every cell stays on one line, so a row reads at a glance; without the device column the table fits the column
  write_table(out, 'TI_arms', df, 'Arms of the Main Study and of Follow-Ups 3 and 9', 'design', colspec='llr',
              tex_df=tdf,
              notes=[ARMS_NOTE + ' ' + MEMORY_NOTE + ' ' + PRIMARY_NOTE + f' Every arm ran jax {next(iter(jx))}. The TPU arms ({", ".join(ARM_NAME[a] for a in tpu)}) ran libtpu '
                     f'{next(iter(ltx))}. '
                     f'Prices are from cloud/prices.csv, read on {", ".join(sorted(set(df["price read on"].str.split(", ").sum()))[:-1])}, and {sorted(set(df["price read on"].str.split(", ").sum()))[-1]}. '
                     'An arm that ran in regions with different prices has two prices (the price of each process region, '
                     'or europe-west4 where no row exists). Follow-up 7 used jax ' + ', '.join(sw_v) + ' (libtpu '
                     + ', '.join(sorted(set(sw.libtpu.dropna().astype(str)))) + ' on the v6e). Provisioning by follow-up is '
                     'in TABREFPROV. The main study ran on demand.'])
  # Reading notes of 2026-11-20: in the body, Table I fills the column exactly at 9 pt, unscaled: the same rows in a
  # tabular* as wide as the line, with the extra space between the columns (\extracolsep{\fill}), so its rules line up
  # with the text. The one-column supplement keeps TI_arms.tex at its natural width.
  body = (out / 'tables' / 'TI_arms.tex').read_text()
  head = '\\begin{tabular}{@{}llr@{}}'
  if body.count(head) != 1 or body.count('\\end{tabular}') != 1:
    raise SystemExit('Table I: the tabular header changed; update TI_arms_col')
  col = body.replace('% TI_arms.', '% TI_arms_col (Table I of the body, as wide as the column).', 1) \
      .replace(head, '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llr@{}}').replace('\\end{tabular}', '\\end{tabular*}')
  # the % after \end{tabular*} drops the end-of-line space, which would make the box 3 pt wider than the line and
  # let \fitcol scale the table to 98.8%
  (out / 'tables' / 'TI_arms_col.tex').write_text(col.rstrip('\n') + '%\n')
  ASSETS['tables/TI_arms_col.tex'] = ('design', 'Table I of the body: TI_arms.tex as wide as the column')
  return df


T2_GROUP = {'recompilation': 'Recompilation alone', 'primary pair': 'Primary pairs',
            'seed yardstick': 'Change of seed (reference)', 'default use': 'Default GPU attention'}   # R57
T2_ROWS = [('recompilation', ['RC(V5)', 'RC(V6)', 'RC(G)']), ('primary pair', ['HW(V5,V6)', 'HW(V6,G)', 'HWminus', 'HW(G,C)']),
           ('seed yardstick', ['SD(V6)']), ('default use', ['DU'])]


def table_headline(out, con, hwrc, main_dir=None):
  rows = []
  ws = read(main_dir, f'{WINNERS}/summary.csv')   # post hoc: the two returned samples of the changed predictions
  wch = ws[(ws.level == 'seed') & (ws.stratum == 'changed')].set_index('contrast')
  we = read(main_dir, f'{WINNERS_EXT}/summary.csv')   # post hoc 16: L4 vs CPU and default attention, main study
  wex = we[(we.source == 'main') & (we.level == 'seed') & (we.stratum == 'changed')].set_index('contrast')
  NONE, NA = 'none', 'n/a'   # no prediction changed; the measure does not apply

  def returned(c, top1):
    if c == 'SD(V6)':   # sample numbers of different seeds are unrelated
      return NA
    for src in (wch, wex):
      if c in src.index:
        return ang(src.loc[c, 'pose_rmsd_median'])
    if float(top1.estimate) == 0:
      return NONE
    raise SystemExit(f'Table II: no returned value for {c}, whose top-1 rate is not 0')
  for role, names in T2_ROWS:
    for c in names:
      if c == 'HWminus':
        d = {s: pick(hwrc, hardware='HW(V6,G)', recompilation='RC(G)', statistic=s) for s in
             ('flip_top1', 'flip_success', 'flip_matters')}
        rows.append({'role': role, 'contrast': 'HW(V6,G) minus RC(G) (points)', 'pairs': int(pick(con, contrast='HW(V6,G)',
                     statistic='flip_top1').n), **{k: diff_ci(d[s]) for k, s in
                     (('top-1 flip (%)', 'flip_top1'), ('success flip (%)', 'flip_success'),
                      ('flips that matter (%)', 'flip_matters'))},
                     'ligand RMSD, median (Å)': NA, 'C-alpha RMSD, median (Å)': NA, 'bit-identical samples': NA,
                     'returned, changed (Å)': NA})
        continue
      g = {s: pick(con, contrast=c, statistic=s) for s in ('flip_top1', 'flip_success', 'flip_matters',
                                                          'm4_ligand_rmsd_median', 'm4_ca_rmsd_global_median',
                                                          'bit_identical_fraction')}
      assert g['flip_top1'].role == role, (c, g['flip_top1'].role)
      b = g['bit_identical_fraction']
      rows.append({'role': role, 'contrast': c, 'pairs': int(g['flip_top1'].n), 'top-1 flip (%)': rate_ci(g['flip_top1']),
                   'success flip (%)': rate_ci(g['flip_success']), 'flips that matter (%)': rate_ci(g['flip_matters']),
                   'ligand RMSD, median (Å)': sig3_ci(g['m4_ligand_rmsd_median']),
                   'C-alpha RMSD, median (Å)': sig3_ci(g['m4_ca_rmsd_global_median']),
                   'bit-identical samples': f'{round(float(b.estimate) * int(b.n))} of {int(b.n)}',
                   'returned, changed (Å)': returned(c, g['flip_top1'])})
  df = pd.DataFrame(rows)
  gc = pick(hwrc, hardware='HW(G,C)', recompilation='RC(G)', statistic='flip_top1')
  if float(gc.ci_low) > 0 or float(gc.estimate) <= 0:
    raise SystemExit('Table II note: HW(G,C) minus RC(G) no longer "exceeds, with an interval that includes 0"')
  if int(wex.loc['HW(G,C)', 'units_with_pose_rmsd']) != 1:
    raise SystemExit('Table II note: the L4 vs CPU returned value no longer rests on one prediction')
  ca = pick(con, contrast='HW(V5,V6)', statistic='m4_ca_rmsd_global_median').estimate
  if round(float(ca), 4) != 0.0445:
    raise SystemExit(f'C-alpha median of HW(V5,V6) is {ca}: superseded folder?')
  # LaTeX: short headers and the RMSD medians without their intervals, so the table fits 7.16 in at \footnotesize
  # unscaled (the intervals stay in the CSV twin and the supplement)
  short = {'top-1 flip (%)': 'top-1 (%)', 'success flip (%)': 'success (%)', 'flips that matter (%)': 'matter (%)',
           'ligand RMSD, median (Å)': 'ligand (Å)', 'C-alpha RMSD, median (Å)': 'C-alpha (Å)',
           'bit-identical samples': 'identical', 'returned, changed (Å)': 'returned'}
  t = df.copy()
  for c in ('ligand RMSD, median (Å)', 'C-alpha RMSD, median (Å)'):
    t[c] = t[c].map(lambda v: v.split(' [')[0] if isinstance(v, str) else v)
  t = t.rename(columns=short)
  # Style round of 2026-11-19: an indented row under v6e vs L4 ('v6e vs L4 beyond recompilation (points)' on one line
  # from 2026-11-17, 'minus recompile L4 (points)' before), so the table fits the text width at \footnotesize unscaled
  t['contrast'] = t['contrast'].replace({'HW(V6,G) minus RC(G) (points)': '\\INDENT beyond recompilation (points)'})
  trows = []
  for role, g in t.groupby('role', sort=False):
    trows.append({'contrast': '\\GROUP ' + T2_GROUP[role]})
    trows += g.drop(columns=['role']).to_dict('records')
  tdf = pd.DataFrame(trows)
  hw = list(tdf.contrast)
  if hw.index('\\INDENT beyond recompilation (points)') != hw.index('HW(V6,G)') + 1:
    raise SystemExit('Table II: the beyond-recompilation row no longer follows v6e vs L4')
  write_table(out, 'TII_headline', df, 'Flips and Structure Differences in the Main Study', ROLE_MAIN,
              # 10.4 pt between columns instead of 12: with the indented row the table is 528 pt wide at the default,
              # 12 pt over the text width, and 515.5 pt with it (2026-11-19)
              colspec='@{\\hspace{10.4pt}}'.join('lrlllllll'), tex_df=tdf,
              # minimal note of 2026-11-19 (captions describe, the text interprets); the seed row keeps one line of
              # legend, and Methods (Contrasts and Statistics) explains it
              # options round of 2026-11-19 (Table II B): what the reader needs to read the columns; the CPU's 6
              # targets are in Results (IV-A) and the column pairs, the bit-identical count in Methods (III-B)
              # reading notes of 2026-11-20 (50 words): the 57 targets of the ligand and the glycans are in Results
              # (IV-A) and the supplement, why the seed row carries no top-1 information in Methods (III-C); the note
              # now says what none and n/a mean
              # reading notes of 2026-11-23: the n/a part has its own verb
              notes=['Flips are percentages of predictions, with 95% bootstrap intervals over targets. The indented '
                     'row subtracts L4 recompilation. Ligand and C-alpha are median RMSDs of paired samples. Returned '
                     '(post hoc) is the median ligand RMSD between differing returned samples. Seed top-1 carries no '
                     'information. None means no change, and n/a means not applicable.'])
  return df


def gpu_followup_cost(main_dir, prices_df):
  """M6 and M7 of the A100 (follow-up 3) and H100 (follow-up 9) arms A and H, computed here with the functions of
  analysis/cost.py, the definitions of plan section 4 (M6, M7) and interpretations 2 to 4 of section 10's entry of
  2026-10-09, on the follow-ups' samples.csv; prices from cloud/prices.csv of the main checkout, region of each run's
  zone. Returns (cost summary, memory) in the format of the main study's cost_summary.csv and memory.csv."""
  sys.path.insert(0, str(REPO / 'analysis'))
  import types
  import cost as costmod  # noqa: E402
  prices = {}
  for r in prices_df.itertuples():
    v = float(r.usd_per_hour) if not (isinstance(r.usd_per_hour, float) and math.isnan(r.usd_per_hour)) else None
    prices[(r.platform, r.region, r.provisioning)] = (v, r.read_on)
  summ, mems = [], []
  for folder, arm in ((f'{AN}/a100_20261018', 'A'), (f'{AN}/h100_20261104', 'H')):
    args = types.SimpleNamespace(samples=str(main_dir / folder / 'samples.csv'), all_settings=False, labels=None,
                                 include=None, exclude=None)
    p = costmod.processes(args)
    p = p[p.platform == arm]
    c = costmod.cost_table(p, prices, costmod.batch_sizes('subset'))
    summ.append(costmod.summary_table(c))
    mems.append(costmod.memory_table(p))
  s, m = pd.concat(summ, ignore_index=True), pd.concat(mems, ignore_index=True)
  # cross-checks against the follow-ups' own outputs: A's fresh median wall per structure (followup34.py) and memory
  both = read(main_dir, f'{AN}/a100_20261018/cost_both_regions.csv')
  amem = read(main_dir, f'{AN}/a100_20261018/memory.csv')
  for b in (256, 512, 768, 1024):
    w = float(both[(both.platform == 'A') & (both.bucket == b) & (~both.warm)].wall_per_structure_s.iloc[0])
    mine = float(s[(s.platform == 'A') & (s.bucket == b) & (s.cache == 'fresh')].median_wall_s.iloc[0])
    if not math.isclose(w, mine, rel_tol=1e-9):
      raise SystemExit(f'A bucket {b}: median wall {mine} against cost_both_regions {w}')
    x = float(amem[(amem.platform == 'A') & (amem.bucket == b)].compiled_peak_plus_args_gib.iloc[0])
    y = float(m[(m.platform == 'A') & (m.bucket == b)].compiled_peak_plus_args_gib.iloc[0])
    if not math.isclose(x, y, rel_tol=1e-9):
      raise SystemExit(f'A bucket {b}: memory {y} against memory.csv {x}')
  NOTES.append('Table III: A and H rows computed with analysis/cost.py (processes, cost_table, summary_table, '
               'memory_table) on the follow-ups\' samples.csv; A agrees with a100_20261018/cost_both_regions.csv (median '
               'wall, fresh) and memory.csv at every bucket.')
  return s, m


def gib_range(v):
  a, b = f'{min(v):.2f}', f'{max(v):.2f}'
  return a if a == b else f'{a} to {b}'


MONTHS = {9: 'September', 10: 'October', 11: 'November'}


def table_q2(out, cost, mem, memlimit, gpu_cost=None, gpu_mem=None, price_dates=None, batch1024='', memlimit_ah=None,
             warm_th=None):
  """Table III (task of 2026-11-10, A3): compilation and inference split, cost per structure fresh, with compilation
  amortized (plan M6, second variant) and from warm reruns, A100 and H100 rows, and one memory metric on every device
  (M7 as the plan defines it: compiled peak plus arguments). warm_th (warm_th_summary, row of 2026-11-11): the warm
  rows of the L4 Triton and the H100, added to the warm column; None leaves both cells empty."""
  if memlimit_ah is not None:
    memlimit = pd.concat([memlimit, memlimit_ah], ignore_index=True)
  cost_all = pd.concat([cost, gpu_cost], ignore_index=True) if gpu_cost is not None else cost
  mem_all = pd.concat([mem, gpu_mem], ignore_index=True) if gpu_mem is not None else mem
  c = cost_all[cost_all.cache == 'fresh']
  w = cost_all[cost_all.cache == 'warm']
  if warm_th is not None:
    clash = sorted(set(warm_th.platform) & set(w[w.bucket == 1024].platform))
    if clash:
      raise SystemExit(f'warm_th: {clash} already have a warm row at 1024')
    w = pd.concat([w, warm_th], ignore_index=True)
  rows = []
  for arm in ('V5', 'V6', 'G', 'GT', 'A', 'H'):
    r = {'arm': arm}
    for b in (256, 1024):
      x = c[(c.platform == arm) & (c.bucket == b)]
      r[f'compile {b} (s)'] = f'{float(x.median_compile_s.iloc[0]):.0f}'
      r[f'inference {b} (s)'] = f'{float(x.median_inference_s.iloc[0]):.0f}'
    x = c[(c.platform == arm) & (c.bucket == 1024)].iloc[0]
    xw = w[(w.platform == arm) & (w.bucket == 1024)]
    r['USD fresh'] = usd3(x.median_usd_per_structure_standard)
    r['USD amortized'] = usd3(x.median_usd_per_structure_amortised_standard)
    r['USD amortized Spot'] = usd3(x.median_usd_per_structure_amortised_spot)
    r['USD warm'] = usd3(xw.median_usd_per_structure_standard.iloc[0]) if len(xw) else ''
    m = mem_all[(mem_all.platform == arm) & (mem_all.bucket == 1024)].iloc[0]
    r['memory 1024 (GiB)'] = sig3(m.compiled_peak_plus_args_gib)
    r['limit (GiB)'] = sig3(m.device_limit_gib)
    r['largest completed; first failed (tokens)'] = memlimit_cell(memlimit[memlimit.arm == arm])
    rows.append(r)
  df = pd.DataFrame(rows)
  c512 = c[(c.platform == 'C') & (c.bucket == 512)].iloc[0]
  ct = c[c.platform == 'C']
  nwarm = w[(w.platform.isin(['V5', 'V6', 'G', 'GT', 'H'])) & (w.bucket == 1024)].processes
  warm_later = ''
  if warm_th is not None:     # the cells stay at the on-demand price, as the table
    warm_later = (' For the L4 Triton and the H100, a separate measurement ran the same warm reruns on 10 pilot '
                  'targets.')
  args_gpu = [float(mem_all[(mem_all.platform == a) & (mem_all.bucket == 1024)].compiled_peak_plus_args_gib.iloc[0]) -
              float(mem_all[(mem_all.platform == a) & (mem_all.bucket == 1024)].compiled_peak_gib.iloc[0])
              for a in ('G', 'GT', 'A', 'H')]
  def day(iso):
    return int(iso[8:10]), MONTHS[int(iso[5:7])]
  md = [day(x) for x in price_dates['main']]
  if len({m for _, m in md}) != 1:
    raise SystemExit(f'Table III note: main-study prices read in more than one month {md}')
  main_read = ' and '.join(str(d_) for d_, _ in md) + f' {md[0][1]} {price_dates["main"][0][:4]}'
  fu_read = ', '.join(f'{k} on ' + ' and '.join(f'{day(x)[0]} {day(x)[1]}' for x in v)
                      for k, v in (('A100', price_dates['a100']), ('H100', price_dates['h100'])))
  gib = gib_range(args_gpu)
  if ' to ' in gib:
    raise SystemExit(f'Table III note: arguments on the GPUs are {gib} GiB, not one value')
  write_table(out, 'TIII_q2', df, 'Time, Cost, and Memory per Structure (Q2)', 'pre-specified (Q2: M6 and M7) for the '
              'v5e, v6e, L4 and L4 Triton; follow-ups 3 and 9 for the A100 and H100; memory limit descriptive'
              + ('; warm cost of the L4 Triton and H100 descriptive, measured later' if warm_th is not None else ''),
              colspec='lrrrrrrrrrrl',
              head=[' & \\multicolumn{2}{c}{256 tokens (s)} & \\multicolumn{2}{c}{1024 tokens (s)} & '
                    '\\multicolumn{4}{c}{USD per structure at 1024} & \\multicolumn{2}{c}{GiB at 1024} & '
                    'memory limit (tokens) \\\\',
                    '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}\\cmidrule(l){12-12}',
                    'arm & compile & inference & compile & inference & fresh & amortized & amort.\\ Spot & warm & '
                    'memory & limit & largest; first failed \\\\'],
              # the note's wording is fixed; the values are computed here
              notes=['A structure is one target with one seed and five samples. Each time is the median over the runs '
                     'that compiled the model from scratch, for inputs padded to 256 or 1024 tokens. A process also loads '
                     'the model and writes its outputs, and its inference time already includes the compilation. '
                     'Cost is the process wall time per structure times the on-demand list price, unless marked Spot. '
                     f'The amortized cost spreads one compilation over the {batch1024} targets of bucket 1024, as the plan '
                     'defines it. The warm cost comes from a new process that loads the model compiled by the first run '
                     'from a saved cache, measured on seed~1 of the 10 pilot targets '
                     f'({int(nwarm.min())} of them at 1024).' + warm_later,
                     f'We read list prices on {main_read} ({fu_read}). They exclude VM setup, idle time, and boot disk. '
                     'Every main-study run was on demand. Memory is the compiled peak plus the arguments on the device, '
                     'as the plan defines it (M7). On GPUs, XLA\'s compiled peak already includes the arguments, so this '
                     'metric counts them twice there. We found the memory limit by padding the input '
                     '7D5C to ever larger buckets. The column gives the largest bucket that ran and the first that failed.'
                     + (' The A100 and H100 limits come from a separate measurement.' if memlimit_ah is not None else '')
                     + ' The A100 and H100 rows are follow-ups. The CPU ran '
                     f'{int(ct[ct.bucket == 256].targets.iloc[0])} target at 256 tokens and {int(c512.targets)} at 512, '
                     'with seeds 1 and 2. The inference time of a fresh process includes the compilation. The amortized '
                     'cost removes 7/8 of each process\'s compile time from its wall time before the median is taken, so '
                     'it can fall below the cost of the median inference time.'])
  # Style round of 2026-11-21 (decisions on the doubts): the size of the double count on GPUs and the CPU's time and
  # cost at 512 moved from the note into the text (Floats Moved From the Main Text); the generator checks the values
  if gib != '1.12' or f'{float(c512.median_wall_s):.0f}' != '3949' or usd3(c512.median_usd_per_structure_standard) != '1.27':
    raise SystemExit(f'TIII_q2: the supplement text prints 1.12 GiB, 3949 s and 1.27 USD; now {gib}, '
                     f'{float(c512.median_wall_s):.0f}, {usd3(c512.median_usd_per_structure_standard)}')
  # Style round of 2026-11-19: the appendix of the main paper prints the same table with a short note (definitions
  # only); the supplement keeps the full note above, with the prices' dates, the GPU arguments, the CPU values and the
  # amortized cost against the median inference time
  write_note(out, 'TIII_q2_app',
             # options round of 2026-11-19 (Table V B): the warm cost, the memory limit and the follow-up rows are
             # in the supplement's full note and in the headers
             # reading notes of 2026-11-20 (61 words); the supplement keeps the full note
             ['A structure is one target with one seed and five samples. Times are medians over runs that compiled the '
              'model from scratch, and inference includes compilation. Cost is on demand unless marked Spot. The '
              f'amortized cost spreads one compilation over the {batch1024} targets of bucket 1024. Memory is the '
              'compiled peak plus the arguments. SUPPTABQTWO has the full note.'],
             'pre-specified (Q2: M6 and M7); appendix of the main paper', 'short note below TIII_q2 in the main paper\'s appendix')
  # Without results/analysis/warm_th_<DATE>/ the warm cells of the L4 Triton and the H100 stay empty and the note
  # keeps a TODO line (warm_th_20261111 exists since 2026-11-11, so the cells and the note's sentence are written).
  if warm_th is None:
    nf = out / 'tables' / 'TIII_q2_note.tex'
    nf.write_text(nf.read_text() + '% TODO warm_th: the warm cost of the L4 Triton and the H100 (empty cells) is written '
                  'when results/analysis/warm_th_<DATE>/ exists; see analysis/paper_assets_isc.py, table_q2.\n')
  return df


def readings_interval(r):
  return 'beyond recompilation' if float(r.ci_low) > 0 else 'not beyond recompilation'


def table_followups(out, d, fu12_targets):
  rows = []

  def add(fu, what, ts, head, reading):
    rows.append({'FU': fu, 'what changed': what, 'targets, seeds': ts, 'headline, 95% interval': head,
                 'reading': reading})
  st = {k: (t, s) for k, _, _, _, t, s in STUDIES}

  def tsx(k):
    return f'{st[k][0]}, {st[k][1]}'
  # FU1
  c, h = d['ppi_con'], d['ppi_hw']
  m = pick(h, hardware='HW(V6,G)', recompilation='RC(G)', statistic='flip_top1')
  s1 = {p: pick(c, contrast=p, statistic='flip_success') for p in ('HW(V5,V6)', 'HW(V6,G)')}
  ones = {p: round(float(r.estimate) * int(r.n)) for p, r in s1.items()}
  assert set(ones.values()) == {1}, ones
  r = s1['HW(V6,G)']
  add('1', 'protein-protein targets, success at DockQ 0.23', tsx('ppi'),
      f'v6e vs L4 minus recompile L4 {diff_ci(m)} points; success flip {rate_ci(r)}% (one of {int(r.n)} predictions)',
      'descriptive; one prediction flipped in each platform pair')
  # FU2
  e = d['embed']
  n0 = {}
  for p in ('V5 r1 - V6 r1', 'V6 r1 - G r1'):
    x = e[e.pair == p]
    n0[p] = (int((x.single_identical & x.pair_identical).sum()), len(x))
  assert len(set(n0.values())) == 1
  k, n = list(n0.values())[0]
  add('2', 'trunk embeddings saved', tsx('embed'), f'trunk embeddings bit-identical on {k} of {n} targets, v5e vs v6e '
      'and v6e vs L4', 'present at the trunk output')
  # FU3
  h = d['a100_hw']
  ga, va = (pick(h, hardware=p, recompilation='RC(A)', statistic='flip_top1') for p in ('HW(G,A)', 'HW(V6,A)'))
  add('3', 'A100 (base, Triton, and highest precision arms)', tsx('a100'),
      f'L4 vs A100 minus recompile A100 {diff_ci(ga)}; v6e vs A100 minus recompile A100 {diff_ci(va)}',
      'no reading rule: L4 vs A100 interval contains 0; v6e vs A100 interval above 0')
  assert float(ga.ci_low) < 0 < float(ga.ci_high) and float(va.ci_low) > 0
  # FU4
  c = d['cpu_con']
  v5c, v6c = (pick(c, contrast=p, statistic='flip_top1') for p in ('HW(V5,C)', 'HW(V6,C)'))
  bits = [pick(d['main_con'], contrast=p, statistic='bit_identical_fraction') for p in ('RC(V5)', 'RC(V6)')]
  assert all(float(b.estimate) == 1.0 for b in bits) and len({int(b.n) for b in bits}) == 1
  nb = int(bits[0].n)
  line = d['cross_line']
  mm = re.match(r'C\s+\(expected identical\): (\d+)/(\d+) \(target, seed\) bit-identical as expected$', line)
  add('4', 'CPU on all 60 targets', tsx('cpu60'), f'v5e vs CPU {rate_ci(v5c)}%; v6e vs CPU {rate_ci(v6c)}%; recompile '
      f'v5e and recompile v6e bit-identical, {nb} of {nb} samples', f'CPU outputs were bit-identical across sessions in '
      f'{mm.group(1)} of {mm.group(2)} predictions, on the 6 targets the CPU ran in both sessions')
  # FU5
  dec = d['feat_decision']
  mi = dec['model_inputs']
  phrase = 'the host is excluded as a source of the HW differences'
  if phrase not in dec['statement']:
    raise SystemExit('feat decision.json: statement changed')
  add('5', 'featurization host (four CPU families), model inputs only', tsx('feat'),
      f'{mi["identical"]} of {mi["compared"]} model-input comparisons identical',
      'the host is excluded as a source of the differences between platforms')
  # FU6
  h, rule = d['msa_hw'], d['msa_rule']
  t1, ftm = (pick(h, hardware='HW(V6o,Go)', statistic=s, targets='all 60') for s in ('flip_top1', 'flip_matters'))
  br = {s: pick(rule, statistic=s, targets='all 60').branch for s in ('flip_top1', 'flip_success', 'flip_matters')}
  holds, fails = 'the main conclusion holds with official inputs', \
      'the main conclusion does not hold with official inputs (an interval not above 0)'
  if [br['flip_top1'], br['flip_success'], br['flip_matters']] != [holds, holds, fails]:
    raise SystemExit(f'msa reading_rule.csv branches changed: {br}')
  add('6', 'official AlphaFold 3 inputs (MSA, templates)', tsx('msa'),
      f'v6e official vs L4 official minus recompile L4 official {diff_ci(t1)}; flips that matter {diff_ci(ftm)}',
      f'top-1 and success: {holds}; flips that matter: {fails}')
  # FU7
  c, k, rule = d['sw_con'], d['sw_cmp'], d['sw_rule']
  b6 = pick(c, contrast='SW(V6)', statistic='bit_identical_fraction', targets='all 60')
  sg = pick(k, software='SW(G)', reference='RC(G)', statistic='flip_top1', targets='all 60')
  rec = 'the software stack moves the top-1 less than the chip'
  hl = rule[(rule.statistic == 'flip_top1') & (rule.targets == 'all 60')]
  if set(hl.branch) != {rec}:
    raise SystemExit(f'sw reading_rule.csv headline changed: {set(hl.branch)}')
  add('7', 'software stack (jax 0.10.1) on the same devices', tsx('sw'),
      f'software stack on the v6e: {round(float(b6.estimate) * int(b6.n))} of {int(b6.n)} samples bit-identical; '
      f'software stack on the L4 minus recompile L4 {diff_ci(sg)}',
      'the software stack moves the top-1 less than v6e vs L4, which measures the platform stack')
  # FU8
  ans = d['seeds_answer']['all 60']['answer']
  r2 = seed_curve(d['q3']).set_index('n_seeds').loc[2]
  add('8', 'seeds 6 to 20 on the v5e and the L4', tsx('seeds'), f'v5e minus L4 success at n = 2: {diff_ci(r2)} (Fig. S3)',
      f'Q3 answer {ans}, not stable')
  # FU9
  h = d['h100_hw']
  cells = {}
  for s in ('flip_top1', 'flip_success', 'flip_matters'):
    for rc in ('RC(H)', 'RC(A)'):
      cells[s, rc] = pick(h, hardware='HW(H,A)', recompilation=rc, statistic=s, targets='all 60')
  rd = {s: {readings_interval(cells[s, rc]) for rc in ('RC(H)', 'RC(A)')} for s in ('flip_top1', 'flip_success',
                                                                                     'flip_matters')}
  if rd != {'flip_top1': {'beyond recompilation'}, 'flip_success': {'not beyond recompilation'},
            'flip_matters': {'not beyond recompilation'}}:
    raise SystemExit(f'FU9 readings changed: {rd}')
  head = '; '.join(f'{lab} {diff_ci(cells[s, "RC(H)"])} against recompile H100, {diff_ci(cells[s, "RC(A)"])} against '
                   'recompile A100'
                   for s, lab in (('flip_top1', 'top-1'), ('flip_success', 'success'), ('flip_matters', 'flips that matter')))
  add('9', 'H100, settings of the A100', tsx('h100'), f'H100 vs A100 minus recompilation: {head}',
      'top-1: beyond recompilation; success and flips that matter: not beyond recompilation (against recompilation on '
      'both GPUs). AlphaFold 3\'s documentation names the A100 and H100 as the GPUs on which numerical accuracy was verified')
  # FU10
  rule = d['large_rule']
  ws, hs = [], []
  for pair in ('HW(V5,V6)', 'HW(V6,G)'):
    r = pick(rule, pair=pair, statistic='flip_top1')
    if r.answer != 'no size difference detected at this sample size':
      raise SystemExit('large reading_rule.csv answer changed')
    hs.append(f'{clab(pair + " minus " + r.reference)} {diff_ci(r)} (width {pc(r.interval_width)})')
  add('10', 'large targets (1025--1536 tokens) against size controls (at most 1024)', tsx('large'),
      'large minus control: ' + '; '.join(hs), 'no size difference detected at this sample size')
  # FU11
  rule, h = d['repl_rule'], d['repl_hw']
  hs = [diff_ci(pick(h, hardware=p, recompilation=rc, statistic='flip_top1', targets='all 72'))
        for p, rc in (('HW(V5,V6)', 'RC(V5)'), ('HW(V6,G)', 'RC(V6)'))]
  hl = rule[rule.role == 'headline']
  if set(hl.answer) != {'replicates'}:
    raise SystemExit('repl headline readings changed')
  nrep = 'does not replicate (interval not above 0)'
  nn, nsec = int((rule.answer == nrep).sum()), int((rule.role == 'secondary').sum())
  assert int((rule[rule.role == 'headline'].answer == nrep).sum()) == 0
  add('11', '72 new targets, three arms rerun', tsx('repl'),
      f'v5e vs v6e minus recompile v5e {hs[0]}; v6e vs L4 minus recompile v6e {hs[1]}',
      f'top-1: replicates (both pairs); secondary: {nn} of {nsec} cells do not replicate (interval not above 0)')
  NOTES.append(f'Table IV FU11: {nn} of {len(rule)} reading cells do not replicate ({nn} of {nsec} secondary cells); '
               'figures_guide.md wrote "4 of 10 secondary cells", the ledger "4 of 10 cells". Table IV prints '
               f'"{nn} of {nsec} cells".')
  # FU12
  h, rule = d['prec_hw'], d['prec_rule']
  hs = [diff_ci(pick(h, hardware='HWh(V6,G)', recompilation='RC(V6)', statistic=s, targets='all 60', seeds='1 to 5'))
        for s in ('flip_top1', 'flip_success', 'flip_matters')]
  a60 = set(rule[(rule.targets == 'all 60') & (rule.seeds == '1 to 5')].reading)
  p39 = set(rule[(rule.targets == 'post-cutoff 39') & (rule.seeds == '1 to 5')].reading)
  if a60 != {'does not shrink to the RC level: attributed to the platform'} or \
     p39 != {'shrinks to the RC level: attributed to the default matmul precision'}:
    raise SystemExit('prec5 reading_rule.csv readings changed')
  nt = fu12_targets.target.nunique()
  # B6 (class B, 2026-11-17): the post-cutoff reading with its numbers, intervals from 0 (prec5 hw_minus_rc.csv)
  h39 = [pick(h, hardware='HWh(V6,G)', recompilation='RC(V6)', statistic=s, targets='post-cutoff 39', seeds='1 to 5')
         for s in ('flip_success', 'flip_matters')]
  if any(float(r.ci_low) != 0.0 for r in h39):
    raise SystemExit('Table IV FU12: a post-cutoff interval no longer starts at 0; the cell says it does')
  add('12', 'matmul precision highest on the v6e and the L4 (v6e highest, L4 highest), seeds 3 to 5', tsx('prec5'),
      f'v6e highest vs L4 highest minus recompile v6e: top-1 {hs[0]}; success {hs[1]}; flips that matter {hs[2]}',
      f'all 60: does not shrink to the recompilation level; 39 post-cutoff targets: success {diff_ci(h39[0])} and flips '
      f'that matter {diff_ci(h39[1])}, intervals from 0, read as shrinking to the recompilation level (events on {nt} '
      f'targets)')
  for row in FU1314_ROWS + FU15_ROW + FU16_ROW + [FU17_21_ROWS[k] for k in sorted(FU17_21_ROWS)]:
    add(*row)
  df = pd.DataFrame(rows)
  write_table(out, 'TIV_followups', df, 'Follow-ups 1 to 21, outside the pre-registered study and specified after the main '
              'results were read (FU10 and FU11 with OSF addenda registered before their runs): what changed, targets and '
              'seeds, headline with 95% interval, and the reading in the record\'s words (for Follow-ups 1 to 4, which have no '
              'reading rule, our description), with the paper\'s names for '
              'platforms and contrasts', ROLE_FU,
              colspec='rp{2.6cm}lp{6.2cm}p{5.0cm}',
              notes=['Rates in %, differences in points. "Minus recompile X" subtracts the recompilation rate on X. '
                     'Pairs of platforms measure the whole platform stack, not the chip alone.'])
  return df


# ---------------------------------------------------------------------------------------------------------------------
# A. Campaign counts and B. log census: the run records of the frozen study sessions

ABSL = re.compile(r'^([WEFID])(\d{4}) \d\d:\d\d:\d\d\.\d+\s+\d+ ([^\]:]+):\d+\] ?(.*)$')
SAMPLE_RE = re.compile(r'seed-\d+_sample-\d+/.*_model\.cif$')
COMPILE_RE = re.compile(r'(Finished |Compiling |Persistent compilation cache)')
# Message classes of the campaign ledger (a process holds a class if any line of its log matches)
CLASSES = [
    ('W compile logging (JAX_LOG_COMPILES)', lambda sev, src, msg, ln: sev == 'W' and src in ('dispatch.py', 'pxla.py')
     and msg.startswith(('Finished ', 'Compiling '))),
    ('UserWarning: int64 in broadcasted_iota truncated to int32', lambda sev, src, msg, ln:
     'Explicitly requested dtype int64 requested in broadcasted_iota' in ln),
    ('E api.py: Failed to run implementation (Tokamax GLU)', lambda sev, src, msg, ln: sev == 'E' and src == 'api.py'
     and 'Failed to run implementation' in msg),
    ('NotImplementedError: Not supported on <device>', lambda sev, src, msg, ln: 'NotImplementedError: Not supported on'
     in ln),
    ('W op.py: Autotuning cache miss for PallasTritonGatedLinearUnit', lambda sev, src, msg, ln: sev == 'W' and
     'Autotuning cache miss for PallasTritonGatedLinearUnit' in msg),
    ('W op.py: Autotuning cache miss for PallasTritonFlashAttention', lambda sev, src, msg, ln: sev == 'W' and
     'Autotuning cache miss for PallasTritonFlashAttention' in msg),
    ('E cuda_timer.cc: Delay kernel timed out', lambda sev, src, msg, ln: sev == 'E' and 'Delay kernel timed out' in msg),
    ('E hugepage_text.cc: File offset incorrectly aligned', lambda sev, src, msg, ln: sev == 'E' and
     'File offset incorrectly aligned' in msg),
    ('E cache.py: Failed to parse autotuning cache file', lambda sev, src, msg, ln: sev == 'E' and
     'Failed to parse autotuning cache file' in msg),
    ('W rdkit_utils.py: Failed to generate conformer', lambda sev, src, msg, ln: sev == 'W' and
     'Failed to generate conformer' in msg),
    ('W compiler.py: Persistent compilation cache hit', lambda sev, src, msg, ln: sev == 'W' and src == 'compiler.py' and
     'Persistent compilation cache hit' in msg),
    ("W hlo_rematerialization: Can't reduce memory use", lambda sev, src, msg, ln: sev == 'W' and
     "Can't reduce memory use below" in msg),
    ('E gpu_hlo_schedule.cc: byte size of input/output arguments', lambda sev, src, msg, ln: sev == 'E' and
     'The byte size of input/output arguments' in msg),
    ('W bfc_allocator: ran out of memory', lambda sev, src, msg, ln: sev == 'W' and 'ran out of memory' in msg and
     'bfc' in src),
    ('JaxRuntimeError: RESOURCE_EXHAUSTED', lambda sev, src, msg, ln: 'JaxRuntimeError: RESOURCE_EXHAUSTED' in ln),
]
# Ledger values (campaign ledger), processes per device in DEVICES order, for comparison only
LEDGER_CLASSES = {
    'W compile logging (JAX_LOG_COMPILES)': (3193, 3064, 5480, 1030, 600, 132),
    'UserWarning: int64 in broadcasted_iota truncated to int32': (3193, 3064, 5480, 1030, 600, 132),
    'E api.py: Failed to run implementation (Tokamax GLU)': (3193, 3064, 0, 0, 0, 132),
    'W op.py: Autotuning cache miss for PallasTritonGatedLinearUnit': (0, 0, 5480, 1030, 600, 0),
    'W op.py: Autotuning cache miss for PallasTritonFlashAttention': (0, 0, 305, 300, 0, 0),
    'E cuda_timer.cc: Delay kernel timed out': (0, 0, 1849, 488, 524, 0),
    'E hugepage_text.cc: File offset incorrectly aligned': (823, 1114, 0, 0, 0, 0),
    'E cache.py: Failed to parse autotuning cache file': (0, 0, 0, 0, 600, 0),
    'W rdkit_utils.py: Failed to generate conformer': (5, 5, 10, 0, 0, 0),
    'W compiler.py: Persistent compilation cache hit': (30, 30, 40, 10, 0, 0),
    "W hlo_rematerialization: Can't reduce memory use": (0, 0, 3, 0, 0, 0),
    'E gpu_hlo_schedule.cc: byte size of input/output arguments': (0, 0, 2, 0, 0, 0),
    'W bfc_allocator: ran out of memory': (0, 0, 2, 0, 0, 0),
    'JaxRuntimeError: RESOURCE_EXHAUSTED': (1, 1, 2, 0, 0, 0),
}
LEDGER_CLASS_LINES = dict(zip([c for c, _ in CLASSES], [7267132, 13499, 198077, 198077, 426672, 29040, 3377, 1937, 600,
                                                          40, 110, 19, 16, 32, 4]))
LEDGER_LEVELS = {'any warning-level line, including compile logging': (3193, 3064, 5480, 1030, 600, 132),
                 'any warning-level line, without compile logging': (35, 35, 5480, 1030, 600, 0),
                 'any error-level line': (3193, 3064, 1850, 488, 600, 132)}
WORDS = ['determin', 'reproduc', 'differ', 'hardware', 'numerical', 'unstable', 'bit-exact', 'fallback', 'fall back',
         'falling back', 'unsupported', 'deprecat']
WORDS_EXPLAINED = ['nan', 'accuracy', 'precision']


# ---------------------------------------------------------------------------------------------------------------------
# Follow-up 3: Q3 and M7 had no generated table; written here from the analysis's
# own q3.csv and memory.csv, cells copied as formatted values.

ROLE_FU3 = 'follow-up 3 (outside the pre-registered study; no reading rule)'
FU3_CONDS = ['V5', 'V6', 'G', 'GT', 'C', 'A', 'AT']


def table_fu03_q3_memory(out, q3, mem):
  q = q3.copy()
  q['n_seeds'] = q.n_seeds.astype(int)
  rows = []
  for n, g in q.groupby('n_seeds'):
    r = {'Seeds': n}
    for c in FU3_CONDS:
      x = g[(g.condition == c) & (g.statistic == 'success_rate')]
      r[c] = pc(x.estimate.iloc[0]) if len(x) else ''
    sp = g[g.statistic == 'spread_max_minus_min']
    r['Spread'] = pc(sp.estimate.iloc[0]) if len(sp) else ''
    rows.append(r)
  write_table(out, 'TS_fu03_q3', pd.DataFrame(rows),
              'Follow-up 3, Q3: success rate of the overall top-ranked sample over the first n seeds, per condition, '
              'with the A100 and the A100 Triton arm added, and the spread between conditions, in %', ROLE_FU3,
              notes=['The table uses 60 targets. The CPU ran seeds 1 and 2 only, so its column stops at n = 2. Pairwise '
                     'differences with intervals are in the next table.'], colspec='r' + 'r' * (len(FU3_CONDS) + 1))
  pairs = [c for c in q[q.statistic == 'success_difference'].condition.unique() if 'A' in c.split('-')]
  drows = []
  for pr in pairs:
    a_, b_ = pr.split('-')
    r = {'Pair': f'{ARM_NAME[a_]} minus {ARM_NAME[b_]}'}
    for n in sorted(q.n_seeds.unique()):
      x = q[(q.n_seeds == n) & (q.condition == pr) & (q.statistic == 'success_difference')]
      r[f'n = {n}'] = diff_ci(x.iloc[0]) if len(x) and not pd.isna(x.ci_low.iloc[0]) else ''
    drows.append(r)
  write_table(out, 'TS_fu03_q3diff', pd.DataFrame(drows),
              'Follow-up 3, Q3: differences in the success rate of the overall top-ranked sample for every pair with '
              'the A100 or the A100 Triton arm, percentage points, with 95% intervals', ROLE_FU3,
              notes=['Q3 asks for the smallest n at which every pairwise interval lies within plus or minus 2 points.'],
              colspec='l' + 'l' * q.n_seeds.nunique())
  mrows = []
  for r in mem.itertuples():
    mrows.append({'Arm': r.platform, 'Bucket': int(r.bucket), 'Processes': int(r.processes),
                  'Peak plus arguments (GiB)': sig3(r.compiled_peak_plus_args_gib),
                  'Compiled peak (GiB)': sig3(r.compiled_peak_gib),
                  'Allocator peak in use (GiB)': sig3(r.alloc_peak_in_use_gib),
                  'Device limit (GiB)': sig3(r.device_limit_gib),
                  'Fraction of limit': f'{float(r.peak_plus_args_fraction_of_limit):.3f}'})
  write_table(out, 'TS_fu03_memory', pd.DataFrame(mrows),
              'Follow-up 3, M7: device memory of the A100, A100 Triton, and A100 highest arms per token bucket against '
              'the device limit', ROLE_FU3,
              notes=['Medians per bucket. On GPUs the compiled peak already includes the arguments, so peak plus '
                     'arguments counts them twice (as TABREFQTWO notes).'], colspec='lrrrrrrr')
  return len(rows), len(drows), len(mrows)


# ---------------------------------------------------------------------------------------------------------------------
# Results that do not exist yet (rows of 2026-11-10): follow-up 13 (label why, analysis/fu13.py) and the memory limit
# on A and H (label memlimit_ah, analysis in its own row). Both readers return None while the output folder is absent,
# so the generator then writes nothing for them and Table III keeps "not measured" for A and H. Neither reads a
# session: they read only the analysis outputs, which exist only after the analysis rows.

ROLE_FU13 = ('follow-up 13 (outside the pre-registered study; specified after the results were read; OSF addendum '
             'registered before its runs)')
FU13_FILES = ['checks.csv', 'completeness.csv', 'p_curve.csv', 'p_eps_star.csv', 'l_summary.csv', 'l_first_depth.csv',
              'l_growth.csv', 'f_contrasts.csv', 'f_reading.csv', 'f_secondary.csv', 'f_success.csv', 'f_cost.csv', 'failures.csv',
              'report.md']
L_STYLE = {'V5-V6': ('o', '-'), 'V6-G': ('s', (0, (3, 1.5)))}
L_COLORS = {'V5-V6': NAVY, 'V6-G': ROSE, 'V5-G': GOLD}
MEMLIMIT_COLUMNS = ['arm', 'session', 'sizes_run', 'largest_completed_size', 'stopping_size', 'cause']


def one_output_dir(main_dir, prefix, required):
  """The one results/analysis/<prefix>_<YYYYMMDD>/ folder that holds every required file, or None if there is none.
  Two such folders, or a folder with only some of the files, stop the generator."""
  base = main_dir / AN
  if not base.is_dir():
    return None
  found = sorted(d for d in base.iterdir() if d.is_dir() and re.fullmatch(rf'{prefix}_\d{{8}}', d.name))
  if not found:
    return None
  if len(found) > 1:
    raise SystemExit(f'{prefix}: more than one output folder {[d.name for d in found]}; name the one to use')
  missing = [f for f in required if not (found[0] / f).is_file()]
  if missing:
    raise SystemExit(f'{found[0].name}: missing {missing}')
  return found[0]


def read_opt(main_dir, rel):
  """read() that returns an empty frame for a CSV written from an empty DataFrame."""
  try:
    return read(main_dir, rel)
  except pd.errors.EmptyDataError:
    return pd.DataFrame()


def memlimit_ah_summary(main_dir):
  """summary.csv of the memory-limit analysis on A and H (format of memlimit_20261014/summary.csv), or None."""
  d = one_output_dir(main_dir, 'memlimit_ah', ['summary.csv'])
  if d is None:
    return None
  sm = read(main_dir, str((d / 'summary.csv').relative_to(main_dir)))
  if [c for c in MEMLIMIT_COLUMNS if c not in sm.columns] or not set(sm.arm) <= {'A', 'H'} or sm.arm.duplicated().any():
    raise SystemExit(f'{d.name}/summary.csv: columns {list(sm.columns)}, arms {list(sm.arm)}')
  return sm


WARM_TH_COLUMNS = ['platform', 'bucket', 'cache', 'processes', 'median_usd_per_structure_standard',
                   'median_usd_per_structure_spot']


def warm_th_summary(main_dir):
  """The warm rows at 1024 of the warm-cost completion on GT and H (results/analysis/warm_th_<DATE>/summary.csv, the
  columns of the main study's cost_summary.csv; analysis/warm_th.py writes it only when every check passed), or None.
  Its checks.csv must read pass on every row; one warm row per arm, GT and H, is required. Row of 2026-11-11."""
  d = one_output_dir(main_dir, 'warm_th', ['summary.csv', 'checks.csv'])
  if d is None:
    return None
  rel = lambda f: str((d / f).relative_to(main_dir))  # noqa: E731
  ck = read(main_dir, rel('checks.csv'))
  if not len(ck) or (ck.detail != 'pass').any():
    raise SystemExit(f'{d.name}/checks.csv: a check did not pass')
  sm = read(main_dir, rel('summary.csv'))
  missing = [c for c in WARM_TH_COLUMNS if c not in sm.columns]
  w = sm[(sm.cache == 'warm') & (sm.bucket == 1024)] if not missing else sm
  if missing or sorted(w.platform) != ['GT', 'H']:
    raise SystemExit(f'{d.name}/summary.csv: columns missing {missing}, warm arms at 1024 {sorted(w.platform)}')
  NOTES.append(f'Table III: warm cells of GT and H from {d.name}/summary.csv (analysis/warm_th.py; every check of '
               'checks.csv passed).')
  return w.reset_index(drop=True)


def memlimit_cell(ml):
  """Table III's last cell for one arm: 'largest completed; first failed' in tokens, 'at least N' when no size failed,
  'not measured' without a row."""
  if ml is None or len(ml) == 0:
    return 'not measured'
  r = ml.iloc[0]
  done = 'none' if pd.isna(r.largest_completed_size) else f'{int(r.largest_completed_size)}'
  if pd.isna(r.stopping_size):
    return f'at least {done}'
  return f'{done}; {int(r.stopping_size)}'


def fu13_dir(main_dir):
  return one_output_dir(main_dir, 'fu13', FU13_FILES)


def fu13_numbers(main_dir, d):
  """The numbers the paper's text and caption take from follow-up 13, one row each (quantity, estimate, interval, note)."""
  rel = lambda f: str((d / f).relative_to(main_dir))  # noqa: E731
  eps = read_opt(main_dir, rel('p_eps_star.csv'))
  fr = read_opt(main_dir, rel('f_reading.csv'))
  fd = read_opt(main_dir, rel('l_first_depth.csv'))
  gr = read_opt(main_dir, rel('l_growth.csv'))
  fc = read_opt(main_dir, rel('f_cost.csv'))
  fails = read_opt(main_dir, rel('failures.csv'))
  rows = []
  for r in eps.itertuples() if len(eps) else []:
    if int(r.n_targets) == 0:
      rows.append({'quantity': f'eps* {r.pair}, {r.statistic}, HW seeds {r.hw_seeds}', 'estimate': np.nan,
                   'ci_low': np.nan, 'ci_high': np.nan, 'note': 'not computed (no common targets)'})
      continue
    rows.append({'quantity': f'eps* {r.pair}, {r.statistic}, HW seeds {r.hw_seeds} (units of u)',
                 'estimate': r.eps_star_u, 'ci_low': r.ci_low_u, 'ci_high': r.ci_high_u,
                 'note': f'{r.status}; HW rate {r.hw_rate}; {int(r.censored_replicates)} of {int(r.replicates)} '
                         f'replicates at a bound'})
  for r in fr.itertuples() if len(fr) else []:
    rows.append({'quantity': f'D_F, {r.statistic} ({r.role})', 'estimate': r.estimate, 'ci_low': r.ci_low,
                 'ci_high': r.ci_high, 'note': f'{r.reading}; HW(V6f,Gf) {r.hw_f}, RC(Gf) {r.rc_f}, HW(V6,G) {r.hw}, '
                                               f'RC(G) {r.rc}; {int(r.n_targets)} targets'})
  for (pair, kind), g in fd.groupby(['pair', 'kind']) if len(fd) else []:
    v = g.first_depth.dropna()
    rows.append({'quantity': f'first depth, recycle 0, {pair}, {kind}', 'estimate': float(v.median()) if len(v) else np.nan,
                 'ci_low': float(v.min()) if len(v) else np.nan, 'ci_high': float(v.max()) if len(v) else np.nan,
                 'note': f'median (range) over {len(v)} of {len(g)} targets with a first depth'})
  for (pair, kind), g in gr.groupby(['pair', 'kind']) if len(gr) else []:
    v = g.slope_bits_per_block.dropna()
    rows.append({'quantity': f'growth, recycle 0, {pair}, {kind} (bits per block)',
                 'estimate': float(v.median()) if len(v) else np.nan, 'ci_low': float(v.min()) if len(v) else np.nan,
                 'ci_high': float(v.max()) if len(v) else np.nan, 'note': f'median (range) over {len(v)} targets'})
  for r in fc[fc.bucket.astype(str) == 'all'].itertuples() if len(fc) else []:
    for m in ('wall', 'compiled_memory', 'peak_reserved'):
      rows.append({'quantity': f'cost ratio {r.new} to {r.reference}, {m}, all buckets', 'estimate': getattr(r, f'{m}_ratio'),
                   'ci_low': np.nan, 'ci_high': np.nan, 'note': f'{int(r.n_new)} and {int(r.n_reference)} processes'})
  if len(fails):
    for c, g in fails.groupby('condition'):
      rows.append({'quantity': f'failed processes, {c}', 'estimate': len(g), 'ci_low': np.nan, 'ci_high': np.nan,
                   'note': '; '.join(sorted(set(str(x) for x in g.failure.fillna('')))[:3])[:300]})
  return pd.DataFrame(rows, columns=['quantity', 'estimate', 'ci_low', 'ci_high', 'note'])


def fu13_rel(main_dir, d):
  return lambda f: str((d / f).relative_to(main_dir))


def fu13_trunk_out(ls_):
  """Median difference at the trunk output of the last recycle (the largest depth captured there), per pair and kind."""
  last = ls_[ls_.recycle == ls_.recycle.max()]
  out = {}
  for (pair, kind), g in last.groupby(['pair', 'kind']):
    out[(pair, kind)] = float(g.loc[g.depth.idxmax(), 'median_u'])
  return out


def fu13_bars(main_dir, d):
  """Panel (b) of the follow-up 13 figure: top-1 flip rates with the bfloat16 trunk (main study; highest precision from
  follow-up 12, seeds 1 to 5) and with the float32 trunk (follow-up 13), on 60 targets with seeds 1 to 5."""
  rel = fu13_rel(main_dir, d)
  main = read(main_dir, f'{MAINV2}/contrasts/contrasts.csv')
  prec = read(main_dir, f'{AN}/prec5_20261107/contrasts.csv')
  prec = prec[(prec.seeds.astype(str) == '1 to 5') & (prec.targets.astype(str) == 'all 60')]
  fc = read(main_dir, rel('f_contrasts.csv'))
  return [('v6e vs L4', 'bf16', pick(main, contrast='HW(V6,G)', statistic='flip_top1')),
          ('v6e vs L4', 'bf16h', pick(prec, contrast='HWh(V6,G)', statistic='flip_top1')),
          ('v6e vs L4', 'f32', pick(fc, contrast='HW(V6f,Gf)', statistic='flip_top1')),
          ('Recompile, L4', 'bf16', pick(main, contrast='RC(G)', statistic='flip_top1')),
          ('Recompile, L4', 'f32', pick(fc, contrast='RC(Gf)', statistic='flip_top1'))]


FU13_BAR_STYLE = {'bf16': ('bfloat16 trunk', NAVY, None), 'bf16h': ('bfloat16 trunk, highest precision', BAND, None),
                  'f32': ('float32 trunk, highest precision', GOLD, None)}


def fu14_bar_rows(main_dir, d):
  """Panel (b) of Fig. 2: top-1 flip with the bfloat16 trunk (default precision; main study, follow-ups 3 and 9) and
  with the float32 trunk (follow-ups 13 and 14) for the four primary pairs of follow-up 14 and recompilation on the L4.
  A pair whose source is absent (synthetic tests) is left out."""
  def row(src, contrast, **kw):
    if not (main_dir / src).is_file():
      return None
    df = read(main_dir, src)
    m = df[(df.contrast == contrast) & (df.statistic == 'flip_top1')]
    for k, v in kw.items():
      m = m[m[k] == v]
    if len(m) != 1:
      raise SystemExit(f'Fig. 2(b): {contrast} in {src}: {len(m)} rows')
    r = m.iloc[0]
    return {'est': r.estimate, 'lo': r.ci_low, 'hi': r.ci_high}

  def agree(pair):
    src = f'{FU14}/agree.csv'
    if not (main_dir / src).is_file():
      return None
    r = pick(read(main_dir, src), pair=pair)
    return {'est': r.flip_top1, 'lo': r.ci_low, 'hi': r.ci_high}
  mc = f'{MAINV2}/contrasts/contrasts.csv'
  f13 = str((d / 'f_contrasts.csv').relative_to(main_dir))
  rows = [('v5e vs v6e', row(mc, 'HW(V5,V6)'), agree('HW(V5f,V6f)')),
          ('v6e vs L4', row(mc, 'HW(V6,G)'), row(f13, 'HW(V6f,Gf)')),
          ('H100 vs A100', row(f'{AN}/h100_20261104/contrasts.csv', 'HW(H,A)', n=300), agree('HW(Hf,Af)')),
          ('L4 vs A100', row(f'{AN}/a100_20261018/contrasts.csv', 'HW(G,A)'), agree('HW(Gf,Af)')),
          ('Recompile, L4', row(mc, 'RC(G)'), row(f13, 'RC(Gf)'))]
  return [r for r in rows if r[1] is not None and r[2] is not None]


FU13_CLEAR_PT = 2.0      # Fig. 2 (R120b): clearance between title, legend, axes and the next panel's decorations
FU13_EDGE_PT = 1.0       # and between the outermost texts and the figure's edges
FU13_AX_IN = (1.15, 0.75)  # axes heights of panels (a) and (b), in inches (R120b; the R120 preview had 1.276, 0.834)
FU13_LEFT = 0.27
GREEN = '#228833'   # no longer drawn (2026-11-19): in grayscale it matched ROSE; the pair marks are GOLD squares


def fu13_prim_seeds12(main_dir, d, main):
  """P22 (class B, 2026-11-17): the band of Fig. 2(a) uses the two primary pairs' top-1 flip rates on seeds 1 and 2,
  the seeds of the perturbation curve, with the study's percentile bootstrap over targets (analysis/contrasts.py
  boot_ci, 10,000 replicates, seed 2026) on the main study's pairs_seed.csv. Checks: the same code on seeds 1 to 5
  reproduces contrasts.csv (estimate and interval), and the seeds 1 and 2 estimate equals hw_rate of follow-up 13's
  p_eps_star.csv. Returns one row per pair with contrast, estimate, ci_low, ci_high."""
  from contrasts import boot_ci, mean as cmean
  sp = read(main_dir, f'{MAINV2}/contrasts/pairs_seed.csv')
  eps = read_opt(main_dir, fu13_rel(main_dir, d)('p_eps_star.csv'))
  rows = []
  for c in ('HW(V5,V6)', 'HW(V6,G)'):
    out = {}
    for seeds in ((1, 2, 3, 4, 5), (1, 2)):
      x = sp[(sp.contrast == c) & (sp.seed_a.isin(seeds))]
      groups = [x[x.target == t].flip_top1.astype(float).values for t in sorted(x.target.unique())]
      lo, hi = boot_ci(groups, cmean, np.random.default_rng(2026), 10000)
      out[seeds] = (cmean(x.flip_top1.astype(float).values), lo, hi)
    reg = pick(main, contrast=c, statistic='flip_top1')
    if any(abs(u - float(v)) > 1e-9 for u, v in zip(out[(1, 2, 3, 4, 5)], (reg.estimate, reg.ci_low, reg.ci_high))):
      raise SystemExit(f'Fig. 2(a): the bootstrap of {c} on seeds 1 to 5 does not reproduce contrasts.csv')
    if len(eps):
      e = eps[(eps.pair == c) & (eps.statistic == 'flip_top1') & (eps.hw_seeds == '1 and 2')]
      if len(e) != 1 or abs(float(e.hw_rate.iloc[0]) - out[(1, 2)][0]) > 1e-9:
        raise SystemExit(f'Fig. 2(a): the seeds 1 and 2 rate of {c} differs from p_eps_star.csv')
    est, lo, hi = out[(1, 2)]
    rows.append(types.SimpleNamespace(contrast=c, estimate=est, ci_low=lo, ci_high=hi))
  NOTES.append('Fig. 2(a) band, seeds 1 and 2: ' + '; '.join(
      f'{r.contrast} {pc(r.estimate)} [{pc(r.ci_low)}, {pc(r.ci_high)}]' for r in rows))
  return rows


def fig_fu13(out, save, plt, main_dir, d):
  """The follow-up 13 figure of the main text, one column: (a) the perturbation curve of the top-1 flip rate against
  eps, with the 95% intervals of the two primary pairs' rates (seeds 1 and 2, fu13_prim_seeds12) as a band and the trunk-output
  differences measured by the layer-capture arm as triangles; (b) top-1 flip rates with the bfloat16 and the float32
  trunk.
  Layout of 2026-11-16: (a) horizontal y label on two lines; the
  measured trunk-output differences on the x axis, single as ROSE triangles and pair as GOLD squares (2026-11-19); ylim 0 to 50 (axis
  and ticks 0 to 40); legend above the panel, under its title, in two rows of three columns, at the left edge of the
  label area. (b) points and 95% intervals as in Fig. 1 (bfloat16 NAVY above, float32 GOLD below) on a square-root x
  scale from 0 to 26; rows at 4, 3, 2, 1 and recompilation on the L4 at -0.25, with thin rules in the label area only;
  one-row legend between the title and the panel. The height is the least that keeps FU13_CLEAR_PT between every title,
  legend, axes and the decorations of the panel above (two passes: draw, measure, place); the generator stops if any
  clearance is short, if a drawn interval leaves its axis, or if a text leaves the figure.
  For space: clearances 2 pt (FU13_CLEAR_PT) and panel heights 1.15 and 0.75 in (FU13_AX_IN), about 3.46 in in all;
  everything else as above."""
  from matplotlib.lines import Line2D
  from matplotlib.patches import Patch
  rel = fu13_rel(main_dir, d)
  curve = read_opt(main_dir, rel('p_curve.csv'))
  ls_ = read_opt(main_dir, rel('l_summary.csv'))
  main = read(main_dir, f'{MAINV2}/contrasts/contrasts.csv')
  prim = fu13_prim_seeds12(main_dir, d, main)
  rcg = pick(main, contrast='RC(G)', statistic='flip_top1')
  g = curve[curve.statistic == 'flip_top1'].sort_values('eps_u')
  marks = fu13_trunk_out(ls_) if len(ls_) else {}
  marks = {k: x for k, x in sorted(marks.items()) if k[0] in ('V5-V6', 'V6-G')}
  bars = fu13_bars(main_dir, d)
  rows = fu14_bar_rows(main_dir, d)
  ys = [4.0, 3.0, 2.0, 1.0, -0.25][:len(rows)] if len(rows) == 5 else list(np.arange(len(rows))[::-1].astype(float))
  # axis limits; the generator stops if a drawn interval leaves them
  A_YMAX, B_XMAX = 50.0, 26.0
  lo, hi = min(float(r.ci_low) for r in prim), max(float(r.ci_high) for r in prim)
  if len(g) and (100 * g.ci_high.max() > A_YMAX or 100 * g.ci_low.min() < 0 or 100 * hi > A_YMAX):
    raise SystemExit(f'Fig. 2(a): an interval leaves the axis 0 to {A_YMAX}')
  for _, b16, f32 in rows:
    for r in (b16, f32):
      if r is not None and (100 * float(r['hi']) > B_XMAX or float(r['lo']) < 0):
        raise SystemExit(f'Fig. 2(b): an interval leaves the axis 0 to {B_XMAX}')
  xa_lim = (2.0 ** -6.6, 2.0 ** 3.6)
  if any(not (xa_lim[0] <= x <= xa_lim[1]) for x in marks.values()):
    raise SystemExit('Fig. 2(a): a measured trunk-output difference leaves the x axis')
  PT = 72.0
  h1, h2 = FU13_AX_IN

  def draw(top, gap, bot, pad_a, pad_b):
    """One drawing with the given margins (points); returns the figure and its measured parts."""
    H = (top + gap + bot) / PT + h1 + h2
    fig = plt.figure(figsize=(3.5, H))
    a = fig.add_axes([FU13_LEFT, (bot / PT + h2 + gap / PT) / H, 0.97 - FU13_LEFT, h1 / H])
    b = fig.add_axes([FU13_LEFT, bot / PT / H, 0.97 - FU13_LEFT, h2 / H])
    twin = []
    # (a)
    a.axhspan(100 * lo, 100 * hi, color=BAND, lw=0, zorder=0)
    for r in prim:
      twin.append({'panel': 'a', 'series': f'{r.contrast} seeds 1 and 2', 'x': np.nan, 'y': r.estimate,
                   'y_low': r.ci_low, 'y_high': r.ci_high})
    a.axhline(100 * float(rcg.estimate), color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    twin.append({'panel': 'a', 'series': 'RC(G) seeds 1 to 5', 'x': np.nan, 'y': rcg.estimate, 'y_low': rcg.ci_low,
                 'y_high': rcg.ci_high})
    a.errorbar(g.eps_u, 100 * g.estimate, yerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)],
               color=NAVY, marker='o', ms=3.2, lw=0.9, elinewidth=0.7, capsize=0, zorder=3)
    twin += [{'panel': 'a', 'series': 'flip_top1', 'x': r.eps_u, 'y': r.estimate, 'y_low': r.ci_low,
              'y_high': r.ci_high} for r in g.itertuples()]
    for (pair, kind), x in marks.items():
      a.plot([x], [0], marker='^' if kind == 'single' else 's', ms=5 if kind == 'single' else 4.2,
             color=ROSE if kind == 'single' else GOLD, mec='none', ls='none',
             clip_on=False, zorder=5)
      twin.append({'panel': 'a', 'series': f'trunk output {pair} {kind}', 'x': x, 'y': np.nan, 'y_low': np.nan,
                   'y_high': np.nan})
    a.set_xscale('log', base=2)
    a.set_xlim(*xa_lim)
    a.set_xticks([2.0 ** k for k in range(-6, 4)])
    a.set_xticklabels(['1/64', '', '1/16', '', '1/4', '', '1', '', '4', ''])
    a.minorticks_off()
    a.set_ylim(0, A_YMAX)
    a.set_yticks([0, 10, 20, 30, 40])
    a.spines['left'].set_bounds(0, 40)
    a.set_xlabel('Perturbation size (units of u)')
    a.set_ylabel('Top-1\nflip (%)', rotation=0, ha='right', va='center', multialignment='center')
    ta = a.set_title('(a) Perturbation arm, v6e', loc='left', fontsize=FS, style='italic', pad=pad_a)
    tri = lambda col, m='^', ms=5: Line2D([], [], marker=m, ms=ms, color=col, mec='none', ls='none')  # noqa: E731
    # two rows of three columns, filled column by column: (Perturbation, Single), (Primary pairs, Pair), (Recompile)
    la = fig.legend([Line2D([], [], color=NAVY, marker='o', ms=3.2, lw=0.9), tri(ROSE), Patch(color=BAND, lw=0),
                     tri(GOLD, 's', 4.2), Line2D([], [], color=INK, lw=0.7, ls=(0, (3, 2)))],
                    ['Perturbation', 'Single, measured', 'Primary pairs', 'Pair, measured', 'Recompile, L4'],
                    loc='lower left', bbox_to_anchor=(FU13_EDGE_PT / PT / 3.5, a.get_position().y1 + FU13_CLEAR_PT / PT / H),
                    ncol=3, fontsize=8, handletextpad=0.4, borderaxespad=0, borderpad=0, labelspacing=0.25,
                    handlelength=1.4, columnspacing=1.2)
    # (b)
    for y, (lab, b16, f32) in zip(ys, rows):
      for dy, r, col in ((0.17, b16, NAVY), (-0.17, f32, GOLD)):
        if r is None:
          continue
        b.plot([100 * float(r['lo']), 100 * float(r['hi'])], [y + dy] * 2, color=col, lw=1.0, solid_capstyle='butt',
               clip_on=False, zorder=3)
        b.plot(100 * float(r['est']), y + dy, marker='o' if col == NAVY else 's', ms=3.8 if col == NAVY else 3.6,
               color=col, mec='white', mew=0.3, ls='none',
               clip_on=False, zorder=4)
        twin.append({'panel': 'b', 'series': f'{lab} {"bfloat16" if col == NAVY else "float32"}', 'x': y,
                     'y': r['est'], 'y_low': r['lo'], 'y_high': r['hi']})
    b.set_xscale('function', functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square))
    b.set_xlim(0, B_XMAX)
    b.set_xticks([0, 1, 5, 10, 20])
    b.set_xticklabels(['0', '1', '5', '10', '20'])
    b.minorticks_off()
    b.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    b.set_yticks(ys)
    b.set_yticklabels([r[0] for r in rows])
    b.tick_params(axis='y', length=0)
    b.spines['left'].set_visible(False)
    b.axvline(0, color=INK, lw=0.6, zorder=1)
    b.set_xlabel('Top-1 flip (%), with 95% interval')
    tb = b.set_title('(b) Float32 trunk arms', loc='left', fontsize=FS, style='italic', pad=pad_b)
    dot = lambda col, m='o', ms=3.8: Line2D([], [], color=col, lw=1.0, marker=m, ms=ms, mec='white', mew=0.3)  # noqa: E731
    lb = fig.legend([dot(NAVY), dot(GOLD, 's', 3.6)], ['bfloat16 trunk', 'float32 trunk, highest'], loc='lower left',
                    bbox_to_anchor=(FU13_LEFT, b.get_position().y1 + FU13_CLEAR_PT / PT / H), ncol=2, fontsize=8,
                    handletextpad=0.4, borderaxespad=0, borderpad=0, handlelength=1.4, columnspacing=1.2)
    # rules between rows, in the label area only
    for y0, y1 in zip(ys[:-1], ys[1:]):
      yf = fig.transFigure.inverted().transform(b.transData.transform((0, (y0 + y1) / 2)))[1]
      fig.add_artist(Line2D([0.0, FU13_LEFT - 0.012], [yf, yf], transform=fig.transFigure, color=RULE_LIGHT, lw=0.3))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = PT / fig.dpi
    m = {'H_pt': H * PT,
         'title_a': [v * k for v in ta.get_window_extent(r).extents],
         'leg_a': [v * k for v in la.get_window_extent(r).extents],
         'ax_a': [v * k for v in a.get_window_extent(r).extents],
         'deco_a': [v * k for v in a.get_tightbbox(r).extents],
         'title_b': [v * k for v in tb.get_window_extent(r).extents],
         'leg_b': [v * k for v in lb.get_window_extent(r).extents],
         'ax_b': [v * k for v in b.get_window_extent(r).extents],
         'deco_b': [v * k for v in b.get_tightbbox(r).extents]}
    return fig, twin, m

  def gaps(m):
    c = {'title a above legend a': m['title_a'][1] - m['leg_a'][3],
         'legend a above panel a': m['leg_a'][1] - m['ax_a'][3],
         'panel a labels above title b': m['deco_a'][1] - m['title_b'][3],
         'title b above legend b': m['title_b'][1] - m['leg_b'][3],
         'legend b above panel b': m['leg_b'][1] - m['ax_b'][3]}
    e = {'title a below the top': m['H_pt'] - m['title_a'][3], 'panel b labels above the bottom': m['deco_b'][1]}
    return c, e
  top, gap, bot, pad_a, pad_b = 40.0, 50.0, 30.0, 27.0, 18.0
  for _ in range(4):
    fig, twin, m = draw(top, gap, bot, pad_a, pad_b)
    c, e = gaps(m)
    placed = [v for k_, v in c.items() if not k_.startswith('legend')]   # the legends sit at their anchors
    if all(abs(v - FU13_CLEAR_PT) < 0.05 for v in placed) and all(abs(v - FU13_EDGE_PT) < 0.05 for v in e.values()):
      break
    plt.close(fig)
    pad_a += FU13_CLEAR_PT - c['title a above legend a']
    pad_b += FU13_CLEAR_PT - c['title b above legend b']
    top += (FU13_CLEAR_PT - c['title a above legend a']) + FU13_EDGE_PT - e['title a below the top']
    gap += (FU13_CLEAR_PT - c['title b above legend b']) + FU13_CLEAR_PT - c['panel a labels above title b']
    bot += FU13_EDGE_PT - e['panel b labels above the bottom']
  else:
    raise SystemExit(f'Fig. 2: the layout did not settle: {c} {e}')
  if any(v < FU13_CLEAR_PT - 0.05 for v in c.values()) or any(v < FU13_EDGE_PT - 0.05 for v in e.values()):
    raise SystemExit(f'Fig. 2: a title, legend or panel touches another: {c} {e}')
  if m['leg_a'][2] > 3.5 * PT or m['leg_b'][2] > 3.5 * PT:
    raise SystemExit('Fig. 2: a legend leaves the figure on the right')
  FU13_LAYOUT.clear()
  FU13_LAYOUT.update({'height_in': round(m['H_pt'] / PT, 3), **{k: round(v, 2) for k, v in {**c, **e}.items()}})
  print(f'NOTE Fig. 2 layout: height {m["H_pt"] / PT:.3f} in; clearances (pt): '
        + ', '.join(f'{k} {v:.2f}' for k, v in {**c, **e}.items()), flush=True)
  save(fig, 'fig_fu13', ROLE_FU13, 'Follow-up 13: perturbation curve against eps with the primary pairs\' rates and '
       'the measured trunk-output differences; top-1 flip rates with the bfloat16 and the float32 trunk')
  write_csv(out, 'fig_fu13', pd.DataFrame(twin, columns=['panel', 'series', 'x', 'y', 'y_low', 'y_high']), ROLE_FU13,
            f'CSV twin of the follow-up 13 figure ({d.name}/p_curve.csv, l_summary.csv, f_contrasts.csv; '
            f'{MAINV2}/contrasts/contrasts.csv; {FU14}/agree.csv; {AN}/a100_20261018/contrasts.csv; '
            f'{AN}/h100_20261104/contrasts.csv; panel b: x is the row position)')
  return {'p_levels': int(len(g)), 'l_drawn': bool(marks), 'marks': marks, 'prim': prim, 'bars': bars}


FU13_LAYOUT = {}


L_PANELS = [('single', 0), ('single', 10), ('pair', 0), ('pair', 10)]


def fig_fu13_l(out, save, plt, main_dir, d):
  """Supplementary figure: the layer-capture arm's median difference (range over the 10 pilot targets as a band) against
  Pairformer depth, single and pair representation, first and last recycle, for the two primary pairs."""
  rel = fu13_rel(main_dir, d)
  ls_ = read_opt(main_dir, rel('l_summary.csv'))
  if not len(ls_):
    return False
  # Note 3 of 18 November 2026: the y labels stand above the panel titles; the figure is 0.4 in taller (0.2 in at the
  # top, 0.2 in between the rows), and the panels keep their size (1.36 in high)
  fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5), sharex=True)
  fig.subplots_adjust(left=0.09, right=0.99, top=0.846, bottom=0.1, wspace=0.22, hspace=0.467)
  twin = []
  for ax, (kind, rec) in zip(axes.flat, L_PANELS):
    for pair in ('V5-V6', 'V6-G'):
      g = ls_[(ls_.pair == pair) & (ls_.kind == kind) & (ls_.recycle == rec)].sort_values('depth')
      g = g[g.median_u > 0]
      col = L_COLORS[pair]
      ax.fill_between(g.depth, g.min_u.clip(lower=g.median_u.min() / 10), g.max_u, color=col, alpha=0.15, lw=0)
      mk, ln = L_STYLE[pair]     # marker and line differ by pair, so the figure reads in grayscale (2026-11-19)
      ax.plot(g.depth, g.median_u, color=col, lw=1.0, ls=ln, marker=mk, ms=2.0, label=pair_name(pair))
      twin += [{'panel': f'{kind} r{rec}', 'series': pair, 'x': r.depth, 'y': r.median_u, 'y_low': r.min_u,
                'y_high': r.max_u} for r in g.itertuples()]
    ax.set_yscale('log')
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.relim()
    ax.autoscale_view()
    lo_, hi_ = ax.get_ylim()
    wide = hi_ / max(lo_, 1e-12) > 30      # over more than about a decade and a half, only the decades are labeled
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _, wide=wide: '' if wide or f'{v:g}'[0] not in '2346' else f'{v:g}'))
    ax.set_title(f'{"Single" if kind == "single" else "Pair"} representation, '
                 f'{"first" if rec == 0 else "last"} recycle', fontsize=FS, style='italic', loc='left', pad=3)
    ax.set_xlim(-1, 49)
    ax.set_xticks([0, 12, 24, 36, 48])
  for ax in axes[:, 0]:
    ax.set_ylabel('Difference (units of u)')
  for ax in axes[1, :]:
    ax.set_xlabel('Pairformer depth (0, its input; 48, the trunk output)')
  h, lab = axes[0, 0].get_legend_handles_labels()
  fig.legend(h, lab, loc='upper center', ncol=2, handlelength=2.0, columnspacing=1.4, borderaxespad=0.2)
  for ax in axes[:, 0]:
    ylabel_top(fig, ax, 'Difference (units of u)', grow=False, align=list(axes[:, 0]))
  save(fig, 'fig_fu13_l', ROLE_FU13, 'Follow-up 13, layer capture: median relative difference against Pairformer depth')
  write_csv(out, 'fig_fu13_l', pd.DataFrame(twin, columns=['panel', 'series', 'x', 'y', 'y_low', 'y_high']), ROLE_FU13,
            f'CSV twin of the supplementary layer-capture figure ({d.name}/l_summary.csv)')
  return True


def u_frac(x):
  """eps in units of u as the text prints it: 1/64, 1/8, 1, 2, 4, 8."""
  x = float(x)
  return f'1/{round(1 / x)}' if x < 1 else f'{x:g}'


def u2(x):
  return f'{float(x):.2g}' if float(x) < 1 else f'{float(x):.1f}'


FU13_CHECK_NAMES = {'1': 'Every process exits 0 or its failure is recorded', '2': 'One machine type per condition',
                    '4': 'Perturbation at size 0 against the main study\'s v6e', '5': 'Layer capture on a TPU against the '
                    'main study', '6': 'Capture files complete, same token subset', '7': 'v6e float32, second run '
                    'against the first'}
FU13_COND = {'GL': 'L4, layer capture', 'Gf': 'L4 float32', 'V5L': 'v5e, layer capture', 'V6L': 'v6e, layer capture',
             'V6f': 'v6e float32', 'P0': 'perturbation, size 0'}


def fu13_tables(out, main_dir, d, fig):
  """The supplement's tables of follow-up 13 (results as analysis/fu13.py wrote them, formatted)."""
  rel = fu13_rel(main_dir, d)
  # checks: one row per check, conditions grouped
  ck = read(main_dir, rel('checks.csv'))
  rows = []
  for num, g in ck.groupby(ck.check.astype(str).str.split(' ').str[0], sort=False):
    conds = list(g.condition)
    perts = [c for c in conds if re.fullmatch(r'P\d+', c)]
    named = [FU13_COND.get(c, pair_name(c)) for c in conds if c not in perts]
    if perts:
      named.append(f'perturbation, {len(perts)} sizes')
    vals = sorted(set(g.value))
    rows.append({'Check': FU13_CHECK_NAMES.get(num, str(g.check.iloc[0])), 'Conditions': '; '.join(named) if num in '12' else
                 '; '.join(vals), 'Result': 'pass' if (g.detail == 'pass').all() else 'FAIL'})
  # Style round of 2026-11-20: the completeness count left the note for the text (Checks and completeness, which
  # cites completeness.csv); the generator still reads it and stops if a planned cell is missing
  done = fu13_completeness(main_dir, d)
  if not any('perturbation, 7 sizes' in r['Conditions'] for r in rows):
    raise SystemExit('follow-up 13: the checks note names 7 perturbation sizes')
  if done.split(' of ')[0] != done.split(' of ')[1]:
    raise SystemExit(f'follow-up 13: {done} planned cells complete; the text says every planned cell is complete')
  write_table(out, 'TS_fu13_checks', pd.DataFrame(rows), 'Follow-up 13: checks before any result', ROLE_FU13,
              notes=['Perturbation, 7 sizes: the perturbation arm at each of its seven sizes.'], colspec='lll')
  # perturbation curve
  pcv = read(main_dir, rel('p_curve.csv'))
  rows = []
  for eps, g in pcv.groupby('eps_u'):
    s = g.set_index('statistic')
    rows.append({'Size (units of u)': u_frac(eps), 'Top-1 flip': rate_ci(s.loc['flip_top1']),
                 'Success flip': rate_ci(s.loc['flip_success']), 'Flips that matter': rate_ci(s.loc['flip_matters'])})
  write_table(out, 'TS_fu13_p', pd.DataFrame(rows), 'Follow-up 13, perturbation arm: flip rates against size', ROLE_FU13,
              notes=['Each row compares the v6e with a perturbation of the given relative size at the trunk output '
                     'against the same runs without one, on 60 targets with seeds 1 and 2 (120 predictions). Rates are '
                     'in %, with 95% intervals over targets. The unit u = 2^-8 is the largest relative rounding error '
                     'of bfloat16.'.replace('2^-8', '1/256')], colspec='rlll')
  # eps*
  es = read(main_dir, rel('p_eps_star.csv'))
  rows = [{'Pair': pair_name(r.pair), 'Statistic': {'flip_top1': 'Top-1 flip', 'flip_success': 'Success flip'}[r.statistic],
           'Seeds': r.hw_seeds, 'Rate': pc(r.hw_rate), 'Size reaching it (u)': f'{u2(r.eps_star_u)} [{u2(r.ci_low_u)}, '
           f'{u2(r.ci_high_u)}]', 'Replicates at a bound': f'{int(r.censored_replicates)} of {fint(r.replicates)}'}
          for r in es.itertuples()]
  write_table(out, 'TS_fu13_eps', pd.DataFrame(rows), 'Follow-up 13, perturbation arm: the size that reaches each '
              'platform rate', ROLE_FU13, notes=[
                  'The size is interpolated on a log2 scale between the first two sizes whose rates bracket the rate '
                  'of the pair, with a 95% interval from the same bootstrap replicates. A replicate whose curve does '
                  'not reach the rate counts at the smallest or largest size. The reading uses seeds 1 and 2, the '
                  'seeds of the perturbation arm. Seeds 1 to 5 are shown next to it. Rates are in %.'],
              colspec='lllrll')
  # layer capture: trunk output after each recycle
  ls_ = read_opt(main_dir, rel('l_summary.csv'))
  rows = []
  for rec in sorted(ls_.recycle.unique()) if len(ls_) else []:
    row = {'Recycle': int(rec)}
    for kind in ('single', 'pair'):
      for pair in ('V5-V6', 'V6-G', 'V5-G', 'G-Gr2'):
        g = ls_[(ls_.pair == pair) & (ls_.kind == kind) & (ls_.recycle == rec)]
        row[f'{pair_name(pair)}, {kind}'] = sig3(g.loc[g.depth.idxmax(), 'median_u']) if len(g) else ''
    rows.append(row)
  if rows:
    write_table(out, 'TS_fu13_l', pd.DataFrame(rows), 'Follow-up 13, layer capture: difference at the trunk output after '
                'each recycle', ROLE_FU13, notes=[
                    'Each cell gives the median over the 10 pilot targets, seed 1, of the relative L2 difference at the '
                    'trunk output, in units of u. Recompile, L4 compares two separately compiled runs on the L4. v5e '
                    'vs L4 is a secondary pair.'], colspec='r' * 9)
  # float32 trunk: contrasts
  fc = read(main_dir, rel('f_contrasts.csv'))
  main = read(main_dir, f'{MAINV2}/contrasts/contrasts.csv')
  prec = read(main_dir, f'{AN}/prec5_20261107/contrasts.csv')
  prec = prec[(prec.seeds.astype(str) == '1 to 5') & (prec.targets.astype(str) == 'all 60')]
  spec = [('v6e float32 vs L4 float32', fc, 'HW(V6f,Gf)'), ('Recompile, L4 float32', fc, 'RC(Gf)'),
          ('Recompile, v6e float32 (10 targets, seed 1)', fc, 'RC(V6f)'), ('v6e vs v6e float32', fc, 'F(V6)'),
          ('L4 vs L4 float32', fc, 'F(G)'), ('v6e vs L4 (main study)', main, 'HW(V6,G)'),
          ('Recompile, L4 (main study)', main, 'RC(G)'), ('v6e highest vs L4 highest (FU12)', prec, 'HWh(V6,G)')]
  rows = []
  for lab, src, c in spec:
    s = src[src.contrast == c].set_index('statistic')
    lig = s.loc['m4_ligand_rmsd_median'] if 'm4_ligand_rmsd_median' in s.index else None
    bit = s.loc['bit_identical_fraction'] if 'bit_identical_fraction' in s.index else None
    rows.append({'Contrast': lab, 'Top-1 flip': rate_ci(s.loc['flip_top1']), 'Success flip': rate_ci(s.loc['flip_success']),
                 'Flips that matter': rate_ci(s.loc['flip_matters']),
                 'Ligand RMSD, median (Å)': '' if lig is None else f'{float(lig.estimate):.4f}',
                 'Bit-identical samples': '' if bit is None else pc(bit.estimate)})
  write_table(out, 'TS_fu13_f', pd.DataFrame(rows), 'Follow-up 13, float32 trunk arm: flip rates per contrast',
              ROLE_FU13, notes=['Rates are in % of 300 predictions (60 targets, seeds 1 to 5), with 95% intervals over '
                                'targets, except the second run of the v6e float32 on 10 pilot targets with seed 1. '
                                'The float32 arms run the trunk in float32 with matrix products at the highest '
                                'precision. The last three rows are the bfloat16 references.'], colspec='llllrr')
  # D_F, secondary comparisons, success
  fr = read(main_dir, rel('f_reading.csv'))
  fs = read(main_dir, rel('f_secondary.csv'))
  su = read(main_dir, rel('f_success.csv'))
  stat = {'flip_top1': 'Top-1 flip', 'flip_success': 'Success flip', 'flip_matters': 'Flips that matter'}
  rows = [{'Quantity': f'D_F, {stat[r.statistic].lower()} ({r.role})', 'Estimate': diff_ci(r),
           'Reading': r.reading} for r in fr.itertuples()]
  sec_lab = {'HW(V6f,Gf) minus HWh(V6,G)': 'v6e float32 vs L4 float32 minus v6e highest vs L4 highest',
             'RC(Gf) minus RC(G)': 'Recompile, L4 float32 minus recompile L4'}
  rows += [{'Quantity': f'{sec_lab[r.comparison]}, {stat[r.statistic].lower()}', 'Estimate': diff_ci(r),
            'Reading': 'descriptive'} for r in fs.itertuples()]
  rows += [{'Quantity': f'Top-1 success, {ARM_NAME[r.condition]}', 'Estimate':
            f'{pc(r.top1_success_rate)} [{pc(r.ci_low)}, {pc(r.ci_high)}]', 'Reading': 'descriptive'}
           for r in su.itertuples()]
  write_table(out, 'TS_fu13_df', pd.DataFrame(rows), 'Follow-up 13, float32 trunk arm: reading and secondary comparisons',
              ROLE_FU13, notes=['D_F is the difference between the v6e and the L4 beyond recompilation on the L4 with the '
                                'float32 trunk, minus the same difference with the bfloat16 trunk, in percentage '
                                'points. It reads reduced if its interval lies below 0. The comparisons below it are '
                                'also in points. Success rates are per seed, in %, on 300 predictions.'], colspec='lll')
  # cost per bucket
  cost = read(main_dir, rel('f_cost.csv'))
  rows = []
  for r in cost.itertuples():
    rows.append({'Arm': ARM_NAME[r.new], 'Bucket': str(r.bucket), 'Processes': f'{int(r.n_new)}, {int(r.n_reference)}',
                 'Wall time (s)': f'{r.wall_median_new:.0f} vs {r.wall_median_reference:.0f}',
                 'Wall time ratio': f'{r.wall_ratio:.2f}', 'Compiled memory ratio': f'{r.compiled_memory_ratio:.2f}',
                 'Peak reservation ratio': '' if pd.isna(r.peak_reserved_ratio) else f'{r.peak_reserved_ratio:.2f}'})
  write_table(out, 'TS_fu13_cost', pd.DataFrame(rows), 'Follow-up 13, float32 trunk arm: time and memory against the '
              'bfloat16 trunk', ROLE_FU13, notes=[
                  'Each ratio divides the median of the float32 arm by the median of the bfloat16 arm on the same '
                  'platform (v6e float32 against the v6e, L4 float32 against the L4), per token bucket. Processes gives '
                  'the two counts. Wall time is the median process wall time. Compiled memory is the device memory of '
                  'the compiled model, and peak reservation is the allocator peak, recorded on the TPU only.'],
              colspec='lrrrrrr')


def fu13_completeness(main_dir, d):
  c = read(main_dir, str((d / 'completeness.csv').relative_to(main_dir)))
  return f'{int((c.status == "complete").sum())} of {len(c)}'


def fu13_caption(info):
  """The caption of the main text's follow-up 13 figure, in sentences, from the values drawn. Without layer-capture
  data the sentence on the triangles is left out."""
  m = info['marks']
  bars = {(g, k): r for g, k, r in info['bars']}
  tri = ''
  if m:
    vals = [m[(p, k)] for p in ('V5-V6', 'V6-G') for k in ('single', 'pair')]
    tri = ('Triangles mark the median differences that the layer-capture arm measured between the platforms at the '
           f'trunk output, open for the single and filled for the pair representation, the latter on a subset of '
           f'tokens ({u2(min(vals))}~$u$ to {u2(max(vals))}~$u$). ')
  f32 = bars[('v6e vs L4', 'f32')]
  rc32 = bars[('Recompile, L4', 'f32')]
  rc_text = 'changed none' if float(rc32.estimate) == 0 else f'changed {pc(rc32.estimate)}\\%'
  return (
      'Follow-up~13 relates the flips to the size of the rounding error at the trunk output. (a) We added a random '
      'perturbation of relative size $\\epsilon$ to the trunk output on the v6e, on 60 targets with seeds 1 and 2. '
      'Circles give the top-1 flip rate against the unperturbed runs, and $u = 2^{-8}$ is the largest relative rounding '
      'error of bfloat16. The grey band spans the 95\\% intervals of the two primary pairs with seeds 1 and 2, and '
      'the dashed line gives recompilation on the L4 with seeds 1 to 5. ' + tri +
      '(b) Top-1 flip rates on 60 targets with seeds 1 to 5. The middle bar keeps the bfloat16 trunk at the highest '
      f'precision (Follow-up~12). With the trunk in float32, the v6e and the L4 differed in {pc(f32.estimate)}\\% of '
      f'predictions, and recompilation on the L4 {rc_text}. Error bars give 95\\% intervals over targets.')


def fu13_caption_l():
  return ('Follow-up 13, layer-capture arm: the relative difference between two platforms after each Pairformer block, '
          'in units of $u = 2^{-8}$, on 10 pilot targets with seed 1. Depth 0 is the input of the Pairformer, after '
          'the input embedding, the recycling embedding, and the MSA module. Lines give the median and bands the range '
          'over targets. The pair representation was captured at even depths on a fixed token subset. Recompilation on '
          'the L4 has a median difference of 0 at every depth and is not drawn on the log scale.')


def fu13_assets(out, save, plt, main_dir):
  """Everything the paper takes from follow-up 13, or nothing (None) while its analysis output does not exist."""
  d = fu13_dir(main_dir)
  if d is None:
    NOTES.append('follow-up 13: no results/analysis/fu13_<DATE>/ yet; no figure, numbers or caption written')
    return None
  nums = fu13_numbers(main_dir, d)
  write_csv(out, 'fu13_numbers', nums, ROLE_FU13, f'Follow-up 13 numbers for the text, from {d.name}')
  info = fig_fu13(out, save, plt, main_dir, d)
  l_fig = fig_fu13_l(out, save, plt, main_dir, d)
  # written as LaTeX directly (tex() would turn the exponent's minus sign into math inside math)
  # Options round of 2026-11-19 (Fig. 2 B): the legend names the band, the dashed line and the markers on the x
  # axis; Methods (III-D) gives the targets and seeds of each arm, Results (IV-C) the seeds of the dashed line and the
  # measured differences
  CAPTIONS['capFigFuThirteen'] = tie_pairs(
      '(a) Top-1 flip rate on the v6e against the size of a random perturbation of the trunk output, in units of '
      '$u = 2^{-8}$ (seeds 1 and 2). (b) Top-1 flip rates with the bfloat16 and the float32 trunk (seeds 1 to 5), on a '
      'square-root scale. All rates use 60 targets and 95\\% intervals over targets.')
  no_em_dash(CAPTIONS['capFigFuThirteen'], 'capFigFuThirteen')
  if l_fig:
    CAPTIONS['capFigFuThirteenL'] = fu13_caption_l()
  fu13_tables(out, main_dir, d, info)
  return {'dir': d.name, 'p_levels': info['p_levels'], 'l_drawn': info['l_drawn'], 'numbers': len(nums)}


# ---------------------------------------------------------------------------------------------------------------------
# Follow-up 14 (label f32, rows of 2026-11-12 and 2026-11-13): the float32 trunk on every accelerator and the
# deterministic flag on the A100 and the H100. The reference run is fu14_20261113b (the analysis row: identical to
# fu14_20261113 except the cost rows of the deterministic flag).

FU14 = f'{AN}/fu14_20261113b'
ROLE_FU14 = ('follow-up 14 (outside the pre-registered study; specified after the results were read; OSF addendum '
             'registered before its runs)')
FU13_DIR = f'{AN}/fu13_20261112'
FU1314_ROWS = []          # filled by fu1314_rows() before the follow-ups table is written
STAT3 = [('flip_top1', 'Top-1 flip'), ('flip_success', 'Success flip'), ('flip_matters', 'Flips that matter')]
F14_LABEL = {'F(V5)': 'v5e vs v5e float32', 'F(A)': 'A100 vs A100 float32', 'F(H)': 'H100 vs H100 float32',
             'RC(Gf)': 'Recompile, L4 float32', 'RC(V5f)': 'Recompile, v5e float32 (10 targets, seed 1)',
             'RC(Ad)': 'Recompile, A100 deterministic', 'RC(Hd)': 'Recompile, H100 deterministic',
             'DET(A)': 'Determinism flag, A100', 'DET(H)': 'Determinism flag, H100',
             'RC(A) pilot': 'Recompile, A100 (same 10 targets)', 'RC(H) pilot': 'Recompile, H100 (same 10 targets)',
             'HWh(G,A)': 'L4 highest vs A100 highest (seeds 1 and 2)'}


def f14_pair(c):
  """'HW(V5f,V6f)' -> 'v5e float32 vs v6e float32'; comparisons with ' minus ' are labeled side by side."""
  if ' minus ' in c:
    a, b = c.split(' minus ')
    return f'{f14_pair(a)} minus {f14_pair(b)}'
  return clab(c)


def fu1314_rows(main_dir):
  """Rows 13 and 14 of the follow-ups table, from the analysis outputs of both follow-ups."""
  fr = read(main_dir, f'{FU13_DIR}/f_reading.csv').set_index('statistic')
  fc = read(main_dir, f'{FU13_DIR}/f_contrasts.csv')
  hw13 = pick(fc, contrast='HW(V6f,Gf)', statistic='flip_top1')
  ag = read(main_dir, f'{FU14}/agree.csv')
  dfx = read(main_dir, f'{FU14}/d_f.csv')
  dt = dfx[dfx.statistic == 'flip_top1'].set_index('pair')
  det = read(main_dir, f'{FU14}/det_identity.csv')
  FU1314_ROWS[:] = [
      ('13', 'perturbation of the trunk output on the v6e; layer capture on the v5e, v6e and L4; float32 trunk on the '
             'v6e and the L4', '60, 2; 10, 1; 60, 5',
       f'v6e float32 vs L4 float32: top-1 {rate_ci(hw13)}; D_F top-1 {diff_ci(fr.loc["flip_top1"])}',
       f'D_F top-1 {fr.loc["flip_top1", "reading"]}; success and flips that matter {fr.loc["flip_success", "reading"]}; '
       'perturbation and layer capture descriptive'),
      ('14', 'float32 trunk on the v5e, A100 and H100; deterministic flag on the A100 and H100', '60, 5; 10, 5',
       f'float32, all 10 pairs: top-1 {pc(ag.flip_top1.min())} to {pc(ag.flip_top1.max())}, largest upper bound '
       f'{pc(ag.ci_high.max())}; D_F top-1 ' + '; '.join(f'{clab(k)} {diff_ci(dt.loc[k])}' for k in dt.index),
       f'agree on every accelerator ({int((ag.reading == "agree").sum())} of {len(ag)} pairs); D_F reduced for '
       f'{int((dt.reading == "reduced").sum())} of {len(dt)} primary pairs; with the flag, recompilation on the A100 '
       'and the H100 identical (' + ' and '.join(str(r.value).split(' ')[0] for r in det.itertuples()) + ' predictions)'),
  ]


def fu14_tables(out, main_dir):
  """The supplement's tables of follow-up 14, formatted from the analysis outputs."""
  ck = read(main_dir, f'{FU14}/checks.csv')
  names = {'1': 'Every process exits 0 or its failure is recorded', '2': 'One machine type per condition',
           '4': 'Settings reached every process', '5': 'v5e float32, second run against the first', '6': 'Memory'}
  rows = []
  for num, g in ck.groupby(ck.check.astype(str).str.split(' ').str[0], sort=False):
    vals = sorted(set(g.value))
    conds = ', '.join(ARM_NAME.get(c, c) for c in g.condition) if num in ('1', '2', '4') else '; '.join(vals)
    rows.append({'Check': names.get(num, str(g.check.iloc[0])), 'Conditions or value': conds,
                 'Result': 'pass' if (g.detail == 'pass').all() else 'FAIL'})
  comp = read(main_dir, f'{FU14}/completeness.csv')
  if (comp.status != 'complete').any():
    raise SystemExit('follow-up 14: a planned cell is incomplete; the text says all 36 are complete')
  write_table(out, 'TS_fu14_checks', pd.DataFrame(rows), 'Follow-up 14: checks before any result', ROLE_FU14,
              notes=['The memory check gives the largest compiled memory as a share of the device limit.'],
              colspec='lp{8cm}l')
  ag = read(main_dir, f'{FU14}/agree.csv')
  rows = [{'Pair': clab(r.pair), 'Role': r.role, 'Top-1 flip': f'{pc(r.flip_top1)} [{pc(r.ci_low)}, {pc(r.ci_high)}]',
           'Reading': r.reading} for r in ag.itertuples()]
  write_table(out, 'TS_fu14_agree', pd.DataFrame(rows), 'Follow-up 14: agreement of the float32 trunk across platforms',
              ROLE_FU14, notes=[f'A pair agrees if the upper bound of its top-1 flip rate lies below '
                                f'{pc(ag.threshold.iloc[0])}% (pre-specified). Rates are in % of 300 predictions (60 '
                                'targets, seeds 1 to 5), with 95% intervals over targets. The pair of the v6e and the L4 '
                                'comes from Follow-up 13.'], colspec='llll')
  dfx = read(main_dir, f'{FU14}/d_f.csv')
  st = dict(STAT3)
  rows = [{'Pair': clab(r.pair), 'Statistic': st[r.statistic], 'D_F (points)': diff_ci(r),
           'Float32 recompilation': 'measured' if str(r.float32_recompilation).startswith('RC') else 'set to 0',
           'Reading': r.reading} for r in dfx.itertuples()]
  write_table(out, 'TS_fu14_df', pd.DataFrame(rows), 'Follow-up 14: the float32 trunk against the bfloat16 trunk',
              ROLE_FU14, notes=['D_F is the difference between two platforms beyond recompilation with the float32 '
                                'trunk, minus the same difference with the bfloat16 trunk at the default precision, in '
                                'percentage points. It reads reduced if its interval lies below 0. Where float32 '
                                'recompilation was not measured on the first-named platform, it is set to 0, so D_F is '
                                'an upper bound.'], colspec='lllll')
  sec = read(main_dir, f'{FU14}/secondary.csv')
  rows = []
  for comp_, g in sec.groupby('comparison', sort=False):
    s2 = g.set_index('statistic')
    rows.append({'Comparison': f14_pair(comp_), **{lab: diff_ci(s2.loc[k]) for k, lab in STAT3}})
  write_table(out, 'TS_fu14_sec', pd.DataFrame(rows), 'Follow-up 14: float32 against bfloat16 trunk, every pair',
              ROLE_FU14, notes=['Descriptive. Each row gives the float32 pair minus the same pair with the bfloat16 '
                                'trunk, at the default precision unless marked highest, in percentage points with 95% '
                                'intervals over targets.'], colspec='llll')
  con = read(main_dir, f'{FU14}/contrasts.csv')
  rows = []
  for c, lab in F14_LABEL.items():
    s2 = con[con.contrast == c].set_index('statistic')
    if not len(s2):
      continue
    rows.append({'Contrast': lab, 'Predictions': int(s2.loc['flip_top1', 'n']),
                 **{l2: rate_ci(s2.loc[k]) for k, l2 in STAT3}})
  write_table(out, 'TS_fu14_f', pd.DataFrame(rows), 'Follow-up 14: float32 trunk, recompilation, and the deterministic '
              'flag', ROLE_FU14, notes=['Rates in % with 95% intervals over targets. The deterministic-flag rows use '
                                        '10 pilot targets with seeds 1 to 5, and the recompilation rows without the '
                                        'flag use the same targets.'], colspec='lrlll')
  su = read(main_dir, f'{FU14}/success.csv')
  rows = []
  for cond, g in su.groupby('condition', sort=False):
    s2 = g.set_index('targets')
    rows.append({'Arm': ARM_NAME[cond], **{t: f'{pc(r.top1_success_rate)} [{pc(r.ci_low)}, {pc(r.ci_high)}]'
                                           for t, r in s2.iterrows()}})
  write_table(out, 'TS_fu14_success', pd.DataFrame(rows), 'Follow-up 14: success rate per arm', ROLE_FU14,
              notes=['Top-1 success per seed, in %, with 95% intervals over targets, on all 60 targets and on the 39 '
                     'released after the training cutoff.'], colspec='lll')
  cost = read(main_dir, f'{FU14}/cost.csv')
  rows = []
  for r in cost.itertuples():
    ref = ARM_NAME.get(str(r.reference).replace('_pilot', ''), r.reference)
    rows.append({'Arm': ARM_NAME[r.new], 'Against': ref + (' (pilot)' if 'pilot' in str(r.reference) else ''),
                 'Bucket': str(r.bucket), 'Processes': f'{int(r.n_new)}, {int(r.n_reference)}',
                 'Wall time (s)': f'{r.wall_median_new:.0f} vs {r.wall_median_reference:.0f}',
                 'Wall time ratio': f'{r.wall_ratio:.2f}', 'Compiled memory ratio': f'{r.compiled_memory_ratio:.2f}'})
  write_table(out, 'TS_fu14_cost', pd.DataFrame(rows), 'Follow-up 14: time and compiled memory against the reference '
              'arm', ROLE_FU14, notes=['Each ratio divides the median of the new arm by the median of its reference on '
                                       'the same platform, per token bucket. The deterministic arms are compared with '
                                       'the runs without the flag on the same pilot targets.'], colspec='llrrrrr')
  mem = read(main_dir, f'{FU14}/memory.csv')
  rows = [{'Arm': ARM_NAME[r.condition], 'Bucket': str(r.bucket), 'Processes': int(r.processes),
           'Largest compiled (GiB)': f'{r.max_compiled_gib:.2f}', 'Device limit (GiB)': f'{r.device_limit_gib:.2f}',
           'Share of limit (%)': pc(r.fraction_of_limit), 'Failed': int(r.failed)} for r in mem.itertuples()]
  write_table(out, 'TS_fu14_memory', pd.DataFrame(rows), 'Follow-up 14: compiled memory per bucket', ROLE_FU14,
              notes=['Failed: processes with a nonzero exit code.'], colspec='lrrrrrr')
  return {'agree': ag, 'df': dfx}


# Post hoc analyses of 2026-11-14 (section 10 rows, labels returned and rare): the quality of the returned structure
# across platforms, and BCa intervals as a sensitivity analysis of the registered readings with few events.

RETURNED = f'{AN}/returned_20261114'
RARE = f'{AN}/rare_20261114'
ROLE_RETURNED = ('post hoc analysis of frozen outputs (section 10, rows of 2026-11-14, label returned), outside the '
                 'pre-registered study; descriptive')
ROLE_RARE = ('post hoc sensitivity analysis (section 10, rows of 2026-11-14, label rare), outside the pre-registered '
             'study; the registered readings stand')
RARE_FAMILY = {'F1': 'Main study, primary pairs', 'F2': 'Follow-up 4, 60 targets', 'F3': 'Follow-up 11',
               'F4 follow-up 12, seeds 1 and 2': 'Follow-up 12, seeds 1 and 2',
               'F4 follow-up 12, seeds 1 to 5': 'Follow-up 12, seeds 1 to 5', 'F5': 'Follow-ups 13 and 14, D_F',
               'F6': 'Follow-up 14, agreement'}
READ_RULE = {'beyond': 'beyond', 'not beyond': 'not beyond', 'agree': 'agree', 'reduced': 'reduced',
             'not reduced': 'not reduced', 'agreement not shown': 'not shown'}


def ang(x, sign=False):
  """Angstrom, two decimals; -0.00 prints as 0.00."""
  v = f'{float(x):.2f}'
  if v in ('-0.00', '0.00'):
    return '0.00'
  return ('+' + v) if sign and not v.startswith('-') else v


def returned_label(source, contrast):
  if contrast == 'HW(G,C)':
    return 'L4 vs CPU (60 targets)' if source == 'cpu60' else 'L4 vs CPU (6 targets)'
  return clab(contrast)


def table_returned(out, main_dir):
  """Supplementary table: ligand RMSD from the experimental pose, top-1 of B minus top-1 of A (label returned)."""
  q = read(main_dir, f'{RETURNED}/quality.csv')
  r = q[q.metric == 'ligand_rmsd']
  rows = []
  for (source, contrast), g in r.groupby(['source', 'contrast'], sort=False):
    a = g[g.stratum == 'all'].iloc[0]
    f = g[g.stratum == 'flipped']
    row = {'Contrast': returned_label(source, contrast), 'Predictions': int(a.n_units),
           'Median, all': f'{ang(a["median"], True)} [{ang(a.median_ci_low)}, {ang(a.median_ci_high)}]',
           'Mean, all': f'{ang(a["mean"], True)} [{ang(a.mean_ci_low)}, {ang(a.mean_ci_high)}]'}
    if len(f) and int(f.iloc[0].n_units):
      f = f.iloc[0]
      row.update({'Changed': int(f.n_units),
                  'Median, changed': f'{ang(f["median"], True)} [{ang(f.median_ci_low)}, {ang(f.median_ci_high)}]',
                  'Mean, changed': f'{ang(f["mean"], True)} [{ang(f.mean_ci_low)}, {ang(f.mean_ci_high)}]'})
    else:
      row.update({'Changed': 0 if len(f) else 'n/a', 'Median, changed': '--', 'Mean, changed': '--'})
    rows.append(row)
  allr = r[r.stratum == 'all']
  flp = r[(r.stratum == 'flipped') & (r.n_units > 0)]
  band = max(allr.median_ci_low.abs().max(), allr.median_ci_high.abs().max())
  bandf = max(flp.median_ci_low.abs().max(), flp.median_ci_high.abs().max())
  if band > 0.1 or bandf > 0.1:
    raise SystemExit('returned: a median interval leaves the 0.1 A band; the paper text says it does not')
  ip = q[(q.metric == 'iptm') & (q.n_units > 0)]
  pl = q[(q.metric == 'mean_plddt') & (q.stratum == 'all')]
  write_table(out, 'TS_returned', pd.DataFrame(rows), 'Ligand RMSD of the returned structure across platforms (post hoc)',
              ROLE_RETURNED,
              notes=['For each prediction, the ligand RMSD from the experimental pose of the top-ranked sample of the '
                     'second-named side minus that of the first-named side, in Å. Medians and means with 95% '
                     'intervals over targets (10000 replicates, random-number seed 2026). Changed: the predictions whose top-ranked '
                     'sample differs. The three glycan targets have no ligand RMSD.'],
              colspec='lrllrll')


def rare_label(q):
  """clab, except that the CPU side of HWh(G,C) has no highest-precision arm of its own (the CPU computes in float32)."""
  if q.startswith('HWh(G,C)'):
    return clab(q.replace('HWh(G,C)', 'HW(G,C)')).replace('L4 vs CPU', 'L4 highest vs CPU', 1)
  return clab(q)


def table_rare(out, main_dir):
  """Supplementary table: registered percentile intervals against BCa intervals on the same replicates (label rare)."""
  d = read(main_dir, f'{RARE}/intervals.csv')
  if (d.reproduced != 'yes').any():
    raise SystemExit('rare: a row does not reproduce its registered interval')
  ruled = d[d.rule != 'none']
  rows, tex_rows, fam_prev = [], [], None
  for r in ruled.itertuples():
    fam = next(v for k, v in RARE_FAMILY.items() if r.family.startswith(k))
    if fam != fam_prev:
      tex_rows.append({'Quantity': '\\GROUP ' + fam})
      fam_prev = fam
    m = re.match(r'D_F (HW\(\w+,\w+\))', r.quantity)
    qty = f'D_F, {clab(m.group(1))}' if m else rare_label(r.quantity)
    ev = ' / '.join(str(int(getattr(r, f'events_{i}'))) for i in range(1, 5)
                    if hasattr(r, f'events_{i}') and getattr(r, f'events_{i}') == getattr(r, f'events_{i}'))
    row = {'Quantity': qty, 'Statistic': dict(STAT3)[r.statistic], 'Events': ev,
                 'Registered': f'{spc(r.estimate)} [{pc(r.registered_low)}, {pc(r.registered_high)}]',
                 'BCa': f'[{pc(r.bca_low)}, {pc(r.bca_high)}]',
                 'Reading': (f'{READ_RULE[r.registered_reading]} → {READ_RULE[r.bca_reading]}' if r.changed == 'yes'
                             else READ_RULE[r.registered_reading])}
    tex_rows.append(row)
    rows.append({'Family': fam, **row})
  changed = ruled[ruled.changed == 'yes']
  weaker = ruled[(ruled.changed == 'yes') & ruled.bca_reading.isin(['not beyond', 'not reduced', 'agreement not shown'])]
  # Style round of 2026-11-20: the counts left the note for the text (Rare-Event Intervals), which says that no reading
  # weakens; the generator stops if that is no longer true
  if len(weaker):
    raise SystemExit(f'rare: {len(weaker)} readings weaken under BCa; the text says none does')
  z = d[d.rule == 'none']
  cp60 = z[z.n_targets == 60].cp_upper_targets.max()
  cp10 = z[z.n_targets == 10].cp_upper_targets.max()
  write_table(out, 'TS_rare', pd.DataFrame(rows), 'Rare-event readings under BCa intervals (post hoc)', ROLE_RARE,
              notes=['Percentage points. Registered: the estimate and the registered 95% percentile interval over '
                     'targets. BCa: the bias-corrected and accelerated interval on the same 10000 replicates. Events: '
                     'the flips on each term of the quantity. Each row is read with its registered rule: beyond if the '
                     'lower limit lies above 0, agree if the upper limit lies below 5.0, reduced if the upper limit '
                     'lies below 0.'],
              colspec='llllll', tex_df=pd.DataFrame(tex_rows))


WINNERS = f'{AN}/winners_20261114'
WINNERS_EXT = f'{AN}/winners_ext_20261116'   # post hoc 16: the contrasts winners did not cover, and two references
ROLE_WINNERS = ('post hoc analysis of frozen outputs (section 10, rows of 2026-11-14, label winners), outside the '
                'pre-registered study; descriptive')
WINNERS_ORDER = ['HW(V5,V6)', 'HW(V6,G)', 'HW(H,A)', 'RC(G)', 'RC(H)', 'RC(A)']


def table_winners(out, main_dir):
  """Supplementary table: the two returned poses compared directly, per seed (changed predictions) and for AlphaFold 3's
  five-seed main output (label winners)."""
  s = read(main_dir, f'{WINNERS}/summary.csv')
  err = read(main_dir, f'{WINNERS}/errors.csv')
  glycans = sorted(err.target.unique())
  if glycans != ['7OZC', '7WDT', '7WJB'] or not err.pose_error.str.contains(r"missing in the prediction: \['O1'\]").all():
    raise SystemExit(f'winners: errors.csv holds more than the three glycan targets: {glycans}')
  rows = []
  for c in WINNERS_ORDER:
    sc = s[(s.contrast == c) & (s.level == 'seed') & (s.stratum == 'changed')].iloc[0]
    ta = s[(s.contrast == c) & (s.level == 'target') & (s.stratum == 'all')].iloc[0]
    rows.append({'Contrast': clab(c), 'Changed': int(sc.units_with_pose_rmsd),
                 'Median [95% CI]': f'{ang(sc.pose_rmsd_median)} [{ang(sc.pose_rmsd_median_ci_low)}, '
                                    f'{ang(sc.pose_rmsd_median_ci_high)}]',
                 '90th pct.': ang(sc.pose_rmsd_p90),
                 'Over 2 Å': f'{int(sc.distinct_n)} of {int(sc.distinct_of)}',
                 'Binding mode': f'{int(sc.binding_mode_change_n)}',
                 'Validity': f'{int(sc.validity_change_n)} ({pc(sc.validity_change_share)}%)',
                 'Main output over 2 Å': f'{int(ta.distinct_n)} of {int(ta.distinct_of)}',
                 'Main output max': f'{ang(ta.pose_rmsd_max)} ({ta.pose_rmsd_max_target})'})
  write_table(out, 'TS_winners', pd.DataFrame(rows), 'The two returned poses compared directly (post hoc)', ROLE_WINNERS,
              notes=['Ligand RMSD between the two returned samples after superposing their pockets, in Å. Changed: the '
                     'predictions (target and seed) whose top-ranked sample differs, with the median and 90th percentile '
                     'of the RMSD (95% interval over targets, 10000 replicates, random-number seed 2026), the poses more than 2 Å '
                     'apart, the binding-mode changes (poses more than 2 Å apart, one within 2 Å of the experimental pose '
                     'and the other beyond 4 Å), and the PoseBusters validity changes. Main output: AlphaFold 3\'s '
                     'top-ranked sample over the five seeds, over all targets. The three glycan targets (7OZC, 7WDT, '
                     '7WJB) have no RMSD, because AlphaFold 3 removes atom O1 from their ligand. In recompilation on the '
                     'H100, glycan predictions with identical coordinates in both runs count with an RMSD of 0.'],
              colspec='lrlrllllll'[:9])



# Post hoc analyses of 2026-11-16 (section 10 rows of that date): winners_ext, pb_flips, best25.

ROLE_PH16 = ('post hoc analysis of frozen outputs (section 10, rows of 2026-11-16, labels winners_ext, pb_flips and '
             'best25), outside the pre-registered study; descriptive')
PB_FLIPS = f'{AN}/pb_flips_20261116'
BEST25 = f'{AN}/best25_20261116'
WINNERS_EXT_ROWS = [('main', 'HW(G,C)', 'L4 vs CPU, main study'), ('cpu60', 'HW(G,C)', 'L4 vs CPU, Follow-up 4'),
                    ('main', 'DU', 'Default attention, L4'), ('rcx', 'RC(GT)', 'Recompile, L4 Triton (FU15)')]


def table_winners_ext(out, main_dir):
  """Supplementary table: the two returned poses compared directly for L4 vs CPU (main study and follow-up 4), default
  attention on the L4, and recompilation of the L4 Triton as a further reference (label winners_ext)."""
  s = read(main_dir, f'{WINNERS_EXT}/summary.csv')
  rows = []
  for src, c, lab in WINNERS_EXT_ROWS:
    sc = s[(s.source == src) & (s.contrast == c) & (s.level == 'seed') & (s.stratum == 'changed')].iloc[0]
    ta = s[(s.source == src) & (s.contrast == c) & (s.level == 'target') & (s.stratum == 'all')].iloc[0]
    one = int(sc.units_with_pose_rmsd) == 1
    rows.append({'Contrast': lab, 'Targets': int(ta.units), 'Changed': int(sc.units_with_pose_rmsd),
                 'Median [95% CI]': ang(sc.pose_rmsd_median) if one else
                 f'{ang(sc.pose_rmsd_median)} [{ang(sc.pose_rmsd_median_ci_low)}, {ang(sc.pose_rmsd_median_ci_high)}]',
                 '90th pct.': 'n/a' if one else ang(sc.pose_rmsd_p90),
                 'Over 2 Å': f'{int(sc.distinct_n)} of {int(sc.distinct_of)}',
                 'Binding mode': f'{int(sc.binding_mode_change_n)}',
                 'Validity': f'{int(sc.validity_change_n)} of {int(sc.validity_change_of)}',
                 'Main output over 2 Å': f'{int(ta.distinct_n)} of {int(ta.distinct_of)}',
                 'Main output max': f'{ang(ta.pose_rmsd_max)} ({ta.pose_rmsd_max_target})'})
  df = pd.DataFrame(rows)
  if df.loc[0, 'Changed'] != 1:
    raise SystemExit('winners_ext: L4 vs CPU in the main study no longer has one changed prediction')
  write_table(out, 'TS_winners_ext', df, 'The two returned poses compared directly, further contrasts (post hoc)', ROLE_PH16,
              notes=['Measures as in SECREFWINNERS, from the same code. Targets: the targets of the contrast. '
                     'L4 vs CPU, main study: the 6 targets the CPU ran, seeds 1 and 2. One prediction changed, so it '
                     'has no interval. L4 vs CPU, Follow-up 4: all 60 targets, seeds 1 and 2. Default attention, L4: '
                     'the L4 against L4 Triton, which uses AlphaFold 3\'s default attention, seeds 1 to 5. Recompile, '
                     'L4 Triton: Follow-up 15, a reference for the default-attention row. Changed: the changed '
                     'predictions with a pose RMSD. The others are on the three glycan targets. Validity: changes of '
                     'PoseBusters validity among them.'],
              colspec='lrrlrlllll'[:10])


def table_pb_flips(out, main_dir):
  """Supplementary table: which success criterion fails in a success flip, and how far beyond its threshold (label
  pb_flips)."""
  s = read(main_dir, f'{PB_FLIPS}/summary.csv')
  fails = [c for c in s.columns if c.startswith('fails: ')]
  if sorted(fails) != ['fails: RMSD', 'fails: minimum_distance_to_protein']:
    raise SystemExit(f'pb_flips: failing criteria are now {fails}; the table and text name only two')
  a = s[s.stratum == 'all flips']
  if (a['criterion: RMSD and PoseBusters'] != 0).any() or (a['criterion: not scored'] != 0).any() or \
      (a['class: not classified'] != 0).any():
    raise SystemExit('pb_flips: a flip fails both criteria, is not scored or is not classified')
  if (a['criterion: PoseBusters'] != a['fails: minimum_distance_to_protein']).any():
    raise SystemExit('pb_flips: a PoseBusters flip fails a check other than the minimum distance to the protein')
  tc = s[s.stratum == 'top-1 changed'].set_index('contrast')
  rows = []
  for _, r in a.iterrows():
    c = r.contrast
    lab = {'group: platform': 'All platform pairs', 'group: recompilation': 'All recompilation'}.get(c) or clab(c)
    grp = c.startswith('group')
    near = int(r['class: near-threshold crossing'])
    exc = pc(r['median worst excess'])
    rows.append({'Contrast': lab, 'Flips': int(r.flips), 'Top-1 changed': int(tc.loc[c, 'flips']),
                 'RMSD': int(r['criterion: RMSD']), 'Minimum distance': int(r['criterion: PoseBusters']),
                 'Near threshold': f'{near} ({pc(r["share near-threshold crossing"])}% '
                                   f'[{pc(r["share near-threshold crossing ci_low"])}, '
                                   f'{pc(r["share near-threshold crossing ci_high"])}])' if grp else f'{near}',
                 'Pose change': int(r['class: pose change']), 'Other': int(r['class: other']),
                 'Median excess (%)': f'{exc} [{pc(r["median worst excess ci_low"])}, '
                                      f'{pc(r["median worst excess ci_high"])}]' if grp else exc})
  df = pd.DataFrame(rows)
  write_table(out, 'TS_pb_flips', df, 'Which success criterion changes in a success flip (post hoc)', ROLE_PH16,
              notes=['Success flips of the main study and of Follow-ups 3 and 9, seeds 1 to 5. Top-1 changed: flips '
                     'whose top-ranked sample index differs. RMSD: the failing side has a ligand RMSD of 2 Å or more '
                     'and is PoseBusters-valid. Minimum distance: the failing side is below 2 Å and fails one '
                     'PoseBusters check, the minimum distance to the protein, which requires every ligand-protein '
                     'distance to be at least 0.75 times the sum of the two van der Waals radii. Excess: how far the '
                     'failing value lies beyond its threshold, as a percentage of the '
                     'threshold. Near threshold: every failing criterion exceeds its threshold by at most 10%. Pose '
                     'change: the two top-ranked poses lie more than 2 Å apart. The groups pool the flips of a target across '
                     'contrasts. The 95% intervals are over targets.'],
              colspec='lrrrrllll'[:9])


def table_best25(out, main_dir):
  """Supplementary table: success of an oracle that finds a correct sample whenever one exists, beside AlphaFold 3's
  main output, per contrast (label best25)."""
  d = read(main_dir, f'{BEST25}/differences.csv')
  order = [('HW(V5,V6)', '1-2-3-4-5'), ('HW(V6,G)', '1-2-3-4-5'), ('HW(H,A)', '1-2-3-4-5'), ('HW(G,C)', '1-2'),
           ('RC(V5)', '1-2-3-4-5'), ('RC(V6)', '1-2-3-4-5'), ('RC(G)', '1-2-3-4-5'), ('RC(H)', '1-2-3-4-5'),
           ('RC(A)', '1-2-3-4-5'), ('RC(G) seeds 1-2', '1-2')]

  def two(r):
    return f'{pc(r.rate_a)}, {pc(r.rate_b)}'
  rows = []
  for c, seeds in order:
    g = d[(d.contrast == c) & (d.seeds == seeds)].set_index(['level', 'measure'])
    to, tm, so = (types.SimpleNamespace(rate_a=r.rate_a, rate_b=r.rate_b, estimate=r.difference,
                                        ci_low=r.difference_ci_low, ci_high=r.difference_ci_high)
                  for r in (g.loc[('target', 'oracle')], g.loc[('target', 'main')], g.loc[('seed', 'oracle')]))
    lab = clab(c.replace(' seeds 1-2', ''))
    rows.append({'Contrast': lab, 'Seeds': seeds.replace('1-2-3-4-5', '1--5').replace('1-2', '1, 2'),
                 'Oracle': two(to), 'Oracle difference': diff_ci(to),
                 'Main output': two(tm), 'Main difference': diff_ci(tm),
                 'Seed oracle': two(so), 'Seed oracle difference': diff_ci(so)})
  df = pd.DataFrame(rows)
  write_table(out, 'TS_best25', df, 'Success of the best of all samples, per platform (post hoc)', ROLE_PH16,
              notes=['Success rates in % of the two sides of each contrast, first-named side first, and their '
                     'difference in points, second side minus first, with 95% intervals over targets. Oracle: at '
                     'least one sample of the target succeeds (25 samples with seeds 1 to 5, 10 with seeds 1 and 2). '
                     'Main output: AlphaFold 3\'s top-ranked sample over the seeds. Seed oracle: at least one of the '
                     'five samples of a seed succeeds, over the target-seed pairs. L4 vs CPU and its recompilation '
                     'reference use the 60 targets of Follow-up 4 with seeds 1 and 2.'],
              colspec='llllllll')


# Post hoc analysis of 2026-11-21 (section 10 rows of that date, label gtat): the L4 against the A100, both with
# AlphaFold 3's default Triton attention (GT of the main study against AT of follow-up 3).

GTAT = f'{AN}/gtat_20261121'
ROLE_GTAT = ('post hoc analysis of frozen outputs (section 10, rows of 2026-11-21, label gtat), outside the '
             'pre-registered study; descriptive in standing, read with the registered HW minus RC rule')
GTAT_READ = {'beyond RC(GT)': 'beyond', 'not beyond RC(GT)': 'not beyond', 'no difference shown': 'no difference shown',
             'larger with Triton': 'larger with Triton', 'smaller with Triton': 'smaller with Triton'}


def table_gtat(out, main_dir):
  """Supplementary table: HW(GT,AT) beside RC(GT) (follow-up 15) and HW(G,A) (follow-up 3, XLA attention), its
  difference beyond RC(GT) and its difference from HW(G,A), on 60 and 57 targets (label gtat). The guards hold the
  readings the text states: on 60 targets the three statistics beyond RC(GT), and no difference shown against HW(G,A)."""
  rates = read(main_dir, f'{GTAT}/rates.csv')
  hwrc = read(main_dir, f'{GTAT}/hw_minus_rc.csv')
  vs = read(main_dir, f'{GTAT}/versus_hw_g_a.csv')
  reg = read(main_dir, f'{GTAT}/registered.csv')
  stats = [k for k, _ in STAT3]
  for c, src in (('HW(G,A)', 'a100 (registered)'), ('RC(GT)', 'rcx (registered)')):
    for st in stats:
      a = rates[(rates.targets == '60 targets') & (rates.contrast == c) & (rates.statistic == st)].iloc[0]
      b = reg[(reg.source == src) & (reg.row == c) & (reg.statistic == st)].iloc[0]
      if any(abs(float(a[k]) - float(b[k])) > 1e-9 for k in ('estimate', 'ci_low', 'ci_high')):
        raise SystemExit(f'{GTAT}: {c} {st} is not its registered row')
  r60 = hwrc[hwrc.targets == '60 targets'].set_index('statistic').reading
  v60 = vs[vs.targets == '60 targets'].set_index('statistic').reading
  if set(r60[stats]) != {'beyond RC(GT)'} or set(v60[stats]) != {'no difference shown'}:
    raise SystemExit(f'{GTAT}: readings changed ({dict(r60)}, {dict(v60)}); the text states beyond RC(GT) on all three '
                     'and no difference shown against HW(G,A)')
  rows, plain = [], []      # rows: the LaTeX table (group and indented rows); plain: the CSV, one row per line
  for tg in ('60 targets', '57 targets (no glycan)'):
    rows.append({'Contrast': f'\\GROUP {tg}'})
    for c, lab in (('HW(GT,AT)', 'L4 Triton vs A100 Triton'), ('RC(GT)', 'Recompile, L4 Triton (FU15)'),
                   ('HW(G,A)', 'L4 vs A100, XLA attention (FU3)')):
      g = rates[(rates.targets == tg) & (rates.contrast == c)].set_index('statistic')
      rows.append({'Contrast': lab, **{l: rate_ci(g.loc[k]) for k, l in STAT3}})
      plain.append({'Targets': tg, 'Contrast': lab, 'Line': 'rate (%)', **{l: rate_ci(g.loc[k]) for k, l in STAT3}})
      if c == 'HW(GT,AT)':
        ev = {l: f'{int(g.loc[k, "events"])} ({int(g.loc[k, "targets_with_event"])})' for k, l in STAT3}
        rows.append({'Contrast': '\\INDENT events (targets)', **ev})
        plain.append({'Targets': tg, 'Contrast': lab, 'Line': 'events (targets)', **ev})
    for df, lab in ((hwrc, 'L4 Triton vs A100 Triton minus recompile L4 Triton'),
                    (vs, 'L4 Triton vs A100 Triton minus L4 vs A100')):
      g = df[df.targets == tg].set_index('statistic')
      rd = {l: GTAT_READ[g.loc[k, 'reading']] for k, l in STAT3}
      rows.append({'Contrast': lab, **{l: diff_ci(g.loc[k]) for k, l in STAT3}})
      rows.append({'Contrast': '\\INDENT reading', **rd})
      plain.append({'Targets': tg, 'Contrast': lab, 'Line': 'difference (points)', **{l: diff_ci(g.loc[k]) for k, l in STAT3}})
      plain.append({'Targets': tg, 'Contrast': lab, 'Line': 'reading', **rd})
  tdf = pd.DataFrame(rows, columns=['Contrast'] + [l for _, l in STAT3])
  df = pd.DataFrame(plain)
  write_table(out, 'TS_gtat', df, 'The L4 and the A100, both with Triton attention (post hoc)', ROLE_GTAT,
              notes=['Rates in % of predictions and differences in points, with 95% intervals over targets. L4 Triton '
                     'and A100 Triton use AlphaFold 3\'s default attention, seeds 1 to 5, first runs. L4 vs A100 uses '
                     'XLA attention. This analysis had no recompilation of the A100 Triton (Follow-up 18 measured it '
                     'later), so its only recompilation reference is that of the L4 Triton. Events (targets): the number of flips and of targets with at '
                     'least one. Beyond: the interval of the difference lies above 0. The 57 targets leave out the '
                     'three glycan targets.'],
              colspec='llll', tex_df=tdf)


# Follow-up 15 (label rcx, rows of 2026-11-14 and 2026-11-15): recompilation of the float32 trunk on the A100 and the
# H100 (part A4) and of the L4 Triton arm (part A3).

RCX = f'{AN}/rcx_20261115'
# Follow-up 19 (section 10, rows of 2026-11-24, label rcf): RC(Af) and RC(Hf) on the 60 targets (r1 of follow-up 14
# against r2 of follow-ups 15 and 19); its rule e fills Table IV with them (round of 2026-11-25).
RCF = f'{AN}/rcf_20261125'
ROLE_FU15 = ('follow-up 15 (outside the pre-registered study; specified after the results were read; OSF addendum '
             'registered before its runs)')


def fu15_tables(out, main_dir):
  """The supplement's tables of follow-up 15 and row 15 of the follow-ups table."""
  con = read(main_dir, f'{RCX}/contrasts.csv')
  ident = read(main_dir, f'{RCX}/identity.csv')
  gvg = read(main_dir, f'{RCX}/rc_gt_vs_g.csv').set_index('statistic')
  dfs = read(main_dir, f'{RCX}/d_f_sensitivity.csv')
  rows = []
  for c in ('RC(Af)', 'RC(Hf)', 'RC(GT)'):
    g = con[con.contrast == c].set_index('statistic')
    b = g.loc['bit_identical_fraction']
    idn = ident[ident.condition.str.startswith(c[3:-1] + ' r2')].value.iloc[0]
    rows.append({'Contrast': clab(c), 'Predictions': int(g.loc['flip_top1', 'n']),
                 **{lab: rate_ci(g.loc[k]) for k, lab in STAT3},
                 'Ligand RMSD, median (Å)': sig3(g.loc['m4_ligand_rmsd_median', 'estimate']),
                 'Identical samples': f'{round(float(b.estimate) * int(b.n))} of {int(b.n)}',
                 'Identical output files': idn.split(',')[0].replace('/', ' of ').replace(' identical', '')})
  write_table(out, 'TS_fu15_rc', pd.DataFrame(rows), 'Follow-up 15: recompilation of the float32 trunk and of the L4 Triton',
              ROLE_FU15, notes=['Rates in % with 95% intervals over targets. The A100 and H100 float32 rows use the 10 '
                                'pilot targets with seeds 1 to 5. The L4 Triton row uses the 60 targets with seeds 1 to 5.'],
              colspec='lrllllll')
  rows = []
  for r in dfs.itertuples():
    rows.append({'Version': r.version, 'Statistic': dict(STAT3)[r.statistic], 'Targets': int(r.n_targets),
                 'D_F (points)': diff_ci(r), 'Reading': r.reading})
  write_table(out, 'TS_fu15_df', pd.DataFrame(rows), 'Follow-up 15: D_F of the H100 and the A100 with recompilation measured',
              ROLE_FU15, notes=['Sensitivity analysis. D_F of H100 float32 vs A100 float32 '
                                'against H100 vs A100, with float32 recompilation on the H100 set to 0 (registered, '
                                'Follow-up 14) or measured on the 10 pilot targets.'], colspec='p{7cm}llll')   # 7 cm: no lone word at 8 pt (style round of 2026-11-20)
  af, hf, gt = (con[(con.contrast == c) & (con.statistic == 'flip_top1')].iloc[0] for c in ('RC(Af)', 'RC(Hf)', 'RC(GT)'))
  reg = dfs[(dfs.statistic == 'flip_top1') & dfs.version.str.startswith('(i)')].iloc[0]
  FU15_ROW[:] = [('15', 'recompilation of the float32 trunk on the A100 and H100; recompilation of the L4 Triton',
                  '10, 5; 60, 5',
                  f'recompile A100 float32 {rate_ci(af)}, H100 float32 {rate_ci(hf)}; recompile L4 Triton {rate_ci(gt)}, '
                  f'minus recompile L4 {diff_ci(gvg.loc["flip_top1"])}',
                  f'float32 recompilation changes no top-ranked sample on the GPUs; L4 Triton recompiles like the L4; '
                  f'D_F of H100 vs A100 with recompilation measured {diff_ci(reg)}, {reg.reading}')]


FU15_ROW = []


HLO = f'{AN}/hlo_20261116b'
ROLE_FU16 = ('follow-up 16 (outside the pre-registered study; specified after the results were read; OSF addendum '
             'registered before its runs; descriptive)')
FU16_STAGES = (('embedding', 'Input embedding'), ('template', 'Templates'), ('msa', 'MSA module'))
FU16_OPS = (('convert', 'Conversions'), ('multiply', 'Multiply'), ('select', 'Select'), ('xor', 'Integer xor'),
            ('constant', 'Constants'))


def fu16_tables(out, main_dir):
  """The supplement's table of follow-up 16 (hlo_20261116b, the rerun with the corrected stage attribution; section 10
  rows of 2026-11-16) and row 16 of the follow-ups table. Cells are v5e/v6e counts in the optimised HLO of bucket 256.
  The guards hold the claims of the supplement's text: outputs identical with the registered processes (check 2),
  identical unoptimised programs (e5), identical r1 and r2 on each chip (e4), the same number and precision of matrix
  products and the same reductions before the Pairformer in every bucket, and the answer of e3."""
  ck = read(main_dir, f'{HLO}/checks.csv')
  if len(ck) != 16 or set(ck.check2_detail) != {'identical'} or set(ck.check1) != {'pass'}:
    raise SystemExit(f'{HLO}/checks.csv: not 16 processes identical with their registered outputs')
  un = read(main_dir, f'{HLO}/unoptimised.csv')
  det = read(main_dir, f'{HLO}/determinism.csv')
  qu = read(main_dir, f'{HLO}/question.csv').set_index('bucket').differences_before_pairformer
  if set(un.unoptimised) != {'identical'} or len(un) != 4 or set(det.r1_r2) != {'identical'} or len(det) != 8:
    raise SystemExit(f'{HLO}: unoptimised programs or r1 and r2 not identical')
  if set(qu) != {'yes'}:
    raise SystemExit(f'{HLO}/question.csv: e3 is not yes everywhere ({dict(qu)})')
  dots = read(main_dir, f'{HLO}/dots.csv')
  ops = read(main_dir, f'{HLO}/opcodes.csv')
  red = read(main_dir, f'{HLO}/reductions.csv')
  red_cols = ['opcode', 'operand_types', 'result_type', 'reducer', 'dimensions', 'count']

  def chip(df, b, c, st):
    return df[(df.bucket == b) & (df.chip == c) & (df.stage == st)]
  changed = {}
  for b in (256, 512, 768, 1024):
    for st, _ in FU16_STAGES:
      d5, d6 = chip(dots, b, 'v5e', st), chip(dots, b, 'v6e', st)
      if d5['count'].sum() != d6['count'].sum():
        raise SystemExit(f'{HLO}/dots.csv: bucket {b}, {st}: different numbers of matrix products')
      prec = [d.fillna('').groupby(['precision', 'algorithm'])['count'].sum().to_dict() for d in (d5, d6)]
      if prec[0] != prec[1]:
        raise SystemExit(f'{HLO}/dots.csv: bucket {b}, {st}: different precision attributes')
      r5, r6 = (chip(red, b, c, st).fillna('')[red_cols].sort_values(red_cols).values.tolist() for c in ('v5e', 'v6e'))
      if r5 != r6:
        raise SystemExit(f'{HLO}/reductions.csv: bucket {b}, {st}: reductions differ')
      cls = [d.fillna('').groupby(['operand_types', 'result_type', 'precision', 'algorithm'])['count'].sum() for d in (d5, d6)]
      diff = cls[0].sub(cls[1], fill_value=0).abs().sum()
      changed[b, st] = int(diff) // 2
  if [changed[b, 'embedding'] for b in (256, 512, 768, 1024)] != [5, 3, 5, 3] or \
     any(changed[b, st] for b in (256, 512, 768, 1024) for st in ('template', 'msa')):
    raise SystemExit(f'{HLO}/dots.csv: products with other input types changed {changed}')
  rows = []
  for st, lab in FU16_STAGES:
    d5, d6 = chip(dots, 256, 'v5e', st), chip(dots, 256, 'v6e', st)
    o = {c: chip(ops, 256, c, st).set_index('opcode')['count'] for c in ('v5e', 'v6e')}
    r = {'Stage': lab, 'Matrix products': f'{int(d5["count"].sum())}/{int(d6["count"].sum())}',
         'With other input types': str(changed[256, st])}
    for k, name in FU16_OPS:
      r[name] = f'{int(o["v5e"].get(k, 0))}/{int(o["v6e"].get(k, 0))}'
    rows.append(r)
  # The 2 products that differ only in operand order: section 10 rerun row of 2026-11-16, item c (reading the HLO).
  write_table(out, 'TS_fu16', pd.DataFrame(rows), 'Follow-up 16: the compiled programs of the v5e and the v6e before '
              'the Pairformer, for inputs padded to 256 tokens', ROLE_FU16,
              notes=['We counted the operations in the program that XLA compiled for each chip, for inputs padded to '
                     '256 tokens (target 7U3J, seed 1). Each cell with two counts gives the v5e count, then the v6e '
                     'count. "With other input types" counts the products whose operand or result types differ between '
                     'the chips. Follow-up 16 is a registered follow-up, not a post hoc analysis, and the comparison is '
                     'descriptive.'],
              colspec='lrrrrrrr')
  # Style round of 2026-11-19: the main paper's appendix prints the table with a short note; the supplement keeps the
  # full note (the convolutions and the status of the follow-up, which the main text gives)
  write_note(out, 'TS_fu16_app',
             # options round of 2026-11-19 (Table VI B): the operand order and the 512 and 1024 buckets are in the
             # supplement's full note
             # reading notes of 2026-11-20 (53 words): the 256 tokens are in the title
             ['We counted the operations in the program that XLA compiled for each chip (target 7U3J, seed 1). Each '
              'cell with two counts gives the v5e count, then the v6e count. "With other input types" counts the '
              'products whose operand or result types differ between the chips. SUPPTABFUSIXTEEN has the full note.'],
             ROLE_FU16, 'short note below TS_fu16 in the main paper\'s appendix')
  FU16_ROW[:] = [('16', 'XLA dumps the HLO of the model on the v5e and the v6e; the computation is unchanged',
                  '4, 1',
                  'descriptive, bucket 256: before the Pairformer, the same matrix products, precision and reductions; '
                  f'conversions, the input types of {changed[256, "embedding"]} embedding products, and some float32 '
                  'and integer operations differ',
                  'the differences appear before the Pairformer in all four buckets')]


FU16_ROW = []


# ---------------------------------------------------------------------------------------------------------------------
# Follow-ups 17 to 21 (section 10, rows of 2026-11-24 with OSF addenda; analyses of 2026-11-24 and 2026-11-25; round of
# 2026-11-25). One supplementary table per follow-up and one row each in the follow-ups table. The readings are copied
# from each analysis's own outputs and checked against the readings the text states.
TRI = f'{AN}/tri_20261124'
AUT = f'{AN}/aut_20261125'
ROLE_FU1721 = ('follow-ups 17 to 21 (outside the pre-registered study; specified after the results were read; OSF addenda '
               'registered before their runs)')
FU17_21_ROWS = {}


def _one(df, **kw):
  m = df
  for k, v in kw.items():
    m = m[m[k] == v]
  if len(m) != 1:
    raise SystemExit(f'follow-ups 17 to 21: {kw}: {len(m)} rows')
  return m.iloc[0]


DTP = f'{AN}/dtp_20261125'


def fu17_tables(out, main_dir):
  """Follow-up 17 (label dtp): the four cells of the pair (V6,G), D_T, the decomposition, RC(Gfd) (rules e1 to e4) and
  part B (rule e5)."""
  cells = read(main_dir, f'{DTP}/cells.csv')
  dt = read(main_dir, f'{DTP}/d_t.csv')
  dec = read(main_dir, f'{DTP}/decomposition.csv')
  rc = read(main_dir, f'{DTP}/recompilation.csv')
  ag = read(main_dir, f'{DTP}/agree.csv')
  pb = read(main_dir, f'{DTP}/part_b.csv')
  att = read(main_dir, f'{DTP}/attribution.csv')
  if len(att) != 1 or att.iloc[0].attribution != 'both are needed':
    raise SystemExit(f'{DTP}: attribution {list(att.attribution)}; the text states "both are needed"')
  if set(ag.reading) != {'agreement not shown'}:
    raise SystemExit(f'{DTP}: part B readings {set(ag.reading)}; the text states that no pair agrees')
  rows = [{'Contrast': '\\GROUP v6e vs L4, four cells (rates, %)'}]
  for c, lab in (('c_bd', 'bfloat16, default precision (main study)'), ('c_bh', 'bfloat16, highest (Follow-up 12)'),
                 ('c_fd', 'float32, default precision (new)'), ('c_fh', 'float32, highest (Follow-up 13)')):
    rows.append({'Contrast': lab, **{l: rate_ci(_one(cells, cell=c, statistic=k)) for k, l in STAT3}})
  rows.append({'Contrast': '\\INDENT reading (top-1)', 'Top-1 flip': _one(ag, pair='HW(V6fd,Gfd)').reading})
  rows.append({'Contrast': '\\GROUP Differences (points)'})
  g = {k: _one(dt, pair='HW(V6fd,Gfd)', statistic=k) for k, _ in STAT3}
  rows.append({'Contrast': 'D_T: float32 at the default precision against bfloat16, beyond recompile L4',
               **{l: diff_ci(g[k]) for k, l in STAT3}})
  rows.append({'Contrast': '\\INDENT reading', **{l: g[k].reading for k, l in STAT3}})
  for t, lab in (('T_d', 'data type at the default precision'), ('P_f', 'precision in float32'),
                 ('I', 'interaction')):
    rows.append({'Contrast': lab, **{l: diff_ci(_one(dec, term=t, statistic=k)) for k, l in STAT3}})
  rows.append({'Contrast': '\\GROUP Recompilation on the L4 (rates, %)'})
  for c, lab in (('RC(Gfd)', 'Recompile, L4 float32 at the default precision'), ('RC(G)', 'Recompile, L4 (main study)')):
    rows.append({'Contrast': lab, **{l: rate_ci(_one(rc, comparison=c, statistic=k)) for k, l in STAT3}})
  tdf = pd.DataFrame(rows, columns=['Contrast'] + [l for _, l in STAT3]).fillna('')
  write_table(out, 'TS_fu17_cells', tdf[~tdf.Contrast.str.startswith('\\GROUP')].reset_index(drop=True),
              'Follow-up 17: the data type of the trunk and the matrix-product precision, v6e vs L4', ROLE_FU1721,
              notes=['Rates in % of predictions and differences in points, with 95% intervals over targets, 60 targets '
                     'with seeds 1 to 5. Each cell is the top-1 flip between the v6e and the L4 with one data type of '
                     'the trunk and one matrix-product precision. Agreement needs an upper bound below 5%. D_T reads '
                     '"reduced" if its interval lies below 0. The three terms of the decomposition are descriptive.'],
              colspec='llll', tex_df=tdf)
  rows = []
  names = {'V6fd': 'v6e', 'Gfd': 'L4', 'Afd': 'A100', 'Hfd': 'H100'}
  for r in ag.itertuples():
    a, b = re.fullmatch(r'HW\((\w+),(\w+)\)', r.pair).groups()
    rows.append({'Pair, float32 at the default precision': f'{names[a]} vs {names[b]}',
                 'Top-1 flip': f'{pc(r.flip_top1)} [{pc(r.ci_low)}, {pc(r.ci_high)}]', 'Reading': r.reading})
  for comp, lab in (('D_T of HW(Hfd,Afd)', 'D_T, H100 vs A100'), ('D_T of HW(Gfd,Afd)', 'D_T, L4 vs A100')):
    r = _one(pb, comparison=comp, statistic='flip_top1')
    rows.append({'Pair, float32 at the default precision': lab, 'Top-1 flip': diff_ci(r), 'Reading': r.reading})
  write_table(out, 'TS_fu17_pairs', pd.DataFrame(rows), 'Follow-up 17: the four accelerators with the float32 trunk at '
              'the default precision', ROLE_FU1721,
              notes=['Top-1 flip in % of predictions (pairs) and in points (D_T), with 95% intervals over targets, 60 '
                     'targets with seeds 1 to 5. The D_T of the H100 and the A100 sets float32 recompilation on the '
                     'H100 to 0, which makes it an upper bound; that of the L4 and the A100 uses the measured '
                     'recompilation on the L4.'], colspec='lll')
  c = _one(cells, cell='c_fd', statistic='flip_top1')
  FU17_21_ROWS['17'] = ('17', 'float32 trunk at the default matmul precision',
                        '60, 5', f'v6e vs L4 {rate_ci(c)}; D_T {diff_ci(g["flip_top1"])}',
                        'agreement not shown; not reduced; both are needed (only the two changes together reduce '
                        'it); no pair of the four accelerators agrees')


def fu18_table(out, main_dir):
  """Follow-up 18 (label tri): the H100 and the A100 with Triton attention, recompilation of both, E_T (rule e1)."""
  con = read(main_dir, f'{TRI}/contrasts.csv')
  et = read(main_dir, f'{TRI}/e_t.csv')
  dp = read(main_dir, f'{TRI}/default_pairs.csv')
  cost = read(main_dir, f'{TRI}/cost.csv')
  e1 = _one(et, comparison='E_T = HW(HT,AT) minus RC(HT)', statistic='flip_top1')
  if e1.reading != 'beyond recompilation':
    raise SystemExit(f'{TRI}: E_T reads {e1.reading}; the text states beyond recompilation')
  rows = [{'Contrast': '\\GROUP Rates (%)'}]
  for c, lab in (('HW(HT,AT)', 'H100 Triton vs A100 Triton'), ('RC(HT)', 'Recompile, H100 Triton'),
                 ('RC(AT)', 'Recompile, A100 Triton'), ('HW(GT,AT)', 'L4 Triton vs A100 Triton'),
                 ('HW(GT,HT)', 'L4 Triton vs H100 Triton')):
    rows.append({'Contrast': lab, **{l: rate_ci(_one(con, contrast=c, statistic=k)) for k, l in STAT3}})
  rows.append({'Contrast': '\\GROUP Differences (points)'})
  for comp, lab in (('E_T = HW(HT,AT) minus RC(HT)', 'H100 Triton vs A100 Triton minus recompile H100 Triton'),
                    ('HW(HT,AT) minus the larger of RC(HT) and RC(AT)', 'the same, minus the larger recompilation')):
    g = {k: _one(et, comparison=comp, statistic=k) for k, _ in STAT3}
    rows.append({'Contrast': lab, **{l: diff_ci(g[k]) for k, l in STAT3}})
    rows.append({'Contrast': '\\INDENT reading', **{l: g[k].reading.replace(' recompilation', '').replace(
        'not distinguishable from', 'not distinguishable') for k, l in STAT3}})
  g = {k: _one(dp, pair='HW(GT,AT)', statistic=k) for k, _ in STAT3}
  rows.append({'Contrast': 'L4 Triton vs A100 Triton minus recompile A100 Triton', **{l: diff_ci(g[k]) for k, l in STAT3}})
  c = _one(cost, new='HT', reference='H', bucket='all')
  df = pd.DataFrame(rows, columns=['Contrast'] + [l for _, l in STAT3])
  write_table(out, 'TS_fu18', df[~df.Contrast.str.startswith('\\GROUP')].reset_index(drop=True),
              'Follow-up 18: the H100 and the A100 with Triton attention', ROLE_FU1721,
              notes=['Rates in % of predictions and differences in points, with 95% intervals over targets, 60 targets '
                     'with seeds 1 to 5. Triton is AlphaFold 3\'s default attention. Beyond: the interval of the '
                     'difference lies above 0. The registered rule reads the first difference, top-1 column. '
                     f'The H100 Triton took {float(c.wall_ratio):.2f} times the time of the H100 with XLA attention.'],
              colspec='llll', tex_df=df)
  FU17_21_ROWS['18'] = ('18', 'Triton attention on the H100 and the A100',
                        '60, 5',
                        f'H100 Triton vs A100 Triton {rate_ci(_one(con, contrast="HW(HT,AT)", statistic="flip_top1"))}, '
                        f'minus recompile H100 Triton {diff_ci(e1)}',
                        f'beyond recompilation; L4 Triton vs A100 Triton not ({diff_ci(g["flip_top1"])})')


def fu19_table(out, main_dir):
  """Follow-up 19 (label rcf): RC(Af) and RC(Hf) on the 60 targets, and D_F of (Hf,Af) with RC(Hf) measured."""
  con = read(main_dir, f'{RCF}/contrasts.csv')
  ident = read(main_dir, f'{RCF}/identity.csv')
  dfs = read(main_dir, f'{RCF}/d_f.csv')
  rows = []
  for c, lab, cond in (('RC(Af)', 'Recompile, A100 float32', 'Af'), ('RC(Hf)', 'Recompile, H100 float32', 'Hf')):
    g = {k: _one(con, contrast=c, statistic=k) for k, _ in STAT3}
    idn = ident[(ident.condition == cond) & (ident.targets == 'all 60')]
    if len(idn) != 1:
      raise SystemExit(f'{RCF}: identity row of {cond}, all 60: {len(idn)}')
    rows.append({'Contrast': lab, 'Predictions': int(g['flip_top1'].n), **{l: rate_ci(g[k]) for k, l in STAT3},
                 'Identical output files': idn.iloc[0].value.replace(' identical', '').replace('/', ' of ')})
  write_table(out, 'TS_fu19_rc', pd.DataFrame(rows), 'Follow-up 19: float32 recompilation on 60 targets', ROLE_FU1721,
              notes=['Rates in % with 95% intervals over targets, 60 targets with seeds 1 to 5: the first runs of '
                     'Follow-up 14 against the second runs of Follow-ups 15 (10 pilot targets) and 19 (50 targets). '
                     'Identical output files counts the predictions whose compared files are all identical.'],
              colspec='lrllll')
  rows = []
  for r in dfs.itertuples():
    rows.append({'Version': r.version, 'Statistic': dict(STAT3)[r.statistic], 'D_F (points)': diff_ci(r),
                 'Reading': r.reading})
  top = dfs[dfs.statistic == 'flip_top1']
  if set(top.reading) != {'reduced'}:
    raise SystemExit(f'{RCF}: D_F top-1 readings {set(top.reading)}; the text states reduced, as registered')
  write_table(out, 'TS_fu19_df', pd.DataFrame(rows), 'Follow-up 19: D_F of the H100 and the A100 with recompilation '
              'measured on 60 targets', ROLE_FU1721,
              notes=['D_F of H100 float32 vs A100 float32 against H100 vs A100, with float32 recompilation on the H100 '
                     'set to 0 (registered, Follow-up 14) or measured on the 60 targets, with the same bootstrap draws. '
                     'The registered reading stands either way.'], colspec='p{6cm}lll')
  af, hf = (_one(con, contrast=c, statistic='flip_top1') for c in ('RC(Af)', 'RC(Hf)'))
  meas = _one(dfs, statistic='flip_top1', version='RC(Hf) measured on the 60 targets')
  FU17_21_ROWS['19'] = ('19', 'float32 recompilation, A100 and H100, 50 more targets', '60, 5',
                        f'recompile A100 float32 {rate_ci(af)}, H100 float32 {rate_ci(hf)}; D_F of H100 vs A100 with '
                        f'recompilation measured {diff_ci(meas)}',
                        'float32 recompilation changes 1 of 300 top-ranked samples on each GPU; D_F reduced, as '
                        'registered')


XVM = f'{AN}/xvm_20261125'


def fu21_table(out, main_dir):
  """Follow-up 21 (label xvm): recompilation on a new VM against the same VM, on the L4 and the A100 (rules e1, e2)."""
  con = read(main_dir, f'{XVM}/contrasts.csv')
  x = read(main_dir, f'{XVM}/x.csv')
  mo = read(main_dir, f'{XVM}/main_output.csv')
  for p in ('G', 'A'):
    r = _one(x, condition=p, statistic='flip_top1')
    if r.reading != 'not distinguishable':
      raise SystemExit(f'{XVM}: X({p}) reads {r.reading}; the text states not distinguishable for the L4 and the A100')
  rows = []
  for p, name in (('G', 'L4'), ('A', 'A100')):
    rows.append({'Contrast': f'\\GROUP {name}'})
    for c, lab in ((f'RC_same({p})', 'Recompile, same VM (r1 vs r2)'), (f'RC_new({p})', 'Recompile, new VM (r1 vs r3)'),
                   (f'RC_new2({p})', 'Recompile, new VM (r2 vs r3)')):
      rows.append({'Contrast': lab, **{l: rate_ci(_one(con, contrast=c, statistic=k)) for k, l in STAT3}})
    g = {k: _one(x, condition=p, statistic=k) for k, _ in STAT3}
    rows.append({'Contrast': 'New VM minus same VM (r1 vs r3 minus r1 vs r2)', **{l: diff_ci(g[k]) for k, l in STAT3}})
    rows.append({'Contrast': '\\INDENT reading', **{l: g[k].reading for k, l in STAT3}})
  tdf = pd.DataFrame(rows, columns=['Contrast'] + [l for _, l in STAT3])
  plain = []
  grp = None
  for r in rows:
    if r['Contrast'].startswith('\\GROUP '):
      grp = r['Contrast'][7:]
      continue
    plain.append({'Platform': grp, **r})
  m = {(r.condition, r.vm, r.sides): (int(r.main_output_differs), int(r.success_differs)) for r in mo.itertuples()}
  gs, as_ = m[('G', 'same VM', 'G r1 against G r2')], m[('A', 'same VM', 'A r1 against A r2')]
  gn, an = m[('G', 'new VM', 'G r1 against G r3')], m[('A', 'new VM', 'A r1 against A r3')]
  write_table(out, 'TS_fu21', pd.DataFrame(plain), 'Follow-up 21: recompilation on a new VM of the same GPU type',
              ROLE_FU1721,
              notes=['Rates in % of predictions and differences in points, with 95% intervals over targets, 60 targets '
                     'with seeds 1 to 5. r1 and r2 ran on one VM per target (main study and Follow-up 3), r3 on new VMs. '
                     'Not distinguishable: the interval of the difference includes 0; this is not a test of '
                     'equivalence. AlphaFold 3\'s main output, the top-ranked of the 25 samples of seeds 1 to 5, '
                     f'differed on {gs[0]} of 60 targets between r1 and r2 of the L4 and on {gn[0]} between r1 and r3, '
                     f'and on {as_[0]} and {an[0]} on the A100.'],
              colspec='llll', tex_df=tdf)
  xg, xa = (_one(x, condition=p, statistic='flip_top1') for p in ('G', 'A'))
  FU17_21_ROWS['21'] = ('21', 'L4 and A100 rerun on new VMs', '60, 5',
                        f'recompile on a new VM minus on the same VM: L4 {diff_ci(xg)}, A100 {diff_ci(xa)}',
                        'not distinguishable on both; the registered recompilation references stand')


def fu20_table(out, main_dir):
  """Follow-up 20 (label aut): the L4 without autotuning (rules e1 to e3), NA(G), HW(V6,Gna) and the cost."""
  con = read(main_dir, f'{AUT}/contrasts.csv')
  rvg = read(main_dir, f'{AUT}/rc_vs_g.csv')
  reg = read(main_dir, f'{AUT}/registered.csv')
  cost = read(main_dir, f'{AUT}/cost.csv')
  ident = read(main_dir, f'{AUT}/identity.csv')
  d1 = _one(rvg, statistic='flip_top1')
  if d1.reading != 'reduced without autotuning':
    raise SystemExit(f'{AUT}: rule 2 reads {d1.reading}; the text states reduced')
  b = _one(con, contrast='RC(Gna)', statistic='bit_identical_fraction')
  if float(b.estimate) != 1.0:
    raise SystemExit(f'{AUT}: RC(Gna) bit-identical fraction {b.estimate}; the text states all 300 bit-identical')
  rows = [{'Contrast': '\\GROUP Rates (%)'}]
  for c, lab, src in (('RC(Gna)', 'Recompile, L4 without autotuning', con), ('RC(G)', 'Recompile, L4 (main study)', reg),
                      ('NA(G)', 'L4 vs L4 without autotuning', con),
                      ('HW(V6,Gna)', 'v6e vs L4 without autotuning', con), ('HW(V6,G)', 'v6e vs L4 (main study)', reg),
                      ('HW(V6,Gd)', 'v6e vs L4 deterministic (main study)', reg)):
    rows.append({'Contrast': lab, **{l: rate_ci(_one(src, contrast=c, statistic=k)) for k, l in STAT3}})
  rows.append({'Contrast': '\\GROUP Difference (points)'})
  g = {k: _one(rvg, statistic=k) for k, _ in STAT3}
  rows.append({'Contrast': 'Recompile, L4 without autotuning minus recompile, L4', **{l: diff_ci(g[k]) for k, l in STAT3}})
  rows.append({'Contrast': '\\INDENT reading', **{l: g[k].reading.replace(' without autotuning', '') for k, l in STAT3}})
  c = _one(cost, new='Gna', reference='G', bucket='all')
  df = pd.DataFrame(rows, columns=['Contrast'] + [l for _, l in STAT3])
  write_table(out, 'TS_fu20', df[~df.Contrast.str.startswith('\\GROUP')].reset_index(drop=True),
              'Follow-up 20: the L4 without autotuning', ROLE_FU1721,
              notes=['Rates in % of predictions and differences in points, with 95% intervals over targets, 60 targets '
                     'with seeds 1 to 5. Without autotuning, all 300 predictions of the recompilation had identical '
                     'output files. Reduced: the interval of the difference lies below 0. The main-study rows are '
                     f'copied as registered. Process time without autotuning was {float(c.wall_ratio):.2f} times that '
                     f'of the L4, compile time {float(c.compile_ratio):.2f} times, and compiled memory '
                     f'{float(c.compiled_memory_ratio):.2f} times.'],
              colspec='llll', tex_df=df)
  FU17_21_ROWS['20'] = ('20', 'XLA autotuning off on the L4', '60, 5',
                        f'recompile L4 without autotuning {rate_ci(_one(con, contrast="RC(Gna)", statistic="flip_top1"))}, '
                        f'all 300 bit-identical; minus recompile L4 {diff_ci(d1)}',
                        'turning autotuning off removed the recompilation differences on the L4, consistent with '
                        'kernel selection by timing as their source')
FOREST_RANGES = []   # Fig. 1: top-1 min, max and success-flip min, max (%), for the text and its % src
SEEDS_FACTS = []     # Fig. 4: estimate min, max (points) and one target in points, for the text


def table_beyond(out, main_dir):
  """Main-text table of Results IV-D (IV-G until 2026-11-19): each follow-up pair's top-1 flip and its excess over recompilation."""
  def con(src):
    rel = f'{src}/contrasts/contrasts.csv' if src.endswith('primary_v2') else f'{src}/contrasts.csv'
    return read(main_dir, rel)
  def hw(src):
    rel = f'{src}/contrasts/hw_minus_rc.csv' if src.endswith('primary_v2') else f'{src}/hw_minus_rc.csv'
    return read(main_dir, rel)
  def rate(src, c, **kw):
    return rate_ci(pick(con(src), contrast=c, statistic='flip_top1', **kw))
  def ex(src, c, rc, **kw):
    return diff_ci(pick(hw(src), hardware=c, recompilation=rc, statistic='flip_top1', **kw))
  A, H, C, R, M, P = (f'{AN}/a100_20261018', f'{AN}/h100_20261104', f'{AN}/cpu60_20261018', f'{AN}/repl_20261106',
                      f'{AN}/msa_20261102', f'{AN}/ppi_20261015')
  # the post hoc L4 vs CPU excess on matched seeds 1 and 2 left this table's note on 2026-11-19; the supplement's
  # Follow-up 4 section prints it, and this guard holds that sentence
  cm = read(main_dir, f'{AN}/cpu_matched_20261114/matched.csv')
  cm = cm[(cm.pair == 'follow-up 4, 60 targets') & (cm.statistic == 'flip_top1')].iloc[0]
  if diff_ci(cm) != '+14.2 [7.5, 20.8]':
    raise SystemExit(f'cpu_matched: L4 vs CPU excess on seeds 1 and 2 is {diff_ci(cm)}; the supplement (Follow-up 4) '
                     'prints +14.2 [7.5, 20.8]')
  sw = read(main_dir, f'{AN}/sw_20261102/comparisons.csv')
  sg = pick(sw, software='SW(G)', reference='RC(G)', statistic='flip_top1', targets='all 60')
  rcx = read(main_dir, f'{RCX}/contrasts.csv')
  gvg = read(main_dir, f'{RCX}/rc_gt_vs_g.csv').set_index('statistic')
  r15 = {c: rate_ci(pick(rcx, contrast=c, statistic='flip_top1')) for c in ('RC(GT)',)}
  # R121 (2026-11-16): the float32 recompilation rows of FU15 (A100, H100) left this table; Table IV (TIII_repro)
  # gives them in its float32 column
  # rows in numerical order of the follow-ups
  rows = [   # excess over the larger recompilation rate of the two platforms (a TPU's is 0)
      ('1', 'v6e vs L4, protein-protein', rate(P, 'HW(V6,G)'), ex(P, 'HW(V6,G)', 'RC(G)')),
      ('3', 'L4 vs A100', rate(A, 'HW(G,A)'), ex(A, 'HW(G,A)', 'RC(A)')),
      ('4', 'v5e vs CPU', rate(C, 'HW(V5,C)'), '= rate'),
      ('4', 'v6e vs CPU', rate(C, 'HW(V6,C)'), '= rate'),
      ('4', 'L4 vs CPU', rate(C, 'HW(G,C)'), ex(C, 'HW(G,C)', 'RC(G)')),
      ('6', 'v6e vs L4, official inputs', rate(M, 'HW(V6o,Go)', targets='all 60'), ex(M, 'HW(V6o,Go)', 'RC(Go)', targets='all 60')),
      ('9', 'H100 vs A100', rate(H, 'HW(H,A)', targets='all 60'), ex(H, 'HW(H,A)', 'RC(A)', targets='all 60')),
      ('11', 'v5e vs v6e, new targets', rate(R, 'HW(V5,V6)', targets='all 72'), '= rate'),
      ('11', 'v6e vs L4, new targets', rate(R, 'HW(V6,G)', targets='all 72'), ex(R, 'HW(V6,G)', 'RC(G)', targets='all 72')),
      ('15', 'Recompile, L4 Triton', r15['RC(GT)'], diff_ci(gvg.loc['flip_top1'])),
  ]
  for c, rc in (('HW(G,A)', 'RC(G)'), ('HW(H,A)', 'RC(H)')):   # the larger reference is the one shown
    src = A if c == 'HW(G,A)' else H
    kw = {} if c == 'HW(G,A)' else {'targets': 'all 60'}
    if float(pick(hw(src), hardware=c, recompilation=rc, statistic='flip_top1', **kw).estimate) < \
       float(pick(hw(src), hardware=c, recompilation='RC(A)', statistic='flip_top1', **kw).estimate):
      raise SystemExit(f'Table of IV-D: for {c}, recompilation on the A100 is no longer the larger reference')
  # headers and labels of 2026-11-17, in words; Style round of 2026-11-19: FU and two-line headers, so the table
  # fits the column at \footnotesize without scaling (it printed at 80% before)
  df = pd.DataFrame(rows, columns=['Follow-up', 'Pair', 'Top-1 flip (%)', 'Beyond recompilation (points)'])
  write_table(out, 'T_beyond', df, 'The Difference Beyond the Main Study', ROLE_FU,
              # 'Beyond recompilation' on one line and '(points)' below it was 262 pt wide, 10 pt over the column:
              # the header takes three lines
              head=['FU & Pair & Top-1 flip & Beyond \\\\', ' &  & (\\%) & recompilation \\\\', ' &  &  & (points) \\\\'],
              # options round of 2026-11-19 (Table III B): official inputs and FU15 are defined in Methods (III-D),
              # and the text below the table cites the supplementary table of every reading
              # reading notes of 2026-11-20 (43 words)
              notes=['Top-1 flip and its excess over the larger recompilation rate of the pair, with 95% intervals '
                     'over targets. FU is the follow-up. "= rate" means a recompilation rate of 0. A CPU pair uses the '
                     'other platform\'s rate, because the CPU ran once.'],
              colspec='rlll')


def table_e_summary(out, main_dir):
  """Supplementary table of exploratory analysis (e), formatted (replaces the unformatted T16b of paper_assets.py)."""
  e = read(main_dir, f'{EXPL}/e_summary.csv')
  rows = [{'Contrast': clab(r.contrast), 'Separated by PoseBusters': int(r.PoseBusters), 'Separated by RMSD': int(r.RMSD),
           'Flips that matter': int(r.flips_that_matter),
           'Median RMSD difference (\u00c5)': '' if pd.isna(r.median_rmsd_difference) else f'{float(r.median_rmsd_difference):.3f}'}
          for r in e.itertuples()]
  write_table(out, 'TS_e_summary', pd.DataFrame(rows), 'Exploratory (e): flips that matter, by the criterion that '
              'separates the two top-ranked structures', ROLE_EXPL, colspec='lrrrr',
              notes=['PoseBusters: both top-ranked ligands lie within 2 \u00c5 of the experimental pose, and only one '
                     'passes the PoseBusters checks. RMSD: the ligand crosses the 2 \u00c5 threshold. The last column is '
                     'the median difference between the two ligand RMSDs to the experimental pose.'])


def table_repro(out, main_dir):
  """Table III of the main text (2026-11-14): the cost of reproducibility per platform.
  Three policies: the default bfloat16 arm, XLA's deterministic flag (GPUs only), and the float32 trunk at the highest
  precision. Residual top-1 flip (recompilation; between platforms), time and compiled memory against the default arm."""
  def flip(src, contrast, **kw):
    df = read(main_dir, src)
    m = df[(df.contrast == contrast) & (df.statistic == 'flip_top1')]
    for k, v in kw.items():
      m = m[m[k] == v]
    if len(m) != 1:
      raise SystemExit(f'Table III (reproducibility): {contrast} in {src}: {len(m)} rows')
    return m.iloc[0]
  mc, a100, h100 = f'{MAINV2}/contrasts/contrasts.csv', f'{AN}/a100_20261018/contrasts.csv', f'{AN}/h100_20261104/contrasts.csv'
  f13c, f13cost = f'{FU13_DIR}/f_contrasts.csv', f'{FU13_DIR}/f_cost.csv'
  f14c, f14cost, f14ag = f'{FU14}/contrasts.csv', f'{FU14}/cost.csv', f'{FU14}/agree.csv'
  cost13, cost14 = read(main_dir, f13cost), read(main_dir, f14cost)

  def ratio(cost, new, col, ref=None):
    m = cost[(cost.new == new) & (cost.bucket.astype(str) == 'all')]
    if ref is not None:
      m = m[m.reference == ref]
    if len(m) != 1:
      raise SystemExit(f'Table III (reproducibility): cost row {new}: {len(m)}')
    return f'{float(m.iloc[0][col]):.2f}'
  cs = read(main_dir, f'{MAINV2}/cost/cost_summary.csv')
  fr = cs[cs.cache == 'fresh']
  g = fr[fr.platform == 'G'].set_index('bucket').median_wall_s
  gd = fr[fr.platform == 'Gd'].set_index('bucket').median_wall_s
  rg = (gd / g).dropna()
  mem = read(main_dir, f'{MAINV2}/cost/memory.csv')
  mg = mem[mem.platform == 'G'].set_index('bucket').compiled_peak_gib
  mgd = mem[mem.platform == 'Gd'].set_index('bucket').compiled_peak_gib
  if not ((mgd / mg).dropna().round(3) == 1.0).all():
    raise SystemExit('Table III (reproducibility): the flag changed the compiled memory on the L4')
  for c in ('Ad', 'Hd'):
    m = cost14[(cost14.new == c) & (cost14.bucket.astype(str) == 'all')]
    if abs(float(m.iloc[0].compiled_memory_ratio) - 1.0) > 0.005:
      raise SystemExit(f'Table III (reproducibility): the flag changed the compiled memory on {c}')
  star = lambda r: pc(r.estimate) + '*'  # noqa: E731
  rows = [
      {'Platform': 'v5e', 'd_rc': pc(flip(mc, 'RC(V5)').estimate), 'x_rc': 'n/a', 'x_t': 'n/a',
       'f_rc': star(flip(f14c, 'RC(V5f)')), 'f_t': ratio(cost14, 'V5f', 'wall_ratio'),
       'f_m': ratio(cost14, 'V5f', 'compiled_memory_ratio')},
      {'Platform': 'v6e', 'd_rc': pc(flip(mc, 'RC(V6)').estimate), 'x_rc': 'n/a', 'x_t': 'n/a',
       'f_rc': star(flip(f13c, 'RC(V6f)')), 'f_t': ratio(cost13, 'V6f', 'wall_ratio'),
       'f_m': ratio(cost13, 'V6f', 'compiled_memory_ratio')},
      {'Platform': 'L4', 'd_rc': pc(flip(mc, 'RC(G)').estimate), 'x_rc': pc(flip(mc, 'RC(Gd)').estimate),
       'x_t': f'{rg.min():.2f}--{rg.max():.2f}', 'f_rc': pc(flip(f13c, 'RC(Gf)').estimate),
       'f_t': ratio(cost13, 'Gf', 'wall_ratio'), 'f_m': ratio(cost13, 'Gf', 'compiled_memory_ratio')},
      {'Platform': 'A100', 'd_rc': f"{pc(flip(a100, 'RC(A)').estimate)} ({star(flip(f14c, 'RC(A) pilot'))})", 'x_rc': star(flip(f14c, 'RC(Ad)')),
       'x_t': ratio(cost14, 'Ad', 'wall_ratio'), 'f_rc': pc(flip(f'{RCF}/contrasts.csv', 'RC(Af)', n=300).estimate), 'f_t': ratio(cost14, 'Af', 'wall_ratio'),
       'f_m': ratio(cost14, 'Af', 'compiled_memory_ratio')},
      {'Platform': 'H100', 'd_rc': f"{pc(flip(h100, 'RC(H)', n=300).estimate)} ({star(flip(f14c, 'RC(H) pilot'))})", 'x_rc': star(flip(f14c, 'RC(Hd)')),
       'x_t': ratio(cost14, 'Hd', 'wall_ratio'), 'f_rc': pc(flip(f'{RCF}/contrasts.csv', 'RC(Hf)', n=300).estimate), 'f_t': ratio(cost14, 'Hf', 'wall_ratio'),
       'f_m': ratio(cost14, 'Hf', 'compiled_memory_ratio')},
  ]
  prim = [flip(mc, 'HW(V5,V6)'), flip(mc, 'HW(V6,G)'), flip(h100, 'HW(H,A)', n=300), flip(a100, 'HW(G,A)')]
  ag = read(main_dir, f14ag)
  # 'Between platforms' until 2026-11-17
  rows.append({'Platform': 'pairs', 'd_rc': f'{pc(min(r.estimate for r in prim))}--{pc(max(r.estimate for r in prim))}',
               'x_rc': pc(flip(mc, 'HW(V6,Gd)').estimate), 'x_t': 'n/a',
               'f_rc': f'{pc(ag.flip_top1.min())}--{pc(ag.flip_top1.max())}', 'f_t': 'n/a', 'f_m': 'n/a'})
  df = pd.DataFrame(rows)
  if (df.astype(str).apply(lambda col: col.str.strip()) == '').any().any():   # R121: n/a, never an empty cell
    raise SystemExit('Table IV (reproducibility): an empty cell; write n/a where the measure does not apply')
  # two italic group labels, as in Table II
  tdf = pd.DataFrame([{'Platform': '\\GROUP Recompilation on one platform'}] + rows[:-1]
                     + [{'Platform': '\\GROUP Change of platform'}, rows[-1]])
  npair_w = {4: 'four'}[len(prim)]     # the bfloat16 pairs of Fig. 2(b), named in the note
  if len(ag) != 10:
    raise SystemExit(f'Table IV note: {len(ag)} float32 pairs, the text of IV-C says all 10')
  write_table(out, 'TIII_repro', df, 'The Cost of Reproducibility per Platform', ROLE_FU14,
              colspec='lrrrrrr', tex_df=tdf,
              head=[' & default & \\multicolumn{2}{c}{deterministic flag} & \\multicolumn{3}{c}{float32 trunk} \\\\',
                    '\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(l){5-7}',
                    'platform & flip & flip & time & flip & time & memory \\\\'],
              # options round of 2026-11-19 (Table IV B): the flag's pair (21.0) is in Results (IV-D), the float32
              # range in IV-C, Follow-up 15 in IV-E
              # reading notes of 2026-11-20: five short sentences instead of the parenthesis with a semicolon
              # reading notes of 2026-11-23: the asterisk has its own sentence that names the symbol
              notes=['Top-1 flip in % of predictions on 60 targets with seeds 1 to 5. An asterisk (*) marks values measured '
                     'on 10 pilot targets. Parentheses give the default on those 10. Time and compiled memory are ratios of medians '
                     'to the default arm. The L4 flag time is a range over buckets. The last row gives ranges over the '
                     f'{npair_w} bfloat16 pairs of FIGREFFUTHIRTEENB and over all {len(ag)} float32 pairs.'])
  return df


# ---------------------------------------------------------------------------------------------------------------------
# Reporting rules (section 10, row of 2026-11-10, label ranking) and campaign simulation (row of 2026-11-10, label
# campaign): follow-up analyses of frozen outputs, descriptive. The tables copy the outputs' values, formatted.

RANKING = f'{AN}/ranking_20261110'
CAMPAIGN = f'{AN}/campaign_20261110'
ROLE_RULES = ('follow-up analysis of frozen outputs (section 10, row of 2026-11-10, ranking rules), outside the '
              'pre-registered study, descriptive; the resampling model is exploratory')
PRIMARY_NOTE = 'The primary pairs are v5e vs v6e, v6e vs L4, and L4 vs CPU.'
ARMS_NOTE = 'Each variant arm differs from its device\'s base arm in one setting. USD/h is the on-demand list price.'
MEMORY_NOTE = 'The A100 and the H100 have 80 GB of memory.'   # Table I lost its device column (2026-11-19)
ROLE_CAMP = ('follow-up analysis of frozen outputs (section 10, row of 2026-11-10, campaign simulation), outside the '
             'pre-registered study, descriptive; the shares of jobs moved are assumptions, not measurements')
RULE_PAIRS = ['HW(V5,V6)', 'HW(V6,G)', 'HW(H,A)', 'RC(G)']          # columns of the main-text table
RULE_ALL = ['HW(V5,V6)', 'HW(V6,G)', 'HW(H,A)', 'RC(G)', 'RC(A)', 'RC(H)']
RULE_T_MAIN = [0.0005, 0.001]
POLICY_SHORT = {1: '(1) A100 → L4', 2: '(2) A100 → H100', 3: '(3) L4 → CPU', 4: '(4) v6e → L4', 5: '(5) v5e → v6e'}
SOURCE_SHORT = {f'{AN}/a100_20261018': 'FU3', f'{AN}/h100_20261104': 'FU9', f'{AN}/cpu60_20261018': 'FU4',
                MAINV2: 'main', f'{AN}/repl_20261106': 'FU11'}


def pts(x):
  """Percentage points already in points, one decimal, with sign."""
  s = f'{float(x):+.1f}'
  return '0.0' if s in ('+0.0', '-0.0') else s


def pts_ci(r):
  return f'{pts(r.estimate)} [{pts(r.ci_low)}, {pts(r.ci_high)}]'.replace('[+', '[').replace(', +', ', ')


def tie_rows(tt, pair, n, t, metric):
  return pick(tt[np.isclose(tt.t, t)], pair=pair, n=n, metric=metric)


def table_rules(out, sc, tt, rd, model, cost):
  """Main-text table: top-1 flip of the reported prediction by rule, at one and five seeds, with the share of units
  in which a tie is declared; supplementary tables with intervals, the seed-consensus series with its cost, the
  rounded score, and the exploratory resampling model."""
  for want, have in ((RULE_ALL, set(tt.pair)), (RULE_ALL, set(rd.pair))):
    if set(want) - have:
      raise SystemExit(f'ranking: pairs {sorted(set(want) - have)} missing')
  rows, tex_rows = [], []
  for n, label in ((1, 'One seed, one process per prediction'), (5, 'Five seeds, five processes per prediction')):
    tex_rows.append({'Rule': f'\\GROUP {label}'})
    rules = [('Full-precision score', lambda p: pick(rd, pair=p, n=n, ranking='ranking_score',
                                                                     metric='flip_top1'), None),
             ('Two-decimal score', lambda p: pick(rd, pair=p, n=n, ranking='ranking_score_summary',
                                                  metric='flip_top1'), None)]
    for t in RULE_T_MAIN:
      rules.append((f'Tie within {t:g}', lambda p, t=t: tie_rows(tt, p, n, t, 'flip_top1'),
                    lambda p, t=t: tie_rows(tt, p, n, t, 'tie_either')))
    for name, flip, tie in rules:
      row = {'Rule': name}
      for p in RULE_PAIRS:
        r = flip(p)
        row[p] = pc(r.estimate)
        rows.append({'n': n, 'rule': name, 'pair': p, 'flip_top1': r.estimate, 'ci_low': r.ci_low,
                     'ci_high': r.ci_high, 'tie_either': tie(p).estimate if tie else np.nan})
      if tie:
        v = [float(tie(p).estimate) for p in RULE_PAIRS]
        row['Tie'] = f'{pc(min(v))}--{pc(max(v))}'
      else:
        row['Tie'] = ''
      tex_rows.append(row)
  w = {pl: float(pick(cost, platform=pl, n=1, quantity='wall_s_per_prediction').estimate) for pl in ('V5', 'G')}
  # the processes per prediction are n, by the row's definition; checked here
  for n in (1, 2, 5, 10, 20):
    for pl in ('V5', 'G'):
      if float(pick(cost, platform=pl, n=n, quantity='processes_per_prediction').estimate) != n:
        raise SystemExit(f'ranking cost: processes for {pl}, n = {n}')
  tdf = pd.DataFrame(tex_rows, columns=['Rule'] + RULE_PAIRS + ['Tie'])
  head = [' & '.join(['Rule'] + [tex(clab(p)) for p in RULE_PAIRS] + ['Tie (\\%)']) + ' \\\\']
  notes = [f'Top-1 flip of the reported prediction, in %, on 60 targets. Rates are per prediction at one seed and '
           f'per target at five. Full-precision score: AlphaFold 3\'s rule. Two-decimal: ties to the lowest seed and sample. '
           f'Tie within t: a tie is declared when the top two '
           f'scores differ by at most t, and a flip needs one sample on each side. Tie: units with a tie on either side.']
  write_table(out, 'T_rules', pd.DataFrame(rows),
              'Reporting rules: how often the reported prediction differs between two platforms, by rule and number '
              'of seeds (follow-up analysis of frozen outputs, descriptive)', ROLE_RULES, notes=notes,
              colspec='lrrrrr', tex_df=tdf, head=head)

  # supplementary: seed consensus with cost
  srows = []
  for r in sc[sc.metric == 'flip_top1'].itertuples():
    fs = pick(sc, pair=r.pair, n=r.n, metric='flip_success')
    c = {pl: {q: float(pick(cost, platform=pl, n=int(r.n), quantity=q).estimate)
              for q in ('wall_s_per_prediction', 'usd_on_demand_per_prediction', 'usd_spot_per_prediction')}
         for pl in ('V5', 'G')}
    srows.append({'Pair': r.pair, 'n': int(r.n), 'Units': int(r.units), 'Top-1 flip': rate_ci(r),
                  'Minus n = 1': f'{spc(r.difference)} [{pc(r.difference_ci_low)}, {pc(r.difference_ci_high)}]',
                  'Success flip': rate_ci(fs), 'Processes': int(r.n),
                  'Wall v5e (s)': f'{c["V5"]["wall_s_per_prediction"]:.0f}',
                  'Wall L4 (s)': f'{c["G"]["wall_s_per_prediction"]:.0f}',
                  'USD v5e': f'{usd3(c["V5"]["usd_on_demand_per_prediction"])} / {usd3(c["V5"]["usd_spot_per_prediction"])}',
                  'USD L4': f'{usd3(c["G"]["usd_on_demand_per_prediction"])} / {usd3(c["G"]["usd_spot_per_prediction"])}'})
  write_table(out, 'TS_rules_seeds', pd.DataFrame(srows),
              'Ranking rules, seed consensus: top-1 flip of the top-ranked sample across n consecutive seeds (disjoint '
              'blocks of seeds 1 to 20; recompilation on the L4 on seeds 1 to 5), and the cost per reported prediction',
              ROLE_RULES,
              notes=['Flips in %, 95% intervals over targets. Cost: n processes; wall time, n times the median wall per '
                     'structure of the target\'s bucket (fresh cache), mean over the 60 targets; USD on demand / Spot, '
                     'fresh cache (amortized values in the CSV of the analysis, cost.csv).'],
              colspec='lrrllllrrrr')

  # supplementary: tie tolerance (flip with interval; tie share and conflict)
  ts = sorted(tt.t.unique())
  frows, srows2 = [], []
  for p in RULE_ALL:
    for n in (1, 5):
      fr_ = {'Pair': p, 'n': n}
      sr_ = {'Pair': p, 'n': n}
      for t in ts:
        f_ = tie_rows(tt, p, n, t, 'flip_top1')
        fr_[f't = {t:g}'] = rate_ci(f_)
        sr_[f't = {t:g}'] = f'{pc(tie_rows(tt, p, n, t, "tie_either").estimate)} / ' \
                            f'{pc(tie_rows(tt, p, n, t, "conflict").estimate)}'
      frows.append(fr_)
      srows2.append(sr_)
  write_table(out, 'TS_rules_ties', pd.DataFrame(frows),
              'Ranking rules, tie tolerance: top-1 flip of the reported prediction, %, when a tie is declared within t',
              ROLE_RULES, notes=['A unit counts as a flip when both sides report one sample and these differ. The 95% intervals '
                                 'are over targets. n = 5: per target, seeds 1 to 5.'],
              colspec='lr' + 'l' * len(ts))
  write_table(out, 'TS_rules_tieshare', pd.DataFrame(srows2),
              'Ranking rules, tie tolerance: % of units with a tie declared on either side / % of units whose reported '
              'sets are disjoint (conflict)', ROLE_RULES, colspec='lr' + 'r' * len(ts))

  # supplementary: rounded score
  rrows = []
  for p in RULE_ALL:
    for n in (1, 5):
      a = pick(rd, pair=p, n=n, ranking='ranking_score', metric='flip_top1')
      b = pick(rd, pair=p, n=n, ranking='ranking_score_summary', metric='flip_top1')
      rrows.append({'Pair': p, 'n': n, 'Full precision': rate_ci(a), 'Two decimals': rate_ci(b),
                    'Difference': f'{spc(b.difference)} [{pc(b.difference_ci_low)}, {pc(b.difference_ci_high)}]',
                    'Success flip, full': rate_ci(pick(rd, pair=p, n=n, ranking='ranking_score',
                                                       metric='flip_success')),
                    'Success flip, two decimals': rate_ci(pick(rd, pair=p, n=n, ranking='ranking_score_summary',
                                                               metric='flip_success'))})
  write_table(out, 'TS_rules_rounded', pd.DataFrame(rrows),
              'Ranking rules: top-1 flip, %, ranking by the full-precision score and by the two-decimal score (ties to '
              'the lowest seed and sample)', ROLE_RULES, colspec='lrlllll')

  # supplementary: models (exploratory)
  mrows = []
  below = above = 0
  for r in model.itertuples():
    below += r.two_sample_model <= r.observed_flip_rate
    above += r.resampling_model >= r.observed_flip_rate
    mrows.append({'Contrast': r.contrast, 'n': int(r.n), 'Units': int(r.units),
                  'Observed': f'{pc(r.observed_flip_rate)} [{pc(r.observed_ci_low)}, {pc(r.observed_ci_high)}]',
                  'Resampling model': f'{pc(r.resampling_model)} [{pc(r.resampling_ci_low)}, {pc(r.resampling_ci_high)}]',
                  'Two-sample model': f'{pc(r.two_sample_model)} [{pc(r.two_sample_ci_low)}, {pc(r.two_sample_ci_high)}]'})
  write_table(out, 'TS_rules_model', pd.DataFrame(mrows),
              'Exploratory: observed top-1 flip rate, %, against the resampling model, which adds the five score '
              'shifts of another target\'s seed together, and the two-sample model of the supplement\'s perturbation figure', ROLE_RULES,
              colspec='lrrlll')
  NOTES.append(f'Ranking rules, models: two-sample model at or below the observed rate in {below} of {len(model)} '
               f'rows; resampling model at or above it in {above} of {len(model)}.')
  return {'w': w, 'below': int(below), 'above': int(above), 'rows': len(model)}


def rerun_name(c):
  """The rerun reference of a campaign row: RC(X) prints the device X; anything else gets the general labels."""
  m = re.fullmatch(r'RC\((\w+)\)', str(c))
  return ARM_NAME[m.group(1)] if m else reader_label(c)


CAMP_PAIR = {1: 'HW(G,A)', 2: 'HW(H,A)', 3: 'HW(G,C)', 4: 'HW(V6,G)', 5: 'HW(V5,V6)'}   # the s = 1 contrast of each policy


def registered_top1(main_dir, src, contrast):
  """The registered top-1 flip rate of a contrast in an analysis folder (all targets), or None."""
  rel = f'{src}/contrasts/contrasts.csv' if src.endswith('primary_v2') else f'{src}/contrasts.csv'
  c = read(main_dir, rel)
  if 'targets' in c.columns:
    c = c[c.targets.astype(str).str.startswith('all')]
  r = c[(c.contrast == contrast) & (c.statistic == 'flip_top1')]
  return r.iloc[0] if len(r) == 1 else None


def same_or_registered(x, reg):
  """A campaign cell whose estimate equals the registered rate of the same quantity takes the registered interval, so a
  number printed in two tables has one interval (the campaign's own replicates resample assignments too)."""
  if reg is not None and abs(float(x.estimate) - float(reg.estimate)) < 1e-9:
    return reg
  return x


def table_campaign_sim(out, camp, refs, main_dir=None):
  """Supplementary tables: share of a campaign's reported samples that change, and points of success, by policy, source
  of the pairs, unit and share of jobs moved, with the rerun reference of each policy."""
  def key(pol):
    return int(pol.split(')')[0].strip('('))
  units = {'a': 'prediction', 'b': 'target'}     # the note defines a target job as the target with all its seeds
  for stat, name, fmt, title in (
      ('changed', 'TS_campaign_changed', rate_ci, 'reported samples that change, % of jobs'),
      ('success_points', 'TS_campaign_points', pts_ci, 'success rate of the mixed campaign minus that of the campaign '
                                                       'on the home device, percentage points')):
    rows = []
    for (pol, src, unit), g in camp[camp.statistic == stat].groupby(['policy', 'source', 'unit'], sort=False):
      k = key(pol)
      row = {'Policy': POLICY_SHORT[k], 'Study': SOURCE_SHORT[src], 'Job': units[unit]}
      for s_ in (0.1, 0.25, 0.5, 1.0):
        cell = pick(g[np.isclose(g.share, s_)], statistic=stat)
        if stat == 'changed' and unit == 'a' and s_ == 1.0 and main_dir is not None:
          cell = same_or_registered(cell, registered_top1(main_dir, src, CAMP_PAIR[k]))
        row[f's = {s_:g}'] = fmt(cell)
      rr = refs[(refs.policy == pol) & (refs.unit == unit) & (refs.statistic == stat)]
      if k in (4, 5):
        rr = rr[rr.source == src]
      if stat == 'changed' and unit == 'a' and main_dir is not None:
        cells = [same_or_registered(x, registered_top1(main_dir, x.source, x.contrast)) for x in rr.itertuples()]
        row['Rerun'] = '; '.join(f'{rerun_name(x.contrast)} {fmt(y)}' for x, y in zip(rr.itertuples(), cells))
      else:
        row['Rerun'] = '; '.join(f'{rerun_name(x.contrast)} {fmt(x)}' for x in rr.itertuples())
      rows.append(row)
    write_table(out, name, pd.DataFrame(rows),
                f'Campaign simulation: {title}, when a share s of jobs runs on the other device', ROLE_CAMP,
                notes=['Policy: home device → other device. s: share of jobs sent to the other device (an assumption). Job: a prediction, or a target with all its seeds. Study: the '
                       'study whose predictions are used (main study 60 targets; FU11 72). Rerun: the same campaign '
                       'rerun on the home device, named in the cell, in a second compilation (for policy 4 also on the L4). '
                       'Point estimates: '
                       'the expectation over assignments. The 95% intervals are over 10000 replicates resampling targets and '
                       'assignments. The CPU ran seeds 1 and 2 once, so policy 3 uses those seeds and has no CPU rerun.'
                       + (' For a prediction job, a cell at s = 1 or a rerun that equals a registered flip rate gives '
                          'the registered interval of that rate.' if stat == 'changed' else '')],
                colspec='llllllll')


@functools.lru_cache(maxsize=400000)
def norm(s):
  """Template of a log message, as the campaign ledger normalized it."""
  s = re.sub(r'Autotuning cache miss for (\w+)\(.*', r'Autotuning cache miss for \1(...)', s)
  s = re.sub(r'(Compiling |Finished ).*', lambda m: m.group(1) + '...', s) if s.startswith(('Compiling', 'Finished')) else s
  s = re.sub(r'\b[0-9A-Z]{4}\b', '<id>', s) if re.search(r'fold job|Processing|Predicting|Fold job|for input', s) else s
  s = re.sub(r"/[^ :']*/", '<path>/', s)
  s = re.sub(r'0x[0-9a-fA-F]+', '0xH', s)
  s = re.sub(r'\d+(\.\d+)?(e[-+]?\d+)?', 'N', s)
  return s[:160]


def scan_log(data):
  """Per-process census of one run_alphafold.log (bytes): severity flags, classes, line counts, templates."""
  res = {'W': 0, 'Wc': 0, 'E': 0, 'F': 0, 'done': False, 'classes': collections.Counter(), 'tmpl': collections.Counter()}
  for ln in data.decode('utf-8', errors='replace').splitlines():
    if not ln.strip():
      continue
    m = ABSL.match(ln)
    if m:
      sev, src, msg = m.group(1), m.group(3), m.group(4)
      key = (sev, src, norm(msg))
      if sev == 'W' and src in ('dispatch.py', 'pxla.py') and COMPILE_RE.match(msg):
        res['Wc'] += 1
      elif sev == 'W':
        res['W'] += 1
      elif sev == 'E':
        res['E'] += 1
      elif sev == 'F':
        res['F'] += 1
    else:
      sev, src, msg = 'raw', '', ln
      key = ('raw', '', norm(ln))
    res['tmpl'][key] += 1
    if ln.startswith('Done running 1 fold jobs'):
      res['done'] = True
    for name, f in CLASSES:
      if f(sev, src, msg, ln):
        res['classes'][name] += 1
  return res


def walk_session(task):
  """Every process of one frozen study session: its runs.jsonl record, session.json, and log census; and SHA-256 of
  each file read."""
  main, session, kind = task
  sd = pathlib.Path(main) / 'results' / 'af3' / session
  out = {'session': session, 'kind': kind, 'proc': [], 'tmpl': collections.Counter(), 'tmpl_dev': {}, 'sha': {}}
  for runs in sorted(sd.glob('*/runs.jsonl')):
    rd = runs.parent
    rel = f'results/af3/{session}/{rd.name}'
    raw = runs.read_bytes()
    out['sha'][f'{rel}/runs.jsonl'] = hashlib.sha256(raw).hexdigest()
    sj = rd / 'session.json'
    host = {}
    if sj.exists():
      b = sj.read_bytes()
      out['sha'][f'{rel}/session.json'] = hashlib.sha256(b).hexdigest()
      host = json.loads(b).get('host', {})
    for line in raw.decode().splitlines():
      if not line.strip():
        continue
      r = json.loads(line)
      t = r.get('pdb_id')
      logp = rd / str(t) / 'run_alphafold.log'
      data = logp.read_bytes()
      out['sha'][f'{rel}/{t}/run_alphafold.log'] = hashlib.sha256(data).hexdigest()
      c = scan_log(data)
      dev = r.get('device_kind')
      out['tmpl'].update(c['tmpl'])
      for k in c['tmpl']:
        out['tmpl_dev'].setdefault(k, collections.Counter())[dev] += 1
      out['proc'].append({
          'session': session, 'kind': kind, 'config': r.get('config'), 'target': t, 'device': dev,
          'exit_code': r.get('exit_code'),
          'failure': r.get('failure'), 'timed_out': r.get('timed_out'),
          'n_samples': sum(1 for o in (r.get('outputs') or []) if SAMPLE_RE.search(o)),
          'wall_s': float(r.get('wall_seconds') or 0), 'machine': host.get('gce_machine_type'), 'zone': host.get('gce_zone'),
          'W': c['W'], 'Wc': c['Wc'], 'E': c['E'], 'F': c['F'], 'done': c['done'], 'classes': dict(c['classes'])})
  return out


def walk_study(main, kinds):
  tasks = [(str(main), s, k) for s, k in sorted(kinds.items()) if k in STUDY_LABELS]
  with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
    res = pool.map(walk_session, tasks, chunksize=2)
  for r in res:
    SOURCES.update(r['sha'])
  check_freeze([p for r in res for p in r['sha']])
  procs = pd.DataFrame([p for r in res for p in r['proc']])
  tmpl, tmpl_dev = collections.Counter(), {}
  for r in res:
    tmpl.update(r['tmpl'])
    for k, v in r['tmpl_dev'].items():
      tmpl_dev.setdefault(k, collections.Counter()).update(v)
  return procs, tmpl, tmpl_dev


def walk_later(main, fz_all):
  """Every process of the 312 sessions of the final freeze outside the counts above (LATER_LABELS: follow-ups 13 to
  16, memlimit_ah, warm_th, follow-ups 17 to 21), read as walk_study reads the others; their files must be in the final
  freeze."""
  later = fz_all[fz_all.kind.isin(LATER_LABELS)]
  tasks = [(str(main), s, k) for s, k in zip(later.session, later.kind)]
  with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
    res = pool.map(walk_session, tasks, chunksize=2)
  for r in res:
    SOURCES.update(r['sha'])
  check_freeze([p for r in res for p in r['sha']])
  return pd.DataFrame([p for r in res for p in r['proc']])


def vm_hours(main, kinds_freeze, kinds_inv, prices):
  """VM hours and list-price estimate per session from launcher records (SC/arms_vm.py, ported)."""
  price = collections.defaultdict(dict)
  for r in prices.itertuples():
    if not (isinstance(r.usd_per_hour, float) and math.isnan(r.usd_per_hour)):
      price[(r.platform, r.machine_type, r.provisioning)][r.region] = float(r.usd_per_hour)
  af3 = main / 'results' / 'af3'
  seen = {}
  for b in (af3, af3 / '.launcher'):
    for s in sorted(os.listdir(b)):
      if s.startswith('.') or s in seen:
        continue
      if (b / s / 'launcher_events.txt').exists():
        seen[s] = b / s
  # Sessions started after freeze 68923012 and not among the 295 counted sessions (follow-ups 13 to 16, memlimit_ah,
  # warm_th and their attempts, from 2026-11-10) are outside the counts; they are left out and listed in README.md.
  # Since the release of 2026-11-17 (main ee327ab) the inventory lists 385 of them, so the inventory no longer decides.
  later = [s for s in seen if re.match(r'\d{8}T\d{6}Z', s) and s[:16] > COUNT_MADE and s not in kinds_freeze]
  for s in later:
    del seen[s]
  if later:
    NOTES.append(f'vm_hours: {len(later)} session(s) started after freeze 68923012 ({COUNT_MADE}) and not among the '
                 f'295 counted sessions are left out of the launcher-record counts: {", ".join(sorted(later))}')
  rows = []
  for s, d in sorted(seen.items()):
    rel = str((d / 'launcher_events.txt').relative_to(main))
    txt = text(main, rel)
    ev = {}
    for line in txt.splitlines():
      m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
      if m:
        try:
          ev.setdefault(f'{m.group(2)}:{m.group(3)}', dt.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ'))
        except ValueError:
          pass
    start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
    lr = {}
    if (d / 'launch_record.txt').exists():
      for line in text(main, str((d / 'launch_record.txt').relative_to(main))).splitlines():
        if '=' in line:
          k, v = line.split('=', 1)
          lr[k] = v
    end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit')
    hours = (end - start).total_seconds() / 3600 if (start and end) else None
    m = re.match(r'\d{8}T\d{6}Z_(v5e|v6e|l4|cpu|a100|h100)', s)
    pl = lr.get('PLATFORM') or (m.group(1) if m else None)
    mt, prov, zone = lr.get('MACHINE_TYPE'), lr.get('PROVISIONING') or 'standard', lr.get('ZONE') or ''
    region = zone.rsplit('-', 1)[0]
    pr = price.get((pl, mt, prov)) or {}
    usd = pr.get(region) or pr.get('europe-west4') or (list(pr.values())[0] if pr else None)
    if usd is None:
      pr = price.get((pl, mt, 'standard')) or {}
      usd = pr.get(region) or pr.get('europe-west4') or (list(pr.values())[0] if pr else None)
    k = kinds_freeze.get(s) or kinds_inv.get(s) or 'unmapped'
    k = 'pilot2_excl' if k.startswith('pilot2, excluded') else k
    rows.append({'session': s, 'kind': k, 'platform': pl, 'provisioning': prov, 'region': region, 'zone': zone,
                 'machine_type': mt, 'has_launch_record': bool(lr), 'has_vm': bool(start),
                 'vm_hours': round(hours, 3) if hours is not None else float('nan'),
                 'usd': round(hours * usd, 3) if (hours is not None and usd) else float('nan')})
  return pd.DataFrame(rows)


# Recorded spend (campaign ledger): (follow-up, item, USD as the row prints it, plan line)
RECORDED = [
    ('FU6', 'Go r1', '27.70', 251), ('FU6', 'V6o', '31.30', 252), ('FU6', 'V5o', '18.50', 253), ('FU6', 'Go r2', '24.53', 254),
    ('FU6', 'smoke tests', '1.02', 254), ('tests', 'eigen_check (outside the follow-up caps)', '0.83', 260),
    ('FU7', 'V6s', '13.28', 266), ('FU7', 'Gs', '17.77', 267), ('FU7', 'sw_smoke', '0.52', 261),
    ('FU8', 'V5x', '40.61', 263), ('FU8', 'Gx', '58.36', 264),
    ('FU9', 'H r1', '54.65', 277), ('FU9', 'H r2', '52.40', 278), ('FU9', 'smoke', '1.08', 277),
    ('FU10', 'V5l', '57.74', 297), ('FU10', 'Gl', '137.15', 298), ('FU10', 'V6l', '37.51', 299),
    ('FU11', 'Gr', '30.77', 287), ('FU11', 'V5r', '33.93', 290), ('FU11', 'V6r', '26.52', 291),
    ('FU12', 'Gh5', '9.11', 293), ('FU12', 'V6h5', '10.54', 294)]
# Sums the rows print themselves, checked against the items: (follow-up, USD, plan line)
RECORDED_SUMS = [('FU6 inference', '102.03', 254, ['Go r1', 'V6o', 'V5o', 'Go r2']),
                 ('FU7', '31.57', 267, ['V6s', 'Gs', 'sw_smoke']), ('FU9', '108.13', 278, ['H r1', 'H r2', 'smoke']),
                 ('FU10', '232.40', 299, ['V5l', 'Gl', 'V6l']), ('FU11', '91.22', 291, ['Gr', 'V5r', 'V6r']),
                 ('FU12', '19.65', 294, ['Gh5', 'V6h5'])]
# Estimated items (not recorded as spend): VM sessions selected from the launcher records
ESTIMATED = [('main', lambda v: v.kind == 'main'), ('pilot2', lambda v: v.kind == 'pilot2'),
             ('superseded, with a launcher record', lambda v: v.kind == 'superseded'),
             ('memlimit', lambda v: v.kind == 'memlimit'), ('ppi (FU1)', lambda v: v.kind == 'ppi'),
             ('embed (FU2)', lambda v: v.kind == 'embed'), ('a100 (FU3)', lambda v: v.kind == 'a100'),
             ('a100 smoke tests', lambda v: (v.kind == 'outside') & v.session.str.contains('_a100_a100_smoke')),
             ('cpu60 (FU4)', lambda v: v.kind == 'cpu60'),
             ('cpu60 preempted', lambda v: (v.kind == 'outside') & v.session.str.contains('_cpu_cpu60_')),
             ('feat (FU5)', lambda v: v.kind == 'feat'), ('failed setup with a VM', lambda v: v.kind == 'attempt')]
LEDGER_COST = {'recorded total': 685.82, 'estimated total': 653.79, 'campaign total': 1339.61}
LEDGER_DEVICE = {'TPU v5 lite': (3193, 15960, 128.4, 137.8, 198.94), 'TPU v6 lite': (3064, 15315, 108.8, 113.2, 229.32),
                 'NVIDIA L4': (5480, 27390, 532.8, 587.2, 424.30), 'NVIDIA A100-SXM4-80GB': (1030, 5150, 39.6, 44.1, 225.14),
                 'NVIDIA H100 80GB HBM3': (600, 3000, 14.3, 16.2, 106.25), 'cpu': (132, 660, 202.9, 214.8, 90.79)}


def campaign_counts(out, main, procs, vm, fz, plan_lines, later=None, fz_all=None):
  rows = []

  def add(section, scope, quantity, value, basis='', ledger=None):
    v = value
    agree = '' if ledger is None else ('yes' if (str(ledger) == str(v) or (isinstance(v, float) and
                                                                         abs(float(ledger) - v) < 0.051)) else 'NO')
    rows.append({'section': section, 'scope': scope, 'quantity': quantity, 'value': v, 'basis': basis,
                 'ledger': '' if ledger is None else ledger, 'agrees': agree})
    if agree == 'NO':
      NOTES.append(f'campaign_counts: {section} {scope} {quantity}: {v} against the ledger\'s {ledger}')
  study = procs
  totals = {'processes': len(study), 'exit 0': int((study.exit_code == 0).sum()),
            'exit 1, OOM': int(((study.exit_code == 1) & (study.failure == 'oom')).sum()),
            'samples': int(study.n_samples.sum()), 'distinct targets': study.target.nunique(),
            'frozen sessions': len(fz), 'frozen local files': int(fz.local_files.sum()),
            'frozen local bytes': int(fz.local_bytes.sum())}
  bad = {k: (v, MUST[k]) for k, v in totals.items() if v != MUST[k]}
  if bad:
    raise SystemExit(f'campaign counts do not reproduce the ledger: {bad}')
  other = len(study) - totals['exit 0'] - totals['exit 1, OOM']
  if other:
    raise SystemExit(f'{other} study processes are neither exit 0 nor exit 1 OOM')
  # per label
  kinds_fz = fz.kind.where(~fz.kind.str.startswith('pilot2, excluded'), 'pilot2, excluded')
  vmk = vm[vm.has_vm]
  for k in STUDY_LABELS + ['feat', 'pilot2, excluded']:
    p = study[study.kind == k]
    add('label', k, 'sessions in the freeze', int((kinds_fz == k).sum()), FREEZE_DIR + '/sessions.tsv')
    add('label', k, 'sessions with a process', p.session.nunique(), 'runs.jsonl')
    add('label', k, 'processes', len(p), 'runs.jsonl')
    add('label', k, 'samples', int(p.n_samples.sum()), 'runs.jsonl outputs')
    add('label', k, 'exit 0', int((p.exit_code == 0).sum()), 'runs.jsonl')
    add('label', k, 'exit 1, OOM', int(((p.exit_code == 1) & (p.failure == 'oom')).sum()), 'runs.jsonl')
    add('label', k, 'distinct targets', p.target.nunique(), 'runs.jsonl')
    add('label', k, 'process wall h', round(p.wall_s.sum() / 3600, 2), 'runs.jsonl wall_seconds')
    vv = vmk[vmk.kind == ('pilot2_excl' if k == 'pilot2, excluded' else k)]
    add('label', k, 'VM sessions with a launcher record', len(vv), 'launcher_events.txt')
    add('label', k, 'VM h', round(vv.vm_hours.sum(), 2), 'launcher_events.txt, VM running to deleted')
    f = fz[kinds_fz == k]
    add('label', k, 'frozen local files', int(f.local_files.sum()), FREEZE_DIR + '/sessions.tsv')
    add('label', k, 'frozen local bytes', int(f.local_bytes.sum()), FREEZE_DIR + '/sessions.tsv')
  for k, v in totals.items():
    add('total', 'all', k, v, 'frozen study sessions', MUST[k])
  add('total', 'all', 'sessions with a process', study.session.nunique(), 'runs.jsonl', 284)
  add('total', 'all', 'study sessions (with feat)', int(kinds_fz.isin(STUDY_LABELS + ['feat']).sum()), 'sessions.tsv', 288)
  add('total', 'all', 'exit 0 with 5 samples', int(((study.exit_code == 0) & (study.n_samples == 5)).sum()),
      'runs.jsonl outputs', 13495)
  add('total', 'all', 'timed out', int((study.timed_out == True).sum()), 'runs.jsonl', 0)  # noqa: E712
  add('total', 'all', 'process wall h', round(study.wall_s.sum() / 3600, 1), 'runs.jsonl', 1026.7)
  add('total', 'all', 'frozen bucket objects', int(fz.bucket_objects.sum()), 'sessions.tsv', 348065)
  add('total', 'all', 'frozen bucket bytes', int(fz.bucket_bytes.sum()), 'sessions.tsv', 340549936510)
  oom = study[study.exit_code == 1]
  for r in oom.itertuples():
    add('OOM process', r.session, 'device; failure', f'{r.device}; {r.failure}', 'runs.jsonl')
  # per device
  incl = vmk[vmk.kind.isin(STUDY_LABELS + ['feat'])]
  tot_vm = tot_usd = 0.0
  for dvc in DEVICES:
    p = study[study.device == dvc]
    pl = [k for k, v in PLATFORM_DEVICE.items() if v == dvc][0]
    vv = incl[incl.platform == pl]
    led = LEDGER_DEVICE[dvc]
    add('device', dvc, 'processes', len(p), 'runs.jsonl', led[0])
    add('device', dvc, 'samples', int(p.n_samples.sum()), 'runs.jsonl', led[1])
    add('device', dvc, 'process wall h', round(p.wall_s.sum() / 3600, 1), 'runs.jsonl', led[2])
    add('device', dvc, 'VM h, included sessions', round(vv.vm_hours.sum(), 1), 'launcher_events.txt (with feat on cpu)',
        led[3])
    add('device', dvc, 'list USD, estimate', round(vv.usd.sum(), 2), 'VM h x cloud/prices.csv', led[4])
    tot_vm += vv.vm_hours.sum()
    tot_usd += vv.usd.sum()
    add('device', dvc, 'machine types', ', '.join(sorted(set(p.machine.dropna()))), 'session.json host')
    zones = sorted(set(p.zone.dropna()))
    add('device', dvc, 'zones', len(zones), 'session.json host')
    add('device', dvc, 'regions', ', '.join(sorted({z.rsplit('-', 1)[0] for z in zones})), 'session.json host')
  add('device', 'all', 'VM h, included sessions', round(tot_vm, 1), 'launcher_events.txt', 1113.3)
  add('device', 'all', 'list USD, estimate', round(tot_usd, 2), 'VM h x cloud/prices.csv', 1274.73)
  zones = sorted(set(study.zone.dropna()))
  add('total', 'all', 'zones', len(zones), 'session.json host', 20)
  add('total', 'all', 'regions', len({z.rsplit('-', 1)[0] for z in zones}), 'session.json host', 9)
  add('total', 'all', 'device kinds', study.device.nunique(), 'runs.jsonl', 6)
  # cost: recorded items, each on its plan line
  rec_usd = {}
  for fu, item, usd, line in RECORDED:
    if f'{usd}' not in plan_lines[line - 1]:
      raise SystemExit(f'recorded spend {fu} {item} {usd} not on line {line} of {PLAN}')
    rec_usd[item] = float(usd)
    add('cost, recorded', fu, item, float(usd), f'{PLAN} line {line} (recorded spend, list prices, VM running to deleted)')
  for name, usd, line, items in RECORDED_SUMS:
    if f'{usd} USD' not in plan_lines[line - 1]:
      raise SystemExit(f'recorded sum {name} {usd} not on line {line} of {PLAN}')
    if abs(sum(rec_usd[i] for i in items) - float(usd)) > 0.005:
      raise SystemExit(f'recorded sum {name}: items do not add to {usd}')
    add('cost, recorded', name, 'sum as the row prints it', float(usd), f'{PLAN} line {line}')
  if 'about 99 USD' not in plan_lines[264 - 1]:
    raise SystemExit('FU8 "about 99 USD" not on line 264')
  rt = round(sum(rec_usd.values()), 2)
  add('cost, recorded', 'all', 'recorded total (sum of the items above)', rt, 'sum', LEDGER_COST['recorded total'])
  et = 0.0
  for name, sel in ESTIMATED:
    v = vmk[sel(vmk)]
    add('cost, estimated', name, 'VM sessions', len(v), 'launcher_events.txt')
    add('cost, estimated', name, 'VM h', round(v.vm_hours.sum(), 2), 'launcher_events.txt')
    add('cost, estimated', name, 'list USD, estimate', round(v.usd.sum(), 2),
        'VM h x cloud/prices.csv (session platform, machine, provisioning, region; europe-west4 stand-in)')
    et += round(v.usd.sum(), 2)
  et = round(et, 2)
  add('cost, estimated', 'all', 'estimated total (not recorded as spend)', et, 'sum', LEDGER_COST['estimated total'])
  add('cost', 'all', 'campaign total, recorded plus estimated', round(rt + et, 2), 'sum', LEDGER_COST['campaign total'])
  add('cost', 'all', 'every session with a launcher record, priced by the rule (check)', round(vmk.usd.sum(), 2),
      'launcher_events.txt', 1336.96)
  # The whole final freeze (decision of 18 November 2026, before arXiv: the Introduction and Section III-H count every
  # run of the paper): the 295 sessions above plus the 312 sessions of LATER_LABELS, read by walk_later
  if later is not None:
    allp = pd.concat([study, later], ignore_index=True)
    odd = allp[~((allp.exit_code == 0) | ((allp.exit_code == 1) & (allp.failure == 'oom')))]
    if len(odd):
      raise SystemExit(f'final freeze: {len(odd)} processes are neither exit 0 nor exit 1 OOM')
    if set(later.session) - set(fz_all.session) or set(later.kind) - set(LATER_LABELS):
      raise SystemExit('final freeze: a later process outside the freeze or its labels')
    basis = f'{FREEZE_DIR}/sessions.tsv ({len(fz_all)} sessions), runs.jsonl'
    add('final freeze', 'all', 'sessions', len(fz_all), f'{FREEZE_DIR}/sessions.tsv')
    add('final freeze', 'all', 'sessions with a process', allp.session.nunique(), basis)
    add('final freeze', 'all', 'processes', len(allp), basis)
    add('final freeze', 'all', 'samples', int(allp.n_samples.sum()), basis + ' outputs')
    add('final freeze', 'all', 'exit 0', int((allp.exit_code == 0).sum()), basis)
    add('final freeze', 'all', 'exit 1, OOM', int(((allp.exit_code == 1) & (allp.failure == 'oom')).sum()), basis)
    add('final freeze', 'all', 'distinct targets', allp.target.nunique(), basis)
    add('final freeze', 'later labels', 'processes', len(later), basis)
    add('final freeze', 'later labels', 'samples', int(later.n_samples.sum()), basis + ' outputs')
    add('final freeze', 'later labels', 'exit 0', int((later.exit_code == 0).sum()), basis)
    for k in LATER_LABELS:
      q = later[later.kind == k]
      add('final freeze', k, 'processes', len(q), basis)
      add('final freeze', k, 'samples', int(q.n_samples.sum()), basis + ' outputs')
      add('final freeze', k, 'exit 0', int((q.exit_code == 0).sum()), basis)
  df = pd.DataFrame(rows)
  write_csv(out, 'campaign_counts', df, ROLE_CAMPAIGN,
            'Campaign counts: per label and in total (sessions, processes, samples, exit codes), per device (processes, '
            'samples, process wall h, VM h, list-price estimate, machine types, regions), frozen files and bytes, and '
            'cost (recorded items on their plan lines; estimated items from VM hours and cloud/prices.csv). Columns '
            'ledger and agrees compare with paper/research/ledger/campaign_scale.md')
  return df


# Supplement tables TS_campaign and TS_provisioning (from the same records as campaign_counts)
CAMPAIGN_ORDER = ['pilot2', 'main', 'memlimit', 'ppi', 'embed', 'a100', 'cpu60', 'feat', 'msa', 'sw', 'seeds', 'h100',
                  'large', 'repl', 'prec5', 'pilot2, excluded']
LABEL_STUDY = {'pilot2': 'repeated pilot (pilot decision only)', 'main': 'main study (pre-registered)',
               'memlimit': 'memory limit (M7)', 'ppi': 'FU1', 'embed': 'FU2', 'a100': 'FU3', 'cpu60': 'FU4',
               'feat': 'FU5 (featurization only)', 'msa': 'FU6', 'sw': 'FU7', 'seeds': 'FU8', 'h100': 'FU9',
               'large': 'FU10', 'repl': 'FU11', 'prec5': 'FU12'}
PROV_NAME = {'standard': 'on demand', 'spot': 'Spot', 'flex_start': 'Flex-start'}
# Arm of a process: (device kind, harness config) gives the base arm; follow-ups 6 to 12 add their suffix (Go, V6s, V5x,
# V5l, V5r, V6h5, ...; supplement section on the platforms).
ARM_BASE = {('TPU v5 lite', 'tpu_xla'): 'V5', ('TPU v6 lite', 'tpu_xla'): 'V6', ('TPU v6 lite', 'tpu_xla_highest'): 'V6h',
            ('NVIDIA L4', 'l4_xla'): 'G', ('NVIDIA L4', 'l4_triton'): 'GT', ('NVIDIA L4', 'l4_xla_det'): 'Gd',
            ('NVIDIA L4', 'l4_xla_highest'): 'Gh', ('NVIDIA A100-SXM4-80GB', 'a100_xla'): 'A',
            ('NVIDIA A100-SXM4-80GB', 'a100_triton'): 'AT', ('NVIDIA A100-SXM4-80GB', 'a100_xla_highest'): 'Ah',
            ('NVIDIA H100 80GB HBM3', 'h100_xla'): 'H', ('cpu', 'cpu_xla'): 'C'}
ARM_ORDER = ['V5', 'V6', 'V6h', 'G', 'GT', 'Gd', 'Gh', 'A', 'AT', 'Ah', 'H', 'C']
ARM_SUFFIX = {'msa': 'o', 'sw': 's', 'seeds': 'x', 'large': 'l', 'repl': 'r', 'prec5': '5'}
# Campaign ledger: (label, arm) -> (sessions, processes, provisioning sessions or None where the ledger has none)
S, F, X = 'standard', 'spot', 'flex_start'
LEDGER_ARMS = {
    ('main', 'C'): (5, 12, {S: 5}), ('main', 'G'): (14, 610, {S: 14}), ('main', 'GT'): (7, 300, {S: 7}),
    ('main', 'Gd'): (16, 610, {S: 16}), ('main', 'Gh'): (3, 120, {S: 3}), ('main', 'V5'): (4, 610, {S: 4}),
    ('main', 'V6'): (3, 610, {S: 3}), ('main', 'V6h'): (1, 120, {S: 1}),
    ('pilot2', 'C'): (4, 12, None), ('pilot2', 'G'): (2, 50, None), ('pilot2', 'Gd'): (2, 50, None),
    ('pilot2', 'Gh'): (2, 20, None), ('pilot2', 'V5'): (1, 50, None), ('pilot2', 'V6'): (1, 50, None),
    ('pilot2', 'V6h'): (1, 20, None),
    ('memlimit', 'G'): (1, 5, {S: 1}), ('memlimit', 'GT'): (1, 5, {S: 1}), ('memlimit', 'V5'): (1, 3, {S: 1}),
    ('memlimit', 'V6'): (1, 4, {S: 1}),
    ('ppi', 'G'): (6, 300, {S: 6}), ('ppi', 'V5'): (1, 150, {S: 1}), ('ppi', 'V6'): (2, 300, {S: 2}),
    ('embed', 'G'): (1, 20, {S: 1}), ('embed', 'Gd'): (1, 10, {S: 1}), ('embed', 'V5'): (1, 10, {S: 1}),
    ('embed', 'V6'): (1, 10, {S: 1}),
    ('cpu60', 'C'): (54, 108, None),
    ('a100', 'A'): (7, 610, {S: 7}), ('a100', 'AT'): (3, 300, {S: 3}), ('a100', 'Ah'): (2, 120, {S: 2}),
    ('msa', 'Go'): (14, 600, {S: 14}), ('msa', 'V5o'): (3, 310, {S: 3}), ('msa', 'V6o'): (3, 310, {S: 3}),
    ('sw', 'Gs'): (7, 300, {F: 7}), ('sw', 'V6s'): (2, 300, {X: 2}),
    ('seeds', 'Gx'): (22, 900, {F: 22}), ('seeds', 'V5x'): (5, 900, {F: 2, S: 3}),
    ('h100', 'H'): (6, 600, {F: 6}),
    ('repl', 'Gr'): (12, 720, {F: 12}), ('repl', 'V5r'): (3, 720, {S: 3}), ('repl', 'V6r'): (3, 720, {X: 3}),
    ('prec5', 'Gh5'): (5, 180, {F: 5}), ('prec5', 'V6h5'): (1, 180, {X: 1}),
    ('large', 'Gl'): (39, 680, {F: 22, S: 17}), ('large', 'V5l'): (6, 440, {S: 6}), ('large', 'V6l'): (4, 440, {X: 4})}
LEDGER_PROVISIONING = {S: 143, F: 134, X: 11}     # campaign ledger, the 288 study sessions


def h1(x):
  """Hours, one decimal, half up from the decimal value (44.05 prints 44.1, as round() gives in campaign_counts)."""
  import decimal
  return str(decimal.Decimal(str(float(x))).quantize(decimal.Decimal('0.1'), rounding=decimal.ROUND_HALF_UP))


def plan_date(plan_lines, line):
  m = re.match(r'\| (\d{4}-\d\d-\d\d) \|', plan_lines[line - 1])
  if not m:
    raise SystemExit(f'{PLAN} line {line} is not a dated row')
  return m.group(1)


def table_campaign(out, cc, procs, fz, plan_lines):
  """TS_campaign: campaign_counts.csv by label, by device, in total, and its cost rows."""
  v = cc.set_index(['section', 'scope', 'quantity']).value
  rows = []

  def row(group, scope, study, sessions='', processes='', samples='', e0='', oom='', wall='', vmh='', usd=''):
    rows.append({'group': group, 'scope': scope, 'description': study, 'sessions': sessions, 'processes': processes,
                 'samples': samples, 'exit 0': e0, 'OOM stops': oom, 'process wall h': wall, 'VM h': vmh,
                 'list USD': usd})
  excl = fz[fz.kind.str.startswith('pilot2, excluded')].kind
  n_int = int(excl.str.contains('interrupted').sum())
  study = dict(LABEL_STUDY, **{'pilot2, excluded': f'pilot2 launches not used ({len(excl) - n_int} ran nothing, '
                                                   f'{n_int} interrupted)'})
  vm_sum = 0.0
  for k in CAMPAIGN_ORDER:
    q = lambda name: v['label', k, name]   # noqa: E731
    row('by label', k, study[k], int(q('sessions in the freeze')), int(q('processes')), int(q('samples')),
        int(q('exit 0')), int(q('exit 1, OOM')), h1(q('process wall h')), h1(q('VM h')))
    vm_sum += float(q('VM h'))
  for dvc in DEVICES:
    p = procs[procs.device == dvc]
    q = lambda name: v['device', dvc, name]   # noqa: E731
    row('by device', DEVICE_LABEL[dvc], q('machine types'), p.session.nunique(), int(q('processes')), int(q('samples')),
        int((p.exit_code == 0).sum()), int(((p.exit_code == 1) & (p.failure == 'oom')).sum()), h1(q('process wall h')),
        h1(q('VM h, included sessions')))
  tv = lambda name: v['total', 'all', name]   # noqa: E731
  row('total', 'all', 'sessions through Follow-up 12', int(tv('frozen sessions')),
      int(tv('processes')), int(tv('samples')), int(tv('exit 0')), int(tv('exit 1, OOM')), h1(tv('process wall h')),
      h1(vm_sum))
  dev_vm = float(v['device', 'all', 'VM h, included sessions'])
  if abs(vm_sum - dev_vm - float(v['label', 'pilot2, excluded', 'VM h'])) > 0.06:
    raise SystemExit(f'TS_campaign: VM h by label {vm_sum} is not the device total {dev_vm} plus pilot2, excluded')
  # cost: recorded items grouped as the plan rows sum them, each item with the date of its row
  by_fu = collections.OrderedDict()
  for fu, item, usd, line in RECORDED:
    smoke = fu == 'FU6' and item == 'smoke tests'
    key = ('FU6 smoke tests' if smoke else 'FU6 inference' if fu == 'FU6' else fu)
    by_fu.setdefault(key, []).append((item, usd, plan_date(plan_lines, line)))
  sums = {n: u for n, u, _, _ in RECORDED_SUMS}
  for key, items in by_fu.items():
    tot = round(sum(float(u) for _, u, _ in items), 2)
    if key in sums and abs(tot - float(sums[key])) > 0.005:
      raise SystemExit(f'TS_campaign: {key} items add to {tot}, the row prints {sums[key]}')
    fu = key.split()[0]
    what = '; '.join(f'{recorded_name(i, fu)} {u} (row of {d})' for i, u, d in items)
    row('cost', 'recorded', (f'{fu}: ' if fu != 'tests' else 'engineering check: ') + what, usd=f'{tot:.2f}')
  row('cost', 'recorded', 'total of the recorded items', usd=f'{float(v["cost, recorded", "all", "recorded total (sum of the items above)"]):.2f}')
  for name, _ in ESTIMATED:
    q = lambda n: v['cost, estimated', name, n]   # noqa: E731
    row('cost', 'estimated', name, int(q('VM sessions')), vmh=h1(q('VM h')), usd=f'{float(q("list USD, estimate")):.2f}')
  row('cost', 'estimated', 'total of the estimated items',
      usd=f'{float(v["cost, estimated", "all", "estimated total (not recorded as spend)"]):.2f}')
  row('cost', 'total', 'campaign total, recorded plus estimated',
      usd=f'{float(v["cost", "all", "campaign total, recorded plus estimated"]):.2f}')
  df = pd.DataFrame(rows)
  heads = {'by label': 'By label', 'by device': 'By device (study processes)', 'total': 'Total',
           'cost': 'Cost: list prices, not a bill; disks, storage and network excluded'}
  trows = []
  for g, d in df.groupby('group', sort=False):
    trows.append({'scope': '\\GROUP ' + heads[g]})
    trows += d.drop(columns=['group']).to_dict('records')
  tdf = pd.DataFrame(trows).fillna('')
  write_table(out, 'TS_campaign', df, 'The campaign by label and by device: sessions, processes, samples, exit codes, '
              'process and VM hours; and its cost at list prices (recorded items with the dates of their plan rows, '
              'estimated items, total)', ROLE_CAMPAIGN, colspec='lp{5.0cm}rrrrrrrr', tex_df=tdf,
              notes=['Sessions by label: sessions of the final freeze, for the main study and FU1 to FU12 (the processes of FU13 to '
                     'FU21 and the 58 of the memory-limit and warm-cost measurements on the A100, the H100, and the L4 '
                     'Triton are given in the text); by device: sessions with a process on that '
                     'device. Process wall h: wall_seconds of runs.jsonl, including featurization and compilation. VM h: '
                     'launcher records, VM running to deleted; device rows include FU5 on the CPU row. OOM stops: the '
                     'planned stops of the memory-limit measurement. Recorded: spend written in the plan rows from the '
                     'session records (VM running to deleted, list prices, interrupted VMs included). Estimated, not '
                     'recorded as spend: VM h times cloud/prices.csv for the session\'s platform, machine type, '
                     'provisioning and region (europe-west4 where a region has no price). List prices, not a bill; '
                     'disks, storage and network excluded, as are sessions without a launcher record (pilot 1, '
                     'engineering), the release VM and the on-premise hosts.'])
  return df


def table_provisioning(out, procs, vm, fz):
  """TS_provisioning: per label and arm, sessions, processes, machine type, zones or regions, provisioning model."""
  vs = vm.set_index('session')
  study_fz = fz[fz.kind.isin(STUDY_LABELS + ['feat'])]
  missing = [s for s in study_fz.session if s not in vs.index or not vs.loc[s, 'has_launch_record']]
  if missing:
    raise SystemExit(f'TS_provisioning: {len(missing)} study sessions without a launch record: {missing[:3]}')
  p = procs.copy()
  key = list(zip(p.device, p.config))
  bad = sorted({k for k in key if k not in ARM_BASE})
  if bad:
    raise SystemExit(f'TS_provisioning: no arm for (device, config) {bad}')
  p['arm'] = [ARM_BASE[k] + ARM_SUFFIX.get(lab, '') for k, lab in zip(key, p.kind)]
  per = p.groupby('session').arm.nunique()
  if (per != 1).any():
    raise SystemExit(f'TS_provisioning: sessions with more than one arm: {list(per[per != 1].index)[:3]}')
  p['provisioning'] = p.session.map(vs.provisioning)

  def prov_cell(sessions):
    c = collections.Counter()
    for s in sorted(sessions):          # session names start with their launch time: first model first
      c[vs.loc[s, 'provisioning']] += 1
    if len(c) == 1:
      return PROV_NAME[next(iter(c))], c
    return ', '.join(f'{PROV_NAME[k]} {n}' for k, n in c.items()), c

  def where(zones):
    zones = sorted(set(zones))
    regions = sorted({z.rsplit('-', 1)[0] for z in zones})
    if len(regions) == 1:
      return ', '.join(zones)
    return ', '.join(regions) + f' ({len(zones)} zones)'
  rows = []
  for lab in CAMPAIGN_ORDER[:-1]:
    if lab == 'feat':
      ss = sorted(study_fz[study_fz.kind == 'feat'].session)
      cell, _ = prov_cell(ss)
      rows.append({'label': lab, 'study': LABEL_STUDY[lab], 'arm': 'hosts', 'sessions': len(ss), 'processes': 0,
                   'machine type': ', '.join(sorted(set(vs.loc[ss, 'machine_type']))),
                   'zones or regions': where(vs.loc[ss, 'zone']), 'provisioning': cell})
      continue
    d = p[p.kind == lab]
    arms = sorted(set(d.arm), key=lambda a: (ARM_ORDER.index(a[:-len(ARM_SUFFIX[lab])] if lab in ARM_SUFFIX else a)))
    for arm in arms:
      a = d[d.arm == arm]
      ss = set(a.session)
      cell, c = prov_cell(ss)
      led = LEDGER_ARMS.get((lab, arm))
      if led is None or led[:2] != (len(ss), len(a)) or (led[2] is not None and led[2] != dict(c)):
        raise SystemExit(f'TS_provisioning {lab} {arm}: {len(ss)} sessions, {len(a)} processes, {dict(c)} against the '
                         f'ledger\'s {led}')
      rows.append({'label': lab, 'study': LABEL_STUDY[lab], 'arm': arm, 'sessions': len(ss), 'processes': len(a),
                   'machine type': ', '.join(sorted(set(a.machine.dropna()))), 'zones or regions': where(a.zone.dropna()),
                   'provisioning': cell})
  if {(r['label'], r['arm']) for r in rows if r['arm'] != 'hosts'} != set(LEDGER_ARMS):
    raise SystemExit('TS_provisioning: arms differ from campaign_scale.md 4.1')
  tot = collections.Counter(vs.loc[s, 'provisioning'] for s in study_fz.session)
  if dict(tot) != LEDGER_PROVISIONING:
    raise SystemExit(f'TS_provisioning: provisioning of the study sessions {dict(tot)}')
  rows.append({'label': 'all', 'study': 'study sessions that ran something', 'arm': '', 'sessions': len(study_fz),
               'processes': len(p), 'machine type': '',
               'zones or regions': f'{p.zone.dropna().str.rsplit("-", n=1).str[0].nunique()} regions, '
                                   f'{p.zone.dropna().nunique()} zones',
               'provisioning': ', '.join(f'{PROV_NAME[k]} {tot[k]}' for k in (S, F, X))})
  df = pd.DataFrame(rows)
  trows = []
  for lab, d in df.groupby('label', sort=False):
    if lab != 'all':
      trows.append({'arm': '\\GROUP ' + f'{lab}: {LABEL_STUDY[lab]}'})
    trows += d.drop(columns=['label', 'study']).assign(
        arm=[('all' if lab == 'all' else a if a in ('hosts', '') else prov_arm_name(a, lab)) for a in d.arm]
    ).to_dict('records')
  tdf = pd.DataFrame(trows).fillna('')
  write_table(out, 'TS_provisioning', df, 'Provisioning by arm and follow-up: sessions, processes, machine type, zones '
              'or regions, and provisioning model', ROLE_CAMPAIGN,
              colspec='lrrp{3.0cm}p{5.5cm}l', tex_df=tdf,
              notes=['Session records of the final freeze: machine type and zone per process (session.json), '
                     'provisioning model per session (launch_record.txt). Regions with the zone count where an arm used '
                     'more than one region; several models: sessions per model in launch order.'])
  return df


def log_census(out, procs, tmpl, tmpl_dev, has_clash):
  rows = []

  def add(section, item, scope, value, ledger=None):
    agree = '' if ledger is None else ('yes' if ledger == value else 'NO')
    rows.append({'section': section, 'item': item, 'device': scope, 'value': value,
                 'ledger': '' if ledger is None else ledger, 'agrees': agree})
    if agree == 'NO':
      NOTES.append(f'log_census: {section} {item} {scope}: {value} against the ledger\'s {ledger}')
  for i, dvc in enumerate(DEVICES + ['all']):
    p = procs if dvc == 'all' else procs[procs.device == dvc]
    lv = (lambda k: LEDGER_LEVELS[k][i]) if dvc != 'all' else (lambda k: None)
    add('level', 'processes', dvc, len(p))
    add('level', 'warning-level lines (absl W), compile logging excluded', dvc, int(p.W.sum()))
    add('level', 'compile-logging lines (absl W from dispatch.py, pxla.py)', dvc, int(p.Wc.sum()))
    add('level', 'error-level lines (absl E)', dvc, int(p.E.sum()))
    add('level', 'fatal lines (absl F)', dvc, int(p.F.sum()), 0)
    add('level', 'processes with any warning-level line, including compile logging', dvc, int(((p.W + p.Wc) > 0).sum()),
        lv('any warning-level line, including compile logging'))
    add('level', 'processes with any warning-level line, without compile logging', dvc, int((p.W > 0).sum()),
        lv('any warning-level line, without compile logging'))
    add('level', 'processes with any error-level line', dvc, int((p.E > 0).sum()), lv('any error-level line'))
  e0 = procs[procs.exit_code == 0]
  add('exit 0', 'processes', 'all', len(e0), 13495)
  add('exit 0', 'with a warning-level line', 'all', int(((e0.W + e0.Wc) > 0).sum()), 13495)
  add('exit 0', 'with an error-level line', 'all', int((e0.E > 0).sum()), 9323)
  add('exit 0', 'with no error-level line', 'all', int((e0.E == 0).sum()), 4172)
  add('exit 0', 'log ends with "Done running 1 fold jobs."', 'all', int(e0.done.sum()), 13495)
  add('exit 1', 'log has "Done running 1 fold jobs."', 'all', int(procs[procs.exit_code != 0].done.sum()), 0)
  for name, _ in CLASSES:
    for i, dvc in enumerate(DEVICES):
      p = procs[procs.device == dvc]
      n = int(p.classes.map(lambda c: name in c).sum())
      add('message class, processes', name, dvc, n, LEDGER_CLASSES.get(name, (None,) * 6)[i])
    add('message class, lines', name, 'all', int(procs.classes.map(lambda c: c.get(name, 0)).sum()),
        LEDGER_CLASS_LINES[name] if name != 'NotImplementedError: Not supported on <device>' else None)
  add('templates', 'distinct (severity, source, normalized message) templates', 'all', len(tmpl), 590)
  for w in WORDS + WORDS_EXPLAINED:
    hit = [k for k in tmpl if w in k[2].lower() or w in k[1].lower()]
    pr = collections.Counter()
    for k in hit:
      pr.update(tmpl_dev[k])
    add('word search', w, 'all', len(hit), 0 if w in WORDS else None)
    if hit:
      add('word search, lines', w, 'all', int(sum(tmpl[k] for k in hit)))
      add('word search, example template', w, 'all', ' | '.join(hit[0])[:150])
  for name, s in has_clash.items():
    add('has_clash', f'{name}: sample rows', 'all', s[0])
    add('has_clash', f'{name}: rows with has_clash not 0', 'all', s[1], 0)
  add('has_clash', 'all eleven analyses: sample rows', 'all', sum(s[0] for s in has_clash.values()), 65900)
  df = pd.DataFrame(rows)
  write_csv(out, 'log_census', df, ROLE_CAMPAIGN,
            'Log census of the run_alphafold.log of every frozen study process: lines and processes by severity and '
            'device, the message classes of campaign_scale.md 5.5, the word search over normalized templates, and '
            'has_clash over the sample rows of the eleven analyses with samples.csv. Absl severity is the leading letter '
            'of each line; compile logging is the JAX_LOG_COMPILES output the harness sets, logged at warning level')
  return df


# ---------------------------------------------------------------------------------------------------------------------
# C, D, E: exploratory mechanism counts, CPU flips, follow-up 12 post-cutoff targets

MECH = ['HW(V5,V6)', 'HW(V6,G)', 'HW(V6,GT)', 'RC(G)', 'DU', 'DET', 'PR(V6)', 'PR(G)', 'HWh(V6,G)']
LEDGER_MECH = {'HW(V5,V6)': (54, 14, 11, 2), 'HW(V6,G)': (62, 23, 10, 2), 'HW(V6,GT)': (67, 23, 12, 5),
               'RC(G)': (22, 4, 5, 2), 'DU': (41, 11, 8, 3), 'DET': (34, 5, 7, 2), 'PR(V6)': (19, 4, 3, 3),
               'PR(G)': (10, 3, 1, 1), 'HWh(V6,G)': (22, 6, 3, 1), 'nine contrasts': (331, 93, 60, 21)}


def mechanism_extra(out, ps):
  rows = []
  for c in MECH + ['nine contrasts']:
    g = ps[ps.contrast.isin(MECH)] if c == 'nine contrasts' else ps[ps.contrast == c]
    if c == 'nine contrasts':
      flips = big = 0
      for cc in MECH:
        h = ps[ps.contrast == cc]
        flips += int(h.flip_top1.sum())
        big += int((h.flip_top1 & (h.margin_a >= h.margin_a.median())).sum())
      med = float('nan')
    else:
      med = float(g.margin_a.median())
      flips, big = int(g.flip_top1.sum()), int((g.flip_top1 & (g.margin_a >= med)).sum())
    sf = int(g.flip_success.sum())
    snt = int((g.flip_success & ~g.flip_top1).sum())
    ftm = int(g.flip_matters.sum())
    led = LEDGER_MECH[c]
    got = (flips, big, sf, snt)
    rows.append({'contrast': c, 'seed pairs': len(g), 'top-1 flips': flips, 'median margin': med,
                 'flips at or above the median margin': big,
                 'share of flips at or above the median margin (%)': pc(big / flips) if flips else '',
                 'success flips': sf, 'flips that matter': ftm, 'success flips with no top-1 flip': snt,
                 'ledger (flips; at or above median; success flips; with no top-1 flip)': '; '.join(map(str, led)),
                 'agrees': 'yes' if got == led else 'NO'})
    if got != led:
      NOTES.append(f'mechanism_extra {c}: {got} against the ledger {led}')
    assert ftm == sf - snt
  df = pd.DataFrame(rows)
  write_csv(out, 'mechanism_extra', df, ROLE_EXPL + '; ledger-only numbers given a generator here',
            f'Exploratory, post hoc: from {MAINV2}/contrasts/pairs_seed.csv, per contrast: top-1 flips; flips at or above '
            'the contrast\'s median margin of the first condition (margin_a); success flips; flips that matter (success '
            'flips with a top-1 flip); success flips with no top-1 flip. The total over nine contrasts counts pairs that '
            'overlap across contrasts; the at-or-above count uses each contrast\'s own median')
  return df


OVERALL_CONTRASTS = [('HW(V5,V6)', ('V5', 'r1'), ('V6', 'r1')), ('HW(V6,G)', ('V6', 'r1'), ('G', 'r1')),
                     ('RC(G)', ('G', 'r1'), ('G', 'r2')), ('RC(V5)', ('V5', 'r1'), ('V5', 'r2')),
                     ('RC(V6)', ('V6', 'r1'), ('V6', 'r2'))]
LEDGER_OVERALL = {'HW(V5,V6)': 16}     # main_q1.md section 0: "V5 against V6 overall top-1 different on 16 of 60 targets"


def overall_top(s):
  """Per target: the overall top-ranked (seed, sample) over seeds 1 to 5 and whether it is a success, by the rule of
  analysis/contrasts.py (highest full-precision ranking_score over all seeds and samples, stable sort; a missing M1
  counts as not a success)."""
  out = {}
  for t, g in s.groupby('target'):
    if sorted(g.seed.unique()) != [1, 2, 3, 4, 5] or len(g) != 25:
      raise SystemExit(f'overall_top1: {t} has seeds {sorted(g.seed.unique())} and {len(g)} samples')
    if (g.ranking_score == g.ranking_score.max()).sum() != 1:
      raise SystemExit(f'overall_top1: tie at the top on {t}')
    b = g.sort_values(['ranking_score'], ascending=False, kind='mergesort').iloc[0]
    out[t] = ((int(b.seed), int(b['sample'])), bool(b.m1_success == True))  # noqa: E712
  return out


def overall_top1(out, main_samples):
  """Post hoc: targets whose overall top-1 over seeds 1 to 5 differs between the two sides, and whose success does."""
  rows = []
  for c, a, b in OVERALL_CONTRASTS:
    ta, tb = (overall_top(main_samples[(main_samples.platform == p) & (main_samples.repetition == r)]) for p, r in (a, b))
    if set(ta) != set(tb) or len(ta) != 60:
      raise SystemExit(f'overall_top1 {c}: target sets differ or are not 60')
    dt = sum(ta[t][0] != tb[t][0] for t in ta)
    ds = sum(ta[t][1] != tb[t][1] for t in ta)
    led = LEDGER_OVERALL.get(c)
    rows.append({'contrast': c, 'side a': f'{a[0]} {a[1]}', 'side b': f'{b[0]} {b[1]}', 'targets': len(ta),
                 'overall top-1 differs (targets)': dt, 'overall top-1 success differs (targets)': ds,
                 'success a (targets)': sum(v[1] for v in ta.values()), 'success b (targets)': sum(v[1] for v in tb.values()),
                 'ledger': '' if led is None else led, 'agrees': '' if led is None else ('yes' if led == dt else 'NO')})
    if led is not None and led != dt:
      NOTES.append(f'overall_top1 {c}: {dt} targets against the ledger\'s {led}')
  df = pd.DataFrame(rows)
  write_csv(out, 'overall_top1', df, ROLE_EXPL,
            f'Post hoc, main study: from {MAINV2}/samples.csv, per contrast, the targets (of 60) whose overall top-ranked '
            '(seed, sample) over seeds 1 to 5 differs between the two sides, and those whose overall top-1 success '
            'differs; ranking rule of analysis/contrasts.py (highest full-precision ranking_score over all seeds and '
            'samples; a missing M1 is not a success). RC uses r2 as side b. Ledger: main_q1.md section 0')
  return df


def embed_summary(out, emb):
  """Follow-up 2: relative L2 distance of the trunk embeddings per pair, over the rows that are not bit-identical."""
  rows = []
  for pair, g in emb.groupby('pair', sort=False):
    for rep in ('single', 'pair'):
      ident = g[f'{rep}_identical'].astype(str) == 'True'
      x = g.loc[~ident, f'{rep}_rel_l2'].astype(float)
      rows.append({'pair': pair, 'embedding': rep, 'targets': len(g), 'bit-identical': int(ident.sum()),
                   'not bit-identical': len(x), 'rel_l2 min': float(x.min()) if len(x) else float('nan'),
                   'rel_l2 median': float(x.median()) if len(x) else float('nan'),
                   'rel_l2 max': float(x.max()) if len(x) else float('nan')})
  df = pd.DataFrame(rows)
  write_csv(out, 'embed_summary', df, 'follow-up 2 (outside the pre-registered study); post hoc summary',
            f'Post hoc: from {AN}/embed_20261015/embeddings.csv, per pair and embedding (single, pair), the targets, '
            'how many are bit-identical, and min, median and max of the relative L2 distance over the rows that are not')
  return df


def cpu_flips(out, cpu_ps, main_samples):
  six = sorted(main_samples[main_samples.platform == 'C'].target.unique())
  rows = []
  for c in ('HW(V5,C)', 'HW(V6,C)'):
    g = cpu_ps[cpu_ps.contrast == c]
    on6 = g.target.isin(six)
    assert g.target.nunique() == 60 and g[on6].target.nunique() == 6
    for st in ('flip_top1', 'flip_success', 'flip_matters'):
      rows.append({'contrast': c, 'statistic': st, 'pairs': len(g), 'all 60 targets': int(g[st].sum()),
                   'on the 6 targets C ran in both sessions': int(g[on6][st].sum()),
                   'on the other 54 targets': int(g[~on6][st].sum())})
  df = pd.DataFrame(rows)
  t1 = df[df.statistic == 'flip_top1']
  got = list(zip(t1['all 60 targets'], t1['on the other 54 targets']))
  if got != [(21, 21), (19, 19)]:
    NOTES.append(f'cpu_flips: top-1 flips (all, on 54) {got} against the ledger (21, 21), (19, 19)')
  write_csv(out, 'cpu_flips', df, 'follow-up 4 (outside the pre-registered study); split post hoc',
            f'Follow-up 4: flips of HW(V5,C) and HW(V6,C) per (target, seed) pair ({AN}/cpu60_20261018/pairs_seed.csv), '
            f'in total, on the 6 targets C ran in both sessions ({", ".join(six)}; main-study C runs), and on the other 54')
  return df, six


def fu12_cutoff_targets(out, prec_ps, dates):
  post = set(dates[dates.post_cutoff.astype(str) == 'True'].pdb_id)
  assert len(post) == 39
  g = prec_ps[(prec_ps.contrast == 'HWh(V6,G)') & prec_ps.target.isin(post)]
  assert len(g) == 39 * 5
  rows = []
  for st, lab in (('flip_success', 'success flip'), ('flip_matters', 'flip that matters')):
    for r in g[g[st]].sort_values(['target', 'seed_a']).itertuples():
      rows.append({'contrast': 'HWh(V6,G)', 'targets': 'post-cutoff 39', 'seeds': '1 to 5', 'event': lab,
                   'target': r.target, 'seed': int(r.seed_a), 'top-1 flip': bool(r.flip_top1)})
  df = pd.DataFrame(rows)
  ev = {lab: list(zip(df[df.event == lab].target, df[df.event == lab].seed)) for lab in ('success flip', 'flip that matters')}
  want = {'success flip': [('7MGT', 1), ('7MGT', 4), ('7Q5I', 3), ('7U3J', 1)],
          'flip that matters': [('7MGT', 4), ('7Q5I', 3), ('7U3J', 1)]}
  if ev != want:
    NOTES.append(f'fu12_cutoff_targets: {ev} against consistency.md S9 {want}')
  write_csv(out, 'fu12_cutoff_targets', df, 'follow-up 12 (outside the pre-registered study); listing post hoc',
            f'Follow-up 12: HWh(V6,G) success flips and flips that matter on the 39 post-cutoff targets, seeds 1 to 5, with '
            f'target and seed ({AN}/prec5_20261107/pairs_seed.csv; post-cutoff from {MAINV2}/contrasts/release_dates.csv)')
  return df


# ---------------------------------------------------------------------------------------------------------------------
# Captions: sentence text is constant, every number comes from the drawn values

def captions(out, d):
  fr, q, quint, logit, pert, sc, con = d['forest'], d['ecdf'], d['quint'], d['logit'], d['pert'], d['seed'], d['main_con']
  main_c = d['main_samples'][d['main_samples'].platform == 'C']
  cap = collections.OrderedDict()
  hw = fr[fr.kind == 'HW']
  t1, sf = hw[hw.statistic == 'flip_top1'], hw[hw.statistic == 'flip_success']
  dd = fr[fr.kind == 'HW minus RC']
  dt1, dsf = dd[dd.statistic == 'flip_top1'], dd[dd.statistic == 'flip_success']
  exc = dt1[dt1.ci_low <= 0]
  if len(exc) != 1:
    raise SystemExit(f'Fig. 2 caption: {len(exc)} top-1 differences with an interval reaching 0')
  e = exc.iloc[0]
  n0 = int(((dsf.ci_low <= 0) & (dsf.ci_high >= 0)).sum())
  # a row without a band: its reference is 0.0 [0.0, 0.0] (not drawn), or the pair has no recompilation row
  # Style round of 2026-11-19: the caption describes what is drawn; the ranges of the rates (Results, the main study's
  # decision) and the exception L4 vs A100 (Results, beyond the main study) are in the text
  # Options round of 2026-11-19 (Fig. 1 B): the legend and the axes give the markers, the band and the unit; the
  # token ranges of FU10 are in Supplementary Table S8 (the follow-ups), and a row without a band is a rate of 0
  # (Table II) or an arm that ran once (the dagger of the legend)
  cap['capFigForest'] = (
      'Top-1 flips (a) and success flips (b) between pairs of platforms, with 95\\% bootstrap intervals over '
      'targets. Bands give recompilation alone on the platform with the larger rate. FU is the follow-up. In FU1, '
      'success is a DockQ of at least 0.23.')
  FOREST_RANGES[:] = [pc(t1.estimate.min()), pc(t1.estimate.max()), pc(sf.estimate.min()), pc(sf.estimate.max())]
  if e.pair != 'HW(G,A)':
    raise SystemExit(f'Fig. 2 caption: the supplementary table cited is follow-up 3\'s, but the exception is {e.pair}')
  qq = q.set_index('contrast')
  med = {c: pick(con, contrast=c, statistic='m4_ligand_rmsd_median') for c in ('HW(V5,V6)', 'HW(V6,G)', 'SD(V6)')}
  r1, r2 = (float(med['SD(V6)'].estimate) / float(med[c].estimate) for c in ('HW(V5,V6)', 'HW(V6,G)'))
  bits = [pick(con, contrast=c, statistic='bit_identical_fraction') for c in ('HW(V5,V6)', 'HW(V6,G)')]
  assert all(float(b.estimate) == 0 for b in bits) and len({int(b.n) for b in bits}) == 1
  tail = max(qq.loc['HW(V5,V6)', 'max'], qq.loc['HW(V6,G)', 'max'])
  rc5 = [pick(con, contrast=c, statistic='bit_identical_fraction').estimate for c in ('RC(V5)', 'RC(V6)')]
  assert rc5 == [1.0, 1.0]
  ntg = d['ligand_targets']
  cap['capFigEcdf'] = (
      'Cumulative distribution of the ligand RMSD between paired samples, for two pairs of platforms, recompilation '
      'on the L4, and a change of seed on the v6e. Recompilation on the v5e and on the v6e is bit-identical and not '
      'drawn. Values of 0 are drawn at $10^{-4}$~\\AA{}. The curves use '
      f'{fint(qq.loc["HW(V5,V6)", "n"], True)} sample pairs on {ntg} targets, and '
      f'{fint(qq.loc["SD(V6)", "n"], True)} for the change of seed. The three glycan targets have no ligand RMSD.')
  lo = {c: pick(logit, contrast=c) for c in ('HW(V5,V6)', 'HW(V6,G)')}
  pe = {c: pick(pert, contrast=c) for c in ('HW(V5,V6)', 'HW(V6,G)')}
  top = {c: quint[(quint.contrast == c) & (quint.quintile == quint.quintile.max())].iloc[0] for c in ('HW(V5,V6)', 'HW(V6,G)')}
  nq = sorted(set(quint[quint.contrast.isin([c for c, *_ in MARGIN_SERIES])].n))
  assert len(nq) == 1, nq

  def orr(r):
    return f'{r.odds_ratio_per_tenfold_margin:.2f} [{r.odds_ratio_ci_low:.2f}, {r.odds_ratio_ci_high:.2f}]'

  def dm(r):
    return f'{r.delta_over_mu:.2f} [{r.ratio_ci_low:.2f}, {r.ratio_ci_high:.2f}]'
  cap['capFigMargin'] = (
      'Exploratory, specified after the main results were read. Per-seed top-1 flip rate by quintile of the first '
      f'condition\'s ranking margin, with 95\\% bootstrap intervals over targets. Each quintile holds {nq[0]} pairs of '
      'predictions.')
  pp = pert.set_index('contrast')

  def po(c):
    return f'{pc(pp.loc[c, "predicted_flip_rate"])}\\% against {pc(pp.loc[c, "observed_flip_rate"])}\\%'
  under = [c for c in pp.index if pp.loc[c, 'observed_ci_low'] > pp.loc[c, 'predicted_flip_rate']]
  if sorted(under) != sorted(['HW(V5,V6)', 'RC(G)', 'DET', 'DU']):
    raise SystemExit(f'Fig. 4 caption: observed interval above the model for {under}')
  npair = sorted(set(pp.n_seed_pairs.astype(int)))
  if npair != [300]:
    raise SystemExit(f'Fig. 4 caption: seed pairs {npair}')
  key = pd.read_csv(out / 'tables' / 'fig4_perturbation.csv').sort_values('number')
  # Style round of 2026-11-19: what is drawn only; the reading (the rate grows with the shift, platform pairs and
  # arithmetic changes on one curve, descriptive and not causal) is in the text of the main paper and the supplement
  # Options round of 2026-11-19 (Fig. 3 A): the key of the numbered points is drawn in the figure (fig_perturbation),
  # the markers are in its legend, the Follow-up 12 points and the model in the supplement's Analysis (c); the
  # appendix figure (Fig. 3) and the supplement's copy (Fig. S4) have the same caption
  if len(key) != 9 or len(PERT_HW + PERT_ONE) != 9:
    raise SystemExit(f'Fig. 3 caption: {len(key)} numbered points, the caption says nine')
  cap['capFigPerturbation'] = (
      'Exploratory. Top-1 flip rate of nine contrasts against the median shift in ranking score, divided by the median '
      'top-two margin. The dashed line is a two-sample model.')
  cap['capFigPerturbationApp'] = cap['capFigPerturbation']
  inside = sc[(sc.ci_low >= -0.02) & (sc.ci_high <= 0.02)]
  if list(inside.n_seeds) != [2]:
    raise SystemExit(f'Fig. 5: criterion met at n = {list(inside.n_seeds)}')
  r2 = sc[sc.n_seeds == 2].iloc[0]
  later = sc[sc.n_seeds >= 3]
  assert ((later.ci_low < -0.02) | (later.ci_high > 0.02)).all()
  one = 100 / int(sc.n_targets.iloc[0])
  # Style round of 2026-11-19: what is drawn only; the range of the estimate, the answer 2 and its instability, and
  # the resolution of one target (1.67 points) are in the text
  # Options round of 2026-11-19 (Fig. 4 B): the diamond at n = 2 and its interval are written in the figure
  cap['capFigSeeds'] = (
      'Follow-up 8, outside the pre-registered study. Difference in the success rate of the overall top-ranked sample '
      'over the first n seeds, v5e minus L4, in points, with 95\\% bootstrap intervals over '
      f'{int(sc.n_targets.iloc[0])} targets. The band is the $\\pm$2 point criterion of Q3.')
  SEEDS_FACTS[:] = [spc(sc.estimate.min()), spc(sc.estimate.max()), f'{one:.2f}']
  if d.get('reserve'):
    c = d['cost']
    c = c[(c.cache == 'fresh') & (c.bucket == 1024)].set_index('platform')

    def u(p, col):
      return usd3(c.loc[p, f'median_usd_per_structure_{col}'])
    cap['capFigCost'] = (
        'Per structure, one target, one seed, five samples, fresh cache. At bucket 1024 the cost was '
        f'{u("V5", "standard")} USD on demand, {u("V5", "spot")} Spot and {u("V5", "flex_start")} Flex-start on the '
        f'v5e; {u("V6", "standard")}, {u("V6", "spot")} and {u("V6", "flex_start")} on the v6e; {u("G", "standard")} '
        f'and {u("G", "spot")} on the L4; {u("GT", "standard")} and {u("GT", "spot")} on the L4 Triton arm. All runs were '
        'on demand: Spot and '
        'Flex-start figures are list prices read on 8 and 9 October 2026 applied to these times, excluding VM setup, '
        'idle time and boot disk. Google publishes no Flex-start price for g2-standard-8.')
    rr = d['large_rule'].set_index('pair')
    cap['capFigSize'] = (
        'Follow-up 10, outside the pre-registered study: for both primary pairs, top-1 flip minus recompilation showed no '
        'size difference detected at this sample size: large minus control, '
        f'{tex(diff_ci(rr.loc["HW(V5,V6)"]))} points for the v5e-v6e difference beyond recompilation on the v5e and '
        f'{tex(diff_ci(rr.loc["HW(V6,G)"]))} points for the v6e-L4 difference beyond recompilation on the v6e '
        f'(interval widths {pc(rr.loc["HW(V5,V6)"].interval_width)} and {pc(rr.loc["HW(V6,G)"].interval_width)}).')
  # numbers that left the captions on 2026-11-19 (style round) and that the text now prints; the text's % src cites
  # this file
  write_csv(out, 'caption_moved', pd.DataFrame(
      [{'figure': 'Fig. 1 (forest)', 'quantity': q, 'value': v} for q, v in
       zip(('top-1 flip, lowest pair (%)', 'top-1 flip, highest pair (%)', 'success flip, lowest pair (%)',
            'success flip, highest pair (%)'), FOREST_RANGES)] +
      [{'figure': 'Fig. 4 (seeds)', 'quantity': q, 'value': v} for q, v in
       zip(('difference, lowest estimate (points)', 'difference, highest estimate (points)', 'one target (points)'),
           SEEDS_FACTS)]), 'captions', 'numbers that left the figure captions on 2026-11-19 and that the text prints')
  lines = ['% Figure and table captions for the ISC paper. Generated by analysis/paper_assets_isc.py; do not edit.',
           '% Each macro holds the caption text after "Fig. N." (LaTeX adds the label).']
  for k, v in list(cap.items()) + list(CAPTIONS.items()):
    lines.append(f'\\newcommand{{\\{k}}}{{{tie_pairs(v)}}}')     # R93: pairs never break
  body = '\n'.join(lines) + '\n'
  no_em_dash(body, 'captions.tex')
  for w in ('chip effect', 'caused by near-ties', 'two seeds suffice', 'no size dependence', 'unchanged', 'no effect'):
    if w in body:
      raise SystemExit(f'captions.tex: forbidden phrase "{w}"')
  (out / 'captions.tex').write_text(body)
  ASSETS['captions.tex'] = ('all', 'one \\newcommand per figure and table caption')
  return cap


# ---------------------------------------------------------------------------------------------------------------------

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default=str(base.main_checkout()))
  ap.add_argument('--out')
  ap.add_argument('--reserve', action='store_true', help='also write reserve figures R1 (cost) and R2 (size)')
  args = ap.parse_args()
  main_dir = pathlib.Path(args.main).expanduser().resolve()
  out = pathlib.Path(args.out).expanduser() if args.out else main_dir / 'results' / 'paper_assets_isc'
  out.mkdir(parents=True, exist_ok=True)
  load_freeze(main_dir)
  # analysis outputs
  con = {k: read(main_dir, f'{v}/contrasts.csv') for k, v in FOLDERS.items()}
  hwrc = {k: read(main_dir, f'{v}/hw_minus_rc.csv') for k, v in FOLDERS.items()}
  assert 'primary_v2' in FOLDERS['main']
  sfold = {'main': MAINV2, 'ppi': f'{AN}/ppi_20261015', 'a100': f'{AN}/a100_20261018', 'cpu60': f'{AN}/cpu60_20261018',
           'msa': f'{AN}/msa_20261102', 'sw': f'{AN}/sw_20261102', 'seeds': f'{AN}/seeds_20261102',
           'h100': f'{AN}/h100_20261104', 'repl': f'{AN}/repl_20261106', 'prec5': f'{AN}/prec5_20261107',
           'large': f'{AN}/large_20261108'}
  samples = {k: read(main_dir, f'{v}/samples.csv') for k, v in sfold.items()}
  has_r2 = {(k, p): bool((s[s.platform == p].repetition == 'r2').any()) for k, s in samples.items()
            for p in s.platform.unique()}
  other = {'memlimit': read(main_dir, f'{AN}/memlimit_20261014/processes.csv'),
           'embed': read(main_dir, f'{AN}/embed_20261015/versus_main.csv'),
           'feat_decision': json.loads(text(main_dir, f'{AN}/feat_20261018/decision.json')),
           'feat_hosts': read(main_dir, f'{AN}/feat_20261018/hosts.csv')}
  runs_tab = fig1_runs(samples, other)
  plt, save = figure_saver(out)
  a100s, h100s = samples['a100'], samples['h100']
  runs = {'table': runs_tab, 'A': (a100s[a100s.platform == 'A'].device_kind.iloc[0],
                                   a100s[a100s.platform == 'A'].machine_type.iloc[0]),
          'H': (h100s.device_kind.iloc[0], h100s.machine_type.iloc[0])}
  table_runs(out, runs, samples['main'], con['main'], samples['large'])
  for stale in ('figures/fig1_design.pdf', 'figures/fig1_design.png', 'tables/TI_arms_primary_note.tex'):
    (out / stale).unlink(missing_ok=True)     # Fig. 1 became Table I(b) on 2026-11-11
  fr = forest_rows(con, hwrc, has_r2)
  fig_forest(out, save, plt, fr)
  pairs = read(main_dir, f'{MAINV2}/contrasts/pairs_sample.csv', usecols=['contrast', 'target', 'm4_ligand_rmsd'])
  q = fig_rmsd_ecdf(out, save, plt, pairs)
  quint = read(main_dir, f'{EXPL}/a_margin_quintiles.csv')
  fig_margin(out, save, plt, quint)
  pert, pert_five = pert_table(main_dir)
  fig_perturbation(out, save, plt, pert)
  table_pert_five(out, pert_five)
  q3 = read(main_dir, f'{AN}/seeds_20261102/q3.csv')
  sc = fig_seeds(out, save, plt, q3)
  cost = read(main_dir, f'{MAINV2}/cost/cost_summary.csv')
  large_rule = read(main_dir, f'{AN}/large_20261108/reading_rule.csv')
  if args.reserve:
    fig_cost(out, save, plt, cost)
    fig_size(out, save, plt, hwrc['large'], large_rule)
  # tables
  prices = read(main_dir, 'cloud/prices.csv')
  table_arms(out, samples, prices)
  table_headline(out, con['main'], hwrc['main'], main_dir)
  gpu_cost, gpu_mem = gpu_followup_cost(main_dir, prices)
  used = read(main_dir, f'{MAINV2}/cost/prices_used.csv')
  main_dates = sorted(set(used.read_on.astype(str)))
  fu_dates = {k: sorted(set(prices[prices.platform == k].read_on.astype(str))) for k in ('a100', 'h100')}
  price_dates = {'main': main_dates, 'a100': fu_dates['a100'], 'h100': fu_dates['h100']}
  sys.path.insert(0, str(REPO / 'analysis'))
  import cost as costmod  # noqa: E402
  table_q2(out, cost, read(main_dir, f'{MAINV2}/cost/memory.csv'), read(main_dir, f'{AN}/memlimit_20261014/summary.csv'),
           gpu_cost, gpu_mem, price_dates=price_dates, batch1024=costmod.batch_sizes('subset')(1024),
           memlimit_ah=memlimit_ah_summary(main_dir), warm_th=warm_th_summary(main_dir))
  dates = read(main_dir, f'{MAINV2}/contrasts/release_dates.csv')
  prec_ps = read(main_dir, f'{AN}/prec5_20261107/pairs_seed.csv')
  fu12 = fu12_cutoff_targets(out, prec_ps, dates)
  cross = text(main_dir, CROSS_SESSION).rstrip('\n').splitlines()
  if cross[-1] != CROSS_SESSION_LINE:
    raise SystemExit(f'{CROSS_SESSION}: last line is "{cross[-1]}"')
  seeds_answer = json.loads(text(main_dir, f'{AN}/seeds_20261102/answer.json'))
  if seeds_answer['all 60']['answer'] != 2:
    raise SystemExit('seeds answer.json changed')
  t4 = {'ppi_con': con['ppi'], 'ppi_hw': hwrc['ppi'], 'embed': read(main_dir, f'{AN}/embed_20261015/embeddings.csv'),
        'a100_hw': hwrc['a100'], 'cpu_con': con['cpu'], 'main_con': con['main'], 'cross_line': cross[-1],
        'feat_decision': other['feat_decision'], 'msa_hw': hwrc['msa'],
        'msa_rule': read(main_dir, f'{AN}/msa_20261102/reading_rule.csv'),
        'sw_con': read(main_dir, f'{AN}/sw_20261102/contrasts.csv'),
        'sw_cmp': read(main_dir, f'{AN}/sw_20261102/comparisons.csv'),
        'sw_rule': read(main_dir, f'{AN}/sw_20261102/reading_rule.csv'), 'seeds_answer': seeds_answer, 'q3': q3,
        'h100_hw': hwrc['h100'], 'large_rule': large_rule, 'repl_rule': read(main_dir, f'{AN}/repl_20261106/reading_rule.csv'),
        'repl_hw': hwrc['repl'], 'prec_hw': hwrc['prec'], 'prec_rule': read(main_dir, f'{AN}/prec5_20261107/reading_rule.csv')}
  fu1314_rows(main_dir)
  fu15_tables(out, main_dir)
  fu16_tables(out, main_dir)
  fu17_tables(out, main_dir)
  fu18_table(out, main_dir)
  fu19_table(out, main_dir)
  fu20_table(out, main_dir)
  fu21_table(out, main_dir)
  table_followups(out, t4, fu12)
  table_fu03_q3_memory(out, read(main_dir, f'{AN}/a100_20261018/q3.csv'),
                      read(main_dir, f'{AN}/a100_20261018/memory.csv'))
  rules = table_rules(out, read(main_dir, f'{RANKING}/seed_consensus.csv'), read(main_dir, f'{RANKING}/tie_tolerance.csv'),
                      read(main_dir, f'{RANKING}/rounded.csv'), read(main_dir, f'{RANKING}/model.csv'),
                      read(main_dir, f'{RANKING}/cost.csv'))
  table_campaign_sim(out, read(main_dir, f'{CAMPAIGN}/campaign.csv'), read(main_dir, f'{CAMPAIGN}/references.csv'),
                     main_dir)
  fu13 = fu13_assets(out, save, plt, main_dir)
  fu14_tables(out, main_dir)
  table_beyond(out, main_dir)
  table_repro(out, main_dir)
  table_e_summary(out, main_dir)
  table_returned(out, main_dir)
  table_rare(out, main_dir)
  table_winners(out, main_dir)
  table_winners_ext(out, main_dir)
  table_pb_flips(out, main_dir)
  table_best25(out, main_dir)
  table_gtat(out, main_dir)
  if fu13:
    print(f'follow-up 13 assets from {fu13["dir"]}', flush=True)
  # C, D
  ps = read(main_dir, f'{MAINV2}/contrasts/pairs_seed.csv')
  mechanism_extra(out, ps)
  cpu_flips(out, read(main_dir, f'{AN}/cpu60_20261018/pairs_seed.csv'), samples['main'])
  overall_top1(out, samples['main'])
  embed_summary(out, t4['embed'])
  # A, B
  fz_all = read(main_dir, f'{FREEZE_DIR}/sessions.tsv', sep='\t')
  fz = fz_all[~fz_all.kind.isin(LATER_LABELS)].reset_index(drop=True)     # the 295 sessions the counts cover
  fz_old = read(main_dir, f'{COUNT_FREEZE_DIR}/sessions.tsv', sep='\t')
  if list(fz.session) != list(fz_old.session) or list(fz.kind) != list(fz_old.kind):
    raise SystemExit(f'{FREEZE_DIR}: its sessions without {LATER_LABELS} are not those of freeze 68923012')
  if set(fz_all.kind) - set(fz.kind) != set(LATER_LABELS):
    raise SystemExit(f'{FREEZE_DIR}: labels {sorted(set(fz_all.kind) - set(fz.kind))}, expected {LATER_LABELS}')
  inv = read(main_dir, 'results/release/inventory/sessions.csv')
  kinds_fz = dict(zip(fz.session, fz.kind))
  print('walking the frozen study sessions ...', flush=True)
  procs, tmpl, tmpl_dev = walk_study(main_dir, kinds_fz)
  vm = vm_hours(main_dir, kinds_fz, dict(zip(inv.session, inv.kind)), prices)
  plan_lines = text(main_dir, PLAN).splitlines()
  print('walking the later sessions of the final freeze ...', flush=True)
  later = walk_later(main_dir, fz_all)
  cc = campaign_counts(out, main_dir, procs, vm, fz, plan_lines, later=later, fz_all=fz_all)
  table_campaign(out, cc, procs, fz, plan_lines)
  table_provisioning(out, procs, vm, fz)
  has_clash = {k: (len(s), int((s.has_clash.fillna(0) != 0).sum())) for k, s in samples.items()}
  log_census(out, procs, tmpl, tmpl_dev, has_clash)
  # captions
  lg = read(main_dir, f'{AN}/large_20261108/samples.csv', usecols=['target', 'num_tokens'])
  lc = con['large']
  ntok = lg.groupby('target').num_tokens.first()
  large_tokens = (1025, 1536, 1024)     # the group definitions of follow-up 10 (row of 2026-11-04), checked here
  lt = lc[(lc.contrast == 'HW(V5,V6)') & (lc.targets == 'all') & (lc.statistic == 'flip_top1')]
  big = ntok[ntok > 1024]
  assert len(big) == int(lt[lt.group == 'large'].n_targets.iloc[0]) and big.max() <= 1536
  assert int((ntok <= 1024).sum()) == int(lt[lt.group == 'control'].n_targets.iloc[0])
  NOTES.append(f'FU10 token counts: large targets {int(big.min())} to {int(big.max())} tokens, controls up to '
               f'{int(ntok[ntok <= 1024].max())}; the text (IV-D) and Supplementary Table S8 print the group definitions 1025--1536 and at most 1024.')
  ls = samples['large']
  cap_data = {'forest': fr, 'ecdf': q, 'quint': quint, 'logit': read(main_dir, f'{EXPL}/a_logistic.csv'),
              'pert': pert, 'seed': sc, 'main_con': con['main'],
              'main_samples': samples['main'], 'seeds_answer': seeds_answer, 'large_tokens': large_tokens,
              'large_r2_targets': ls[(ls.platform == 'V5') & (ls.repetition == 'r2')].target.nunique(),
              'large_targets': ls.target.nunique(), 'reserve': args.reserve, 'cost': cost, 'large_rule': large_rule,
              'ligand_targets': pairs[(pairs.contrast == 'HW(V5,V6)') & pairs.m4_ligand_rmsd.notna()].target.nunique()}
  captions(out, cap_data)
  if CLIPPED:
    raise SystemExit(f'text cut at the figure edges: {CLIPPED}')
  # sources and README
  lines = ['path\tsha256\tin_freeze']
  for p, h in sorted(SOURCES.items()):
    fr_ = f'yes (freeze {FREEZE_ID[:8]})' if FROZEN.get(p) == h else (f'manifest of freeze {FREEZE_ID[:8]}' if p == (
        f'{FREEZE_DIR}/manifest_sha256.tsv') else 'no')
    lines.append(f'{p}\t{h}\t{fr_}')
  (out / 'sources.tsv').write_text('\n'.join(lines) + '\n')
  readme = ['# Paper assets, ISC 2027 paper', '',
            'Generated by analysis/paper_assets_isc.py (specification: paper/research/figures_guide.md sections 2, 5, 6, 8; '
            'amendments: paper/research/decisions.md). sources.tsv lists every input with its SHA-256, including the Arial '
            'font file the PDFs embed and every run record and log read for the campaign counts and the log census; files '
            f'under results/af3/ of frozen sessions are checked against freeze {FREEZE_ID[:8]} before any output is written.', '',
            'Figures: vector PDF for the paper, PNG for review only. Widths 3.5 or 7.16 in, Arial 9 pt, TrueType embedded. '
            'Each figure has a CSV twin under tables/. Captions are in captions.tex as \\newcommand macros; numbers are '
            'formatted from the drawn values.', '',
            'Derived numbers computed here: Fig. 2 caption ranges and counts (over the rows drawn); Fig. 3 ratios (SD(V6) '
            'median over each HW median, rounded), zero share, tail maximum; Fig. 5 the n at which the interval lies '
            'within plus or minus 2 points; Table IV counts (FU1 pairs, FU2 targets, FU11 reading cells, FU12 targets); '
            'all of campaign_counts.csv, log_census.csv, mechanism_extra.csv, cpu_flips.csv, fu12_cutoff_targets.csv, '
            'overall_top1.csv, embed_summary.csv; Fig. 3 at_least_1A and at_least_2A; TS_campaign and TS_provisioning '
            '(from the campaign records).', '',
            'Each line: asset, [role], what it shows.', '']
  readme += [f'- `{k}` [{role}]: {what}' for k, (role, what) in ASSETS.items()]
  readme += ['', '## Checks and disagreements with the ledgers', '']
  readme += [f'- {n}' for n in NOTES] or ['- none']
  body = '\n'.join(readme) + '\n'
  no_em_dash(body, 'README.md')
  (out / 'README.md').write_text(body)
  print(f'{len(ASSETS)} assets, {len(SOURCES)} inputs -> {out}')
  for n in NOTES:
    print('NOTE', n)


if __name__ == '__main__':
  main()
