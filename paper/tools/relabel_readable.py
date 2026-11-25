"""Save-time steps of relabel_figures.py that make the supplement figures F1 to F15 readable at their printed width
(style round of 2026-11-20), and the dump of the drawn data that shows the steps change no drawn value.

Imported by the boot script that relabel_figures.py runs around each generator; never run alone. Layout only: no step
here changes a drawn value (line data, error bars, bars, points, limits, the strings of texts).
"""
import json
import math

TEXTWIDTH_PT = 516.0     # \textwidth of the supplement (onecolumn IEEEtran)
MIN_PT = 8.0             # smallest printed text
# Markers in order of first appearance of a series label ('x' is kept free: F8 marks a failed size with it), dashes for
# lines without markers, hatches for bars
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '<', '>', 'p', 'h']
DASHES = ['-', (0, (5, 2)), (0, (1, 1.5)), (0, (6, 1.5, 1, 1.5)), (0, (10, 2)), (0, (3, 1.5, 1, 1.5, 1, 1.5))]
HATCHES = ['', '///', '...', 'xxx']
LEGEND_KEYS = ('loc', 'ncols', 'ncol', 'frameon', 'bbox_to_anchor', 'handlelength', 'columnspacing', 'handletextpad',
               'borderaxespad', 'title', 'labelspacing')


def record_legends():
  """Keep the arguments of every legend call, so that a legend can be drawn again with larger text and the current
  styles of its handles."""
  import matplotlib.axes
  import matplotlib.figure
  for cls in (matplotlib.axes.Axes, matplotlib.figure.Figure):
    orig = cls.legend
    if getattr(orig, '_relabel_recorded', False):
      continue

    def legend(self, *a, _orig=orig, **k):
      leg = _orig(self, *a, **k)
      leg._relabel_call = (_orig, self, a, dict(k))
      return leg
    legend._relabel_recorded = True
    cls.legend = legend


def glyph_ratio(t):
  """Smallest glyph of a text over its font size: 1 for plain text, 0.7 for a mathtext superscript such as the 10^-4
  of a log axis."""
  s = t.get_text()
  if not s.strip():
    return 1.0
  s2, ismath = t._preprocess_math(s)
  if ismath is not True:
    return 1.0
  from matplotlib.mathtext import MathTextParser
  fp = t.get_fontproperties().copy()
  fp.set_size(10.0)
  g = MathTextParser('path').parse(s2, 72, fp).glyphs
  return min(x[1] for x in g) / 10.0 if g else 1.0


def shown_ticks(axis, which):
  """(location, label) of the tick labels drawn: visible, with text, at a tick inside the view limits."""
  lo, hi = sorted(axis.get_view_interval())
  eps = 1e-9 * abs(hi - lo)
  ticks = axis.get_minor_ticks() if which == 'minor' else axis.get_major_ticks()
  locs = axis.get_minorticklocs() if which == 'minor' else axis.get_majorticklocs()
  out = []
  for k, tk in enumerate(ticks):
    if k >= len(locs) or not (lo - eps <= locs[k] <= hi + eps):
      continue
    for lab in (tk.label1, tk.label2):
      if lab.get_visible() and lab.get_text().strip():
        out.append((float(locs[k]), lab))
  return out


def shown_ticklabels(ax, axis, which):
  return [lab for _, lab in shown_ticks(axis, which)]


def drawn_texts(fig):
  """Every text drawn: (text, kind) with kind 'tick', 'legend' or 'text'."""
  from matplotlib.text import Text
  fig.canvas.draw()
  ticks, hidden = [], set()
  for ax in fig.axes:
    for axis in (ax.xaxis, ax.yaxis):
      for which in ('major', 'minor'):
        shown = shown_ticklabels(ax, axis, which)
        ticks += shown
        allt = axis.get_minor_ticks() if which == 'minor' else axis.get_major_ticks()
        hidden |= {id(x) for tk in allt for x in (tk.label1, tk.label2)} - {id(x) for x in shown}
  legs = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
  inleg = {id(t) for g in legs for t in g.findobj(Text)}
  tick_ids = {id(t) for t in ticks}
  out = [(t, 'tick') for t in ticks]
  for t in fig.findobj(Text):
    if id(t) in tick_ids or id(t) in hidden or not t.get_visible() or not t.get_text().strip():
      continue
    out.append((t, 'legend' if id(t) in inleg else 'text'))
  return out


def up(x):
  return math.ceil(x * 10 - 1e-6) / 10


def series(fig, kind):
  """(label, artist) of every series in drawing order: labeled lines, error bars and bar groups."""
  from matplotlib.container import BarContainer, ErrorbarContainer
  out = []
  for ax in fig.axes:
    if kind == 'hatch':
      out += [(c.get_label(), c) for c in ax.containers if isinstance(c, BarContainer)]
      continue
    inbar = set()
    for c in ax.containers:
      if isinstance(c, ErrorbarContainer) and not c.get_label().startswith('_'):
        out.append((c.get_label(), c.lines[0]))
        inbar.add(id(c.lines[0]))
    out += [(ln.get_label(), ln) for ln in ax.get_lines()
            if id(ln) not in inbar and not ln.get_label().startswith('_')]
  return out


def gray(fig, kind, log, name):
  """One marker, dash or hatch per series label, in order of first appearance, the same in every panel."""
  order = {}
  for label, a in series(fig, kind):
    i = order.setdefault(label, len(order))
    if kind == 'marker':
      a.set_marker(MARKERS[i])
    elif kind == 'dash':
      a.set_linestyle(DASHES[i])
    else:
      for p in a.patches:
        p.set_hatch(HATCHES[i])
        p.set_hatchcolor('black')
        p.set_hatch_linewidth(0.6)
  if len(order) > {'marker': len(MARKERS), 'dash': len(DASHES), 'hatch': len(HATCHES)}[kind]:
    raise SystemExit(f'{name}: more series than {kind} styles')
  log.append((f'series of {name}', f'one {kind} each ({len(order)} labels)'))


def redraw_legends(fig, size, below, log, name):
  """Draw every legend again, with the current styles of its handles and text of at least `size`. below maps the
  index of an axes to the number of columns of its legend drawn under the axes, as a figure legend."""
  legs = [(g, None) for g in list(fig.legends)] + [(ax.get_legend(), ax) for ax in fig.axes
                                                    if ax.get_legend() is not None]
  for leg, ax in legs:
    call = getattr(leg, '_relabel_call', None)
    if call is None:
      raise SystemExit(f'{name}: a legend was made without the recorded call')
    orig, owner, a, k = call
    k = dict(k)
    fs = leg.get_texts()[0].get_fontsize() if leg.get_texts() else size
    k['fontsize'] = max(fs, size)
    if leg.get_title().get_text().strip():
      k['title_fontsize'] = max(leg.get_title().get_fontsize(), size)
    leg.remove()
    if below and ax is not None and fig.axes.index(ax) in below:
      # the axes legend becomes a figure legend under the axes; the figure grows by its height, so the axes keep
      # their height, and relayout leaves it its room
      h, lab = ax.get_legend_handles_labels()
      kk = {x: k[x] for x in ('frameon', 'fontsize', 'handlelength', 'columnspacing') if x in k}
      g = fig.legend(h, lab, loc='lower center', borderaxespad=0.2, ncols=below[fig.axes.index(ax)], **kk)
      fig.canvas.draw()
      W, H = fig.get_size_inches()
      fig.set_size_inches(W, H + g.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi + 4 / 72)
      log.append((f'legend of {name}', 'under the axes (inside the axes it covered points or texts)'))
    else:
      orig(owner, *a, **k)


def relayout(fig):
  """tight_layout, as the generators lay out their figures, with room for a figure legend at the top or bottom."""
  fig.tight_layout()
  if not fig.legends:
    return
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  H = fig.bbox.height
  pad = 4 * fig.dpi / 72
  top, bottom = 1.0, 0.0
  for g in fig.legends:
    bb = g.get_window_extent(r)
    if (bb.y0 + bb.y1) / 2 > H / 2:
      top = min(top, (bb.y0 - pad) / H)
    else:
      bottom = max(bottom, (bb.y1 + pad) / H)
  fig.tight_layout(rect=(0, bottom, 1, top))


def readable(fig, name, conf, kind, below, log):
  """The step of the style round of 2026-11-20 (see relabel_figures.py)."""
  import matplotlib.pyplot as plt
  frac, width = conf
  changed = False
  if width and abs(fig.get_size_inches()[0] - width) > 1e-9:
    fig.set_size_inches(width, fig.get_size_inches()[1])
    changed = True
    log.append((f'width of {name}', f'{width} in, printed at {frac} \\textwidth'))
  scale = frac * TEXTWIDTH_PT / (fig.get_size_inches()[0] * 72)
  base = plt.rcParams['font.size']

  def need(t):
    return MIN_PT / (scale * glyph_ratio(t))

  texts = drawn_texts(fig)
  raised = []
  for ax in fig.axes:
    for axname, axis in (('x', ax.xaxis), ('y', ax.yaxis)):
      for which in ('major', 'minor'):
        labs = shown_ticklabels(ax, axis, which)
        if not labs:
          continue
        req = max(need(t) for t in labs)
        fs = min(t.get_fontsize() for t in labs)
        if fs < req - 1e-9:
          ax.tick_params(axis=axname, which=which, labelsize=up(max(base, req)))
          raised.append(f'{axname} {which} tick labels {fs:g}')
  leg_size = 0.0
  for t, k in texts:
    if k == 'legend' and t.get_fontsize() < need(t) - 1e-9:
      leg_size = max(leg_size, up(max(base, need(t))))
    elif k == 'text' and t.get_fontsize() < need(t) - 1e-9:
      raised.append(f'{t.get_text()[:20]!r} {t.get_fontsize():g}')
      t.set_fontsize(up(max(base, need(t))))
  if raised or leg_size:
    changed = True
    log.append((f'text of {name}', f'raised to print at {MIN_PT:g} pt or more ({len(raised)} groups'
                f'{", legends" if leg_size else ""})'))
  if kind:
    gray(fig, kind, log, name)
    changed = True
  if kind or leg_size or below:
    redraw_legends(fig, leg_size, below, log, name)
    changed = True
  if changed:
    relayout(fig)


def transpose_targets(fig, name, size_in, log):
  """F12 drawn again with the targets on the y axis, so that their names print horizontally (decisions of
  2026-11-21 on the supplement's doubts): the same values with x and y swapped. The generator draws one axes with the
  range of every sample as vertical lines (vlines), one line of markers per condition, the two thresholds as
  horizontal lines with a text each, and the target names as x tick labels at 90 degrees. Here: horizontal range lines
  (hlines), the same markers at (DockQ, target), vertical threshold lines with their text above the axes, the names as
  y tick labels from top to bottom in draw order. Anything else in the figure stops the script."""
  from matplotlib.collections import LineCollection
  if len(fig.axes) != 1 or fig.legends or fig.texts:
    raise SystemExit(f'{name}: expected one axes and no figure legend or text')
  ax = fig.axes[0]
  cols = [c for c in ax.collections]
  if len(cols) != 1 or not isinstance(cols[0], LineCollection) or ax.patches:
    raise SystemExit(f'{name}: expected one collection of range lines and no patch')
  rng = cols[0]
  segs = [s.tolist() for s in rng.get_segments()]
  if any(len(s) != 2 or s[0][0] != s[1][0] for s in segs):
    raise SystemExit(f'{name}: a range line is not vertical')
  pts = [ln for ln in ax.get_lines() if str(ln.get_linestyle()) in ('None', '', ' ')]
  thr = [ln for ln in ax.get_lines() if ln not in pts]
  if any(ln.get_xydata()[:, 1].min() != ln.get_xydata()[:, 1].max() for ln in thr) or len(thr) != len(ax.texts):
    raise SystemExit(f'{name}: expected horizontal threshold lines, one text each')
  ticks = [(float(p), t.get_text()) for p, t in zip(ax.get_xticks(), ax.get_xticklabels())]
  xlim, ylim, xlabel, ylabel = ax.get_xlim(), ax.get_ylim(), ax.get_xlabel(), ax.get_ylabel()
  leg = ax.get_legend()
  call = getattr(leg, '_relabel_call', None) if leg else None
  texts = [(t.get_text(), t.get_position()[1], t.get_fontsize()) for t in ax.texts]
  P = [dict(xy=ln.get_xydata().copy(), marker=ln.get_marker(), ms=ln.get_markersize(), mfc=ln.get_markerfacecolor(),
            mec=ln.get_markeredgecolor(), color=ln.get_color(), label=ln.get_label()) for ln in pts]
  T = [dict(v=float(ln.get_xydata()[0, 1]), color=ln.get_color(), lw=ln.get_linewidth(), ls=ln.get_linestyle())
       for ln in thr]
  fs_tick = ax.get_xticklabels()[0].get_fontsize() if ticks else 8
  fig.clf()
  fig.set_size_inches(*size_in)
  ax = fig.add_subplot()
  ax.hlines([s[0][0] for s in segs], [s[0][1] for s in segs], [s[1][1] for s in segs], colors=rng.get_colors(),
            lw=float(rng.get_linewidths()[0]), label=rng.get_label())
  for p in P:
    ax.plot(p['xy'][:, 1], p['xy'][:, 0], marker=p['marker'], ls='', ms=p['ms'], mfc=p['mfc'], mec=p['mec'],
            color=p['color'], label=p['label'])
  for t in T:
    ax.axvline(t['v'], color=t['color'], lw=t['lw'], ls=t['ls'])
  ax.set_yticks([p for p, _ in ticks], [s for _, s in ticks], fontsize=fs_tick)
  ax.set_ylim(xlim[1], xlim[0])   # the first target of the draw at the top
  ax.set_xlim(*ylim)
  ax.set_xlabel(ylabel)
  ax.set_ylabel(xlabel)
  for (s, v, fs), t in zip(texts, T):
    if abs(v - t['v']) > 1e-12:
      raise SystemExit(f'{name}: the text {s!r} is not at its threshold')
    ax.annotate(s, (v, 1), xycoords=('data', 'axes fraction'), xytext=(0, 2), textcoords='offset points',
                ha='center', va='bottom', fontsize=fs, annotation_clip=False)
  if call is not None:
    _, _, a, k = call
    ax.legend(*a, **k)   # through the recorded legend (record_legends), so that redraw_legends can draw it again
  fig.tight_layout()
  log.append((f'axes of {name}', 'transposed: targets on the y axis, names horizontal (same values, x and y swapped)'))


def check_printed(fig, name, frac):
  """Every drawn text prints at 8 pt or more at the printed width; no axis label is rotated."""
  scale = frac * TEXTWIDTH_PT / (fig.get_size_inches()[0] * 72)
  small = [(t.get_text()[:30], round(t.get_fontsize() * glyph_ratio(t) * scale, 2)) for t, _ in drawn_texts(fig)
           if t.get_fontsize() * glyph_ratio(t) * scale < MIN_PT - 1e-6]
  if small:
    raise SystemExit(f'{name}: text printed below {MIN_PT:g} pt: {small[:4]}')
  for ax in fig.axes:
    for lab in (ax.xaxis.label, ax.yaxis.label):
      if lab.get_visible() and lab.get_text().strip() and round(lab.get_rotation()) % 360:
        raise SystemExit(f'{name}: rotated axis label {lab.get_text()!r}')
  # decisions of 2026-11-21: no text, legend included, may leave the figure (a legend under F12 was cut at the right)
  r = fig.canvas.get_renderer()
  W, H = fig.bbox.width, fig.bbox.height
  out = [t.get_text()[:30] for t, _ in drawn_texts(fig) if (lambda b: b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5
                                                                     or b.y1 > H + 0.5)(t.get_window_extent(r))]
  if out:
    raise SystemExit(f'{name}: text outside the figure: {out[:4]}')


def _num(a):
  import numpy as np
  a = np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)
  return [[None if v != v else float(v) for v in row] for row in np.atleast_2d(a).tolist()]


def dump(fig, name, path):
  """The drawn data of a figure at save time, as JSON: 'data' holds every value a layout step must not change, 'style'
  the series' styles (compared apart) and 'layout' the size and the text sizes."""
  from matplotlib.container import ErrorbarContainer
  from matplotlib.patches import Rectangle
  from matplotlib.text import Annotation
  fig.canvas.draw()
  data, style = [], []
  for ax in fig.axes:
    A = {'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()), 'xscale': ax.get_xscale(),
         'yscale': ax.get_yscale(), 'title': [ax.get_title(loc=x) for x in ('left', 'center', 'right')],
         'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel()}
    A['lines'] = [{'label': ln.get_label(), 'xy': _num(ln.get_xydata())} for ln in ax.get_lines()]
    A['collections'] = [{'type': type(c).__name__, 'label': c.get_label(), 'offsets': _num(c.get_offsets()),
                         'paths': [_num(p.vertices) for p in c.get_paths()]} for c in ax.collections]
    A['patches'] = []
    for p in ax.patches:
      if isinstance(p, Rectangle):
        A['patches'].append({'type': 'Rectangle', 'xywh': [p.get_x(), p.get_y(), p.get_width(), p.get_height()]})
      else:
        A['patches'].append({'type': type(p).__name__,
                             'path': _num(p.get_patch_transform().transform(p.get_path().vertices))})
    A['texts'] = [{'text': t.get_text(), 'xy': list(t.xy) if isinstance(t, Annotation) else list(t.get_position())}
                  for t in ax.texts]
    A['ticks'] = {f'{axn} {w}': [[loc, t.get_text()] for loc, t in shown_ticks(axis, w)]
                  for axn, axis in (('x', ax.xaxis), ('y', ax.yaxis)) for w in ('major', 'minor')}
    A['errorbar containers'] = sum(isinstance(c, ErrorbarContainer) for c in ax.containers)
    leg = ax.get_legend()
    A['legend'] = [t.get_text() for t in leg.get_texts()] if leg else []
    data.append(A)
    st = []
    for ln in ax.get_lines():
      st.append({'label': ln.get_label(), 'color': str(ln.get_color()), 'marker': str(ln.get_marker()),
                 'linestyle': str(ln.get_linestyle()), 'dashes': str(ln._dash_pattern), 'mfc': str(ln.get_mfc())})
    for p in ax.patches:
      st.append({'patch': type(p).__name__, 'facecolor': str(p.get_facecolor()), 'hatch': str(p.get_hatch())})
    style.append(st)
  fig_data = {'axes': data, 'figure legends': [[t.get_text() for t in g.get_texts()] for g in fig.legends],
              'figure texts': [t.get_text() for t in fig.texts]}
  layout = {'size_in': [float(x) for x in fig.get_size_inches()],
            'text sizes': sorted({round(t.get_fontsize() * glyph_ratio(t), 3) for t, _ in drawn_texts(fig)})}
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({'data': fig_data, 'style': style, 'layout': layout}, indent=1, sort_keys=True,
                             default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
