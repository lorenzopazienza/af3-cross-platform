"""Follow-up 13 figures from the CSVs of analysis/fu13.py (one for P, one for L), as PDF and PNG.

    analysis/figures/.venv/bin/python analysis/fu13_figures.py --dir results/analysis/fu13_<DATE>

Run with the figures environment (analysis/requirements-figures.txt: the analysis environment plus matplotlib), as
analysis/paper_assets.py is; it reads only the CSVs, so no number is computed here.
  fig_fu13_p: top-1 and success flip rates against eps (log2 axis, in units of u = 2^-8) with 95% intervals, the
              main study's HW(V5,V6) and HW(V6,G) on seeds 1 and 2 as horizontal lines, eps* marked.
  fig_fu13_l: median relative L2 difference (units of u, log axis) against depth, recycle 0 and recycle 10, single
              and pair, for V5-V6, V6-G and G-Gr2; the shaded band is the range over targets.
"""
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {'V5-V6': '#1b6ca8', 'V6-G': '#c4501b', 'G-Gr2': '#6b6b6b', 'flip_top1': '#1b6ca8', 'flip_success': '#c4501b'}


def save(fig, d, name):
  fig.savefig(d / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Producer': None})
  fig.savefig(d / f'{name}.png', dpi=200, metadata={'Software': None})
  plt.close(fig)


def fig_p(d):
  c = pd.read_csv(d / 'p_curve.csv')
  e = pd.read_csv(d / 'p_eps_star.csv')
  fig, ax = plt.subplots(figsize=(4.2, 3.0))
  for st, label in (('flip_top1', 'top-1 flip'), ('flip_success', 'success flip')):
    g = c[c.statistic == st].sort_values('eps_u')
    ax.errorbar(g.eps_u, 100 * g.estimate, yerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)],
                marker='o', ms=3, capsize=2, lw=1, color=COLORS[st], label=label)
  hw = e[(e.statistic == 'flip_top1') & (e.hw_seeds == '1 and 2') & (e.n_targets > 0)]
  for r, ls in zip(hw.itertuples(), ('--', ':')):
    ax.axhline(100 * r.hw_rate, ls=ls, lw=0.8, color='k', label=f'{r.pair}, seeds 1 and 2')
    if r.status == 'interpolated':
      ax.axvline(r.eps_star_u, ls=ls, lw=0.6, color='0.5')
  ax.set_xscale('log', base=2)
  ax.set_xlabel('perturbation size eps at the trunk output (u)')
  ax.set_ylabel('flip rate against eps = 0 (%)')
  ax.legend(fontsize=6, frameon=False)
  fig.tight_layout()
  save(fig, d, 'fig_fu13_p')


def fig_l(d):
  s = pd.read_csv(d / 'l_summary.csv')
  fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
  for i, kind in enumerate(('single', 'pair')):
    for j, rec in enumerate((0, 10)):
      ax = axes[i][j]
      for name in ('V5-V6', 'V6-G', 'G-Gr2'):
        g = s[(s.kind == kind) & (s.recycle == rec) & (s.pair == name) & (s.median_u > 0)].sort_values('depth')
        if len(g):
          ax.plot(g.depth, g.median_u, marker='.', ms=3, lw=1, color=COLORS[name], label=name)
          ax.fill_between(g.depth, g.min_u.clip(lower=1e-6), g.max_u, color=COLORS[name], alpha=0.12, lw=0)
      ax.set_yscale('log')
      ax.set_title(f'{kind}, recycle {rec}', fontsize=8)
      if j == 0:
        ax.set_ylabel('relative L2 (units of u)')
  fig.supxlabel('depth (0 = Pairformer input, 48 = trunk output)', fontsize=9)
  axes[0][0].legend(fontsize=6, frameon=False)
  fig.tight_layout()
  save(fig, d, 'fig_fu13_l')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--dir', required=True)
  d = pathlib.Path(ap.parse_args().dir)
  fig_p(d)
  fig_l(d)
  print(f'wrote {d}/fig_fu13_p.pdf, fig_fu13_l.pdf (and .png)')


if __name__ == '__main__':
  main()
