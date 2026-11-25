"""analysis/paper_assets_isc.py: formatting, the log census on synthetic logs, and (when the checkout with results/ is
present: AF3_MAIN, else the repository's main working tree) one full run of the generator, checked for fonts, sizes,
widths, the CSV twins against their sources, the Fig. 2 selection rule, wording rules, and byte-identical PDFs across
two runs."""
import hashlib
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import paper_assets_isc as isc  # noqa: E402

MAIN = pathlib.Path(os.environ.get('AF3_MAIN') or isc.base.main_checkout())
HAVE_MAIN = (MAIN / isc.MAINV2 / 'contrasts' / 'contrasts.csv').exists() and (MAIN / isc.FREEZE_DIR).exists()
needs_main = pytest.mark.skipif(not HAVE_MAIN, reason='checkout with results/ (AF3_MAIN) is absent')


# ---- formatting

def test_rates_and_differences():
  assert isc.pc(0.18) == '18.0' and isc.pc(-0.0001) == '0.0' and isc.pc(0) == '0.0'
  assert isc.spc(0.183) == '+18.3' and isc.spc(-0.0176) == '-1.8' and isc.spc(0.0) == '0.0'
  r = pd.Series({'estimate': -0.0176, 'ci_low': -0.1059, 'ci_high': 0.0647})
  assert isc.diff_ci(r) == '-1.8 [-10.6, 6.5]'          # bounds carry a sign only when negative
  r = pd.Series({'estimate': 0.18, 'ci_low': 0.1433, 'ci_high': 0.2167})
  assert isc.rate_ci(r) == '18.0 [14.3, 21.7]'
  assert isc.sig3(0.07655) == '0.0766' and isc.sig3(0.022) == '0.0220' and isc.sig3(0) == '0'


def test_tex_minus_thousands_ranges():
  assert isc.tex('-1.8 [-10.6, 6.5]') == '$-$1.8 [$-$10.6, 6.5]'
  assert isc.tex('26550 of 26550') == '26\\,550 of 26\\,550'
  assert isc.tex('1500 of 1500 and 1024 tokens') == '1500 of 1500 and 1024 tokens'     # four digits: no separator
  assert isc.tex('1025--1536 tokens') == '1025--1536 tokens'
  assert isc.tex('HW(V5,V6) 18.0%') == 'HW(V5,V6) 18.0\\%'
  assert isc.tex('0.0165 Å') == '0.0165 \\AA{}'
  assert isc.fint(260930, tex=True) == '260\\,930' and isc.fint(260930) == '260930'


def test_wording_guard():
  with pytest.raises(SystemExit):
    isc.no_em_dash('a \u2014 b', 'x')
  with pytest.raises(SystemExit):
    isc.no_em_dash('RC(C) was bit-identical', 'x')


# ---- log census on synthetic lines

LOG = '\n'.join([
    'I1014 10:00:00.000000  1 run_alphafold.py:100] Processing fold job 7BTT',
    'W1014 10:00:01.000000  1 dispatch.py:50] Finished tracing + transforming foo for pjit in 0.1 sec',
    'W1014 10:00:01.000000  1 pxla.py:60] Compiling jit_apply_fn with global shapes',
    'W1014 10:00:02.000000  1 compiler.py:125] Persistent compilation cache hit for \'jit_apply_fn\'',
    'E1014 10:00:03.000000  1 api.py:116] Failed to run implementation triton',
    'NotImplementedError: Not supported on TPU v5 lite.',
    '/x/featurization.py:113: UserWarning: Explicitly requested dtype int64 requested in broadcasted_iota is not '
    'available, and will be truncated to dtype int32.',
    'Done running 1 fold jobs.', ''])


def test_scan_log_levels_and_classes():
  r = isc.scan_log(LOG.encode())
  assert (r['Wc'], r['W'], r['E'], r['F']) == (2, 1, 1, 0)      # compile logging is counted apart from other W
  assert r['done']
  c = r['classes']
  assert c['W compile logging (JAX_LOG_COMPILES)'] == 2
  assert c['W compiler.py: Persistent compilation cache hit'] == 1
  assert c['E api.py: Failed to run implementation (Tokamax GLU)'] == 1
  assert c['NotImplementedError: Not supported on <device>'] == 1
  assert c['UserWarning: int64 in broadcasted_iota truncated to int32'] == 1
  assert ('I', 'run_alphafold.py', 'Processing fold job <id>') in r['tmpl']


def test_norm_removes_paths_and_digits():
  assert isc.norm('Compiling jit_x with shapes 123') == 'Compiling ...'
  assert isc.norm('read /home/u/a/b.json in 0.25 s') == 'read <path>/b.json in N s'


def test_ecdf_series_differ_by_dash():
  """Every series of the ligand RMSD figure has its own line style, so it reads in grayscale (2026-11-20)."""
  styles = [str(ls) for _, _, ls in isc.ECDF_SERIES]
  assert len(set(styles)) == len(styles)


def test_plain_log_ticks():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.set_xscale('log')
  isc.plain_log_ticks(ax.xaxis)
  ax.set_xlim(1e-4, 12)
  fig.canvas.draw()
  labs = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
  assert {'0.0001', '0.001', '0.01', '0.1', '1', '10'} <= set(labs) and not any('$' in x for x in labs)
  plt.close(fig)


def test_palette_grayscale():
  """Navy, rose and band differ by at least 15 L*, from the hex values and after PIL's convert('L')."""
  from PIL import Image

  def lstar(h):
    rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return 116 * (y ** (1 / 3) if y > 0.008856 else 7.787 * y + 16 / 116) - 16
  ls = {k: lstar(h) for k, h in (('navy', isc.NAVY), ('rose', isc.ROSE), ('band', isc.BAND))}
  for k, v in ls.items():
    assert abs(v - isc.LSTAR[k]) < 1.5, (k, v)
  vals = sorted(ls.values())
  assert all(b - a >= 15 for a, b in zip(vals, vals[1:]))
  grays = sorted(Image.new('RGB', (1, 1), h).convert('L').getpixel((0, 0)) for h in (isc.NAVY, isc.ROSE, isc.BAND))
  assert all(b - a >= 15 * 255 / 100 for a, b in zip(grays, grays[1:]))


# ---- one full run

@pytest.fixture(scope='module')
def run(tmp_path_factory):
  if not HAVE_MAIN:
    pytest.skip('checkout with results/ is absent')
  out = tmp_path_factory.mktemp('isc1')
  argv = sys.argv
  sys.argv = ['paper_assets_isc.py', '--main', str(MAIN), '--out', str(out), '--reserve']
  try:
    isc.main()
  finally:
    sys.argv = argv
  return out


@needs_main
def test_widths_font_sizes_and_fonts(run):
  pdfs = sorted((run / 'figures').glob('*.pdf'))
  assert {p.stem for p in pdfs} == {'fig2_forest', 'fig3_rmsd_ecdf', 'fig4_margin_exploratory', 'fig4_perturbation',
                                    'fig5_seed_curve', 'figR1_cost', 'figR2_size', 'fig_fu13', 'fig_fu13_l'}
  for name, texts in isc.FIG_TEXTS.items():
    assert texts and min(s for _, s in texts) >= 8, name
  # superscripts and subscripts too (style round of 2026-11-20), except in the reserve figure R1, which is not printed
  assert set(isc.FIG_SCRIPTS) == set(isc.FIG_TEXTS)
  for name, texts in isc.FIG_SCRIPTS.items():
    if name not in isc.SCRIPT_EXEMPT:
      assert texts and min(s for _, s in texts) >= 8, name
  assert isc.SCRIPT_EXEMPT == {'figR1_cost'}
  for p in pdfs:
    info = subprocess.run(['pdfinfo', str(p)], capture_output=True, text=True, check=True).stdout
    w = float(re.search(r'Page size:\s+([\d.]+) x', info).group(1)) / 72
    assert round(w, 2) in isc.WIDTHS, (p.name, w)
    assert 'CreationDate' not in info and 'ModDate' not in info
    fonts = subprocess.run(['pdffonts', str(p)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
    assert fonts, p.name
    for line in fonts:
      assert 'Type 3' not in line and re.search(r'\+Arial', line), (p.name, line)
      assert line.split()[-5:-2] == ['yes', 'yes', 'yes'], (p.name, line)     # embedded, subset, unicode


@needs_main
def test_main_study_read_from_primary_v2(run):
  src = pd.read_csv(run / 'sources.tsv', sep='\t')
  main = [p for p in src.path if p.startswith('results/analysis/main_20261014/') and 'exploratory' not in p]
  assert main and all('/primary_v2/' in p for p in main)
  assert src.sha256.str.fullmatch(r'[0-9a-f]{64}').all()
  t2 = pd.read_csv(run / 'tables' / 'TII_headline.csv')
  assert t2.loc[t2.contrast == 'HW(V5,V6)', 'C-alpha RMSD, median (Å)'].iloc[0].startswith('0.0445')


@needs_main
def test_forest_twin_equals_sources_and_selection_rule(run):
  fr = pd.read_csv(run / 'tables' / 'fig2_forest.csv')
  for r in fr.itertuples():
    s = pd.read_csv(MAIN / r.source)
    if r.kind == 'HW minus RC':
      m = s[(s.hardware == r.pair) & (s.recompilation == r.reference) & (s.statistic == r.statistic)]
    else:
      m = s[(s.contrast == (r.pair if r.kind == 'HW' else r.reference)) & (s.statistic == r.statistic)]
    assert any(math.isclose(v, r.estimate, abs_tol=1e-12) for v in m.estimate), r
  ref = fr[fr.kind == 'RC'].set_index(['group', 'pair', 'statistic']).sort_index().reference
  assert ref['A100', 'HW(G,A)', 'flip_top1'] == 'RC(A)' and ref['A100', 'HW(G,A)', 'flip_success'] == 'RC(G)'
  assert ref['H100', 'HW(H,A)', 'flip_top1'] == 'RC(A)' and ref['H100', 'HW(H,A)', 'flip_success'] == 'RC(H)'
  d = fr[(fr.kind == 'HW minus RC') & (fr.pair == 'HW(G,A)') & (fr.statistic == 'flip_success')].estimate.iloc[0]
  assert isc.pc(d) == '0.3'
  dag = fr.drop_duplicates(['group', 'pair']).set_index(['group', 'pair']).dagger.fillna('')
  assert dag['CPU', 'HW(V5,C)'] == '†' and dag['Main', 'HW(V6,G)'] == '' and dag['Official inputs', 'HW(V6o,Go)'] == '†'


@needs_main
def test_other_twins_equal_sources(run):
  q = pd.read_csv(run / 'tables' / 'fig4_margin.csv')
  src = pd.read_csv(MAIN / isc.EXPL / 'a_margin_quintiles.csv')
  m = q.merge(src, on=['contrast', 'quintile'], suffixes=('', '_src'))
  assert len(m) == len(q) == 15 and (m.flip_rate == m.flip_rate_src).all() and (m.ci_high == m.ci_high_src).all()
  s = pd.read_csv(run / 'tables' / 'fig5_seed_curve.csv')
  q3 = isc.seed_curve(pd.read_csv(MAIN / isc.AN / 'seeds_20261102' / 'q3.csv'))
  assert (s.estimate.values == q3.estimate.values).all() and (s.ci_low.values == q3.ci_low.values).all()
  e = pd.read_csv(run / 'tables' / 'fig3_rmsd_quantiles.csv').set_index('contrast')
  con = pd.read_csv(MAIN / isc.MAINV2 / 'contrasts' / 'contrasts.csv')
  for c in ('HW(V5,V6)', 'HW(V6,G)', 'SD(V6)'):
    med = con[(con.contrast == c) & (con.statistic == 'm4_ligand_rmsd_median')].iloc[0]
    assert math.isclose(e.loc[c, 'median'], med.estimate, rel_tol=1e-3) and e.loc[c, 'n'] == med.n


@needs_main
def test_ledger_values_reproduced(run):
  cc = pd.read_csv(run / 'tables' / 'campaign_counts.csv')
  tot = cc[(cc.section == 'total') & (cc.scope == 'all')].set_index('quantity').value
  for k, v in isc.MUST.items():
    assert int(tot[k]) == v, k
  assert not (cc.agrees == 'NO').any(), cc[cc.agrees == 'NO']
  assert set(cc.section) >= {'cost, recorded', 'cost, estimated', 'label', 'device'}
  me = pd.read_csv(run / 'tables' / 'mechanism_extra.csv')
  assert (me.agrees == 'yes').all()
  cpu = pd.read_csv(run / 'tables' / 'cpu_flips.csv')
  t1 = cpu[cpu.statistic == 'flip_top1']
  assert list(t1['on the 6 targets C ran in both sessions']) == [0, 0]
  fu12 = pd.read_csv(run / 'tables' / 'fu12_cutoff_targets.csv')
  assert sorted(set(fu12.target)) == ['7MGT', '7Q5I', '7U3J']
  lc = pd.read_csv(run / 'tables' / 'log_census.csv')
  words = lc[(lc.section == 'word search') & lc.item.isin(isc.WORDS)]
  assert len(words) == len(isc.WORDS) and (words.value.astype(int) == 0).all()


@needs_main
def test_wording_in_every_output(run):
  for p in run.rglob('*'):
    if p.suffix in ('.csv', '.tex', '.md'):
      t = p.read_text()
      assert '\u2014' not in t and 'RC(C)' not in t, p
  t4 = (run / 'tables' / 'TIV_followups.csv').read_text()
  assert ('CPU outputs were bit-identical across sessions in 12 of 12 predictions, on the 6 targets the CPU ran in both '
          'sessions') in t4
  assert 'moves the top-1 less than v6e vs L4, which measures the platform stack' in t4
  assert 'beyond recompilation' in t4 and 'A100 and H100' in t4
  fu7 = pd.read_csv(run / 'tables' / 'TIV_followups.csv').set_index('FU').loc[7]
  assert 'chip' not in ' '.join(map(str, fu7.values))
  caps = (run / 'captions.tex').read_text()
  # no thousands separator ("1,500")
  assert caps.count('\\newcommand') >= 9 and not re.search(r'\d,\d', re.sub(r'\[[^\]]*\]', '', caps))


@needs_main
def test_two_runs_give_identical_pdfs(run, tmp_path):
  out = tmp_path / 'isc2'
  py = sys.executable
  subprocess.run([py, str(REPO / 'analysis' / 'paper_assets_isc.py'), '--main', str(MAIN), '--out', str(out),
                  '--reserve'], check=True, capture_output=True)
  for p in sorted((run / 'figures').glob('*.pdf')):
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256((out / 'figures' / p.name).read_bytes()).hexdigest(), p
  for p in sorted((run / 'tables').glob('*')):
    assert p.read_bytes() == (out / 'tables' / p.name).read_bytes(), p
  assert (run / 'captions.tex').read_bytes() == (out / 'captions.tex').read_bytes()
  shutil.rmtree(out)


# ---- table files: tabular only, note apart (added 2026-11-09)

def test_write_table_puts_the_note_in_its_own_file(tmp_path):
  df = pd.DataFrame({'arm': ['V5'], 'processes': [13499]})
  isc.write_table(tmp_path, 'TI_arms', df, 'Caption', 'design', notes=['A note, 12345 rows.'])
  body = (tmp_path / 'tables' / 'TI_arms.tex').read_text()
  assert 'parbox' not in body and body.rstrip().endswith('\\end{tabular}') and '13\\,499' in body
  note = (tmp_path / 'tables' / 'TI_arms_note.tex').read_text()
  assert 'parbox' not in note and 'A note, 12\\,345 rows.' in note
  isc.write_table(tmp_path, 'TI_arms', df, 'Caption', 'design')
  assert not (tmp_path / 'tables' / 'TI_arms_note.tex').exists()     # a stale note never survives



def test_tie_pairs_binds_whole_pairs_only(tmp_path):
  """Reading note R93 (2026-11-12): a pair "X vs Y" in a caption or note never breaks across two lines."""
  assert isc.tie_pairs('The pairs are v5e vs v6e, v6e vs L4 Triton, and v6e highest vs L4 highest.') == \
      'The pairs are v5e~vs~v6e, v6e~vs~L4~Triton, and v6e~highest~vs~L4~highest.'
  assert isc.tie_pairs('The row v6e vs L4 minus recompile L4') == 'The row v6e~vs~L4 minus recompile L4'
  assert isc.tie_pairs('except L4 vs A100 (Supplementary') == 'except L4~vs~A100 (Supplementary'
  assert isc.tie_pairs('versus and vs alone stay') == 'versus and vs alone stay'
  df = pd.DataFrame({'pair': ['v5e vs v6e']})
  isc.write_table(tmp_path, 'TI_arms', df, 'Caption', 'design', notes=['Pairs: H100 vs A100.'])
  assert 'v5e vs v6e' in (tmp_path / 'tables' / 'TI_arms.tex').read_text()      # cells are not touched
  assert 'H100~vs~A100.' in (tmp_path / 'tables' / 'TI_arms_note.tex').read_text()


@needs_main
def test_no_breakable_pair_in_captions_or_notes(run):
  for f in [run / 'captions.tex', *sorted((run / 'tables').glob('*_note.tex'))]:
    assert not isc.PAIR.search(f.read_text()), f.name


@needs_main
def test_tables_are_tabular_only_with_notes_apart(run):
  for p in (run / 'tables').glob('T*.tex'):
    t = p.read_text()
    if p.stem.endswith('_note'):
      assert '\\begin' not in t and 'parbox' not in t, p
    else:
      # a supplement table taller than a page is a longtable (style round of 2026-11-20)
      ends = ('\\end{longtable}',) if isc.SUPP_LAYOUT.get(p.stem, {}).get('longtable') else ('\\end{tabular}', '\\end{tabular*}%')
      assert 'parbox' not in t and t.rstrip().endswith(ends), p
  for n in ('TI_arms', 'TII_headline', 'TIII_q2', 'TIV_followups', 'TS_campaign', 'TS_provisioning'):
    assert (run / 'tables' / f'{n}_note.tex').exists(), n
  t2 = (run / 'tables' / 'TII_headline.tex').read_text()
  assert 'top-1 (\\%)' in t2 and 'C-alpha (\\AA{})' in t2 and '0.0445 &' in t2 and '0.0445 [' not in t2
  t4 = pd.read_csv(run / 'tables' / 'TIV_followups.csv').set_index('FU').loc[11, 'reading']
  assert t4.endswith('secondary: 4 of 8 cells do not replicate (interval not above 0)')


@needs_main
@pytest.mark.skipif(shutil.which('pdflatex') is None, reason='pdflatex absent')
def test_natural_widths_at_footnotesize(run, tmp_path):
  """At \\footnotesize, unscaled (style round of 2026-11-19: every table of the body prints at 8 pt): Tables I, III
  (T_beyond) and IV (TIII_repro) within the 3.5 in column, Table II, TIII_q2 and TIV_followups within the 7.16 in text
  width (IEEEtran)."""
  lines = ['\\documentclass[letterpaper,conference]{IEEEtran}', '\\usepackage[T1]{fontenc}\\usepackage{booktabs}'
           '\\usepackage{array}\\usepackage{amsmath}', '\\newsavebox{\\mb}', '\\begin{document}']
  for n in ('TI_arms', 'TII_headline', 'TIII_q2', 'TIV_followups', 'T_beyond', 'TIII_repro'):
    lines.append(f'\\sbox{{\\mb}}{{\\footnotesize\\input{{{run}/tables/{n}.tex}}}}\\typeout{{WIDTH {n} \\the\\wd\\mb}}')
  lines += ['x', '\\end{document}']
  (tmp_path / 'w.tex').write_text('\n'.join(lines) + '\n')
  subprocess.run(['pdflatex', '-interaction=nonstopmode', 'w.tex'], cwd=tmp_path, capture_output=True, check=True)
  got = dict(re.findall(r'WIDTH (\S+) ([\d.]+)pt', (tmp_path / 'w.log').read_text()))
  assert len(got) == 6
  for n in ('TI_arms', 'T_beyond', 'TIII_repro'):
    assert float(got[n]) <= 3.5 * 72.27, (n, got[n])
  for n in ('TII_headline', 'TIII_q2', 'TIV_followups'):
    assert float(got[n]) <= 7.16 * 72.27, (n, got[n])


@needs_main
def test_natural_widths_at_small(run, tmp_path):
  """At \\small (9 pt, style round of 2026-11-19), unscaled, with the column spacing the paper sets:
  Table I in the 3.5 in column, and its body copy TI_arms_col exactly as wide as the column (reading notes of
  2026-11-20); Tables III (T_beyond) and IV (TIII_repro) side by side in one full-width float, with
  \\tabcolsep 3.5 pt, in minipages of 0.508 and 0.478 of the 516 pt text width (sections/results.tex, start of IV-D)."""
  lines = ['\\documentclass[letterpaper,conference]{IEEEtran}', '\\usepackage[T1]{fontenc}\\usepackage{booktabs}'
           '\\usepackage{array}\\usepackage{amsmath}', '\\newsavebox{\\mb}', '\\begin{document}']
  for n, sep in (('TI_arms', '6pt'), ('T_beyond', '3.5pt'), ('TIII_repro', '3.5pt'), ('TI_arms_col', '6pt')):
    lines.append(f'\\sbox{{\\mb}}{{\\small\\setlength{{\\tabcolsep}}{{{sep}}}\\input{{{run}/tables/{n}.tex}}}}'
                 f'\\typeout{{WIDTH {n} \\the\\wd\\mb}}')
  lines += ['\\typeout{WIDTH column \\the\\columnwidth}', 'x', '\\end{document}']
  (tmp_path / 'w.tex').write_text('\n'.join(lines) + '\n')
  subprocess.run(['pdflatex', '-interaction=nonstopmode', 'w.tex'], cwd=tmp_path, capture_output=True, check=True)
  log = (tmp_path / 'w.log').read_text()
  got = {k: float(v) for k, v in re.findall(r'WIDTH (\S+) ([\d.]+)pt', log)}
  assert len(got) == 5
  assert got['TI_arms'] <= 3.5 * 72.27, got
  # reading notes of 2026-11-20: the body's Table I (TI_arms_col) is exactly as wide as the column, with no overfull
  # alignment (its natural width, TI_arms above, fits)
  assert got['TI_arms_col'] == got['column'] and 'Overfull' not in log, got
  assert got['T_beyond'] <= 0.508 * 516, got
  assert got['TIII_repro'] <= 0.478 * 516, got


# ---- supplement tables at 8 pt (style round of 2026-11-20)

BODY_TABLES = {'TI_arms', 'TI_arms_col', 'TII_headline', 'TIII_q2', 'TIV_followups', 'T_beyond', 'TIII_repro', 'TS_fu16',
               'T_rules', 'TI_runs'}


def test_supplement_layout_never_touches_tables_of_the_main_text():
  assert not set(isc.SUPP_LAYOUT) & BODY_TABLES
  assert all(n.startswith('TS_') for n in isc.SUPP_LAYOUT)


@needs_main
def test_supplement_layout_keeps_every_cell(run, tmp_path, monkeypatch):
  """A run without the supplement layout writes plain tabulars; the laid-out tables of the full run read back (nested
  header lines joined, the repeated longtable header left out) to the same cells, row by row, and every other table
  and the captions are byte-identical."""
  layout = dict(isc.SUPP_LAYOUT)
  monkeypatch.setattr(isc, 'SUPP_LAYOUT', {})
  monkeypatch.setattr(sys, 'argv', ['paper_assets_isc.py', '--main', str(MAIN), '--out', str(tmp_path), '--reserve'])
  isc.main()
  lay = isc.table_layout()

  def cells(text):
    text = re.sub(r'\\endfirsthead\n.*?\\endhead\n', '', text, flags=re.S)
    return [[lay.read_back(c) for c in r] for r in map(lay.split_row, text.splitlines()) if r]
  laid = set()
  for p in sorted((run / 'tables').glob('*.tex')):
    plain = (tmp_path / 'tables' / p.name).read_text()
    if p.stem in layout:
      laid.add(p.stem)
      assert p.read_text() != plain and cells(p.read_text()) == cells(plain), p.stem
    else:
      assert p.read_text() == plain, p.stem
  assert laid == set(layout)
  assert (run / 'captions.tex').read_text() == (tmp_path / 'captions.tex').read_text()


@needs_main
@pytest.mark.skipif(shutil.which('pdflatex') is None, reason='pdflatex absent')
def test_supplement_tables_fit_the_text_width_at_footnotesize(run, tmp_path):
  """At \\footnotesize (8 pt), unscaled: every supplement-only table that table_layout lays out fits the 516 pt text
  width of the one-column supplement, a longtable by the column widths it records (TS_rules_seeds with the 4 pt column
  spacing the supplement sets)."""
  sep = {'TS_rules_seeds': '4pt'}
  lines = ['\\documentclass[conference,onecolumn]{IEEEtran}', '\\usepackage[T1]{fontenc}\\usepackage{booktabs}'
           '\\usepackage{array}\\usepackage{amsmath}\\usepackage{longtable}', '\\newsavebox{\\mb}', '\\begin{document}']
  for n, o in isc.SUPP_LAYOUT.items():
    if o.get('longtable'):
      lines.append(f'{{\\footnotesize\\input{{{run}/tables/{n}.tex}}}}')
    else:
      lines.append(f'\\sbox{{\\mb}}{{\\footnotesize\\setlength{{\\tabcolsep}}{{{sep.get(n, "6pt")}}}'
                   f'\\input{{{run}/tables/{n}.tex}}}}\\typeout{{WIDTH {n} \\the\\wd\\mb}}')
  lines += ['\\typeout{WIDTH text \\the\\textwidth}', 'x', '\\end{document}']
  (tmp_path / 'w.tex').write_text('\n'.join(lines) + '\n')
  for _ in range(2):
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'w.tex'], cwd=tmp_path, capture_output=True, check=True)
  got = {k: float(v) for k, v in re.findall(r'WIDTH (\S+) ([\d.]+)pt', (tmp_path / 'w.log').read_text())}
  aux = (tmp_path / 'w.aux').read_text()
  # the column widths longtable writes to the aux file: \gdef \LT@i {\LT@entry {1}{38.2pt}...}, one per longtable
  longs = [sum(float(w) for w in re.findall(r'\{\d+\}\{([\d.]+)pt\}', chunk))
           for chunk in aux.split('\\gdef ')[1:] if re.match(r'\\LT@[ivxl]+\s*\{', chunk)]
  assert got['text'] == 516.0
  assert len(got) - 1 == sum(not o.get('longtable') for o in isc.SUPP_LAYOUT.values()), got
  assert len(longs) == sum(bool(o.get('longtable')) for o in isc.SUPP_LAYOUT.values()), aux
  for n, w in got.items():
    assert w <= got['text'], (n, w)
  for w in longs:
    assert 0 < w <= got['text'], longs


# ---- supplement tables TS_campaign and TS_provisioning

@needs_main
def test_ts_campaign(run):
  c = pd.read_csv(run / 'tables' / 'TS_campaign.csv', dtype=str).fillna('')
  tot = c[c.group == 'total'].iloc[0]
  assert (tot.sessions, tot.processes, tot.samples, tot['exit 0'], tot['OOM stops']) == ('295', '13499', '67475', '13495', '4')
  assert tot['process wall h'] == '1026.7' and tot['VM h'] == '1113.3'
  lab = c[c.group == 'by label'].set_index('scope')
  assert int(lab.sessions.astype(int).sum()) == 295 and int(lab.processes.astype(int).sum()) == 13499
  assert lab.loc['memlimit', 'OOM stops'] == '4' and lab.loc['feat', 'processes'] == '0'
  dev = c[c.group == 'by device'].set_index('scope')
  assert dev.loc['A100 80 GB', 'VM h'] == '44.1' and lab.loc['a100', 'VM h'] == '44.1'     # rounding half up, both
  cost = c[c.group == 'cost'].set_index('description')
  assert cost.loc['total of the recorded items', 'list USD'] == '685.82'
  assert cost.loc['total of the estimated items', 'list USD'] == '653.79'
  assert cost.loc['campaign total, recorded plus estimated', 'list USD'] == '1339.61'
  rec = c[(c.group == 'cost') & (c.scope == 'recorded')]
  assert 'L4 official r1 27.70 (row of 2026-10-27)' in rec.description.iloc[0] and rec.description.str.contains('row of').iloc[:-1].all()
  t = (run / 'tables' / 'TS_campaign.tex').read_text()
  assert 'list prices, not a bill; disks, storage and network excluded' in t
  assert '13\\,499 & 67\\,475 & 13\\,495 & 4 & 1026.7 & 1113.3' in t and '& 1339.61 \\\\' in t and '2992 &' in t
  assert '\\newcommand{\\capTabCampaign}' in (run / 'captions.tex').read_text()


@needs_main
def test_ts_provisioning(run):
  p = pd.read_csv(run / 'tables' / 'TS_provisioning.csv', dtype=str).fillna('').set_index(['label', 'arm'])
  assert p.loc[('seeds', 'V5x'), 'provisioning'] == 'Spot 2, on demand 3'
  assert p.loc[('large', 'Gl'), 'provisioning'] == 'Spot 22, on demand 17'
  assert p.loc[('sw', 'V6s'), 'provisioning'] == 'Flex-start' and p.loc[('main', 'G'), 'provisioning'] == 'on demand'
  assert p.loc[('main', 'V5'), 'zones or regions'] == 'europe-west4-b' and p.loc[('main', 'V5'), 'machine type'] == \
      'n2d-48-24-v5lite-tpu'
  assert p.loc[('h100', 'H'), 'zones or regions'] == 'us-central1-a, us-central1-b'
  a = p.loc[('all', '')]
  assert (a.sessions, a.processes) == ('288', '13499') and a.provisioning == 'on demand 143, Spot 134, Flex-start 11'
  assert a['zones or regions'] == '9 regions, 20 zones'
  arms = p.drop(index=[('all', ''), ('feat', 'hosts')])
  assert len(arms) == len(isc.LEDGER_ARMS) and arms.processes.astype(int).sum() == 13499
  t = (run / 'tables' / 'TS_provisioning.tex').read_text()
  assert '\\emph{seeds: FU8}' in t and '13\\,499' in t
  assert '\\newcommand{\\capTabProvisioning}' in (run / 'captions.tex').read_text()


# ---- post hoc tables (added 2026-11-09)

@needs_main
def test_overall_top1_embed_summary_and_rmsd_counts(run):
  o = pd.read_csv(run / 'tables' / 'overall_top1.csv').set_index('contrast')
  assert o.loc['HW(V5,V6)', 'overall top-1 differs (targets)'] == 16       # main_q1.md section 0
  assert (o.targets == 60).all()
  assert o.loc['RC(V5)', 'overall top-1 differs (targets)'] == 0 and o.loc['RC(V6)', 'overall top-1 differs (targets)'] == 0
  assert (o['success a (targets)'].loc[['HW(V5,V6)', 'RC(V5)']] == 45).all()     # V5 75.0% of 60 (main_q1.md)
  e = pd.read_csv(run / 'tables' / 'embed_summary.csv').set_index(['pair', 'embedding'])
  for pair, lo, hi in (('V5 r1 - V6 r1', '0.0027', '0.0056'), ('V6 r1 - G r1', '0.0031', '0.0058'),
                       ('G r1 - G r2', '0.0026', '0.0033')):
    r = e.loc[(pair, 'single')]
    assert (f'{r["rel_l2 min"]:.4f}', f'{r["rel_l2 max"]:.4f}') == (lo, hi), pair
  assert e.loc[('G r1 - G r2', 'single'), 'not bit-identical'] == 6
  q = pd.read_csv(run / 'tables' / 'fig3_rmsd_quantiles.csv').set_index('contrast')
  pairs = pd.read_csv(MAIN / isc.MAINV2 / 'contrasts' / 'pairs_sample.csv', usecols=['contrast', 'm4_ligand_rmsd'])
  for c in q.index:
    x = pairs[pairs.contrast == c].m4_ligand_rmsd.dropna()
    assert q.loc[c, 'at_least_1A'] == (x >= 1).sum() and q.loc[c, 'at_least_2A'] == (x >= 2).sum()


@needs_main
def test_perturbation_figure_and_table_iii(run):
  """Task of 2026-11-10: Fig. 4 is the perturbation-to-margin model (A1); Table III has A and H rows and one memory
  metric (A3)."""
  f = pd.read_csv(run / 'tables' / 'fig4_perturbation.csv').set_index('contrast')
  src = pd.read_csv(MAIN / isc.EXPL / 'c_perturbation.csv').set_index('contrast')
  five = pd.read_csv(MAIN / isc.PREC5_EXPL / 'exploratory_c.csv').set_index('contrast')
  assert sorted(f.index) == sorted(src.index) and len(f) == 9 and sorted(five.index) == sorted(isc.PERT_FIVE)
  assert (f.n_seed_pairs == 300).all()
  src = pd.concat([src.drop(index=isc.PERT_FIVE), five])    # row of 2026-11-10: five seeds for the precision contrasts
  for c in f.index:
    assert f.loc[c, 'observed_flip_rate'] == src.loc[c, 'observed_flip_rate']
    assert f.loc[c, 'delta_over_mu'] == src.loc[c, 'delta_over_mu']
  t3 = pd.read_csv(run / 'tables' / 'TIII_q2.csv', dtype=str)
  assert list(t3.arm) == ['V5', 'V6', 'G', 'GT', 'A', 'H']
  cost = pd.read_csv(MAIN / isc.MAINV2 / 'cost' / 'cost_summary.csv')
  g = cost[(cost.platform == 'G') & (cost.bucket == 1024) & (cost.cache == 'fresh')].iloc[0]
  row = t3[t3.arm == 'G'].iloc[0]
  assert row['compile 1024 (s)'] == f'{g.median_compile_s:.0f}' and row['USD fresh'] == isc.usd3(g.median_usd_per_structure_standard)
  mem = pd.read_csv(MAIN / isc.MAINV2 / 'cost' / 'memory.csv')
  v5 = mem[(mem.platform == 'V5') & (mem.bucket == 1024)].iloc[0]
  assert t3[t3.arm == 'V5'].iloc[0]['memory 1024 (GiB)'] == isc.sig3(v5.compiled_peak_plus_args_gib)
  assert 'capFigPerturbation' in (run / 'captions.tex').read_text()
  # Options round of 2026-11-19: the appendix figure and the supplement's copy share a short caption; the key of the
  # numbered points is drawn in the figure, not written in the caption
  caps = (run / 'captions.tex').read_text()
  app = caps.split('\\newcommand{\\capFigPerturbationApp}')[1].split('\n')[0]
  sup = caps.split('\\newcommand{\\capFigPerturbation}')[1].split('\n')[0]
  assert app == sup and 'nine contrasts' in app and 'numbered' not in app
  assert set(isc.PERT_KEY) == set(isc.PERT_HW + isc.PERT_ONE)
