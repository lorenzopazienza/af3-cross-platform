"""analysis/paper_assets.py: LaTeX escaping, table writing, the memory-limit table from session records, and a full,
deterministic regeneration from the frozen outputs (figures only where matplotlib is installed: the figures
environment, analysis/requirements-figures.txt)."""

import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

import paper_assets as pa

REPO = pathlib.Path(__file__).resolve().parents[2]
needs_outputs = pytest.mark.skipif(not (pa.PRIMARY / 'contrasts' / 'contrasts.csv').exists()
                                   or not (pa.EXPLORATORY / 'c_perturbation.csv').exists(),
                                   reason='frozen analysis outputs not present (results/ is gitignored)')


def test_tex_escape():
  assert pa.tex_escape('a_b & 5% $x$ #1 {y} ~^\\') == \
      r'a\_b \& 5\% \$x\$ \#1 \{y\} \textasciitilde{}\textasciicircum{}\textbackslash{}'


def test_write_table(tmp_path, monkeypatch):
  monkeypatch.setitem(pa.ROLES, 'T99', 'exploratory (test)')
  df = pd.DataFrame({'contrast': ['HW(V5,V6)', 'RC(G)'], 'targets': [60, 60], 'rate (%)': ['18.0 [14.3, 21.7]', '7.3']})
  pa.write_table(tmp_path, 'T99', df, 'a caption with 95% and _')
  csv = (tmp_path / 'tables' / 'T99.csv').read_text().splitlines()
  tex = (tmp_path / 'tables' / 'T99.tex').read_text().splitlines()
  assert len(csv) == 3 and csv[0] == 'contrast,targets,rate (%)'
  assert r'\begin{tabular}{lrl}' in tex and r'HW(V5,V6) & 60 & 18.0 [14.3, 21.7] \\' in tex
  assert tex[0].startswith('% T99: a caption with 95\\% and \\_')
  assert tex[1] == '% Role: exploratory (test)' and pa.ASSETS['tables/T99'] == ('exploratory (test)', 'a caption with 95% and _')


def test_role_order_puts_primary_pairs_first():
  df = pd.DataFrame({'contrast': ['SD(G)', 'RC(G)', 'HW(V5,V6)', 'RC(Gd)', 'HW(V6,G)'],
                     'role': ['seed yardstick', 'recompilation', 'primary pair', 'recompilation, secondary arm',
                              'primary pair']})
  assert list(pa.by_role(df).contrast) == ['HW(V5,V6)', 'HW(V6,G)', 'RC(G)', 'RC(Gd)', 'SD(G)']


def _memlimit_session(root, name, config, sizes, stop=None):
  for size in sizes:
    d = root / name / f'{config}_bucket{size}_rep1_seed1_fresh'
    (d / '7D5C').mkdir(parents=True)
    (d / 'session.json').write_text(json.dumps({'label': 'memlimit', 'config_name': config, 'af3_buckets': [size],
                                                'host': {'gce_zone': 'europe-west4-a'}}))
    failed = size == stop
    (d / '7D5C' / 'run.json').write_text(json.dumps({
        'af3_buckets': [size], 'padded_bucket': size, 'num_tokens': 1023, 'seeds': [1], 'exit_code': 1 if failed else 0,
        'failure': 'oom' if failed else None, 'wall_seconds': 60.0 if failed else 600.0,
        'inference_seconds_by_seed': {} if failed else {'1': 500.0}, 'compile': {'model_seconds': 40.0},
        'failure_detail': ('jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory '
                           'required for HLO temporaries (20.95G) exceeds available HBM (15.75G).') if failed else None,
        'memory': {} if failed else {'compiled_peak_memory_in_bytes': size * 2 ** 22,
                                     'compiled_argument_size_in_bytes': 2 ** 30, 'bytes_limit': int(15.75 * 2 ** 30)}}))


def test_memlimit_table_from_session_records(tmp_path, monkeypatch):
  monkeypatch.setattr(pa, 'AF3', tmp_path / 'af3')
  _memlimit_session(tmp_path / 'af3', '20261015T000000Z_v5e_memlimit_v5', 'tpu_xla', [1280, 1536, 2048], stop=2048)
  _memlimit_session(tmp_path / 'af3', '20261015T000000Z_l4_memlimit_g', 'l4_xla', pa.MEMLIMIT_SIZES)
  recs = pa.t18_memlimit(tmp_path / 'out')
  t = pd.read_csv(tmp_path / 'out' / 'tables' / 'T18_memlimit.csv', dtype=str, keep_default_na=False).set_index('arm')
  assert len(recs) == 3 + 9
  assert t.loc['V5', 'largest completed size'] == '1536' and t.loc['V5', 'first failing size'] == '2048'
  assert t.loc['V5', 'cause'] == 'oom: 20.95 GiB required, 15.75 available'
  assert t.loc['G', 'largest completed size'] == '5120' and t.loc['G', 'first failing size'] == 'none (all sizes ran)'
  assert t.loc['V6', 'largest completed size'] == 'pending'
  b = pd.read_csv(tmp_path / 'out' / 'tables' / 'T18b_memlimit_processes.csv', dtype=str, keep_default_na=False)
  assert ((b.arm == 'V5') & (b['size'] == '2048') & (b['memory in the error (GiB)'] == '20.95 required of 15.75')).any()


def run(out, tables_only):
  cmd = [sys.executable, str(REPO / 'analysis' / 'paper_assets.py'), '--out', str(out)] + (['--tables_only'] if tables_only else [])
  subprocess.run(cmd, check=True, capture_output=True)
  return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob('*')) if p.is_file()}


@needs_outputs
def test_tables_regenerate_identically(tmp_path):
  a, b = run(tmp_path / 'a', True), run(tmp_path / 'b', True)
  assert a == b
  tables = [k for k in a if k.startswith('tables/')]
  assert len(tables) == 2 * 32 and 'sources.tsv' in a and 'README.md' in a
  # every asset is listed once in README.md with its role, and every role is one of ROLES
  readme = a['README.md'].decode().splitlines()
  listed = [l for l in readme if l.startswith('- `')]
  assert len(listed) == 32
  assert all('` [' in l and ']: ' in l for l in listed)
  assert pa.FREEZE_ID in a['README.md'].decode()
  src = pd.read_csv(tmp_path / 'a' / 'sources.tsv', sep='\t')
  assert (src[src.path.str.startswith('results/af3/')].in_freeze == 'yes').all()
  assert {'results/analysis/ppi_20261015/contrasts.csv', 'results/analysis/embed_20261015/summary.csv'} <= set(src.path)
  t19 = pd.read_csv(tmp_path / 'a' / 'tables' / 'T19b_ppi_contrasts.csv', keep_default_na=False)
  r = t19[(t19.contrast == 'HW(V6,G)') & (t19.statistic == 'top-1 flip per seed (%)')].iloc[0]
  assert r['main study, 60 targets'] == '20.7 [16.3, 25.3]' and r['ppi, 30 targets'].startswith('30.0')
  assert (t19[t19.statistic.str.contains('0.49|overall')]['main study, 60 targets'] == '').all()
  t04 = pd.read_csv(tmp_path / 'a' / 'tables' / 'T04_flip_rates.csv')
  assert list(t04.contrast[:3]) == ['HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)']
  assert t04.set_index('contrast').loc['HW(V5,V6)', 'top-1 flip (%)'] == '18.0 [14.3, 21.7]'


@needs_outputs
def test_figures_regenerate_identically(tmp_path):
  pytest.importorskip('matplotlib')
  a, b = run(tmp_path / 'a', False), run(tmp_path / 'b', False)
  figs = sorted(k for k in a if k.startswith('figures/'))
  assert len(figs) == 2 * 13
  readme = a['README.md'].decode()
  assert all(f'`figures/F{i}' in readme for i in range(1, 14))
  assert all(a[k] == b[k] and len(a[k]) > 1000 for k in figs)


def test_every_asset_has_a_role():
  import re
  names = set(re.findall(r"write_table\(out, '([^']+)'", (REPO / 'analysis' / 'paper_assets.py').read_text()))
  names |= set(re.findall(r"save\(fig, '([^']+)'", (REPO / 'analysis' / 'paper_assets.py').read_text()))
  assert names and names <= set(pa.ROLES)
  assert all(pa.ROLES[n] for n in names)


def test_freeze_check_refuses_a_changed_session_file(tmp_path, monkeypatch):
  import hashlib
  m = tmp_path / 'manifest_sha256.tsv'
  m.write_text('session\tpath\tbytes\tsha256\tcrc32c\nS1\ta/run.json\t3\t' + 'a' * 64 + '\tX\n')
  monkeypatch.setattr(pa, 'FREEZE', m)
  monkeypatch.setattr(pa, 'FREEZE_ID', hashlib.sha256(m.read_bytes()).hexdigest())
  assert pa.check_freeze([('results/af3/S1/a/run.json', 'a' * 64), ('inputs/manifest.csv', 'b' * 64)]) == \
      {'results/af3/S1/a/run.json'}
  with pytest.raises(SystemExit, match='differ from freeze'):
    pa.check_freeze([('results/af3/S1/a/run.json', 'c' * 64)])
  with pytest.raises(SystemExit, match='differ from freeze'):
    pa.check_freeze([('results/af3/S2/b.json', 'a' * 64)])
  monkeypatch.setattr(pa, 'FREEZE_ID', '0' * 64)
  with pytest.raises(SystemExit, match='is not freeze'):
    pa.check_freeze([])
