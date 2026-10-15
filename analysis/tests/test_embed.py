"""analysis/embed.py (follow-up experiment 2, section 10, 2026-10-14) on synthetic arrays and session folders."""

import json
import math

import numpy as np
import pandas as pd
import pytest

import embed


def test_rel_l2():
  a = np.array([[3.0, 4.0], [0.0, 0.0]])
  assert embed.rel_l2(a, a) == 0.0
  assert embed.rel_l2(a, a * 1.1) == pytest.approx(0.1)
  assert embed.rel_l2(a, np.zeros_like(a)) == pytest.approx(1.0)
  assert embed.rel_l2(np.zeros(3), np.ones(3)) == math.inf
  with pytest.raises(ValueError):
    embed.rel_l2(np.zeros(3), np.zeros(4))


def test_token_max_single_and_pair():
  a = np.ones((4, 384))
  b = a.copy()
  b[2] *= 1.5                                   # one token 50% off
  assert embed.token_max(a, b) == pytest.approx(0.5)
  assert embed.rel_l2(a, b) == pytest.approx(0.25)   # sqrt(1/4) * 0.5
  p = np.ones((3, 3, 128))
  q = p.copy()
  q[0, 2, :64] = -1                             # one token pair: half its channels flipped
  assert embed.token_max(p, q) == pytest.approx(np.sqrt(64 * 4) / np.sqrt(128))
  assert embed.token_max(p, p) == 0.0
  z = np.zeros((2, 4))
  assert embed.token_max(z, z) == 0.0 and embed.token_max(z, z + 1) == math.inf


def test_compare_embeddings_float16_identity():
  rng = np.random.default_rng(0)
  e = {'single_embeddings': rng.normal(size=(5, 384)).astype(np.float16),
       'pair_embeddings': rng.normal(size=(5, 5, 128)).astype(np.float16)}
  same = embed.compare_embeddings(e, {k: v.copy() for k, v in e.items()})
  assert same['single_identical'] and same['pair_identical'] and same['pair_rel_l2'] == 0.0
  other = {k: v.copy() for k, v in e.items()}
  other['pair_embeddings'][1, 1, 0] = np.float16(other['pair_embeddings'][1, 1, 0] + np.float16(0.5))
  d = embed.compare_embeddings(e, other)
  assert d['single_identical'] and not d['pair_identical'] and 0 < d['pair_rel_l2'] < d['pair_token_max']


def test_load_embeddings(tmp_path):
  p = tmp_path / 'x_embeddings.npz'
  np.savez_compressed(p, single_embeddings=np.ones((3, 384), np.float16), pair_embeddings=np.ones((3, 3, 128), np.float16))
  e = embed.load_embeddings(p)
  assert e['single_embeddings'].shape == (3, 384) and e['pair_embeddings'].dtype == np.float16


def _process(root, session, run, config, target, exit_code=0):
  d = root / session / run
  (d / target / 'af3_output' / target).mkdir(parents=True)
  (d / 'session.json').write_text(json.dumps({'label': 'embed', 'config_name': config}))
  (d / target / 'run.json').write_text(json.dumps({'pdb_id': target, 'exit_code': exit_code}))
  return d / target / 'af3_output' / target


def test_processes_maps_sessions_to_conditions(tmp_path):
  _process(tmp_path, '20261015T000000Z_v5e_embed_v5', 'tpu_xla_rep1_seed1_fresh', 'tpu_xla', '7U3J')
  _process(tmp_path, '20261015T000000Z_v6e_embed_v6', 'tpu_xla_rep1_seed1_fresh', 'tpu_xla', '7U3J')
  _process(tmp_path, '20261015T000000Z_l4_embed_g', 'l4_xla_rep2_seed1_fresh', 'l4_xla', '7U3J')
  _process(tmp_path, '20261015T000000Z_l4_embed_gd', 'l4_xla_det_rep1_seed1_fresh', 'l4_xla_det', '7U3J')
  _process(tmp_path, '20261015T000000Z_l4_embed_gd', 'l4_xla_det_rep1_seed1_fresh', 'l4_xla_det', '7BTT', exit_code=1)
  assert sorted(embed.processes(tmp_path)) == [('G', 'r2', '7U3J'), ('Gd', 'r1', '7U3J'), ('V5', 'r1', '7U3J'),
                                               ('V6', 'r1', '7U3J')]


def test_versus_main_compares_outputs_byte_for_byte(tmp_path):
  def outputs(folder, score, stamp):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / '7U3J_ranking_scores.csv').write_text(f'seed,sample,ranking_score\n1,0,{score}\n')
    for k in embed.SAMPLES:
      s = folder / f'seed-1_sample-{k}'
      s.mkdir(exist_ok=True)
      (s / f'7U3J_seed-1_sample-{k}_model.cif').write_text(f'data_x\n_ma_model_list.model_group_name "{stamp}"\nATOM 1\n')
      (s / f'7U3J_seed-1_sample-{k}_summary_confidences.json').write_text('{}')
  main = tmp_path / 'main' / '7U3J'
  outputs(main, 0.9, 'AlphaFold 2026-10-13')
  same = tmp_path / 'emb_same'
  outputs(same, 0.9, 'AlphaFold 2026-10-15')          # only the timestamp line differs
  other = tmp_path / 'emb_other'
  outputs(other, 0.8, 'AlphaFold 2026-10-15')
  csv = tmp_path / 'samples.csv'
  pd.DataFrame([{'platform': 'V6', 'repetition': 'r1', 'target': '7U3J', 'seed': 1, 'session': 'main_s',
                 'model_path': str(main / 'seed-1_sample-0' / '7U3J_seed-1_sample-0_model.cif')}]).to_csv(csv, index=False)
  vm = embed.versus_main({('V6', 'r1', '7U3J'): same}, csv)
  assert bool(vm.identical.iloc[0]) and vm.compared_files.iloc[0] == 11
  vm = embed.versus_main({('V6', 'r1', '7U3J'): other, ('V5', 'r1', '7U3J'): same}, csv)
  v = vm.set_index('condition')
  assert v.loc['V6', 'differ'] == '7U3J_ranking_scores.csv' and not v.loc['V6', 'identical']
  assert v.loc['V5', 'identical'] is None                  # no main-study run of that condition in the table


def test_embeddings_path_is_alphafold3s_name(tmp_path):
  # third_party/alphafold3/run_alphafold.py writes seed-<S>_embeddings/<job>_seed-<S>_embeddings.npz.
  assert embed.embeddings_path(tmp_path, '7U3J') == tmp_path / 'seed-1_embeddings' / '7U3J_seed-1_embeddings.npz'


def test_summarise_per_pair():
  import pandas as pd
  pair = 'G r1 - G r2'
  emb = pd.DataFrame([{'pair': pair, 'target': t, 'tokens': 10, 'single_rel_l2': v, 'single_token_max': v,
                       'pair_rel_l2': v, 'pair_token_max': v, 'single_identical': v == 0, 'pair_identical': v == 0}
                      for t, v in (('A', 0.0), ('B', 0.2), ('C', 0.4))])
  st = pd.DataFrame([{'pair': pair, 'target': t, 'm4_ca_rmsd_median': v, 'm4_ligand_rmsd_median': v,
                      'm5_ranking_score_median': v, 'samples_bit_identical': 5 if v == 0 else 0, 'same_top1': v < 0.3}
                     for t, v in (('A', 0.0), ('B', 0.2), ('C', 0.4))])
  s = embed.summarise(emb, st)
  assert list(s.pair) == [pair]
  r = s.iloc[0]
  assert r.targets == 3 and r.single_rel_l2_median == 0.2 and r.pair_token_max_max == 0.4
  assert r.single_identical_targets == 1 and r.pair_identical_targets == 1
  assert r.samples_bit_identical == 5 and r.samples == 15 and r.same_top1_targets == 2
