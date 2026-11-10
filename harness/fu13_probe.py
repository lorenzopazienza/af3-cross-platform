"""Follow-up 13 (label why): perturbation, layer-wise capture and float32 trunk, without editing AlphaFold3.

notes/analysis_plan.md section 10, row of 2026-11-10 (follow-up experiment 13), item a. harness/af3_entry.py imports
this module only when one of the AF3_FU13_* variables below is set in its environment (the harness configurations
tpu_xla_fu13_*, l4_xla_fu13_* in harness/configs.yaml set them), so every earlier configuration runs exactly as
before. Any other AF3_FU13_* name is refused. Each process must run one target and one seed (harness/run_plan.py
runs every (config, rep, seed) as its own harness run, harness/run_af3.py one process per target), and this module
refuses a process with more than one seed when the perturbation or the capture is on.

  AF3_FU13_BFLOAT16=none      AlphaFold3's own GlobalConfig.bfloat16 (src/alphafold3/model/model_config.py: 'all'
                              by default; 'intermediate' is rejected by the Evoformer's assert), set on the config
                              run_alphafold.make_model_config returns. 'none' runs the input embedding, the trunk
                              and the confidence head's Pairformer on float32 activations; the stored bfloat16 weights
                              are used as they are. Used with JAX_DEFAULT_MATMUL_PRECISION=highest (item a3).
  AF3_FU13_PERTURB_EPS=E      E = 0 or 2^-K (K a whole number, for example 2^-8): the relative size of the
  AF3_FU13_PERTURB_SEED=S     perturbation of the trunk output, and its seed (2026 in the row). Before the diffusion,
                              confidence and distogram heads read the trunk output, each of the single and pair arrays
                              X (float32) becomes X + E * (norm(m X) / norm(m Z)) * (m Z): m the token mask (single)
                              or its outer product (pair), Frobenius norms, Z standard normal of X's shape from
                              jax.random with key fold_in(fold_in(PRNGKey(S), h), seed), h the first 4 bytes of the
                              SHA-256 of the input's name (big-endian unsigned), seed the process's AlphaFold3 seed,
                              split once into the single key and the pair key (item a1). The wrapper edits the
                              embeddings dict that Model.__call__ passes to _sample_diffusion and then hands to the
                              confidence and distogram heads (src/alphafold3/model/model.py lines 321 to 343), so the
                              three heads read the same perturbed arrays; AlphaFold3's own random stream is not used.
  AF3_FU13_CAPTURE=1          the layer-wise capture (item a2): an ordered jax.debug.callback at the input of every
                              trunk Pairformer block and on the Evoformer output of every recycle. Depth d = 0 is the
                              input of the 48-block Pairformer, d = 1 to 47 the input of block d + 1 (the output of
                              block d), d = 48 the Evoformer's output. Kept: recycles 0 and the last (num_recycles),
                              single at every depth over the real tokens, pair at even depths on the token subset S
                              (every real non-protein token plus 64 protein tokens at positions
                              numpy.round(numpy.linspace(0, n_protein - 1, 64)) among the protein tokens in token
                              order; every protein token if there are at most 64), and the trunk output (d = 48) of
                              every recycle. Values as the trunk computes them: bfloat16 stored as its 16-bit
                              patterns (uint16), float32 as float32. Written to <run folder>/fu13_capture_seed-<S>.npz
                              (next to af3_output/, so the AlphaFold3 output folder holds exactly its usual files).

What was applied is written into af3_entry_stats.json under "fu13" (the dict RECORD).
"""

import hashlib
import json
import os
import re

import numpy as np

KNOWN = ('AF3_FU13_BFLOAT16', 'AF3_FU13_PERTURB_EPS', 'AF3_FU13_PERTURB_SEED', 'AF3_FU13_CAPTURE')
PAIR_PROTEIN_TOKENS = 64
PAIRFORMER_BLOCKS = 48
RECORD = {}
_STATE = {'name': None, 'seed': None, 'tokens': None, 'pair_idx': None, 'blocks': 0, 'recycles': 0,
          'events': [], 'kept': {}, 'errors': []}


def settings(environ):
  """The follow-up 13 settings of an environment, validated; {} if none is set."""
  ours = {k: v for k, v in environ.items() if k.startswith('AF3_FU13_')}
  unknown = sorted(set(ours) - set(KNOWN))
  if unknown:
    raise SystemExit(f'fu13_probe: unknown variable(s) {unknown}; known: {list(KNOWN)}')
  out = {}
  if 'AF3_FU13_BFLOAT16' in ours:
    if ours['AF3_FU13_BFLOAT16'] not in ('all', 'none'):
      raise SystemExit('fu13_probe: AF3_FU13_BFLOAT16 must be all or none (the Evoformer rejects intermediate)')
    out['bfloat16'] = ours['AF3_FU13_BFLOAT16']
  if ('AF3_FU13_PERTURB_EPS' in ours) != ('AF3_FU13_PERTURB_SEED' in ours):
    raise SystemExit('fu13_probe: AF3_FU13_PERTURB_EPS and AF3_FU13_PERTURB_SEED go together')
  if 'AF3_FU13_PERTURB_EPS' in ours:
    out['perturb_eps_text'] = ours['AF3_FU13_PERTURB_EPS']
    out['perturb_eps'] = parse_eps(ours['AF3_FU13_PERTURB_EPS'])
    if not re.fullmatch(r'\d+', ours['AF3_FU13_PERTURB_SEED']):
      raise SystemExit('fu13_probe: AF3_FU13_PERTURB_SEED must be a whole number')
    out['perturb_seed'] = int(ours['AF3_FU13_PERTURB_SEED'])
  if 'AF3_FU13_CAPTURE' in ours:
    if ours['AF3_FU13_CAPTURE'] != '1':
      raise SystemExit('fu13_probe: AF3_FU13_CAPTURE must be 1')
    out['capture'] = True
  return out


def parse_eps(text):
  """'0' -> 0.0, '2^-K' -> 2.0 ** -K (exact in float32 for the levels used)."""
  if text == '0':
    return 0.0
  m = re.fullmatch(r'2\^-(\d+)', text)
  if not m:
    raise SystemExit(f'fu13_probe: AF3_FU13_PERTURB_EPS {text!r}: 0 or 2^-K')
  return 2.0 ** -int(m.group(1))


def name_hash(name):
  """h of item a1: the first 4 bytes of the SHA-256 of the input's name, big-endian unsigned."""
  return int.from_bytes(hashlib.sha256(name.encode('utf-8')).digest()[:4], 'big')


def pair_subset(seq_mask, is_protein, k=PAIR_PROTEIN_TOKENS):
  """Indices of S (item a2), ascending: every real non-protein token plus k protein tokens, evenly spaced."""
  real = np.flatnonzero(np.asarray(seq_mask) > 0)
  prot = real[np.asarray(is_protein)[real] > 0]
  other = real[np.asarray(is_protein)[real] == 0]
  if len(prot) <= k:
    chosen = prot
  else:
    chosen = prot[np.round(np.linspace(0, len(prot) - 1, k)).astype(int)]
  return np.unique(np.concatenate([chosen, other])).astype(np.int32)


def perturb(x, mask, key, eps):
  """x + eps * (norm(m x) / norm(m z)) * (m z); x float32, mask broadcastable to x (jax arrays)."""
  import jax
  import jax.numpy as jnp
  z = jax.random.normal(key, x.shape, jnp.float32)
  m = jnp.broadcast_to(mask.astype(jnp.float32), x.shape)
  mz = z * m
  scale = jnp.float32(eps) * jnp.sqrt(jnp.sum(jnp.square(x * m))) / jnp.sqrt(jnp.sum(jnp.square(mz)))
  return x + scale * mz


def perturbation_keys(seed_value, name, af3_seed):
  import jax
  key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed_value), np.uint32(name_hash(name))),
                           np.uint32(af3_seed))
  ks, kp = jax.random.split(key)
  return ks, kp


def _host_value(a):
  a = np.asarray(a)
  if a.dtype.name == 'bfloat16':
    return a.view(np.uint16), 'bfloat16'
  return a, a.dtype.name


def _on_block(single, pair):
  """Host side of the ordered callback at a Pairformer block input (d = 0 to 47 of recycle r)."""
  s = _STATE
  r, d = divmod(s['blocks'], PAIRFORMER_BLOCKS)
  s['blocks'] += 1
  if r != s['recycles']:
    s['errors'].append(f'block event {s["blocks"] - 1} in recycle {r} before the output of recycle {s["recycles"]}')
  if r in (0, s['last']):
    s['kept'][f'single_r{r:02d}_d{d:02d}'] = _host_value(single)
    if d % 2 == 0:
      s['kept'][f'pair_r{r:02d}_d{d:02d}'] = _host_value(pair)


def _on_output(single, pair):
  """Host side of the ordered callback on the Evoformer output of recycle r (d = 48)."""
  s = _STATE
  r = s['recycles']
  s['recycles'] += 1
  if s['blocks'] != PAIRFORMER_BLOCKS * (r + 1):
    s['errors'].append(f'recycle {r} output after {s["blocks"]} block events (expected {PAIRFORMER_BLOCKS * (r + 1)})')
  s['kept'][f'single_r{r:02d}_d48'] = _host_value(single)
  s['kept'][f'pair_r{r:02d}_d48'] = _host_value(pair)


def _emit(fn, single, pair):
  """Sends the real-token single rows and the S x S pair block to the host, in program order."""
  import jax
  import jax.numpy as jnp
  n = _STATE['tokens']
  idx = jnp.asarray(_STATE['pair_idx'])
  pair_s = jnp.take(jnp.take(pair, idx, axis=0), idx, axis=1)
  jax.debug.callback(fn, single[:n], pair_s, ordered=True)


def install(run_alphafold, record=RECORD, environ=None):
  """Applies the settings of the environment to the imported run_alphafold module; returns them ({} = nothing)."""
  cfg = settings(os.environ if environ is None else environ)
  if not cfg:
    return cfg
  record.update({k: v for k, v in cfg.items()})
  from alphafold3.model import model as model_mod
  from alphafold3.model.network import evoformer as evoformer_mod
  from alphafold3.model.network import modules as modules_mod

  if 'bfloat16' in cfg:
    original_make = run_alphafold.make_model_config

    def make_model_config(*args, **kwargs):
      config = original_make(*args, **kwargs)
      config.global_config.bfloat16 = cfg['bfloat16']
      record['global_config_bfloat16'] = config.global_config.bfloat16
      return config
    run_alphafold.make_model_config = make_model_config

  original_predict = run_alphafold.predict_structure

  def predict_structure(fold_input, *args, **kwargs):
    seeds = list(fold_input.rng_seeds)
    if ('perturb_eps' in cfg or cfg.get('capture')) and len(seeds) != 1:
      raise SystemExit(f'fu13_probe: one seed per process, got {seeds}')
    _STATE['name'], _STATE['seed'] = fold_input.name, seeds[0]
    record.update({'input_name': fold_input.name, 'seed': seeds[0]})
    if 'perturb_eps' in cfg:
      record['name_hash'] = name_hash(fold_input.name)
    result = original_predict(fold_input, *args, **kwargs)
    if cfg.get('capture'):
      _write_capture(record)
    return result
  run_alphafold.predict_structure = predict_structure

  if 'perturb_eps' in cfg:
    original_sample = model_mod.Model._sample_diffusion

    def _sample_diffusion(self, batch, embeddings, *, sample_config):
      ks, kp = perturbation_keys(cfg['perturb_seed'], _STATE['name'], _STATE['seed'])
      mask = batch.token_features.mask
      embeddings['single'] = perturb(embeddings['single'], mask[:, None], ks, cfg['perturb_eps'])
      embeddings['pair'] = perturb(embeddings['pair'], (mask[:, None] * mask[None, :])[:, :, None], kp,
                                   cfg['perturb_eps'])
      record['perturbed'] = ['single', 'pair']
      return original_sample(self, batch, embeddings, sample_config=sample_config)
    model_mod.Model._sample_diffusion = _sample_diffusion

  if cfg.get('capture'):
    original_run = run_alphafold.ModelRunner.run_inference

    def run_inference(self, featurised_example, rng_key):
      _STATE['tokens'] = int(np.asarray(featurised_example['seq_mask']).sum())
      _STATE['pair_idx'] = pair_subset(featurised_example['seq_mask'], featurised_example['is_protein'])
      _STATE['last'] = int(self._model_config.num_recycles)
      record.update({'capture_tokens': _STATE['tokens'], 'capture_pair_tokens': len(_STATE['pair_idx']),
                     'capture_last_recycle': _STATE['last']})
      return original_run(self, featurised_example, rng_key)
    run_alphafold.ModelRunner.run_inference = run_inference

    base_pairformer = modules_mod.PairFormerIteration

    class CapturePairFormerIteration(base_pairformer):
      def __call__(self, act, pair_mask, single_act=None, seq_mask=None):
        if self.name == 'trunk_pairformer':
          _emit(_on_block, single_act, act)
        return super().__call__(act, pair_mask, single_act=single_act, seq_mask=seq_mask)
    modules_mod.PairFormerIteration = CapturePairFormerIteration

    base_evoformer = evoformer_mod.Evoformer

    class CaptureEvoformer(base_evoformer):
      def __call__(self, batch, prev, target_feat, key):
        out = super().__call__(batch=batch, prev=prev, target_feat=target_feat, key=key)
        _emit(_on_output, out['single'], out['pair'])
        return out
    evoformer_mod.Evoformer = CaptureEvoformer
  return cfg


def _write_capture(record):
  import jax
  jax.effects_barrier()            # every ordered callback has reached the host
  s = _STATE
  out_dir = os.path.dirname(os.path.normpath(str(run_alphafold_output_dir())))
  path = os.path.join(out_dir, f'fu13_capture_seed-{s["seed"]}.npz')
  arrays = {k: v[0] for k, v in s['kept'].items()}
  dtypes = {k: v[1] for k, v in s['kept'].items()}
  meta = {'input_name': s['name'], 'seed': s['seed'], 'tokens': s['tokens'], 'last_recycle': s['last'],
          'block_events': s['blocks'], 'recycle_events': s['recycles'], 'dtypes': dtypes, 'errors': s['errors']}
  np.savez(path, pair_tokens=s['pair_idx'], meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
  record.update({'capture_file': os.path.basename(path), 'capture_arrays': len(arrays),
                 'capture_block_events': s['blocks'], 'capture_recycle_events': s['recycles'],
                 'capture_errors': s['errors']})


def run_alphafold_output_dir():
  import run_alphafold
  return run_alphafold._OUTPUT_DIR.value
