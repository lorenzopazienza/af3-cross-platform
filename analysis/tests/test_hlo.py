"""Follow-up 16 (label hlo): analysis/hlo.py on synthetic HLO dumps, the plans, the configuration, the hook in
harness/af3_entry.py, the budget rules and the session check."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import hlo  # noqa: E402
from run_af3 import read_configs  # noqa: E402

# ------------------------------------------------------------------------------------------------ synthetic HLO

TABLES = '''FileNames
1 "/af3/src/alphafold3/model/network/evoformer.py"
2 "/af3/src/alphafold3/model/network/diffusion_head.py"
3 "/af3/src/alphafold3/model/model.py"

FunctionNames
1 "Evoformer.__call__"
2 "Evoformer._embed_process_msa"
3 "sample"
4 "Model._sample_diffusion"

FileLocations
1 {file_name_id=1 function_name_id=1 line=260 end_line=260 column=4 end_column=9}
2 {file_name_id=1 function_name_id=2 line=220 end_line=220 column=4 end_column=9}
3 {file_name_id=2 function_name_id=3 line=300 end_line=300 column=4 end_column=9}
4 {file_name_id=3 function_name_id=4 line=110 end_line=110 column=4 end_column=9}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=4 parent_frame_id=1}
4 {file_location_id=3 parent_frame_id=4}

'''

MD = 'metadata={{op_name="jit(apply_fn)/diffuser/{p}" stack_frame_id={f}}}'


def md(path, frame=1):
  return MD.format(p=path, f=frame)


def module(chip='v5e', variant=None):
  """A small optimised module with one unit per stage of interest; variant changes one stage:
  'layout_msa', 'fusion_pairformer', 'types_diffusion', 'shapes_diffusion', 'backend_confidence', 'embedding'."""
  v = variant or ''
  lay = '{0,1}' if v == 'layout_msa' else '{1,0}'
  red_t = 'bf16' if v == 'types_diffusion' else 'f32'
  red_dims = '{0}' if v == 'shapes_diffusion' else '{1}'
  red_out = 'f32[4]' if v == 'shapes_diffusion' else 'f32[8]'
  be = '{"tiling":"256x256"}' if v == 'backend_confidence' else '{"tiling":"128x128"}'
  emb_op = 'multiply' if v == 'embedding' else 'add'
  pf_fused = v != 'fusion_pairformer'
  pf = (f'''
%fused_pf (p0: f32[8,8], p1: f32[8,8]) -> f32[8,8] {{
  %p0 = f32[8,8]{{1,0}} parameter(0)
  %p1 = f32[8,8]{{1,0}} parameter(1)
  %m = f32[8,8]{{1,0}} multiply(%p0, %p1), {md("evoformer/__layer_stack_no_per_layer/trunk_pairformer/triangle/mul")}
  ROOT %e = f32[8,8]{{1,0}} exponential(%m), {md("evoformer/__layer_stack_no_per_layer/trunk_pairformer/triangle/exp")}
}}
''' if pf_fused else f'''
%fused_pf (p0: f32[8,8], p1: f32[8,8]) -> f32[8,8] {{
  %p0 = f32[8,8]{{1,0}} parameter(0)
  %p1 = f32[8,8]{{1,0}} parameter(1)
  ROOT %m = f32[8,8]{{1,0}} multiply(%p0, %p1), {md("evoformer/__layer_stack_no_per_layer/trunk_pairformer/triangle/mul")}
}}
''')
  pf_entry = ('  %pf = f32[8,8]{1,0} fusion(%emb, %emb), kind=kLoop, calls=%fused_pf, '
              + md('evoformer/__layer_stack_no_per_layer/trunk_pairformer/triangle/exp') + '\n') if pf_fused else (
      '  %pf0 = f32[8,8]{1,0} fusion(%emb, %emb), kind=kLoop, calls=%fused_pf, '
      + md('evoformer/__layer_stack_no_per_layer/trunk_pairformer/triangle/mul') + '\n'
      + '  %pf = f32[8,8]{1,0} exponential(%pf0), '
      + md('evoformer/__layer_stack_no_per_layer/trunk_pairformer/triangle/exp') + '\n')
  return (f'HloModule jit_apply_fn, is_scheduled=true, entry_computation_layout={{(f32[8,8]{{1,0}})->f32[8]{{0}}}}\n\n'
          + TABLES + f'''
%region_add ({red_t[0]}a: {red_t}[], b: {red_t}[]) -> {red_t}[] {{
  %a = {red_t}[] parameter(0)
  %b = {red_t}[] parameter(1)
  ROOT %s = {red_t}[] add(%a, %b)
}}
''' + pf + f'''
ENTRY %main.1 (x.1: f32[8,8]) -> f32[8] {{
  %x.1 = f32[8,8]{{1,0}} parameter(0), metadata={{op_name="x"}}
  %emb = f32[8,8]{{1,0}} {emb_op}(%x.1, %x.1), {md("evoformer/left_single/add")}
  %msa = f32[8,8]{lay} transpose(%emb), dimensions={{1,0}}, {md("evoformer/msa_stack/transpose", 2)}
  %shuf = f32[8,8]{{1,0}} negate(%msa), metadata={{op_name="jit(apply_fn)/diffuser/evoformer/neg" stack_frame_id=2}}
''' + pf_entry + f'''  %d0 = {red_t}[8,8]{{1,0}} convert(%pf), {md("diffusion_head/convert", 4)}
  %c0 = {red_t}[] constant(0)
  %red = {red_out}{{0}} reduce(%d0, %c0), dimensions={red_dims}, to_apply=%region_add, {md("sample/reduce_sum", 4)}
  %glue = f32[8,8]{{1,0}} add(%shuf, %pf), metadata={{op_name="jit(apply_fn)/diffuser/add"}}
  %conf = f32[8,8]{{1,0}} custom-call(%glue), custom_call_target="tpu_custom", {md("confidence_head/pae_logits/custom")}, backend_config={be}
  %nometa = f32[8,8]{{1,0}} sqrt(%conf)
  %r8 = f32[8]{{0}} reduce(%nometa, %c0), dimensions={{1}}, to_apply=%region_add, {md("confidence_head/sum")}
  ROOT %out = (f32[8]{{0}}, /*index=1*/{red_out}{{0}}) tuple(%r8, %red)
}}
''')


def units(text):
  return hlo.units(hlo.Module(text))


# ------------------------------------------------------------------------------------------------ parsing

def test_parser_reads_computations_tables_and_attributes():
  m = hlo.Module(module())
  assert m.entry == 'main.1'
  assert set(m.comps) == {'region_add', 'fused_pf', 'main.1'}
  assert m.files[2].endswith('diffusion_head.py') and m.functions[4] == 'Model._sample_diffusion'
  assert [f for _, f, _ in m.stack(4)] == ['sample', 'Model._sample_diffusion']
  red = m.by_name[('main.1', 'red')]
  assert red.opcode == 'reduce' and red.operands == ['d0', 'c0'] and red.attrs['dimensions'] == '{1}'
  assert red.op_name == 'jit(apply_fn)/diffuser/sample/reduce_sum' and red.frame == 4
  out = m.by_name[('main.1', 'out')]
  assert out.opcode == 'tuple' and out.root and out.shape.startswith('(f32[8]')
  conf = m.by_name[('main.1', 'conf')]
  assert conf.attrs['backend_config'] == '{"tiling":"128x128"}' and conf.attrs['custom_call_target'] == '"tpu_custom"'


def test_parser_takes_operands_printed_with_shapes_and_quoted_commas():
  line = ('  %f = bf16[2,3]{1,0:T(8,128)(2,1)} fusion(f32[2,3]{1,0} %a, s32[] %b), kind=kOutput, calls=%c, '
          'metadata={op_name="jit(apply_fn)/diffuser/x/\\"y, z\\"" stack_frame_id=7}, '
          'backend_config="{\\"k\\": [1, 2]}"')
  i = hlo.parse_instruction('c0', line)
  assert i.operands == ['a', 'b'] and i.attrs['kind'] == 'kOutput' and i.frame == 7
  assert hlo.logical(i.shape) == 'bf16[2,3]'
  assert hlo.parse_instruction('c0', '  %k = f32[] constant(0.5)').attrs['literal'] == '0.5'


def test_logical_shapes_drop_layouts_and_tiling():
  assert hlo.logical('f32[4,8]{1,0}') == 'f32[4,8]'
  assert hlo.logical('(f32[4]{0}, bf16[2,2]{0,1:T(8,128)(2,1)})') == '(f32[4], bf16[2,2])'
  assert hlo.dtypes('(f32[4]{0}, bf16[2,2]{1,0})') == ['f32', 'bf16']


def test_gzip_dumps_are_read(tmp_path):
  import gzip
  p = tmp_path / 'module_0001.jit_apply_fn.after_optimizations.txt.gz'
  with gzip.open(p, 'wt') as f:
    f.write(module())
  assert hlo.read_text(p) == module()


# ------------------------------------------------------------------------------------------------ stages

@pytest.mark.parametrize('op_name,stack,stage', [
    ('jit(apply_fn)/diffuser/evoformer/__layer_stack_no_per_layer/trunk_pairformer/x/dot_general', [], 'pairformer'),
    ('jit(apply_fn)/diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/x/add', [], 'msa'),
    ('jit(apply_fn)/diffuser/evoformer/msa_activations/dot_general', [], 'msa'),
    ('jit(apply_fn)/diffuser/evoformer/neg', [('evoformer.py', 'Evoformer._embed_process_msa', 1)], 'msa'),
    ('jit(apply_fn)/diffuser/evoformer/template_embedding/single_template_embedding/add', [], 'template'),
    ('jit(apply_fn)/diffuser/evoformer/left_single/dot_general', [], 'embedding'),
    ('jit(apply_fn)/diffuser/evoformer_conditioning/x/add', [], 'embedding'),
    ('jit(apply_fn)/diffuser/diffusion_head/diffusion/x/add', [], 'diffusion'),
    ('jit(apply_fn)/diffuser/while/body/add', [('model.py', 'Model._sample_diffusion', 110)], 'diffusion'),
    ('jit(apply_fn)/diffuser/confidence_head/confidence_pairformer/x/add', [], 'confidence'),
    ('jit(apply_fn)/diffuser/while/body/closed_call/vmap(confidence_head)/while/body/confidence_pairformer/x/add', [],
     'confidence'),
    ('jit(apply_fn)/diffuser/evoformer_conditioning_atom_transformer_encoder/x/add', [], 'embedding'),
    ('jit(apply_fn)/diffuser/x/add', [('stateful.py', 'scan', 644), ('confidence_head.py', 'pairformer_fn', 146)],
     'confidence'),
    ('jit(apply_fn)/diffuser/distogram_head/half_logits/dot_general', [], 'distogram'),
    ("params['diffuser/evoformer/left_single']['weights']", [], 'params'),
    ('jit(apply_fn)/diffuser/add', [], 'other'),
    ('', [], None),
])
def test_stage_of(op_name, stack, stage):
  assert hlo.stage_of(op_name, stack) == stage


def test_module_path_drops_transformations_and_the_primitive():
  assert hlo.module_path('jit(apply_fn)/diffuser/evoformer/while/body/trunk_pairformer/triangle_multiplication/'
                         'jit(relu)/max') == 'evoformer/trunk_pairformer/triangle_multiplication'


def test_units_attribute_stages_with_inheritance():
  us = {(u['stage'], u['opcode']) for u in units(module())}
  assert ('embedding', 'add') in us and ('msa', 'transpose') in us and ('msa', 'negate') in us
  assert ('pairformer', 'multiply') in us and ('pairformer', 'exponential') in us and ('pairformer', 'fusion') in us
  assert ('diffusion', 'reduce') in us and ('diffusion', 'convert') in us
  assert ('confidence', 'sqrt') in us          # no metadata: inherits from its operand %conf
  assert all(u['opcode'] != 'add' or u['stage'] != 'unattributed' for u in units(module()))
  assert not any(u['stage'] == 'diffusion' and u['opcode'] == 'add' and u['semantic'] is None for u in units(module()))


def test_reducer_computations_enter_the_signature_not_the_units():
  us = units(module())
  red = [u for u in us if u['opcode'] == 'reduce' and u['stage'] == 'diffusion'][0]
  assert 'reducer=add:f32' in red['semantic'] and 'reducer=add:f32' in red['types']
  assert not [u for u in us if u['opcode'] == 'parameter' and u['shape'].startswith('f32[]')]


# ------------------------------------------------------------------------------------------------ levels

def levels(variant):
  return {s: v['level'] for s, v in hlo.compare(units(module()), units(module(variant=variant))).items()}


def test_identical_modules_are_identical_everywhere():
  assert set(levels(None).values()) == {'identical'}


def test_each_kind_of_difference_lands_at_its_level_and_stage():
  lv = levels('layout_msa')
  assert lv['msa'] == 'layout' and lv['embedding'] == lv['pairformer'] == lv['diffusion'] == 'identical'
  lv = levels('fusion_pairformer')
  assert lv['pairformer'] == 'fusion' and lv['msa'] == 'identical'
  lv = levels('backend_confidence')
  assert lv['confidence'] == 'backend' and lv['diffusion'] == 'identical'
  c = hlo.compare(units(module()), units(module(variant='types_diffusion')))
  assert c['diffusion']['level'] == 'semantic' and c['diffusion']['detail'] == 'types'
  c = hlo.compare(units(module()), units(module(variant='shapes_diffusion')))
  assert c['diffusion']['level'] == 'semantic' and c['diffusion']['detail'] == 'shapes'
  assert levels('embedding')['embedding'] == 'semantic'


def test_names_and_metadata_do_not_count():
  a = module()
  b = a.replace('%emb', '%emb_renamed').replace('left_single/add', 'left_single/add_other_name')
  assert set(levels_between(a, b).values()) == {'identical'}


def levels_between(a, b):
  return {s: v['level'] for s, v in hlo.compare(units(a), units(b)).items()}


def test_question_rule_e3():
  base = units(module())
  assert hlo.question(hlo.compare(base, units(module()))) == 'no'
  assert hlo.question(hlo.compare(base, units(module(variant='layout_msa')))) == 'layout only'
  assert hlo.question(hlo.compare(base, units(module(variant='embedding')))) == 'yes'
  assert hlo.question(hlo.compare(base, units(module(variant='fusion_pairformer')))) == 'no'
  lv = lambda **k: {s: {'level': v} for s, v in k.items()}  # noqa: E731
  assert hlo.question(lv(embedding='backend', msa='identical')) == 'backend only'
  assert hlo.question(lv(embedding='layout', msa='backend', template='identical')) == 'layout only'
  assert hlo.question(lv(embedding='layout', msa='fusion')) == 'yes'
  assert hlo.question(lv(template='semantic', msa='layout')) == 'yes'
  assert hlo.question(lv(embedding='identical', template='identical', msa='identical', pairformer='semantic')) == 'no'
  four = {256: 'yes', 512: 'yes', 768: 'yes', 1024: 'yes'}
  assert hlo.overall(four) == 'yes'
  mixed = hlo.overall({**four, 768: 'layout only'})
  assert mixed.startswith('mixed') and '768: layout only' in mixed and '256: yes' in mixed


def test_module_diffs_name_the_module_path():
  # the variant changes the reducer computation that the diffusion and confidence reductions share, the convert and
  # the constant: those, and only those, differ
  d = hlo.module_diffs(units(module()), units(module(variant='types_diffusion')))
  assert set(d) == {('diffusion', 'sample', 1, 1), ('diffusion', 'diffusion_head', 1, 1),
                    ('confidence', 'confidence_head', 1, 1), ('unattributed', '', 1, 1)}
  d = hlo.module_diffs(units(module()), units(module(variant='layout_msa')), 'layout')
  assert d == [('msa', 'evoformer', 1, 1)]


# ------------------------------------------------------------------------------------------------ end to end

def make_tree(tmp_path, variants, rep2_variant=None, outputs_ok=True):
  """results/af3 with two hlo sessions and the main-study references; variants: {chip: variant}."""
  af3 = tmp_path / 'af3'
  sessions = {'v5e': '20261117T080000Z_v5e_hlo_v5_01', 'v6e': '20261117T080000Z_v6e_hlo_v6_01'}
  main = {'v5e': '20261013T132954Z_v5e_main_v5_01', 'v6e': '20261013T205257Z_v6e_main_v6_01'}
  for chip, s in sessions.items():
    for rep in hlo.REPS:
      for t, b in hlo.TARGETS.items():
        d = af3 / s / hlo.RUN.format(rep=rep) / t
        (d / 'xla_dump').mkdir(parents=True)
        (d / 'af3_output' / t).mkdir(parents=True)
        (d / 'af3_output' / t / 'x.txt').write_text('same' if outputs_ok else 'other')
        (d / 'run.json').write_text(json.dumps({'exit_code': 0, 'device_kind': hlo.DEVICE[chip],
                                                'entry_stats': {'hlo_dump': {'xla_flags': hlo.XLA_FLAGS}}}))
        (d.parent / 'session.json').write_text(json.dumps({
            'probe': {'packages': dict(hlo.STACK), 'python': '3.12.16 (main)'},
            'tpu_runtime': {'build_label': hlo.LIBTPU_BUILD}, 'host': {'gce_accelerator_type': 'x'}}))
        v = rep2_variant if (rep == 2 and rep2_variant and chip == 'v6e') else variants.get(chip)
        (d / 'xla_dump' / 'module_0101.jit_apply_fn.tpu_after_optimizations.txt').write_text(module(chip, v))
        (d / 'xla_dump' / 'module_0101.jit_apply_fn.tpu_after_optimizations-buffer-assignment.txt').write_text('x')
        (d / 'xla_dump' / 'module_0101.jit_apply_fn.before_optimizations.txt').write_text(module(chip, None))
    for t in hlo.TARGETS:
      r = af3 / main[chip] / hlo.REF_RUN / t / 'af3_output' / t
      r.mkdir(parents=True)
      (r / 'x.txt').write_text('same')
  sm = tmp_path / 'sessions_main.txt'
  sm.write_text('\n'.join(main.values()) + '\n20261013T143640Z_v6e_main_v6h_01\n')
  return af3, sm, list(sessions.values())


def fake_identical(a, b, seed):
  same = (pathlib.Path(a) / 'x.txt').read_text() == (pathlib.Path(b) / 'x.txt').read_text()
  return {'status': 'identical' if same else 'differ', 'differ': [] if same else ['x.txt'], 'missing': []}


def test_end_to_end_valid_run(tmp_path):
  af3, sm, sessions = make_tree(tmp_path, {'v6e': 'layout_msa'})
  procs = hlo.processes(sessions, af3)
  assert len(procs) == 16
  rows = hlo.checks(procs, af3, sm, fake_identical)
  assert all(r['check1'] == r['check2'] == r['check3'] == r['check4'] == 'pass' for r in rows)
  assert rows[0]['reference'].startswith('20261013T') and rows[0]['python'] == '3.12.16'
  valid, q = hlo.analyse(procs, tmp_path / 'out', rows)
  assert valid and q == {256: 'layout only', 512: 'layout only', 768: 'layout only', 1024: 'layout only'}
  text = (tmp_path / 'out' / 'summary.md').read_text()
  assert 'Rule e1: VALID' in text and '- overall: layout only' in text
  det = (tmp_path / 'out' / 'determinism.csv').read_text()
  assert det.count('identical') == 8
  assert 'identical' in (tmp_path / 'out' / 'unoptimised.csv').read_text()
  for f in ('stages', 'question', 'opcodes', 'fusions', 'dtypes', 'dots', 'reductions', 'module_diffs', 'checks'):
    assert (tmp_path / 'out' / f'{f}.csv').is_file()
  assert 'reduce' in (tmp_path / 'out' / 'reductions.csv').read_text()


def test_outputs_that_differ_make_the_run_invalid(tmp_path):
  af3, sm, sessions = make_tree(tmp_path, {}, outputs_ok=False)
  procs = hlo.processes(sessions, af3)
  rows = hlo.checks(procs, af3, sm, fake_identical)
  assert all(r['check2'] == 'FAIL' for r in rows)
  valid, _ = hlo.analyse(procs, tmp_path / 'out', rows)
  assert not valid and 'INVALID' in (tmp_path / 'out' / 'summary.md').read_text()


def test_a_missing_dump_fails_check_1(tmp_path):
  af3, sm, sessions = make_tree(tmp_path, {})
  for p in (af3 / sessions[0]).rglob('*after_optimizations.txt'):
    p.unlink()
    break
  rows = hlo.checks(hlo.processes(sessions, af3), af3, sm, fake_identical)
  assert sum(r['check1'] == 'FAIL' for r in rows) == 1


def test_within_chip_differences_are_reported_and_qualify_the_cross_chip_level(tmp_path):
  af3, sm, sessions = make_tree(tmp_path, {'v6e': 'layout_msa'}, rep2_variant='fusion_pairformer')
  procs = hlo.processes(sessions, af3)
  hlo.analyse(procs, tmp_path / 'out', hlo.checks(procs, af3, sm, fake_identical))
  det = (tmp_path / 'out' / 'determinism.csv').read_text()
  assert 'differ' in det and 'pairformer fusion' in det
  stages = (tmp_path / 'out' / 'stages.csv').read_text()
  assert re.search(r'\nmsa|,msa,layout', stages)


def test_reference_is_the_main_study_r1_of_the_same_chip(tmp_path):
  af3, sm, _ = make_tree(tmp_path, {})
  assert '_v6e_main_v6_01' in str(hlo.reference_dir('v6e', '7D5C', af3, sm))
  assert '_v5e_main_v5_01' in str(hlo.reference_dir('v5e', '7U3J', af3, sm))


def test_main_needs_one_session_per_chip(tmp_path):
  f = tmp_path / 's.txt'
  f.write_text('# header\n20261117T080000Z_v5e_hlo_v5_01\n')
  out = subprocess.run([sys.executable, str(REPO / 'analysis' / 'hlo.py'), '--sessions', str(f), '--out',
                        str(tmp_path / 'o')], capture_output=True, text=True)
  assert out.returncode != 0 and 'one v5e and one v6e' in out.stderr


# ------------------------------------------------------------------------------------------------ plans, configuration

def test_configuration_is_tpu_xla_plus_the_dump_variable_only():
  c = read_configs(REPO / 'harness' / 'configs.yaml')
  new, base = dict(c['tpu_xla_hlo']), dict(c['tpu_xla'])
  env = new.pop('env')
  new.pop('description'), base.pop('description')
  assert new == base
  assert env == {'AF3_HLO_DUMP_FLAGS': '--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn'}


def test_plans_are_up_to_date_and_as_specified():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_hlo_plans.py'), '--check'],
                       capture_output=True, text=True, cwd=REPO)
  assert out.returncode == 0, out.stdout
  plans = sorted((REPO / 'harness' / 'plans' / 'hlo').glob('*.yaml'))
  assert [p.name for p in plans] == ['hlo_v5_01.yaml', 'hlo_v6_01.yaml']
  for p, pl in zip(plans, ('v5e', 'v6e')):
    t = p.read_text()
    assert f'configs:\n  {pl}: tpu_xla_hlo\n' in t and 'default: [7U3J, 7NPL, 7XQZ, 7D5C]' in t
    assert 'seeds: [1]\nfresh_reps: 2\nwarm_rerun_seed: none\n' in t and 'num_recycles: default' in t
    assert 'label: hlo' in t and 'eigen_source: archive' in t and 'launch_by: 2026-11-17' in t


def test_targets_are_the_pilot_target_with_the_most_tokens_of_each_bucket():
  import csv
  sys.path.insert(0, str(REPO / 'harness' / 'plans'))
  import make_hlo_plans as mk
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  assert mk.pilot_targets(manifest) == mk.TARGETS == list(hlo.TARGETS) == ['7U3J', '7NPL', '7XQZ', '7D5C']
  assert [int(manifest[k]['bucket']) for k in mk.TARGETS] == list(hlo.TARGETS.values()) == [256, 512, 768, 1024]


def test_parts_and_cost_within_the_cap():
  import csv
  ps = list(csv.DictReader(open(REPO / 'harness' / 'plans' / 'hlo_parts.csv')))
  assert {p['provisioning'] for p in ps} == {'standard'}
  assert sum(float(p['lifetime_usd_planned']) for p in ps) <= 15
  assert {p['platform']: float(p['usd_per_h_on_demand']) for p in ps} == {'v5e': 1.56, 'v6e': 2.97}


def test_af3_entry_sets_the_dump_before_importing_alphafold3_and_only_under_the_variable():
  src = (REPO / 'harness' / 'af3_entry.py').read_text()
  assert src.index("if os.environ.get('AF3_HLO_DUMP_FLAGS')") < src.index('import run_alphafold')
  assert src.index("if os.environ.get('AF3_HLO_DUMP_FLAGS')") < src.index('from absl import app')
  assert "if _HLO:    # follow-up 16 only" in src


AF3_PY = REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python'
FAKE_AF3 = '''from absl import flags
import jax, jax.numpy as jnp, numpy as np, os
_OUT = flags.DEFINE_string('output_dir', None, 'out')
def main(argv):
  def apply_fn(x):
    return jnp.tanh(x @ x.T).astype(jnp.bfloat16).sum(-1)
  y = jax.jit(apply_fn)(jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64)
  np.save(os.path.join(_OUT.value, 'y.npy'), np.asarray(y, np.float32))
'''


@pytest.mark.skipif(not AF3_PY.is_file(), reason='no AlphaFold3 venv (jax 0.10.2) here')
def test_hook_dumps_into_the_process_folder_and_leaves_the_result_unchanged(tmp_path):
  af3 = tmp_path / 'af3'
  af3.mkdir()
  (af3 / 'run_alphafold.py').write_text(FAKE_AF3)
  entry = str(REPO / 'harness' / 'af3_entry.py')
  base = {k: v for k, v in os.environ.items() if not k.startswith(('XLA_', 'AF3_'))}
  base['JAX_PLATFORMS'] = 'cpu'
  results = {}
  for name, extra in (('plain', {}), ('dump', {'AF3_HLO_DUMP_FLAGS': '--xla_dump_hlo_as_text '
                                               '--xla_dump_hlo_module_re=jit_apply_fn'})):
    proc = tmp_path / name
    proc.mkdir()
    env = dict(base, AF3_ENTRY_STATS=f'../{name}/af3_entry_stats.json', **extra)
    r = subprocess.run([str(AF3_PY), entry, f'--output_dir=../{name}'], cwd=af3, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    results[name] = (proc / 'y.npy').read_bytes()
    stats = json.loads((proc / 'af3_entry_stats.json').read_text())
    if name == 'plain':
      assert 'hlo_dump' not in stats and not (proc / 'xla_dump').exists()
    else:
      h = stats['hlo_dump']
      assert h['xla_flags'] == ('--xla_dump_to=<process folder>/xla_dump --xla_dump_hlo_as_text '
                                '--xla_dump_hlo_module_re=jit_apply_fn')
      assert len(h['model_after_optimizations']) == 1 and h['files'] > 0
      opt, pre = hlo.dump_files(proc)
      assert len(opt) == 1 and len(pre) == 1
      assert hlo.Module(hlo.read_text(opt[0])).entry
  assert results['plain'] == results['dump']


# ------------------------------------------------------------------------------------------------ budget

def budget(args, tmp_path, **env):
  e = dict(os.environ, RESULTS_AF3=str(tmp_path / 'af3'), HLO_MARKER=str(tmp_path / 'stop.txt'), **env)
  return subprocess.run([sys.executable, str(REPO / 'scripts' / 'monitor' / 'hlo_budget.py')] + args,
                        capture_output=True, text=True, env=e, cwd=REPO)


def session(tmp_path, name, events):
  d = tmp_path / 'af3' / name
  d.mkdir(parents=True)
  (d / 'launcher_events.txt').write_text('\n'.join(events) + '\n')


def test_budget_check_and_guard(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['check', '40.47'], tmp_path, NOW='2026-11-16T18:00:00Z')
  assert r.returncode == 0 and r.stdout.strip().endswith('GO')
  r = budget(['check', '30'], tmp_path, NOW='2026-11-16T18:00:00Z')
  assert r.returncode == 1 and 'floor' not in r.stdout.splitlines()[-1] or '< 25' in r.stdout
  r = budget(['check', '40.47'], tmp_path, NOW='2026-11-18T00:10:00Z')
  assert r.returncode == 1 and 'launch after 2026-11-17' in r.stdout
  r = budget(['guard', 'harness/plans/hlo/hlo_v6_01.yaml'], tmp_path, NOW='2026-11-16T18:00:00Z')
  assert r.returncode == 0 and 'projected 3.88' in r.stdout
  r = budget(['guard', 'harness/plans/rcx/rcx_af_r2_01.yaml'], tmp_path, NOW='2026-11-16T18:00:00Z')
  assert r.returncode == 1 and 'not a follow-up 16 part' in r.stdout
  session(tmp_path, '20261116T180000Z_v5e_hlo_v5_01', [
      '2026-11-16T18:00:00Z  launch: session x', '2026-11-16T18:03:00Z  vm: x running in europe-west4-b',
      '2026-11-16T18:30:00Z  delete: done', '2026-11-16T18:30:01Z  launcher: exit code 0'])
  r = budget(['spent'], tmp_path, NOW='2026-11-16T19:00:00Z')
  assert r.stdout.strip() == f'{27 / 60 * 1.56:.2f}'
  r = budget(['guard', 'harness/plans/hlo/hlo_v6_01.yaml'], tmp_path, NOW='2026-11-16T19:00:00Z')
  assert r.returncode == 0 and 'still needed by the other part 0.00' in r.stdout
  (tmp_path / 'stop.txt').write_text('x')
  r = budget(['guard', 'harness/plans/hlo/hlo_v6_01.yaml'], tmp_path, NOW='2026-11-16T19:00:00Z')
  assert r.returncode == 1 and 'marker' in r.stdout


def test_credit_check_usage():
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_hlo.sh'), 'abc'], capture_output=True,
                     text=True, cwd=REPO)
  assert r.returncode == 2


def test_check_sessions_and_reproduce_name_this_follow_up():
  cs = (REPO / 'scripts' / 'analysis_runs' / 'hlo' / 'check_sessions.py').read_text()
  assert 'hlo_parts.csv' in cs and 'xla_dump' in cs
  rp = (REPO / 'scripts' / 'analysis_runs' / 'hlo' / 'reproduce.sh').read_text()
  assert 'analysis/hlo.py' in rp and 'test_hlo.py' in rp and 'hlo_${DATE' in rp
  out = subprocess.run(['bash', '-n', str(REPO / 'scripts' / 'analysis_runs' / 'hlo' / 'reproduce.sh')])
  assert out.returncode == 0


HAIKU_MODEL = '''import haiku as hk, jax, jax.numpy as jnp, numpy as np, sys
class Blk(hk.Module):
  def __call__(self, x):
    y = hk.LayerNorm(-1, True, True, name='ln')(x)
    y = hk.Linear(32, name='proj')(y.astype(jnp.bfloat16).astype(jnp.float32))
    return x + jax.nn.gelu(y).astype(jnp.bfloat16).astype(jnp.float32)
class Evo(hk.Module):
  def __call__(self, x):
    x = hk.Linear(32, name='left_single')(x)
    x = hk.experimental.layer_stack(2)(lambda z: Blk(name='msa_stack')(z))(x)
    x = hk.experimental.layer_stack(4)(lambda z: Blk(name='trunk_pairformer')(z))(x)
    return x
class Diffuser(hk.Module):
  def __call__(self, x):
    evo = Evo(name='evoformer')
    e = evo(x) if hk.running_init() else hk.fori_loop(0, 3, lambda i, a: evo(a), x)
    d = Blk(name='diffusion_head')(e)
    c = hk.Linear(4, name='confidence_head')(d).sum(-1)
    return c
t = hk.transform(lambda x: Diffuser(name='diffuser')(x))
x = jnp.ones((16, 32))
p = t.init(jax.random.PRNGKey(0), x)
apply_fn = jax.jit(t.apply)
print(np.asarray(apply_fn(p, None, x)).sum())
'''


@pytest.mark.skipif(not AF3_PY.is_file(), reason='no AlphaFold3 venv (jax 0.10.2, haiku) here')
def test_a_real_xla_dump_of_a_haiku_model_with_alphafold3_module_names(tmp_path):
  """XLA's own text dump (jax 0.10.2 on CPU) of a small Haiku model named like AlphaFold 3's modules (layer stacks,
  a recycling fori_loop): parsed, every stage found, and two compilations identical at every level."""
  (tmp_path / 'm.py').write_text(HAIKU_MODEL)
  env = {k: v for k, v in os.environ.items() if not k.startswith(('XLA_', 'AF3_'))}
  mods = []
  for k in (1, 2):
    d = tmp_path / f'dump{k}'
    e = dict(env, JAX_PLATFORMS='cpu',
             XLA_FLAGS=f'--xla_dump_to={d} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn')
    r = subprocess.run([str(AF3_PY), '-I', str(tmp_path / 'm.py')], env=e, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    (d.parent / f'proc{k}').mkdir()
    d.rename(d.parent / f'proc{k}' / 'xla_dump')
    opt, pre = hlo.dump_files(d.parent / f'proc{k}')
    assert len(opt) == 1 and len(pre) == 1
    m = hlo.Module(hlo.read_text(opt[0]))
    assert m.entry and len(m.frames) > 10
    mods.append(hlo.units(m))
  stages = {u['stage'] for u in mods[0]}
  assert {'embedding', 'msa', 'pairformer', 'diffusion', 'confidence', 'params'} <= stages
  assert {u['path'] for u in mods[0] if u['stage'] == 'pairformer'} >= {'evoformer/trunk_pairformer/ln',
                                                                         'evoformer/trunk_pairformer/proj'}
  assert {v['level'] for v in hlo.compare(mods[0], mods[1]).values()} == {'identical'}
  # the stack frames, read with parent_frame_id less one, are the Python call stack of the operation
  m = hlo.Module(hlo.read_text(opt[0]))
  dots = [i for c in m.comps.values() for i in c if i.op_name.endswith('trunk_pairformer/proj/dot_general') and i.frame]
  assert dots
  funcs = [fn for _, fn, _ in m.stack(dots[0].frame)]
  assert 'Linear.__call__' in funcs and 'Blk.__call__' in funcs and 'Evo.__call__' in funcs
  assert funcs.index('Blk.__call__') < funcs.index('Evo.__call__') < funcs.index('Diffuser.__call__')


def test_stack_frames_follow_parent_id_less_one():
  m = hlo.Module(module())
  assert [f for _, f, _ in m.stack(2)] == ['Evoformer._embed_process_msa', 'Evoformer.__call__']
  assert [f for _, f, _ in m.stack(1)] == ['Evoformer.__call__']


def test_checks_3_and_4_catch_another_device_and_other_flags(tmp_path):
  af3, sm, sessions = make_tree(tmp_path, {})
  rj = af3 / sessions[1] / hlo.RUN.format(rep=1) / '7U3J' / 'run.json'
  j = json.loads(rj.read_text())
  j['device_kind'] = 'TPU v5 lite'
  j['entry_stats']['hlo_dump']['xla_flags'] = '--xla_dump_to=<process folder>/xla_dump'
  rj.write_text(json.dumps(j))
  rows = hlo.checks(hlo.processes(sessions, af3), af3, sm, fake_identical)
  bad = [r for r in rows if r['check3'] == 'FAIL' or r['check4'] == 'FAIL']
  assert [(r['chip'], r['target'], r['rep'], r['check3'], r['check4']) for r in bad] == [('v6e', '7U3J', 1, 'FAIL',
                                                                                           'FAIL')]
  valid, _ = hlo.analyse(hlo.processes(sessions, af3), tmp_path / 'out', rows)
  assert valid                                   # reported beside the result, not a stop (item c)


TPU_NAMES = {   # the file names of one process's xla_dump/ on the TPU (follow-up 16 run of 2026-11-16, from ls)
    256: ('0530', '0533'), 512: ('0534', '0537')}
TPU_COMPANIONS = ('after_optimizations-buffer-assignment.txt', 'after_optimizations-buffer-assignment-values.txt',
                  'after_optimizations-live-range.txt', 'after_optimizations-memory-usage-report.txt',
                  'after_optimizations_after_buffer_assignment.txt', 'after_optimizations_before_buffer_assignment.txt',
                  'after_codegen.txt', 'debug_options', 'execution_options.txt', 'flagfile', 'hlo_module_config.txt',
                  'target_arguments.txt', 'tpu_comp_env.txt', 'basic_compiler_metadata.txt', 'transfer_stats.txt')


@pytest.mark.parametrize('bucket', [256, 512])
def test_tpu_dump_names_with_the_cl_segment_one_of_each_kind(tmp_path, bucket):
  pre_id, opt_id = TPU_NAMES[bucket]
  d = tmp_path / 'proc' / 'xla_dump'
  d.mkdir(parents=True)
  (d / f'module_{pre_id}.jit_apply_fn.cl_970057490.before_optimizations.txt').write_text('')
  (d / f'module_{opt_id}.jit_apply_fn.cl_970057490.after_optimizations.txt').write_text('')
  (d / f'module_{pre_id}.jit_apply_fn.cl_970057490').write_text('')
  for ident in (pre_id, opt_id):
    for c in TPU_COMPANIONS:
      (d / f'module_{ident}.jit_apply_fn.cl_970057490.{c}').write_text('')
  (d / 'module_0002.jit__threefry_seed.cl_123.after_optimizations.txt').write_text('')   # another module
  opt, pre = hlo.dump_files(tmp_path / 'proc')
  assert [p.name for p in opt] == [f'module_{opt_id}.jit_apply_fn.cl_970057490.after_optimizations.txt']
  assert [p.name for p in pre] == [f'module_{pre_id}.jit_apply_fn.cl_970057490.before_optimizations.txt']
  sys.path.insert(0, str(REPO / 'scripts' / 'analysis_runs' / 'hlo'))
  import check_sessions
  import fetch_dumps
  assert check_sessions.dumps(tmp_path / 'proc') == (1, 1)
  assert check_sessions.OPT is hlo.DUMP_OPT and fetch_dumps.OPT is hlo.DUMP_OPT and fetch_dumps.PRE is hlo.DUMP_PRE


def test_cpu_and_gzip_dump_names_are_still_recognised():
  for name in ('module_0006.jit_apply_fn.cpu_after_optimizations.txt', 'module_0006.jit_apply_fn.after_optimizations.txt',
               'module_0533.jit_apply_fn.cl_970057490.after_optimizations.txt.gz'):
    assert hlo.DUMP_OPT.search(name), name
  for name in ('module_0006.jit_apply_fn.before_optimizations.txt',
               'module_0530.jit_apply_fn.cl_970057490.before_optimizations.txt.gz'):
    assert hlo.DUMP_PRE.search(name), name
  for name in ('module_0006.jit_apply_fn.cpu_after_optimizations-buffer-assignment.txt',
               'module_0533.jit_apply_fn.cl_970057490.after_optimizations_after_buffer_assignment.txt',
               'module_0533.jit_apply_fn.cl_970057490.after_optimizations_before_buffer_assignment.txt',
               'module_0533.jit_apply_fn.cl_970057490.after_codegen.txt',
               'module_0533.jit_apply_fn.cl_x.after_optimizations.txt', 'module_0530.jit_stage.before_optimizations.txt'):
    assert not hlo.DUMP_OPT.search(name) and not hlo.DUMP_PRE.search(name), name
