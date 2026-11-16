"""Follow-up 16 (label hlo; notes/analysis_plan.md section 10, row of 2026-11-16): the HLO of AlphaFold 3's model as XLA
compiles it on the v5e and on the v6e, compared per token bucket and per AlphaFold 3 module.

    analysis/.venv/bin/python analysis/hlo.py --sessions scripts/analysis_runs/hlo/sessions.txt \
        --out results/analysis/hlo_DATE

Inputs: the sessions listed (one per chip, label hlo), each holding, per target (7U3J, 7NPL, 7XQZ and 7D5C: per token
bucket 256, 512, 768 and 1,024, the pilot target with the most tokens) and repetition (r1, r2; seed 1), the process folder <session>/tpu_xla_hlo_rep<k>_seed1_fresh/<target>/ with
af3_output/ and xla_dump/ (harness/af3_entry.py, AF3_HLO_DUMP_FLAGS): XLA's text dump of the model's module
jit_apply_fn before (module_*.jit_apply_fn.before_optimizations.txt) and after its optimisations
(module_*.jit_apply_fn.*after_optimizations.txt). Plain or gzip-compressed (.txt.gz) files are read.

Checks (item c of the row), before any HLO is compared (3 and 4 are reported beside the result and do not stop it):
  1. every planned process exited 0 (run.json) and has exactly one optimised and one unoptimised dump of jit_apply_fn;
  2. validity: every process's outputs are identical bit for bit with the registered process of the same chip, target
     and seed (main study, V5 or V6 r1 seed 1; contrasts.outputs_identical, every output file, mmCIF timestamp line
     masked). A process that differs makes the run invalid: the HLO comparison is then written but marked INVALID and
     not read (rule e1);
  3. device: run.json device_kind TPU v5 lite on the v5e and TPU v6 lite on the v6e;
  4. stack and dump: session.json libtpu build libtpu_lts_20260630_b_RC06 and jax, jaxlib 0.10.2, libtpu 0.0.43.2;
     af3_entry_stats.json hlo_dump.xla_flags as the configuration sets them (XLA_FLAGS); the Python version is listed.
HLO units. Every instruction of every computation that runs (the entry, loop bodies and conditions, called and fused
computations; not the reducer computations of reduce-like operations, which enter their caller's signature) is one
unit, with:
  stage     the AlphaFold 3 module it belongs to (STAGES, from its op_name and the stack frames of its metadata, see
            stage_of): embedding (input embedding and the evoformer's pair and single embedding), template, msa,
            pairformer (the trunk's 48 blocks), diffusion, confidence, distogram, params (reading the weights: op_name params[...], whatever module owns them), other
            (model glue), unattributed (no metadata: the stage of its fusion, else of its first operand that has one);
  semantic  for arithmetic, the operation as it rounds: opcode, element types and dimensions of its result and
            operands, and every attribute that sets the computation (precision, algorithm, contracting, batch and
            reduced dimensions, windows, comparison, the reducer's operations and types, custom-call target), never
            names, layouts, metadata or backend configuration; None for operations that move data exactly (MOVE) and
            for control (CONTROL);
  types     the semantic signature without dimensions (opcode, element types, precision, algorithm, reducer);
  fusion    the semantic signatures of the innermost fusion the unit sits in, with its kind (or "unfused");
  layout    opcode and result shape with its layout (and tiling), with the non-semantic attributes, for every unit
            that is not control;
  backend   the backend configuration of the unit, if it has one.
Levels (item e2), per bucket and stage, the first that differs between the two chips' multisets of units, in this
order: semantic (which arithmetic operations run, with which types, shapes, precisions and reductions), fusion (the
same operations grouped into other kernels), layout (the same kernels with other layouts or data movement), backend
(the same HLO with another backend configuration), identical. A semantic difference is marked "types" if the types
signatures differ as well (element types, precision, algorithm or reducer), else "shapes".
Reading rules (fixed in the row before the run):
  e1  the run is valid if checks 1 and 2 pass for all 16 processes; otherwise nothing below is read;
  e2  the level of each (bucket, stage), as above;
  e3  pre-specified question, per bucket: the differences appear before the Pairformer "yes" if the highest level among
      embedding, template and msa is semantic or fusion; "layout only" if it is layout; "backend only" if it is
      backend; "no" if all three are identical. Overall: the answer if every bucket gives the same, else "mixed" with
      each bucket's;
  e4  within-chip determinism, per chip and bucket: r1 and r2 "identical" if their optimised dumps are byte-identical,
      else the highest level at which they differ, per stage; a cross-chip level in a stage at or below the
      within-chip level of that stage on either chip is reported as "not attributable to the chip";
  e5  the unoptimised dumps of the two chips, per bucket: "identical" if byte-identical after the first line (the module
      header), else the level, per stage, as in e2.
Descriptive tables (no reading rule): counts of units per opcode, fusion kind, element type, the dot and convolution
precisions and types, and the reductions (opcode, types, reducer), per bucket, chip and stage; the module paths (the
op_name's Haiku modules below the stage) whose units differ, with the counts on each side.
Writes: checks.csv, stages.csv, question.csv, determinism.csv, unoptimised.csv, opcodes.csv, fusions.csv, dtypes.csv,
dots.csv, reductions.csv, module_diffs.csv, summary.md.
"""
import argparse
import collections
import csv
import gzip
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TARGETS = {'7U3J': 256, '7NPL': 512, '7XQZ': 768, '7D5C': 1024}
CHIPS = ('v5e', 'v6e')
REPS = (1, 2)
SEED = 1
RUN = 'tpu_xla_hlo_rep{rep}_seed1_fresh'
REF_RUN = 'tpu_xla_rep1_seed1_fresh'
REF_PATTERN = {'v5e': r'_v5e_main_v5_\d+$', 'v6e': r'_v6e_main_v6_\d+$'}
STAGES = ('embedding', 'template', 'msa', 'pairformer', 'diffusion', 'confidence', 'distogram', 'params', 'other',
          'unattributed')
BEFORE_PAIRFORMER = ('embedding', 'template', 'msa')
DEVICE = {'v5e': 'TPU v5 lite', 'v6e': 'TPU v6 lite'}
STACK = {'jax': '0.10.2', 'jaxlib': '0.10.2', 'libtpu': '0.0.43.2'}
LIBTPU_BUILD = 'libtpu_lts_20260630_b_RC06'
XLA_FLAGS = '--xla_dump_to=<process folder>/xla_dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn'
LEVELS = ('semantic', 'fusion', 'layout', 'backend', 'identical')
MOVE = {'bitcast', 'copy', 'copy-start', 'copy-done', 'reshape', 'transpose', 'broadcast', 'slice', 'dynamic-slice',
        'dynamic-update-slice', 'concatenate', 'pad', 'reverse', 'iota', 'gather', 'bitcast-convert'}
CONTROL = {'parameter', 'tuple', 'get-tuple-element', 'fusion', 'while', 'conditional', 'call', 'after-all',
           'add-dependency', 'opt-barrier', 'domain', 'partition-id', 'replica-id', 'async-start', 'async-update',
           'async-done', 'send', 'send-done', 'recv', 'recv-done', 'infeed', 'outfeed'}
# attributes that never enter a semantic signature: names, metadata, layouts, backend and scheduling data
NOT_SEMANTIC = {'metadata', 'backend_config', 'frontend_attributes', 'sharding', 'calls', 'to_apply', 'body',
                'condition', 'branch_computations', 'control-predecessors', 'statistics', 'origin', 'kind',
                'operand_layout_constraints', 'output_to_operand_aliasing', 'schedule', 'scheduling_name',
                'async_execution_thread', 'channel_id', 'is_host_transfer'}
TYPE_ATTRS = {'operand_precision', 'precision_config', 'algorithm', 'comparison_type', 'exponent_bits',
              'mantissa_bits', 'custom_call_target', 'rng_algorithm', 'distribution'}
STRUCT_LAYOUT_DROP = {'metadata', 'backend_config', 'frontend_attributes', 'calls', 'to_apply', 'body', 'condition',
                      'branch_computations', 'control-predecessors', 'statistics', 'origin'}
# op_name components that name transformations, not Haiku modules
TRANSFORM = re.compile(r'^(jit|pjit|jvp|vjp|transpose|vmap|remat\d*|checkpoint|while|body|cond|scan|closed_call|'
                       r'core_call|custom_jvp_call|custom_vjp_call|branch_\d+|named\(.*\))(\(.*\))?$')


# ------------------------------------------------------------------------------------------------ parsing

def read_text(path):
  path = pathlib.Path(path)
  if path.suffix == '.gz':
    with gzip.open(path, 'rt') as f:
      return f.read()
  return path.read_text()


def split_top(s, sep=','):
  """Splits s at sep outside (), {}, [] and double quotes (backslash escapes inside quotes)."""
  out, depth, cur, q, esc = [], 0, [], False, False
  for ch in s:
    if q:
      cur.append(ch)
      if esc:
        esc = False
      elif ch == '\\':
        esc = True
      elif ch == '"':
        q = False
      continue
    if ch == '"':
      q = True
    elif ch in '({[':
      depth += 1
    elif ch in ')}]':
      depth -= 1
    if ch == sep and depth == 0:
      out.append(''.join(cur).strip())
      cur = []
    else:
      cur.append(ch)
  if ''.join(cur).strip():
    out.append(''.join(cur).strip())
  return out


def balanced(s, i):
  """Index after the bracket group that starts at s[i] (quotes respected)."""
  pairs = {'(': ')', '{': '}', '[': ']'}
  stack, q, esc = [], False, False
  for j in range(i, len(s)):
    ch = s[j]
    if q:
      if esc:
        esc = False
      elif ch == '\\':
        esc = True
      elif ch == '"':
        q = False
      continue
    if ch == '"':
      q = True
    elif ch in pairs:
      stack.append(pairs[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
      if not stack:
        return j + 1
  raise ValueError(f'unbalanced: {s[i:i + 80]}')


COMMENT = re.compile(r'/\*.*?\*/')
COMP_HEAD = re.compile(r'^(ENTRY\s+)?%?([^\s(]+)\s*\(.*\)\s*->\s*.*\{\s*$')
INSTR_HEAD = re.compile(r'^\s*(ROOT\s+)?%?([^\s=]+)\s*=\s*')
OPCODE = re.compile(r'\s*([a-z][a-z0-9\-]*)\(')
LAYOUT = re.compile(r'\{[^{}]*\}')
SHAPE1 = re.compile(r'([a-z]+[0-9]*)\[([^\]]*)\]')


class Instr:
  __slots__ = ('comp', 'name', 'shape', 'opcode', 'operands', 'attrs', 'root', 'op_name', 'frame')

  def __init__(self, comp, name, shape, opcode, operands, attrs, root):
    self.comp, self.name, self.shape, self.opcode = comp, name, shape, opcode
    self.operands, self.attrs, self.root = operands, attrs, root
    md = attrs.get('metadata', '')
    m = re.search(r'op_name="((?:[^"\\]|\\.)*)"', md)
    self.op_name = m.group(1) if m else ''
    m = re.search(r'stack_frame_id=(\d+)', md)
    self.frame = int(m.group(1)) if m else None


class Module:
  """An HLO module from XLA's text dump: computations of instructions, and the stack-frame tables."""

  def __init__(self, text):
    self.text = text
    self.files, self.functions, self.locations, self.frames = {}, {}, {}, {}
    self.comps, self.entry = collections.OrderedDict(), None
    self._parse(text)
    self.by_name = {(c, i.name): i for c, instrs in self.comps.items() for i in instrs}

  def _parse(self, text):
    table, comp = None, None
    for raw in text.splitlines():
      line = COMMENT.sub('', raw).rstrip()
      if comp is None:
        if line in ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'):
          table = line
          continue
        if table and re.match(r'^\d+ ', line):
          k, v = line.split(' ', 1)
          k = int(k)
          if table == 'FileNames':
            self.files[k] = v.strip().strip('"')
          elif table == 'FunctionNames':
            self.functions[k] = v.strip().strip('"')
          else:
            d = dict(re.findall(r'(\w+)=(-?\d+)', v))
            (self.locations if table == 'FileLocations' else self.frames)[k] = {a: int(b) for a, b in d.items()}
          continue
        table = None if not line.strip() else table
        m = COMP_HEAD.match(line)
        if m and not line.lstrip().startswith('HloModule'):
          comp = m.group(2)
          self.comps[comp] = []
          if m.group(1):
            self.entry = comp
        continue
      if line.strip() == '}':
        comp = None
        continue
      ins = parse_instruction(comp, line)
      if ins:
        self.comps[comp].append(ins)

  def stack(self, frame):
    """[(file, function, line)] from the frame to the root (innermost first)."""
    out, seen = [], set()
    while frame and frame in self.frames and frame not in seen:
      seen.add(frame)
      f = self.frames[frame]
      loc = self.locations.get(f.get('file_location_id'), {})
      out.append((self.files.get(loc.get('file_name_id'), ''), self.functions.get(loc.get('function_name_id'), ''),
                  loc.get('line')))
      parent = f.get('parent_frame_id')
      if parent == frame:
        break
      frame = parent
    return out


def parse_instruction(comp, line):
  m = INSTR_HEAD.match(line)
  if not m:
    return None
  rest = line[m.end():]
  if rest.startswith('('):
    j = balanced(rest, 0)
    shape, rest = rest[:j], rest[j:]
  else:
    sp = rest.find(' ')
    shape, rest = rest[:sp], rest[sp:]
    while rest.startswith('{'):          # a layout printed after a space is not expected; kept for safety
      j = balanced(rest, 0)
      shape, rest = shape + rest[:j], rest[j:]
  mo = OPCODE.match(rest)
  if not mo:
    return None
  opcode = mo.group(1)
  start = mo.end() - 1
  end = balanced(rest, start)
  inner = rest[start + 1:end - 1]
  operands = [] if opcode == 'constant' else [o.split()[-1].lstrip('%') for o in split_top(inner) if o.split()]
  attrs = {}
  tail = rest[end:].lstrip(', ')
  for item in split_top(tail):
    k, eq, v = item.partition('=')
    attrs[k.strip()] = v.strip() if eq else ''
  if opcode == 'constant':
    attrs['literal'] = inner if len(inner) <= 64 else '{...}'
  return Instr(comp, m.group(2), shape, opcode, operands, attrs, bool(m.group(1)))


def logical(shape):
  """The shape without layouts (and tiling): f32[4,8]{1,0:T(8,128)} -> f32[4,8]."""
  prev = None
  while prev != shape:
    prev, shape = shape, re.sub(r'(\])\{[^{}]*(\{[^{}]*\})*[^{}]*\}', r'\1', shape)
  return shape


def dtypes(shape):
  return [t for t, _ in SHAPE1.findall(logical(shape))]


def names_out(v):
  return re.sub(r'%[\w.\-]+', '%', v)


# ------------------------------------------------------------------------------------------------ stages

def op_tokens(op_name):
  return [t for t in op_name.split('/') if t]


def stage_of(op_name, stack):
  """The AlphaFold 3 stage of an instruction from its op_name (Haiku module path) and its stack frames
  [(file, function, line)]; None if it has neither."""
  if op_name.startswith('params['):       # reading a weight (its op_name names the module that owns it)
    return 'params'
  toks = set(op_tokens(op_name))
  files = {pathlib.PurePath(f).name for f, _, _ in stack}
  funcs = {fn for _, fn, _ in stack}
  if 'confidence_head' in toks or 'confidence_head.py' in files:
    return 'confidence'
  if 'distogram_head' in toks or 'distogram_head.py' in files:
    return 'distogram'
  if ('diffusion_head' in toks or files & {'diffusion_head.py', 'diffusion_transformer.py'}
          or any(fn.endswith('_sample_diffusion') for fn in funcs)):
    return 'diffusion'
  if 'trunk_pairformer' in toks or any('pairformer_fn' in fn for fn in funcs):
    return 'pairformer'
  if toks & {'msa_stack', 'msa_activations', 'extra_msa_target_feat'} or any(fn.endswith('_embed_process_msa')
                                                                           for fn in funcs):
    return 'msa'
  if 'template_embedding' in toks or 'template_modules.py' in files or any(fn.endswith('_embed_template_pair')
                                                                           for fn in funcs):
    return 'template'
  if toks & {'evoformer', 'evoformer_conditioning'} or 'evoformer.py' in files:
    return 'embedding'
  if op_name or stack:
    return 'other'
  return None


def module_path(op_name, depth=4):
  """The Haiku module path of an op_name below the model ('diffuser'), transformations and the primitive left out."""
  toks = [t for t in op_tokens(op_name) if not TRANSFORM.match(t)]
  if toks and not op_name.startswith('params['):
    toks = toks[:-1]                     # the primitive (dot_general, add, ...)
  if toks and toks[0] == 'diffuser':
    toks = toks[1:]
  return '/'.join(toks[:depth])


# ------------------------------------------------------------------------------------------------ units

def units(mod):
  """[dict] one per counted instruction: stage, path, opcode, semantic, types, fusion, layout, backend."""
  # computations: fused (with their fusion instruction), reducers (not counted), and the rest
  fused_by, reducer_comps = {}, set()
  for c, instrs in mod.comps.items():
    for i in instrs:
      if i.opcode == 'fusion' and 'calls' in i.attrs:
        fused_by[i.attrs['calls'].lstrip('%')] = i
      if 'to_apply' in i.attrs and i.opcode != 'call':
        reducer_comps.add(i.attrs['to_apply'].lstrip('%'))
  reducer_comps -= set(fused_by)

  def reducer_sig(name):
    instrs = mod.comps.get(name.lstrip('%'), [])
    return '+'.join(sorted(f'{i.opcode}:{"".join(dtypes(i.shape))}' for i in instrs if i.opcode != 'parameter'))

  # stage of every instruction; inherited when it has no metadata
  stage, path = {}, {}

  def own(i):
    return stage_of(i.op_name, mod.stack(i.frame) if i.frame else [])
  for c, instrs in mod.comps.items():
    for i in instrs:
      stage[(c, i.name)] = own(i)
      path[(c, i.name)] = module_path(i.op_name) if i.op_name else ''

  def resolve(c, i, seen=()):
    s = stage[(c, i.name)]
    if s:
      return s, path[(c, i.name)]
    if (c, i.name) in seen:
      return 'unattributed', ''
    seen = seen + ((c, i.name),)
    if c in fused_by:
      f = fused_by[c]
      return resolve(f.comp, f, seen)
    for o in i.operands:
      oi = mod.by_name.get((c, o))
      if oi is not None and stage[(c, o)]:
        return stage[(c, o)], path[(c, o)]
    return 'unattributed', ''

  def sem(i, with_dims):
    if i.opcode in MOVE or i.opcode in CONTROL:
      return None
    ops = [mod.by_name.get((i.comp, o)) for o in i.operands]
    if with_dims:
      parts = [i.opcode, logical(i.shape), '(' + ','.join(logical(o.shape) if o else '?' for o in ops) + ')']
    else:
      parts = [i.opcode, ''.join(dtypes(i.shape)), '(' + ','.join(''.join(dtypes(o.shape)) if o else '?' for o in ops)
               + ')']
    for k in sorted(i.attrs):
      if k in NOT_SEMANTIC or (not with_dims and k not in TYPE_ATTRS):
        continue
      parts.append(f'{k}={names_out(i.attrs[k])}')
    if 'to_apply' in i.attrs:
      parts.append(f'reducer={reducer_sig(i.attrs["to_apply"])}')
    return ' '.join(parts)

  ctx = {}
  for c, f in fused_by.items():
    sigs = sorted(s for s in (sem(i, True) for i in mod.comps.get(c, [])) if s)
    ctx[c] = f'{f.attrs.get("kind", "")}[' + ' | '.join(sigs) + ']'
  out = []
  for c, instrs in mod.comps.items():
    if c in reducer_comps:
      continue
    for i in instrs:
      st, p = resolve(c, i)
      s = sem(i, True)
      lay = None
      if i.opcode not in CONTROL:
        extra = ' '.join(f'{k}={names_out(v)}' for k, v in sorted(i.attrs.items()) if k not in STRUCT_LAYOUT_DROP)
        lay = f'{i.opcode} {i.shape} {extra}'.strip()
      be = i.attrs.get('backend_config')
      out.append({'stage': st, 'path': p, 'opcode': i.opcode, 'shape': i.shape, 'semantic': s,
                  'types': sem(i, False), 'fusion': ctx.get(c, 'unfused') if s else None, 'layout': lay,
                  'backend': f'{i.opcode} {logical(i.shape)} {be}' if be else None,
                  'fusion_kind': i.attrs.get('kind') if i.opcode == 'fusion' else None,
                  'attrs': i.attrs, 'operand_types': [''.join(dtypes(mod.by_name[(c, o)].shape))
                                                      if (c, o) in mod.by_name else '?' for o in i.operands],
                  'reducer': reducer_sig(i.attrs['to_apply']) if 'to_apply' in i.attrs and i.opcode != 'call' else None})
  return out


def multisets(us):
  """{stage: {level key: Counter}} for the comparison levels."""
  out = collections.defaultdict(lambda: {k: collections.Counter() for k in ('semantic', 'types', 'fusion', 'layout',
                                                                            'backend')})
  for u in us:
    m = out[u['stage']]
    if u['semantic']:
      m['semantic'][u['semantic']] += 1
      m['types'][u['types']] += 1
      m['fusion'][(u['semantic'], u['fusion'])] += 1
    if u['layout']:
      m['layout'][u['layout']] += 1
    if u['backend']:
      m['backend'][u['backend']] += 1
  return out


def sym(a, b):
  return sum((a - b).values()) + sum((b - a).values())


def compare(ua, ub):
  """{stage: {'level', 'detail', counts}} between two lists of units (rule e2)."""
  ma, mb = multisets(ua), multisets(ub)
  out = {}
  for st in STAGES:
    a, b = ma.get(st), mb.get(st)
    if a is None and b is None:
      continue
    a = a or multisets([])['x']
    b = b or multisets([])['x']
    diffs = {k: sym(a[k], b[k]) for k in ('semantic', 'types', 'fusion', 'layout', 'backend')}
    level = next((k for k in ('semantic', 'fusion', 'layout', 'backend') if diffs[k]), 'identical')
    detail = ('types' if diffs['types'] else 'shapes') if level == 'semantic' else ''
    out[st] = {'level': level, 'detail': detail, 'units_a': sum(1 for u in ua if u['stage'] == st),
               'units_b': sum(1 for u in ub if u['stage'] == st), **{f'diff_{k}': v for k, v in diffs.items()}}
  return out


def rank(level):
  return LEVELS.index(level)


def question(stages):
  """Rule e3 for one bucket: yes, layout only, backend only or no."""
  levels = [stages[s]['level'] for s in BEFORE_PAIRFORMER if s in stages]
  if not levels:
    return 'no units before the Pairformer'
  top = min(levels, key=rank)
  return {'semantic': 'yes', 'fusion': 'yes', 'layout': 'layout only', 'backend': 'backend only'}.get(top, 'no')


def overall(answers):
  vals = set(answers.values())
  return vals.pop() if len(vals) == 1 else 'mixed (' + '; '.join(f'{b}: {a}' for b, a in sorted(answers.items())) + ')'


def module_diffs(ua, ub, key='semantic'):
  """[(stage, path, only_a, only_b)] of module paths whose units differ at a level key."""
  def by(us):
    d = collections.defaultdict(collections.Counter)
    for u in us:
      k = u[key] if key != 'fusion' else ((u['semantic'], u['fusion']) if u['semantic'] else None)
      if k:
        d[(u['stage'], u['path'])][k] += 1
    return d
  a, b = by(ua), by(ub)
  out = []
  for k in sorted(set(a) | set(b)):
    x, y = a.get(k, collections.Counter()), b.get(k, collections.Counter())
    if x != y:
      out.append((k[0], k[1], sum((x - y).values()), sum((y - x).values())))
  return out


def first_line_dropped(text):
  return text.split('\n', 1)[1] if '\n' in text else ''


# ------------------------------------------------------------------------------------------------ descriptive

def describe(us):
  """Counters per stage for the descriptive tables."""
  t = collections.defaultdict(collections.Counter)
  for u in us:
    st = u['stage']
    t['opcodes'][(st, u['opcode'])] += 1
    for d in dtypes(u['shape']):
      t['dtypes'][(st, d)] += 1
    if u['fusion_kind']:
      t['fusions'][(st, u['fusion_kind'])] += 1
    if u['opcode'] in ('dot', 'convolution'):
      a = u['attrs']
      prec = a.get('operand_precision') or a.get('precision_config') or 'default'
      t['dots'][(st, u['opcode'], ','.join(u['operand_types']), ''.join(dtypes(u['shape'])), prec,
                 a.get('algorithm', ''))] += 1
    if u['reducer'] is not None and u['opcode'] != 'call':
      t['reductions'][(st, u['opcode'], ','.join(u['operand_types']), ''.join(dtypes(u['shape'])), u['reducer'],
                       u['attrs'].get('dimensions', ''))] += 1
  return t


# ------------------------------------------------------------------------------------------------ processes

def listed(path):
  return [l.split('#')[0].strip() for l in pathlib.Path(path).read_text().splitlines() if l.split('#')[0].strip()]


def chip_of(session):
  m = re.search(r'_(v5e|v6e)_', session)
  return m.group(1) if m else None


def dump_files(proc):
  """(optimised, unoptimised) dump paths of jit_apply_fn in a process folder, as lists."""
  d = pathlib.Path(proc) / 'xla_dump'
  files = sorted(p for p in d.glob('*') if p.is_file()) if d.is_dir() else []
  opt = [p for p in files if re.search(r'module_\d+\.jit_apply_fn\.(?:[a-z0-9_]+_)?after_optimizations\.txt(\.gz)?$',
                                       p.name)]
  pre = [p for p in files if re.search(r'module_\d+\.jit_apply_fn\.before_optimizations\.txt(\.gz)?$', p.name)]
  return opt, pre


def processes(sessions, af3=None):
  """{(chip, target, rep): {'dir', 'session', 'run_json'}} of the hlo sessions listed."""
  af3 = pathlib.Path(af3 or REPO / 'results' / 'af3')
  out = {}
  for s in sessions:
    chip = chip_of(s)
    for rep in REPS:
      for t in TARGETS:
        d = af3 / s / RUN.format(rep=rep) / t
        out[(chip, t, rep)] = {'dir': d, 'session': s, 'run_json': d / 'run.json'}
  return out


def reference_dir(chip, target, af3=None, sessions_main=None):
  """The registered output folder of the same chip, target and seed: main study, V5 or V6, r1, seed 1."""
  af3 = pathlib.Path(af3 or REPO / 'results' / 'af3')
  sessions_main = sessions_main or REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt'
  for s in pathlib.Path(sessions_main).read_text().split():
    if re.search(REF_PATTERN[chip], s):
      d = af3 / s / REF_RUN / target / 'af3_output' / target
      if d.is_dir():
        return d
  return None


def checks(procs, af3=None, sessions_main=None, identical=None):
  """Checks 1 and 2 per process: [dict]."""
  import json
  if identical is None:
    sys.path.insert(0, str(REPO / 'analysis'))
    from contrasts import outputs_identical as identical
  rows = []
  for (chip, t, rep), p in sorted(procs.items()):
    r = {'chip': chip, 'target': t, 'bucket': TARGETS[t], 'rep': rep, 'session': p['session']}
    try:
      rj = json.loads(p['run_json'].read_text())
      r['exit_code'] = rj.get('exit_code')
    except (OSError, ValueError):
      rj, r['exit_code'] = {}, None
    opt, pre = dump_files(p['dir'])
    r['optimised_dumps'], r['unoptimised_dumps'] = len(opt), len(pre)
    r['check1'] = 'pass' if r['exit_code'] == 0 and len(opt) == 1 and len(pre) == 1 else 'FAIL'
    # checks 3 and 4: device, stack, and the dump flags the process ran with
    try:
      sj = json.loads((p['dir'].parent / 'session.json').read_text())
    except (OSError, ValueError):
      sj = {}
    stats = rj.get('entry_stats') or {}
    pk = (sj.get('probe') or {}).get('packages') or {}
    r['device_kind'] = rj.get('device_kind')
    r['machine_type'] = (sj.get('host') or {}).get('gce_accelerator_type') or (sj.get('host') or {}).get('gce_machine_type')
    r['libtpu_build'] = (sj.get('tpu_runtime') or {}).get('build_label')
    r['stack'] = ' '.join(f'{k} {pk.get(k)}' for k in STACK)
    r['python'] = ((sj.get('probe') or {}).get('python') or '').split()[0] if (sj.get('probe') or {}).get('python') else ''
    r['xla_flags'] = (stats.get('hlo_dump') or {}).get('xla_flags')
    r['check3'] = 'pass' if r['device_kind'] == DEVICE.get(chip) else 'FAIL'
    r['check4'] = 'pass' if (r['libtpu_build'] == LIBTPU_BUILD and all(pk.get(k) == v for k, v in STACK.items())
                             and r['xla_flags'] == XLA_FLAGS) else 'FAIL'
    ref = reference_dir(chip, t, af3, sessions_main)
    if ref is None:
      r['reference'], r['check2'], r['check2_detail'] = '', 'FAIL', 'no registered reference'
    else:
      res = identical(p['dir'] / 'af3_output' / t, ref, SEED)
      r['reference'] = str(ref.relative_to(pathlib.Path(af3 or REPO / 'results' / 'af3')))
      r['check2'] = 'pass' if res['status'] == 'identical' else 'FAIL'
      r['check2_detail'] = res['status'] + (f'; differ: {" ".join(res["differ"][:5])}' if res['differ'] else '') + (
          f'; missing: {" ".join(res["missing"][:5])}' if res['missing'] else '')
    rows.append(r)
  return rows


# ------------------------------------------------------------------------------------------------ analysis

def write_csv(path, rows, fields):
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k, '') for k in fields})


def analyse(procs, out, check_rows):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  valid = len(check_rows) == len(TARGETS) * len(CHIPS) * len(REPS) and all(
      r['check1'] == 'pass' and r['check2'] == 'pass' for r in check_rows)
  write_csv(out / 'checks.csv', check_rows, ['chip', 'target', 'bucket', 'rep', 'session', 'exit_code',
                                             'optimised_dumps', 'unoptimised_dumps', 'check1', 'reference', 'check2',
                                             'check2_detail', 'device_kind', 'machine_type', 'check3', 'libtpu_build',
                                             'stack', 'python', 'xla_flags', 'check4'])
  mods, texts = {}, {}
  for key, p in procs.items():
    opt, pre = dump_files(p['dir'])
    if len(opt) == 1 and len(pre) == 1:
      texts[key] = (read_text(opt[0]), read_text(pre[0]))
      mods[key] = units(Module(texts[key][0]))
  stage_rows, q, det_rows, pre_rows, md_rows = [], {}, [], [], []
  desc = collections.defaultdict(list)
  within = {}
  for t, b in TARGETS.items():
    for chip in CHIPS:
      k1, k2 = (chip, t, 1), (chip, t, 2)
      if k1 in texts and k2 in texts:
        same = texts[k1][0] == texts[k2][0]
        cmp = {} if same else compare(mods[k1], mods[k2])
        within[(chip, b)] = cmp
        det_rows.append({'bucket': b, 'chip': chip, 'r1_r2': 'identical' if same else 'differ',
                         'stages': '; '.join(f'{s} {v["level"]}' for s, v in cmp.items() if v['level'] != 'identical')})
      else:
        det_rows.append({'bucket': b, 'chip': chip, 'r1_r2': 'not determined (dump missing)'})
      for name, ctr in describe(mods.get(k1, [])).items():
        for key, n in ctr.items():
          desc[name].append((b, chip) + key + (n,))
    a, c = ('v5e', t, 1), ('v6e', t, 1)
    if a not in mods or c not in mods:
      continue
    cmp = compare(mods[a], mods[c])
    for st, v in cmp.items():
      lim = [within.get((ch, b), {}).get(st, {}).get('level', 'identical') for ch in CHIPS]
      within_top = min(lim, key=rank)
      note = ('not attributable to the chip' if v['level'] != 'identical' and within_top != 'identical'
              and rank(v['level']) >= rank(within_top) else '')
      stage_rows.append({'bucket': b, 'stage': st, 'level': v['level'], 'detail': v['detail'], 'units_v5e': v['units_a'],
                         'units_v6e': v['units_b'], **{k: v[k] for k in v if k.startswith('diff_')},
                         'within_chip_level': within_top, 'note': note})
    q[b] = question(cmp)
    for lvl in ('semantic', 'fusion', 'layout', 'backend'):
      for st, pth, na, nb in module_diffs(mods[a], mods[c], lvl):
        md_rows.append({'bucket': b, 'level': lvl, 'stage': st, 'module_path': pth or '(none)', 'only_v5e': na,
                        'only_v6e': nb})
    pa, pc = first_line_dropped(texts[a][1]), first_line_dropped(texts[c][1])
    if pa == pc:
      pre_rows.append({'bucket': b, 'unoptimised': 'identical', 'stages': ''})
    else:
      pcmp = compare(units(Module(texts[a][1])), units(Module(texts[c][1])))
      pre_rows.append({'bucket': b, 'unoptimised': 'differ', 'stages': '; '.join(
          f'{s} {v["level"]}' for s, v in pcmp.items() if v['level'] != 'identical')})
  fields = ['bucket', 'stage', 'level', 'detail', 'units_v5e', 'units_v6e', 'diff_semantic', 'diff_types',
            'diff_fusion', 'diff_layout', 'diff_backend', 'within_chip_level', 'note']
  write_csv(out / 'stages.csv', stage_rows, fields)
  qrows = [{'bucket': b, 'differences_before_pairformer': a} for b, a in sorted(q.items())]
  if q:
    qrows.append({'bucket': 'overall', 'differences_before_pairformer': overall(q)})
  write_csv(out / 'question.csv', qrows, ['bucket', 'differences_before_pairformer'])
  write_csv(out / 'determinism.csv', det_rows, ['bucket', 'chip', 'r1_r2', 'stages'])
  write_csv(out / 'unoptimised.csv', pre_rows, ['bucket', 'unoptimised', 'stages'])
  write_csv(out / 'module_diffs.csv', md_rows, ['bucket', 'level', 'stage', 'module_path', 'only_v5e', 'only_v6e'])
  heads = {'opcodes': ['stage', 'opcode'], 'dtypes': ['stage', 'dtype'], 'fusions': ['stage', 'kind'],
           'dots': ['stage', 'opcode', 'operand_types', 'result_type', 'precision', 'algorithm'],
           'reductions': ['stage', 'opcode', 'operand_types', 'result_type', 'reducer', 'dimensions']}
  for name, cols in heads.items():
    fields = ['bucket', 'chip'] + cols + ['count']
    write_csv(out / f'{name}.csv', [dict(zip(fields, r)) for r in sorted(desc.get(name, []), key=str)], fields)
  summary(out, valid, check_rows, stage_rows, q, det_rows, pre_rows, md_rows)
  return valid, q


def summary(out, valid, check_rows, stage_rows, q, det_rows, pre_rows, md_rows):
  L = ['# Follow-up 16 (label hlo): HLO of the model, v5e against v6e (analysis/hlo.py)', '']
  L.append(f'Rule e1: {"VALID" if valid else "INVALID: a check failed; the comparisons below are not read"}.')
  L += ['', '## Checks (1: exit 0 and one optimised and one unoptimised dump; 2: outputs identical with the registered '
        'process of the same chip; 3: device; 4: libtpu build, jax, jaxlib, libtpu and the dump flags; 3 and 4 are '
        'reported and do not stop the reading)', '',
        '| chip | target | bucket | rep | check 1 | check 2 | detail | check 3 | check 4 | Python |',
        '|---|---|---|---|---|---|---|---|---|---|']
  L += [f'| {r["chip"]} | {r["target"]} | {r["bucket"]} | r{r["rep"]} | {r["check1"]} | {r["check2"]} | '
        f'{r.get("check2_detail", "")} | {r.get("check3", "")} | {r.get("check4", "")} | {r.get("python", "")} |'
        for r in check_rows]
  L += ['', '## Question (rule e3): do the HLO differences appear before the Pairformer?', '']
  L += [f'- bucket {b}: {a}' for b, a in sorted(q.items())]
  if q:
    L.append(f'- overall: {overall(q)}')
  L += ['', '## Levels per bucket and stage (rule e2; within-chip level from rule e4)', '',
        '| bucket | stage | level | detail | units v5e | units v6e | within chip | note |', '|---|---|---|---|---|---|---|---|']
  L += [f'| {r["bucket"]} | {r["stage"]} | {r["level"]} | {r["detail"]} | {r["units_v5e"]} | {r["units_v6e"]} | '
        f'{r["within_chip_level"]} | {r["note"]} |' for r in stage_rows]
  L += ['', '## Within-chip determinism (rule e4)', '']
  L += [f'- {r["chip"]} bucket {r["bucket"]}: {r["r1_r2"]}' + (f' ({r["stages"]})' if r.get('stages') else '')
        for r in det_rows]
  L += ['', '## Unoptimised HLO, v5e against v6e (rule e5)', '']
  L += [f'- bucket {r["bucket"]}: {r["unoptimised"]}' + (f' ({r["stages"]})' if r['stages'] else '') for r in pre_rows]
  sem = [r for r in md_rows if r['level'] == 'semantic']
  L += ['', f'## Module paths with semantic differences ({len(sem)}; all levels in module_diffs.csv)', '']
  L += [f'- bucket {r["bucket"]}, {r["stage"]}, {r["module_path"]}: {r["only_v5e"]} units only on v5e, '
        f'{r["only_v6e"]} only on v6e' for r in sem[:60]]
  (pathlib.Path(out) / 'summary.md').write_text('\n'.join(L) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--sessions', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  sessions = listed(args.sessions)
  chips = sorted(chip_of(s) for s in sessions)
  if chips != list(CHIPS):
    raise SystemExit(f'{args.sessions} must list one v5e and one v6e session, not {sessions}')
  procs = processes(sessions)
  rows = checks(procs)
  valid, q = analyse(procs, args.out, rows)
  print((pathlib.Path(args.out) / 'summary.md').read_text())
  return 0 if valid else 1


if __name__ == '__main__':
  sys.exit(main())
