"""Layout of a generated booktabs tabular for the supplement, cells unchanged (style round of 2026-11-20: no table of
the supplement prints below 8 pt). Used by relabel_tables.py and center_guides.py; analysis/paper_assets_isc.py has
the same rules for its supplement tables.

    layout(text, wrap=None, spec=None, stack=None, group=None, longtable=False, name='') -> text

- wrap: header cells longer than `wrap` printed characters are set on several lines, broken at spaces (inside a long
  identifier, after an underscore), in a nested tabular aligned like the column.
- spec: a new column specification with the same number of columns; P{w} is a ragged-right paragraph column of
  width w.
- stack: {column index: 'bracket' | 'semicolon'}: a body cell "estimate [low, high]" (or "estimate (x)") prints the
  bracket on a second line; with 'semicolon', each item of a list "a; b" goes on its own line.
- group: (label, sub, join) for a header that repeats a qualifier across columns: label(cell) is the group of a
  header cell (None: no group), sub(cell) the text left in the column, and join(label, sub) must give the cell back.
  Consecutive columns of one group get one spanning header line above them, with a \\cmidrule under it.
- longtable: the table becomes a longtable that breaks across pages, with the header repeated on every page. Its
  first row is \\tablecaptionrow (empty by default; the supplement defines it as the table's \\caption and \\label),
  and the repeated header is preceded by \\tablecontinued.

Every change is checked: the cells read back from the new text (nested tabulars removed, their lines joined with a
space, or with nothing after a broken identifier) must equal the old cells one by one, or the script stops.
"""
import re

PCOL = r'>{\raggedright\arraybackslash}p{%s}'
NEST = re.compile(r'\\begin\{tabular\}\[[tb]\]\{@\{\}[lrc]@\{\}\}(.*?)\\end\{tabular\}')


def expand_spec(spec):
  """P{w} -> ragged-right p column."""
  return re.sub(r'P\{([^}]*)\}', lambda m: PCOL % m.group(1), spec)


def columns(spec):
  """Alignment letter of every column of a tabular spec (a p column counts as l)."""
  s = re.sub(r'[@>]\{(?:[^{}]|\{[^{}]*\})*\}', '', spec)
  s = re.sub(r'[pPm]\{[^}]*\}', 'l', s)
  return [c for c in s if c in 'lrc']


def visible(s):
  """Printed characters, roughly: a command counts as one."""
  s = re.sub(r'\\[A-Za-z]+(\{\})?', 'x', s)
  return len(re.sub(r'[{}$\\]', '', s))


def balanced(s):
  return s.count('{') == s.count('}') and s.count('$') % 2 == 0


def wrap_lines(text, width):
  """Greedy lines of at most `width` printed characters. Returns [(line, glued)]: glued means the line continues a
  word broken after \\_ (read back without a space)."""
  tokens = []      # (piece, glued to the previous piece)
  for word in text.split(' '):
    parts = [word]
    if visible(word) > width and '\\_' in word:
      parts = [p for p in re.split(r'(?<=\\_)', word) if p]
    tokens += [(p, i > 0) for i, p in enumerate(parts)]
  merged = []      # never break inside braces or math, nor around "=" ("n = 1" stays on one line)
  for p, glued in tokens:
    if merged and (not balanced(merged[-1][0]) or p == '=' or merged[-1][0].endswith(' =')):
      merged[-1] = (merged[-1][0] + ('' if glued else ' ') + p, merged[-1][1])
    else:
      merged.append((p, glued))
  lines = []
  for p, glued in merged:
    joined = lines[-1][0] + ('' if glued else ' ') + p if lines else None
    if lines and visible(joined) <= width:
      lines[-1] = (joined, lines[-1][1])
    else:
      lines.append((p, glued))
  return lines


def nest(lines, align, pos):
  """Lines [(text, glued)] in a nested tabular; one line stays as it is."""
  if len(lines) == 1:
    return lines[0][0]
  out = lines[0][0]
  for text, _ in lines[1:]:
    out += ('\\\\\\relax ' if text.startswith(('[', '*')) else '\\\\') + text
  return f'\\begin{{tabular}}[{pos}]{{@{{}}{align}@{{}}}}{out}\\end{{tabular}}'


def read_back(cell):
  """A cell as printed text: nested tabulars joined (a space between lines, none after a broken identifier)."""
  def join(m):
    parts = m.group(1).replace('\\\\\\relax ', '\\\\').split('\\\\')
    out = parts[0]
    for p in parts[1:]:
      out += ('' if out.endswith('\\_') else ' ') + p
    return out
  return NEST.sub(join, cell).strip()


def split_row(line):
  """Cells of a tabular row 'a & b \\\\' (None for rules, \\multicolumn rows and other lines)."""
  s = line.strip()
  if not s.endswith('\\\\') or s.startswith(('\\multicolumn', '\\toprule', '\\midrule', '\\bottomrule')):
    return None
  return [c.strip() for c in s[:-2].rstrip().split(' & ')]


def stack_cell(cell, how, align):
  if how == 'bracket':
    m = re.fullmatch(r'(.*\S) ((?:\[|\().*(?:\]|\)))', cell)
    if not m or not balanced(m.group(1)) or not balanced(m.group(2)):
      return cell
    return nest([(m.group(1), False), (m.group(2), False)], align, 't')
  if how == 'semicolon':
    items = cell.split('; ')
    return nest([(i + ';', False) for i in items[:-1]] + [(items[-1], False)], align, 't')
  raise ValueError(how)


def layout(text, wrap=None, spec=None, stack=None, group=None, longtable=False, name=''):
  lines = text.split('\n')
  i0 = next(i for i, l in enumerate(lines) if l.startswith('\\begin{tabular}{'))
  old_spec = re.fullmatch(r'\\begin\{tabular\}\{(.*)\}', lines[i0]).group(1)
  new_spec = expand_spec(spec) if spec else old_spec
  aligns = columns(new_spec)
  if len(aligns) != len(columns(old_spec)):
    raise SystemExit(f'{name}: new column spec has {len(aligns)} columns, the table {len(columns(old_spec))}')
  lines[i0] = '\\begin{tabular}{' + new_spec + '}'
  ih = lines.index('\\toprule', i0) + 1
  if lines[ih + 1] != '\\midrule':
    raise SystemExit(f'{name}: expected one header row')
  head = split_row(lines[ih])
  old_body = [split_row(l) for l in lines[ih + 2:]]
  groups = [None] * len(head)
  if group:
    label, sub, join = group
    groups = [label(c) for c in head]
    top, rules, j = [], [], 0
    while j < len(head):
      k = j
      while groups[j] is not None and k + 1 < len(head) and groups[k + 1] == groups[j]:
        k += 1
      if groups[j] is None:
        top.append('')
      else:
        top.append(f'\\multicolumn{{{k - j + 1}}}{{c}}{{{groups[j]}}}')
        rules.append(f'\\cmidrule(lr){{{j + 1}-{k + 1}}}')
      j = k + 1
    cells = [sub(c) if g is not None else c for c, g in zip(head, groups)]
    new_head = [' & '.join(top) + ' \\\\', ' '.join(rules)]
  else:
    cells, new_head = list(head), []
  if wrap:
    cells = [nest(wrap_lines(c, wrap), a, 'b') for c, a in zip(cells, aligns)]
  new_head.append(' & '.join(cells) + ' \\\\')
  lines[ih:ih + 1] = new_head
  im = ih + len(new_head)
  if stack:
    for i in range(im + 1, len(lines)):
      row = split_row(lines[i])
      if row is None:
        continue
      for j, how in stack.items():
        row[j] = stack_cell(row[j], how, aligns[j])
      lines[i] = ' & '.join(row) + ' \\\\'
  # check: header and body read back as before
  back = [read_back(c) for c in split_row(lines[im - 1])]
  back = [c if g is None else join(g, c) for c, g in zip(back, groups)]
  if back != head:
    raise SystemExit(f'{name}: header cells changed by the layout:\n  {head}\n  {back}')
  new_body = [split_row(l) for l in lines[im + 1:]]
  if [r and [read_back(c) for c in r] for r in new_body] != old_body:
    raise SystemExit(f'{name}: body cells changed by the layout')
  if longtable:
    n = len(aligns)
    ib = max(i for i, l in enumerate(lines) if l == '\\bottomrule')
    ie = lines.index('\\end{tabular}', ib)
    head_block = lines[ih - 1:im + 1]          # \toprule ... \midrule
    edge = '@{}' if new_spec.startswith('@{}') else ''
    # a group heading row (\multicolumn) never ends a page: \\* keeps it with the row below
    for i in range(im + 1, ib):
      if lines[i].startswith('\\multicolumn') and lines[i].endswith(' \\\\'):
        lines[i] += '*'
    lines = (lines[:i0] +
             ['\\providecommand{\\tablecaptionrow}{}\\providecommand{\\tablecontinued}{(continued)}%',
              '\\begin{longtable}{' + new_spec + '}', '\\tablecaptionrow'] + head_block + ['\\endfirsthead',
              f'\\multicolumn{{{n}}}{{{edge}l}}{{\\tablecontinued}} \\\\'] + head_block +
             ['\\endhead', '\\bottomrule', '\\endfoot'] + lines[im + 1:ib] + ['\\end{longtable}'] + lines[ie + 1:])
  return '\n'.join(lines)
