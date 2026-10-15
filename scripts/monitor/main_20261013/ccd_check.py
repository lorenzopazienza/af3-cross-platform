"""CCD check (no outcome): every CCD code of the frozen inputs, its entry in two components.cif files.

    python3 ccd_check.py OLD_components.cif NEW_components.cif CODES.json OUT.tsv

For each code: targets using it, _chem_comp.pdbx_initial_date and pdbx_modified_date, the SHA-256 of its whole
data_ block in each file, and whether the blocks are identical. Also the latest initial/modified date in the
whole NEW file (the release it belongs to).
"""
import hashlib, json, sys

old_path, new_path, codes_path, out = sys.argv[1:5]
codes = json.load(open(codes_path))
want = set(codes['ligands']) | set(codes['standard_residues'])


def blocks(path, latest=False):
  out, cur, buf, dates = {}, None, [], {'initial': '', 'modified': ''}
  with open(path, 'rb') as f:
    for line in f:
      if line.startswith(b'data_'):
        if cur in want:
          out[cur] = b''.join(buf)
        cur, buf = line[5:].strip().decode(), []
      if cur in want:
        buf.append(line)
      if latest and line.startswith((b'_chem_comp.pdbx_initial_date', b'_chem_comp.pdbx_modified_date')):
        k = 'initial' if b'initial' in line else 'modified'
        v = line.split()[1].decode()
        if v[:1].isdigit() and v > dates[k]:
          dates[k] = v
  if cur in want:
    out[cur] = b''.join(buf)
  return out, dates


old, _ = blocks(old_path)
new, latest = blocks(new_path, latest=True)


def field(b, name):
  for line in b.splitlines():
    if line.startswith(name):
      return line.split()[1].decode()
  return ''


rows = []
for c in sorted(want, key=lambda c: (c not in codes['ligands'], c)):
  o, n = old.get(c), new.get(c)
  rows.append([c, 'ligand' if c in codes['ligands'] else 'standard residue', ' '.join(codes['ligands'].get(c, ['all 60'])),
               field(n or b'', b'_chem_comp.pdbx_initial_date'), field(n or b'', b'_chem_comp.pdbx_modified_date'),
               hashlib.sha256(o).hexdigest()[:16] if o else 'MISSING', hashlib.sha256(n).hexdigest()[:16] if n else 'MISSING',
               'yes' if o and o == n else 'NO'])
with open(out, 'w') as f:
  f.write('code\tkind\ttargets\tinitial_date\tmodified_date\tblock_sha256_2026-10-08\tblock_sha256_2026-10-13\tidentical\n')
  f.writelines('\t'.join(r) + '\n' for r in rows)
print(f'{len(rows)} codes; identical blocks {sum(r[-1] == "yes" for r in rows)}; '
      f'latest modified date among them {max(r[4] for r in rows)}; '
      f'latest dates in the whole file: initial {latest["initial"]}, modified {latest["modified"]}')
