#!/bin/bash
# CCD recheck (row (a) of the draft section 10 rows): downloads the current components.cif from wwPDB (the URL
# libcifpp's build uses), and compares it with the 2026-10-08 copy: whole file (size, SHA-256), then the 80 entry
# blocks of the frozen inputs' CCD codes (ccd_check.py). Writes ccd_recheck_<UTC date>.txt next to this script.
# No output of any run is read. Run from the repository root: bash scripts/monitor/main_20261013/ccd_recheck.sh
set -eu
H=results/analysis/main_monitor_20261013
S=scripts/monitor/main_20261013   # ccd_check.py; outputs go under $H
OLD=third_party/alphafold3/.venv/lib/python3.12/site-packages/share/libcifpp/components.cif
URL=https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
D=$(mktemp -d "${TMPDIR:-/tmp}/ccd_recheck.XXXXXX")
NOW=$(date -u +%Y-%m-%dT%H:%MZ)
OUT="$H/ccd_recheck_$(date -u +%Y%m%d).txt"
LM=$(curl -sI "$URL" | tr -d '\r' | sed -n 's/^[Ll]ast-[Mm]odified: //p')
curl -sf -o "$D/components.cif.gz" "$URL"
gunzip "$D/components.cif.gz"
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
{
  echo "CCD recheck at $NOW: $URL (Last-Modified: ${LM:-not given})"
  echo "  2026-10-08 copy: $(wc -c < "$OLD" | tr -d ' ') bytes, SHA-256 $(sha "$OLD")"
  echo "  current copy:    $(wc -c < "$D/components.cif" | tr -d ' ') bytes, SHA-256 $(sha "$D/components.cif")"
  if cmp -s "$OLD" "$D/components.cif"; then echo "  whole file: UNCHANGED"; else echo "  whole file: CHANGED"; fi
  python3 "$S/ccd_check.py" "$OLD" "$D/components.cif" "$H/ccd_codes.json" "$H/ccd_codes_recheck_$(date -u +%Y%m%d).tsv" | sed 's/^/  entries: /'
  awk -F'\t' 'NR > 1 && $8 != "yes" { print "  CHANGED entry: " $1 " (" $2 ", " $3 "), modified " $5 }' "$H/ccd_codes_recheck_$(date -u +%Y%m%d).tsv"
} | tee "$OUT"
rm -rf "$D"
