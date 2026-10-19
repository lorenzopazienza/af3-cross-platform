#!/usr/bin/env python3
"""Inputs for follow-up 6 (label `msa`): the 60 frozen PoseBusters inputs without MSAs or templates.

notes/analysis_plan.md section 10, row "Follow-up experiment 6, AlphaFold3's own data pipeline" (2026-10-19).
Each frozen input (inputs/manifest.csv) is copied with the protein's `unpairedMsa`, `pairedMsa` and `templates`
removed, so that AlphaFold3's data pipeline (run_alphafold.py --norun_inference) searches MSAs and templates
itself. Name, model seeds, protein sequence, chain ids, ligand CCD codes, dialect and version are unchanged.
The frozen input's SHA-256 is checked first, and the protein sequence must keep the SHA-256 in the manifest.

Writes data/inputs_official/<pdb_id>.json, inputs/official/manifest.csv and data/inputs_official.tar.gz
(the 60 JSONs plus manifest.csv, for the transfer to the NAS). Run from the repository root:
    python3 inputs/official/build_official_inputs.py
"""
import csv
import hashlib
import io
import json
import os
import sys
import tarfile

CUTOFF = "2021-09-30"
OUT_DIR = "data/inputs_official"
MANIFEST = "inputs/official/manifest.csv"
TARBALL = "data/inputs_official.tar.gz"
DROP = ("unpairedMsa", "pairedMsa", "templates")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    frozen = list(csv.DictReader(open("inputs/manifest.csv")))
    targets = {r["pdb_id"]: r for r in csv.DictReader(open("targets/posebusters_subset.csv"))}
    if len(frozen) != 60:
        sys.exit(f"expected 60 frozen inputs, found {len(frozen)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for r in frozen:
        raw = open(r["input_path"], "rb").read()
        if sha256_bytes(raw) != r["input_sha256"]:
            sys.exit(f"{r['pdb_id']}: frozen input SHA-256 mismatch")
        d = json.loads(raw)
        proteins = 0
        for entry in d["sequences"]:
            (kind, body), = entry.items()
            if kind == "protein":
                proteins += 1
                for k in DROP:
                    body.pop(k, None)
                if sha256_bytes(body["sequence"].encode()) != r["sequence_sha256"]:
                    sys.exit(f"{r['pdb_id']}: sequence SHA-256 mismatch")
                extra = set(body) - {"id", "sequence"}
                if extra:
                    sys.exit(f"{r['pdb_id']}: unexpected protein fields {sorted(extra)}")
            elif kind != "ligand":
                sys.exit(f"{r['pdb_id']}: unexpected entry {kind}")
        if proteins != 1:
            sys.exit(f"{r['pdb_id']}: expected one protein entry, found {proteins}")
        out = (json.dumps(d, indent=2) + "\n").encode()
        path = os.path.join(OUT_DIR, f"{r['pdb_id']}.json")
        open(path, "wb").write(out)
        t = targets[r["pdb_id"]]
        rows.append({
            "pdb_id": r["pdb_id"],
            "ligand_ccd": r["ligand_ccd"],
            "pilot": r["pilot"],
            "sequence_length": t["sequence_length"],
            "num_tokens": r["num_tokens"],
            "bucket": r["bucket"],
            "rcsb_initial_release": t["rcsb_initial_release"],
            "released_on_or_before_cutoff": "yes" if t["rcsb_initial_release"][:10] <= CUTOFF else "no",
            "frozen_input_sha256": r["input_sha256"],
            "sequence_sha256": r["sequence_sha256"],
            "official_input_path": path,
            "official_input_sha256": sha256_bytes(out),
            "official_input_bytes": len(out),
        })
    # Order for the NAS: the 3 longest sequences first (timing and memory), then the rest by pdb_id.
    rows.sort(key=lambda x: x["pdb_id"])
    longest = sorted(rows, key=lambda x: (-int(x["sequence_length"]), x["pdb_id"]))[:3]
    order = longest + [x for x in rows if x not in longest]
    for i, x in enumerate(order, 1):
        x["nas_order"] = i
    fields = ["nas_order"] + [k for k in order[0] if k != "nas_order"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(order)
    manifest = buf.getvalue().encode()
    open(MANIFEST, "wb").write(manifest)
    with tarfile.open(TARBALL, "w:gz") as tar:
        for x in order:
            tar.add(x["official_input_path"], arcname=f"inputs_official/{x['pdb_id']}.json")
        ti = tarfile.TarInfo("inputs_official/manifest.csv")
        ti.size = len(manifest)
        tar.addfile(ti, io.BytesIO(manifest))
    print(f"60 inputs in {OUT_DIR}; manifest {MANIFEST} sha256 {sha256_bytes(manifest)}")
    print(f"tarball {TARBALL} sha256 {sha256_bytes(open(TARBALL, 'rb').read())}")
    print("first 3 (longest):", [(x["pdb_id"], x["sequence_length"]) for x in longest])
    print("released on or before cutoff:", sum(x["released_on_or_before_cutoff"] == "yes" for x in rows))
    print("pilot:", [x["pdb_id"] for x in order if x["pilot"] == "yes"])


if __name__ == "__main__":
    main()
