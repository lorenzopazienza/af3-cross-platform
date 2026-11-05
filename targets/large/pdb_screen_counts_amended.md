# Follow-up 10 (large) on the whole PDB: counts per step (targets/large/pdb_screen.py, ligand rule amended)

Screen run 2026-11-04T16:07:09+00:00; RCSB search 2026-11-04T12:46:26+00:00: 724 entries of 950 to 2048 residues and 33596 of at most 1024 (released after 2021-09-30, X-ray or EM, resolution <= 3.0 A, protein only, one protein entity, a non-polymer entity).

## Large band (950 to 2,048 residues in the search): every entry checked

| step | entries | X-ray | EM | clusters (30%) |
|---|---|---|---|---|
| from the search | 723 | 157 | 566 | 158 |
| rule 1: one polypeptide(L) entity, standard letters, no nucleic acid, not a PoseBusters study entry | 712 | 147 | 565 | 157 |
| rule 2: a drug-like ligand in the AlphaFold3 CCD, an instance touching the protein | 204 | 54 | 150 | 53 |
| rule 3: single-chain pocket (6 A) | 168 | 53 | 115 | 46 |
| rule 4: featurised | 168 | 53 | 115 | 46 |
| tokens 1,025 to 1,536 (large) | 128 | 37 | 91 | 34 |
| tokens 1,537 to 2,048 (information) | 19 | 2 | 17 | 10 |
| tokens at most 1,024 | 16 | 14 | 2 | 5 |

Entries without an RCSB 30% cluster (each its own cluster): 17.

Rule 3 removes 36 entries of the large band.

## Selected (rule 5: one per 30% cluster)

| group | targets | variant A (one deposited chain) | by stratum | by bucket |
|---|---|---|---|---|
| large 1,025 to 1,536 | 34 | 17 | EM 2.0-2.5: 6, EM 2.5-3.0: 15, EM <=2.0: 1, X-ray 2.0-2.5: 4, X-ray 2.5-3.0: 1, X-ray <=2.0: 7 | 1280: 24, 1536: 10 |
| control at most 1,024 | 34 | 11 | EM 2.0-2.5: 6, EM 2.5-3.0: 15, EM <=2.0: 1, X-ray 2.0-2.5: 4, X-ray 2.5-3.0: 1, X-ray <=2.0: 7 | 256: 3, 512: 20, 768: 7, 1024: 4 |
| information 1,537 to 2,048 | 10 | 8 | EM 2.0-2.5: 4, EM 2.5-3.0: 4, X-ray 2.5-3.0: 2 | 2048: 10 |

Control: 62 entries checked in draw order; reasons: pocket on more than one chain 22, no instance of a drug-like ligand in model 1 6, eligible 34
