# Follow-up 10 (large) on the whole PDB: counts per step (targets/large/pdb_screen.py, ligand rule screen)

Screen run 2026-11-04T16:09:37+00:00; RCSB search 2026-11-04T12:46:26+00:00: 724 entries of 950 to 2048 residues and 33596 of at most 1024 (released after 2021-09-30, X-ray or EM, resolution <= 3.0 A, protein only, one protein entity, a non-polymer entity).

## Large band (950 to 2,048 residues in the search): every entry checked

| step | entries | X-ray | EM | clusters (30%) |
|---|---|---|---|---|
| from the search | 723 | 157 | 566 | 158 |
| rule 1: one polypeptide(L) entity, standard letters, no nucleic acid, not a PoseBusters study entry | 712 | 147 | 565 | 157 |
| rule 2: a drug-like ligand in the AlphaFold3 CCD, an instance touching the protein | 354 | 80 | 274 | 90 |
| rule 3: single-chain pocket (6 A) | 255 | 77 | 178 | 76 |
| rule 4: featurised | 255 | 77 | 178 | 76 |
| tokens 1,025 to 1,536 (large) | 203 | 57 | 146 | 60 |
| tokens 1,537 to 2,048 (information) | 28 | 3 | 25 | 13 |
| tokens at most 1,024 | 19 | 17 | 2 | 7 |

Entries without an RCSB 30% cluster (each its own cluster): 17.

Rule 3 removes 99 entries of the large band.

## Selected (rule 5: one per 30% cluster)

| group | targets | variant A (one deposited chain) | by stratum | by bucket |
|---|---|---|---|---|
| large 1,025 to 1,536 | 60 | 30 | EM 2.0-2.5: 10, EM 2.5-3.0: 31, EM <=2.0: 2, X-ray 2.0-2.5: 7, X-ray 2.5-3.0: 1, X-ray <=2.0: 9 | 1280: 38, 1536: 22 |
| control at most 1,024 | 60 | 23 | EM 2.0-2.5: 10, EM 2.5-3.0: 31, EM <=2.0: 2, X-ray 2.0-2.5: 7, X-ray 2.5-3.0: 1, X-ray <=2.0: 9 | 256: 7, 512: 28, 768: 20, 1024: 5 |
| information 1,537 to 2,048 | 13 | 8 | EM 2.0-2.5: 6, EM 2.5-3.0: 4, X-ray 2.5-3.0: 3 | 2048: 13 |

Control: 132 entries checked in draw order; reasons: pocket on more than one chain 72, eligible 60
