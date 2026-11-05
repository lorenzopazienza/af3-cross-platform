# Section 10 draft: follow-up 10 on the whole PDB (label `large`), specification

Drafted 2026-11-04 in branch `fu10` (worktree `~/Desktop/af3-fu10`, from `789841a`). Nothing here is in
`notes/analysis_plan.md` yet. No input was built: no MSA, no input JSON. Lorenzo enters the row, unchanged or amended,
before any input of follow-up 10 is built.

Files of this draft (untracked until Lorenzo commits them):

- `targets/large/pdb_screen.py`: the screen (rules 1 to 7), read-only on the RCSB PDB, every answer cached in
  `data/pdb_large_screen/` (main checkout, ignored). It reruns offline in about 30 s. Rerun twice, it writes byte-identical outputs apart from the
  header line (run time).
- `targets/large/pdb_targets_amended.csv`, `pdb_screen_entries_amended.csv`, `pdb_screen_counts_amended.md`: the
  amended ligand rule (proposed).
- `targets/large/pdb_targets.csv`, `pdb_screen_entries.csv`, `pdb_screen_counts.md`: the literal rule, for comparison.
- `drafts/fu10_cost.py`: VM hours and cost of the design.

## Decision for Lorenzo: the ligand rule

The task's rule 2 says a ligand is drug-like if it is in the CCD, is not an additive, ion or cofactor "as the screen's
lists", and weighs 150 to 1,000 Da. Read literally, with the list of `~/Desktop/af3_drafts/fu10_pdb/pdb_large_screen.py`
unchanged, it admits many components that are not drug-like. In the large group of the literal selection (60 targets) these include
UDP, MGD, ACO (acetyl-CoA, twice), PNS (4'-phosphopantetheine, twice), SEP (phosphoserine, a modified residue), BTB
(a buffer) and SPM (spermine). Across the eligible entries there are also many lipids (EIC 22 times, PTY, PSF, PCW),
CoA thioesters and nucleotides. The control group had RET (retinal, covalent) 4 times.

The proposed amended rule keeps the screen's list and also excludes:

- components whose CCD type is not `non-polymer`, or with a PSI-MOD annotation (post-translational groups);
- components without carbon, or with an element other than C, H, N, O, S, P, F, Cl, Br, I, B, Se, Si (ions, metal
  clusters, metal complexes);
- an acyclic chain of 8 or more carbons (fatty acids, phospholipids, detergents, alkanes);
- a nucleoside 5', 2' or 3' phosphate (nucleotides, CoA and its thioesters, FAD, NAD and analogues, cyclic
  nucleotides);
- a run of three ethylene-glycol units (PEG);
- a named list (`EXTRA` in the script): buffers and chelators (BTB, BCN, B3P, NTA, EDT, CXS, NHE, PIN), a detergent
  (CA7), polyamines (SPM, SPK), tartrate (TLA), cofactors and chromophores without a nucleotide (CLA, BLA, BLR, LBV,
  IE5, CYC, PLP, PMP, H4B, TPP, MTE, FOL, THH, ZXF, FNR, P1T, PC, G3H, AGP), and ergosterol (ERG, as cholesterol is
  on the screen's list);
- at the instance level, instances that RCSB annotates `HAS_COVALENT_LINKAGE` (PoseBusters' ligands are
  non-covalent).

The rule was refined in four passes while the lists were read, each time because the output still held a component of
a category the rule already meant to exclude: first lipids, nucleotides and covalent ligands; then ions, metal
clusters, buffers and PEG; then bilins and free PLP; then cyclic AMP and ergosterol. No prediction exists for any of
these targets, so this is not selection on results. Each pass changed the control draw, because it changed the set of
clusters each stratum's permutation runs over. The large group was the same 34 targets in the last three passes.

| | literal rule | amended rule (proposed) |
|---|---|---|
| large targets (one per 30% cluster) | 60 (variant A 30) | 34 (variant A 17) |
| control targets | 60 (variant A 23) | 34 (variant A 11) |
| expected half-width of the headline difference, HW(V5,V6) / HW(V6,G) | 5.2 / 6.2 points | 6.9 / 8.2 points |
| cost as proposed, expected | 316.94 USD | 184.47 USD |

Recommendation: the amended rule. Its 34 large targets are above the floor of 25. The literal rule's extra 26 large
targets are mostly lipid, nucleotide-cofactor and buffer complexes. With them, the comparison would mix the size question
with ligand class.

## Row D: follow-up 10 on the whole PDB, specification

| 2026-11-0X | Follow-up experiment 10, larger inputs on the whole PDB (label `large`), specified after the main-study and follow-up outcomes were read and before any input of its targets was built or any run was made; reported as a follow-up, outside the pre-registered study, whatever it shows. It replaces the deferred design of 2026-10-29 and follows the Runs N' Poses screen of 2026-11-04 (11 independent large targets, not run). Question: do hardware-induced top-1 flips beyond recompilation grow or shrink when the input grows from at most 1,024 tokens to 1,025 to 1,536? (a) Targets: `targets/large/pdb_screen.py --ligand_rule amended`, read-only on the RCSB PDB (search of 2026-11-04T12:46:26Z; every answer cached under `data/pdb_large_screen/`, so the screen reruns offline with identical outputs). Source: entries released after 2021-09-30, X-ray or EM, resolution at most 3.0 Å, one protein entity, no nucleic acid, at least one non-polymer entity. Rule, fixed before any prediction: 1. one polypeptide(L) entity whose RCSB canonical sequence has the 20 standard letters only; not one of the main study's 60 or follow-up 11's 72 entries. 2. one ligand: the candidates are the entry's non-polymer components of 150 to 1,000 Da in AlphaFold 3's CCD that are not on the screen's additive, ion and cofactor list and not excluded by the amended rule (CCD type other than non-polymer; a PSI-MOD annotation; no carbon, or an element other than C, H, N, O, S, P, F, Cl, Br, I, B, Se, Si; an acyclic chain of 8 or more carbons; a nucleoside 5', 2' or 3' phosphate; three consecutive ethylene-glycol units; the named list `EXTRA` of buffers, chelators, a detergent, polyamines, cofactors, chromophores and ergosterol). Of all their instances in model 1 of the deposited mmCIF that RCSB does not annotate as covalently linked, the one with the most protein heavy atoms within 4 Å of its heavy atoms; ties, the lowest `label_asym_id`. An entry whose instance touches no protein atom is excluded. 3. Single-chain pocket: every protein heavy atom within 6 Å of that instance's heavy atoms (all alternate locations counted) is in one chain. 4. Size: AlphaFold 3's token count of that chain plus one copy of the ligand by CCD code (empty MSA, no templates; `inputs/af3_featurise.py`): large 1,025 to 1,536; control at most 1,024; 1,537 to 2,048 recorded for information, not run. 5. One target per RCSB 30% sequence cluster (Search API grouping; an entity not yet clustered is its own cluster): the member that meets rules 1 to 4 with the best resolution, then the earliest release, then the lowest PDB ID. 6. Control: clusters not in the large group, in strata of method (X-ray, EM) and resolution band (at most 2.0, 2.0 to 2.5, 2.5 to 3.0 Å), as many as the large group has in each; `numpy.random.default_rng(2026)` permutes each stratum's cluster IDs sorted as strings (strata in a fixed order); each cluster in turn gives its first member in that stratum that meets rules 1 to 4, until the stratum is full. 7. Variant A: the targets with exactly one deposited polymer chain. Counts in the large band (950 to 2,048 residues, every entry checked): 723 entries from the search (158 clusters); after rule 1, 712; rule 2, 204 (53 clusters); rule 3, 168 (46 clusters; it removes 36 entries and 7 clusters); 1,025 to 1,536 tokens, 128 entries in 34 clusters. Selected: large 34 (X-ray 12, EM 22; bucket 1,280: 24, 1,536: 10; variant A 17); control 34 (the same strata; buckets 256: 3, 512: 20, 768: 7, 1,024: 4; variant A 11; 62 entries checked in draw order); 1,537 to 2,048, 10 (not run). Every target was released after the training cutoff. List: `targets/large/pdb_targets_amended.csv`. Under the screen's lists alone (the literal rule), lipids, nucleotide cofactors, buffers, a modified residue and covalent ligands pass rule 2; that selection (60 large) is kept in `targets/large/pdb_targets.csv` for the record and not used. (b) Inputs: as the main study's: unpaired ColabFold MSAs from the public server (mode `env`), no templates, the ligand by CCD code, seeds 1 to 5, frozen with SHA-256 before any run. A target is replaced only for a permanent cause (AlphaFold 3 rejects its input, or its token count differs from the screen's): by the next member of its cluster that meets rules 1 to 4 (large) or the next cluster of its stratum's permutation (control), recorded in a row before any run. A server refusal or timeout is not a cause; the build stops and reruns from the cache. (c) Arms, main-study settings unchanged: V5 (`tpu_xla`, v5e, on demand), V6 (`tpu_xla`, v6e, Flex-start), G (`l4_xla`, L4, Spot). r1 on all 68 targets. r2 on G for all 68. r2 on V5 and V6 for a fixed subset of 10 large and 10 control targets, drawn by `numpy.random.default_rng(2026).choice` without replacement from each group's PDB IDs sorted as strings, large first: 7DIA, 7QUR, 8C8V, 8SX7, 8TZG, 8V1G, 8XB3, 9BWT, 9GZ2, 9X5O (large); 7UMP, 8BLP, 8CEY, 9ETQ, 9HKX, 9OHN, 9OMM, 9P4Y, 9ZB9, 9ZNO (control). TPU recompilation was bit-identical on all 60 main-study targets; if any subset target's r2 on V5 or V6 is not bit-identical to its r1, r2 on that arm is extended to all 68 targets, recorded in a row before that run, with its own budget check. No warm reruns; C is not run. (d) Contrasts, with the statistics of section 5 and the target-clustered bootstrap (10,000 replicates, seed 2026). Primary pairs HW(V5,V6) and HW(V6,G), each with HW minus RC of its first platform (RC(V5), RC(V6)) in each group; the reading against the second platform's RC is reported next to it. RC(V5) and RC(V6) are computed on the subset's targets of each group, so HW minus RC is the difference between HW over all the group's targets and RC over its subset targets, each resampled within its own targets. RC(G) uses all targets, paired. Headline, for each primary pair: the large minus control difference in HW minus RC for the top-1 flip rate, with a 95% interval from resampling targets within each group independently. Secondary: the success flip rate, flips that matter, HW(V5,G) and SD on each platform. Reading rule, per pair: if the interval lies above 0, the hardware effect grows with input size; if below 0, it shrinks; otherwise "no size difference detected at this sample size", stated with the interval's width. From the main study's per-target spread, the expected half-width is about 6.9 points for HW(V5,V6) and 8.2 for HW(V6,G). Variant A: the same statistics on the variant A targets (17 large, 11 control), a sensitivity analysis reported next to the primary result and not read with the rule. Exploratory, no reading rule: the per-seed top-1 flip of each primary pair against token count over the 68 targets, by logistic regression (slope per 256 tokens, 95% interval from the target-clustered bootstrap), with and without method as a covariate. M1: the reference is the rule-2 instance of the deposited model. Every M1 result (success flip, flips that matter, success rates) is flagged for EM (44 of 68 targets) and also reported on the X-ray subset (12 large, 12 control). The headline uses no reference structure. All targets are post-cutoff, so there is no post-cutoff subset. (e) Order: after follow-ups 11 and 12, under the parallel-arms row: on TPU, V5 then V6, after V6h5; on GPU, G after Gh5. Every run finishes by 2026-11-14 (results freeze 2026-11-15), else the arm is reported as incomplete. Launch by 2026-11-11 for the TPU arms (65.0 VM h in sequence) and by 2026-11-13 for G (21.6 h on 8 L4). (f) Cost, from `cloud/prices.csv` list prices (`drafts/fu10_cost.py`; process walls: main-study medians up to 1,024 tokens, the memory-limit runs of 2026-10-14 for 1,280 and 1,536; parts up to 8 h on TPU and 4 h on GPU, one target's runs in one part), expected and worst case: V5 on demand, 5 parts, 37.1 VM h, 57.88 and 79.56 USD; V6 Flex-start, 4 parts, 27.9 VM h, 37.63 and 50.98; G Spot, 39 parts, 165.3 VM h, 88.96 and 125.86. Total 184.47 expected, 256.40 worst. Cap 221 USD (1.2 times expected); floor 25 USD, after subtracting the expected cost of every running arm. The contingent r2 extension would add 32.66 (V5) and 21.02 (V6) and needs a cap amendment. (g) Stop rules: the inputs are frozen, the implementation is committed and the OSF addendum is registered before the first run; no run is dropped or selected on results; a preempted part is relaunched unchanged; each launch passes the budget rule; every run finishes by 2026-11-14. | Hardware flip rates above 1,024 tokens, on enough independent proteins |

## Counts per step (amended rule)

Large band (950 to 2,048 residues in the search), every entry checked:

| step | entries | X-ray | EM | clusters (30%) |
|---|---|---|---|---|
| from the search | 723 | 157 | 566 | 158 |
| rule 1: one polypeptide(L) entity, standard letters, no nucleic acid, not a study entry | 712 | 147 | 565 | 157 |
| rule 2: a drug-like ligand, an instance touching the protein | 204 | 54 | 150 | 53 |
| rule 3: single-chain pocket (6 Å) | 168 | 53 | 115 | 46 |
| rule 4, tokens 1,025 to 1,536 (large) | 128 | 37 | 91 | 34 |
| rule 4, tokens 1,537 to 2,048 (information) | 19 | 2 | 17 | 10 |
| rule 4, tokens at most 1,024 (control pool) | 16 | 14 | 2 | 5 |

Rule 3 removes 36 entries (7 clusters). Under the literal rule it removes 99 entries (14 clusters). 17 large-band
entries have no RCSB 30% cluster yet; none of them is selected.

Control band (at most 1,024 residues in the search): rules 1 and 2 are applied to every entry from the RCSB record;
rules 3 and 4 only to the entries drawn.

| step | entries | X-ray | EM | clusters (30%) |
|---|---|---|---|---|
| from the search | 33,596 | 31,525 | 2,025 | 5,782 |
| rule 1 | 33,064 | 31,064 | 2,000 | 5,642 |
| rule 2, from the RCSB record (before the instance check) | 16,039 | 15,554 | 485 | 1,827 |
| drawn and checked (rules 2 to 4) | 62 | | | |
| eligible and taken | 34 | 12 | 22 | 34 |

Of the 62 drawn: 22 had a pocket on more than one chain and 6 had no instance of a candidate ligand in model 1.

Draws per stratum (target / entries checked / clusters in the stratum): X-ray at most 2.0 Å, 7 / 7 / 1,173;
X-ray 2.0 to 2.5, 4 / 7 / 919; X-ray 2.5 to 3.0, 1 / 1 / 560; EM at most 2.0, 1 / 1 / 6; EM 2.0 to 2.5, 6 / 10 / 60;
EM 2.5 to 3.0, 15 / 36 / 126.

Selected:

| group | targets | variant A | X-ray / EM | by stratum | by bucket | tokens (min, median, max) |
|---|---|---|---|---|---|---|
| large | 34 | 17 (8 X-ray) | 12 / 22 | X-ray: at most 2.0: 7, 2.0 to 2.5: 4, 2.5 to 3.0: 1; EM: at most 2.0: 1, 2.0 to 2.5: 6, 2.5 to 3.0: 15 | 1,280: 24, 1,536: 10 | 1,032, 1,200, 1,523 |
| control | 34 | 11 (5 X-ray) | 12 / 22 | the same | 256: 3, 512: 20, 768: 7, 1,024: 4 | 195, 454, 938 |
| information | 10 | 8 | 2 / 8 | X-ray 2.5 to 3.0: 2; EM 2.0 to 2.5: 4, 2.5 to 3.0: 4 | 2,048: 10 | |

## Target lists (amended rule)

Ordered by method, then resolution. TPU r2: the seeded subset of 10 large and 10 control targets.

### Large group (1,025 to 1,536 tokens)

| PDB ID | ligand (CCD) | ligand name | tokens | bucket | method | resolution (Å) | released | chains | variant A | cluster (30%) | TPU r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9E0D | A1BDV | (2H-1,3-benzodioxol-5-yl)methanol | 1,246 | 1,280 | X-ray | 1.33 | 2024-11-27 | 2 | no | 1297_30 |  |
| 8SVL | WRC | N-[(1R)-2-(hydroxyamino)-2-oxo-1-(3',4... | 1,119 | 1,280 | X-ray | 1.504 | 2024-04-24 | 1 | yes | 1532_30 |  |
| 7DIA | H8O | (S)-[2,8-bis(trifluoromethyl)quinolin-... | 1,184 | 1,280 | X-ray | 1.55 | 2021-12-01 | 1 | yes | 11846_30 | yes |
| 9EJQ | A1BIX | (4S)-4-(3-ethoxyphenyl)-3-methyl-1-[(4... | 1,171 | 1,280 | X-ray | 1.87 | 2025-02-26 | 1 | yes | 703_30 |  |
| 8EXL | 799 | 2-methyl-2-(4-{2-[3-methyl-1-(propan-2... | 1,114 | 1,280 | X-ray | 1.989 | 2022-11-30 | 1 | yes | 596_30 |  |
| 8J79 | 141 | Oxypurinol | 1,343 | 1,536 | X-ray | 1.99 | 2023-05-24 | 2 | no | 1479_30 |  |
| 7Z3J | I3P | D-MYO-INOSITOL-1,4,5-TRIPHOSPHATE | 1,200 | 1,280 | X-ray | 2.0 | 2022-07-20 | 1 | yes | 4073_30 |  |
| 9FTR | A1IGB | amide modified swainsonine-configured ... | 1,053 | 1,280 | X-ray | 2.14 | 2024-10-09 | 1 | yes | 2251_30 |  |
| 8C8V | ILA | N-[ISOLEUCINYL]-N'-[ADENOSYL]-DIAMINOS... | 1,063 | 1,280 | X-ray | 2.2 | 2023-08-16 | 1 | yes | 3561_30 | yes |
| 9MFP | A1BK0 | 1-ethyl-N-[3-(methanesulfonamido)pheny... | 1,032 | 1,280 | X-ray | 2.27 | 2025-05-07 | 1 | yes | 4454_30 |  |
| 7NSN | MVL | (5R,6R,7S,8R)-5-(HYDROXYMETHYL)-5,6,7,... | 1,425 | 1,536 | X-ray | 2.29 | 2022-09-21 | 2 | no | 21276_30 |  |
| 9WLN | A1EXG | 7-(dimethylamino)-3-[4-[[7-[2-[(2~{S},... | 1,074 | 1,280 | X-ray | 2.9 | 2026-09-09 | 4 | no | 1203_30 |  |
| 9OVL | A1AR1 | 5-acetamido-8-O-(5-acetamido-9-O-acety... | 1,388 | 1,536 | EM | 1.68 | 2026-05-27 | 3 | no | 1949_30 |  |
| 9MXL | A1BUJ | (6R,12R)-6-(5-bromofuran-2-yl)-7,9-dim... | 1,516 | 1,536 | EM | 2.1 | 2025-08-06 | 1 | yes | 1774_30 |  |
| 7QUR | GE9 | 2,3,5-tris(iodanyl)benzamide | 1,271 | 1,280 | EM | 2.27 | 2022-06-01 | 3 | no | 24_30 | yes |
| 9QHP | A1JBK | (2~{R})-2-(4-methoxyphenyl)-5,7-bis(ox... | 1,430 | 1,536 | EM | 2.35 | 2026-02-11 | 4 | no | 1385_30 |  |
| 8SI6 | ZY8 | (4bS,8R,8aS,14bR)-7-(cyclopropylmethyl... | 1,310 | 1,536 | EM | 2.44 | 2023-05-17 | 4 | no | 8471_30 |  |
| 9BFT | A1AOF | (2S)-1-{[(1R,5R)-3-azabicyclo[3.1.0]he... | 1,069 | 1,280 | EM | 2.44 | 2024-05-08 | 3 | no | 873_30 |  |
| 9BWT | HCZ | 6-chloro-3,4-dihydro-2H-1,2,4-benzothi... | 1,037 | 1,280 | EM | 2.5 | 2024-09-04 | 2 | no | 2693_30 | yes |
| 8X94 | EZI | 4-(7-Hydroxy-2-isopropyl-4-oxoquinazol... | 1,135 | 1,280 | EM | 2.55 | 2024-08-14 | 4 | no | 1612_30 |  |
| 9Z22 | A1CY8 | Necrocide 1 | 1,241 | 1,280 | EM | 2.58 | 2026-06-10 | 4 | no | 2505_30 |  |
| 9VU1 | A1ETS | Liquiritin apioside | 1,158 | 1,280 | EM | 2.59 | 2026-07-15 | 4 | no | 8529_30 |  |
| 9X5O | 4KU | 2,2'-ethane-1,2-diylbis{5-[(sulfanylme... | 1,146 | 1,280 | EM | 2.61 | 2026-10-28 | 2 | no | 3673_30 | yes |
| 8SX7 | ZWY | 17-oxoandrost-5-en-3beta-yl hydrogen s... | 1,350 | 1,536 | EM | 2.7 | 2024-01-17 | 1 | yes | 2073_30 | yes |
| 8TZG | A1N | (2~{R},6~{S})-2,6-dimethyl-4-[6-[5-(1-... | 1,218 | 1,280 | EM | 2.7 | 2023-12-06 | 1 | yes | 2574_30 | yes |
| 8Y85 | 4KU | 2,2'-ethane-1,2-diylbis{5-[(sulfanylme... | 1,260 | 1,280 | EM | 2.73 | 2024-07-31 | 2 | no | 2168_30 |  |
| 9VT2 | TPF | 2-(2,4-DIFLUOROPHENYL)-1,3-DI(1H-1,2,4... | 1,523 | 1,536 | EM | 2.79 | 2026-09-30 | 1 | yes | 1636_30 |  |
| 8XB3 | YMN | Iobenguane | 1,256 | 1,280 | EM | 2.8 | 2024-06-26 | 1 | yes | 964_30 | yes |
| 9JO3 | A1L4E | 5-azanyl-2-[2,3,5,6-tetrakis(fluoranyl... | 1,142 | 1,280 | EM | 2.8 | 2025-02-26 | 4 | no | 2478_30 |  |
| 8K3X | BGI | (2S)-{[(2S,3S,4S)-2-amino-4-hydroxy-4-... | 1,166 | 1,280 | EM | 2.86 | 2023-10-18 | 2 | no | 18594_30 |  |
| 9GZ2 | XB2 | Mavacamten | 1,165 | 1,280 | EM | 2.9 | 2025-03-12 | 1 | yes | 5382_30 | yes |
| 9MGY | A1BLR | (2M)-2-(6-{[(3R)-1-methylpiperidin-3-y... | 1,061 | 1,280 | EM | 2.9 | 2025-02-26 | 1 | yes | 1578_30 |  |
| 8V1G | Y6F | MMV009108 | 1,498 | 1,536 | EM | 2.98 | 2024-11-20 | 1 | yes | 46971_30 | yes |
| 9OE4 | R0Z | elacridar | 1,323 | 1,536 | EM | 3.0 | 2026-06-03 | 1 | yes | 804_30 |  |

### Control group (at most 1,024 tokens)

| PDB ID | ligand (CCD) | ligand name | tokens | bucket | method | resolution (Å) | released | chains | variant A | cluster (30%) | TPU r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7KFO | IAC | 1H-INDOL-3-YLACETIC ACID | 195 | 256 | X-ray | 1.3 | 2021-11-10 | 1 | yes | 17083_30 |  |
| 8V4J | J7D | 1-deoxy-1-[formyl(hydroxy)amino]-5-O-p... | 236 | 256 | X-ray | 1.31 | 2023-12-13 | 1 | yes | 620_30 |  |
| 9LZG | A1L8C | 7,8-bis(oxidanyl)chromen-2-one | 500 | 512 | X-ray | 1.5 | 2026-02-25 | 2 | no | 248_30 |  |
| 9ETQ | A1H63 | 5-[4-[(7-ethyl-6-oxidanylidene-5~{H}-1... | 382 | 512 | X-ray | 1.59 | 2024-12-04 | 2 | no | 1198_30 | yes |
| 9DRA | A1BA2 | (6R)-3,4-dichloro-6-hydroxycyclohex-3-... | 329 | 512 | X-ray | 1.62 | 2024-10-09 | 2 | no | 4762_30 |  |
| 7UMP | A1Z | Vadadustat | 237 | 256 | X-ray | 1.8 | 2022-05-18 | 1 | yes | 2773_30 | yes |
| 8CEY | UEF | ~{N}-(3,3-dimethylcyclobutyl)imidazo[5... | 335 | 512 | X-ray | 1.95 | 2024-02-21 | 3 | no | 2682_30 | yes |
| 8AHH | M4X | 5-cyano-~{N}-methyl-pyrazolo[1,5-a]pyr... | 312 | 512 | X-ray | 2.037 | 2022-09-14 | 1 | yes | 4533_30 |  |
| 9OMM | GMP | GUANOSINE | 299 | 512 | X-ray | 2.25 | 2025-10-29 | 2 | no | 28281_30 | yes |
| 8T7N | ZT3 | N-(4-tert-butylphenyl)-7,8-dimethyl-5,... | 454 | 512 | X-ray | 2.26 | 2024-08-28 | 2 | no | 1107_30 |  |
| 8ITA | R0U | 3-(4-hydroxyphenyl)-6-methoxy-5,7-bis(... | 489 | 512 | X-ray | 2.5 | 2023-08-23 | 1 | yes | 1234_30 |  |
| 9SDI | A1JNJ | 6-(1~{H}-pyrazol-4-yl)pyridine-3-carbo... | 291 | 512 | X-ray | 2.53 | 2025-12-31 | 2 | no | 934_30 |  |
| 9ZB9 | A1C1E | {[(2-{3-[(3-fluoro-4-methoxyphenyl)car... | 338 | 512 | EM | 1.95 | 2026-09-09 | 6 | no | 9176_30 | yes |
| 8EEP | IHP | INOSITOL HEXAKISPHOSPHATE | 267 | 512 | EM | 2.2 | 2023-02-15 | 1 | yes | 416_30 |  |
| 9JDC | LH6 | (2~{S})-2-(methylamino)-3-[4-[(1~{E})-... | 500 | 512 | EM | 2.33 | 2025-01-01 | 2 | no | 1028_30 |  |
| 8X1S | I2U | 5-chloranyl-2-oxidanyl-3-thiophen-2-yl... | 796 | 1,024 | EM | 2.37 | 2025-05-14 | 2 | no | 1887_30 |  |
| 7MIP | ZG4 | N-(1-{2-[(cyclopropanesulfonyl)amino]-... | 624 | 768 | EM | 2.4 | 2021-10-13 | 4 | no | 1507_30 |  |
| 9LQY | A1EK3 | Azelaprag | 417 | 512 | EM | 2.4 | 2026-02-04 | 2 | no | 3413_30 |  |
| 9OHN | A1CBF | (1P)-1-{4-(benzylamino)-2-methyl-1-[2-... | 861 | 1,024 | EM | 2.46 | 2026-05-27 | 12 | no | 643_30 | yes |
| 7UG0 | TB1 | (3S)-3-(BENZYLOXY)-L-ASPARTIC ACID | 435 | 512 | EM | 2.55 | 2023-03-29 | 1 | yes | 722_30 |  |
| 9P4Y | A1CG9 | (4R)-8-bromo-2-(furan-2-yl)-N-(2-methy... | 549 | 768 | EM | 2.56 | 2025-07-30 | 1 | yes | 3193_30 | yes |
| 9OKG | UKX | (2S)-2-(2-oxidanylidenepyrrolidin-1-yl... | 616 | 768 | EM | 2.58 | 2025-12-10 | 1 | yes | 4140_30 |  |
| 8BLP | 5D3 | 10-(4-ethylphenyl)sulfonyl-~{N}-(thiop... | 397 | 512 | EM | 2.6 | 2023-10-04 | 3 | no | 7544_30 | yes |
| 9UWL | A1EQN | SRX246 | 438 | 512 | EM | 2.6 | 2026-02-04 | 2 | no | 9440_30 |  |
| 7X01 | 7XQ | (1R,2S,3S,4R,5R)-3-(6-aminopurin-9-yl)... | 573 | 768 | EM | 2.62 | 2022-08-10 | 12 | no | 14834_30 |  |
| 9OSW | A1CER | (3M)-2'-chloro-N-{5-[(1S,2S)-2-(4-cyan... | 938 | 1,024 | EM | 2.67 | 2025-09-17 | 4 | no | 4613_30 |  |
| 7SIL | 9IG | 3-(2-chlorophenyl)-N-[(1R)-1-(3-methox... | 899 | 1,024 | EM | 2.7 | 2022-01-19 | 2 | no | 1997_30 |  |
| 8W0R | A1AEU | 1-methyl-1'-[(oxan-4-yl)methyl]-5-(tri... | 265 | 512 | EM | 2.8 | 2024-06-19 | 2 | no | 9298_30 |  |
| 8JPD | STU | STAUROSPORINE | 723 | 768 | EM | 2.81 | 2023-08-09 | 1 | yes | 3697_30 |  |
| 9HKX | ADN | ADENOSINE | 491 | 512 | EM | 2.83 | 2025-02-19 | 4 | no | 382_30 | yes |
| 9ZNN | A1C3G | 4-({(2S,3S)-1-amino-3-{[(2S,3R,4S,5R)-... | 481 | 512 | EM | 2.9 | 2026-02-04 | 2 | no | 4820_30 |  |
| 9ZNO | A1C3G | 4-({(2S,3S)-1-amino-3-{[(2S,3R,4S,5R)-... | 427 | 512 | EM | 2.91 | 2026-02-04 | 2 | no | 2456_30 | yes |
| 8GOF | LYA | 2-{4-[2-(2-AMINO-4-OXO-4,7-DIHYDRO-3H-... | 575 | 768 | EM | 3.0 | 2022-10-05 | 1 | yes | 5224_30 |  |
| 9BIQ | A1APN | N-(7-benzoyl-2,3-dihydro-1,4-benzodiox... | 657 | 768 | EM | 3.0 | 2025-02-05 | 2 | no | 7010_30 |  |

### Information band (1,537 to 2,048 tokens; not run)

| PDB ID | ligand (CCD) | ligand name | tokens | bucket | method | resolution (Å) | released | chains | variant A | cluster (30%) | TPU r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9FU2 | A1IGE | (3~{a}~{S})-1-(2-methoxypyridin-4-yl)-... | 1,997 | 2,048 | X-ray | 2.581 | 2025-07-09 | 1 | yes | 511_30 |  |
| 8BCH | QA9 | 1-(4-aminophenyl)sulfonylguanidine | 1,761 | 2,048 | X-ray | 2.87 | 2023-03-08 | 1 | yes | 1355_30 |  |
| 8F0P | X7L | N-[6-(cyclopentylmethoxy)-1,3-benzothi... | 1,645 | 2,048 | EM | 2.2 | 2023-04-12 | 1 | yes | 1368_30 |  |
| 9SEA | PSF | 1,2-DICAPROYL-SN-PHOSPHATIDYL-L-SERINE | 1,797 | 2,048 | EM | 2.23 | 2025-11-05 | 1 | yes | 3782_30 |  |
| 8GKC | X5O | denifanstat | 1,703 | 2,048 | EM | 2.45 | 2023-05-24 | 2 | no | 2865_30 |  |
| 8W2L | M05 | 4-(4-chlorobenzyl)-1-(7H-pyrrolo[2,3-d... | 1,558 | 2,048 | EM | 2.45 | 2024-04-17 | 4 | no | 8471_30 |  |
| 9VSU | FK5 | 8-DEETHYL-8-[BUT-3-ENYL]-ASCOMYCIN | 1,558 | 2,048 | EM | 2.82 | 2026-09-30 | 1 | yes | 1636_30 |  |
| 9LGC | R1Q | vincristine | 1,618 | 2,048 | EM | 2.95 | 2025-10-01 | 1 | yes | 2569_30 |  |
| 9WZU | A1E06 | Enfumafungin | 1,926 | 2,048 | EM | 2.97 | 2026-02-18 | 1 | yes | 5520_30 |  |
| 7Y1J | BJX | Repaglinide | 1,578 | 2,048 | EM | 3.0 | 2023-06-14 | 1 | yes | 2100_30 |  |

## Budget and calendar

Cost per arm (`analysis/.venv/bin/python drafts/fu10_cost.py --targets targets/large/pdb_targets_amended.csv`):

| arm | parts | processes | VM h | on demand USD | Spot USD | Flex-start USD | worst case, on demand |
|---|---|---|---|---|---|---|---|
| V5 | 5 | 440 | 37.1 | 57.88 | 20.87 | - | 79.56 |
| V6 | 4 | 440 | 27.9 | 82.79 | 49.67 | 37.63 | 112.17 |
| G | 39 | 680 | 165.3 | 148.29 | 88.96 | - | 209.80 |

As proposed (V5 on demand, V6 Flex-start, G Spot): 184.47 USD expected, 256.40 worst case (the worst case at the
proposed price: V6 50.98, G 125.86). Cap 221 USD. With r2 on V5 and V6 for all 68 targets (the contingency, or a
choice now): V5 90.54, V6 58.65 Flex-start, total 226.16 expected.

The process walls above 1,024 tokens come from one process each (7D5C padded to the bucket, memory-limit runs of
2026-10-14): V5 399 and 604 s, V6 290 and 421 s, G 1,143 and 1,751 s for buckets 1,280 and 1,536. A G target of
1,536 tokens with r1 and r2 (10 processes) needs about 4.9 h, more than the 4 h part size, so it gets a part of its
own (the longest G part is 5.2 h).

Credit (to be read from the console before each launch): 988.30 USD before H r2 (row of 2026-11-04). H r2 spent
52.40, so about 935.90 now.

| step | expected USD | worst USD | balance after, expected | balance after, worst |
|---|---|---|---|---|
| balance, about | | | 935.90 | 935.90 |
| follow-up 11 (V5r, V6r, Gr) | 84.31 | 116.98 | 851.59 | 818.92 |
| follow-up 12 (V6h5, Gh5) | 20.03 | 27.75 | 831.56 | 791.17 |
| follow-up 10 (V5, V6, G) | 184.47 | 256.40 | 647.09 | 534.77 |

Calendar, at most one TPU arm and one GPU arm at a time (row of 2026-11-04). The dates assume that follow-up 11 is
launched on 2026-11-05 and that the follow-up 10 inputs are frozen by 2026-11-08.

| platform | arm | VM h | parts at once | wall, about | window |
|---|---|---|---|---|---|
| TPU | V5r, V6r, V6h5 | 48.6 | 1 | 49 h | 2026-11-05 to 11-07 |
| TPU | V5 (follow-up 10) | 37.1 | 1 | 37 h | 2026-11-08 to 11-09 |
| TPU | V6 (follow-up 10) | 27.9 | 1 | 28 h | 2026-11-09 to 11-10 |
| GPU | Gr, Gh5 | 63.6 | 8 | 12 h | 2026-11-05 to 11-06 |
| GPU | G (follow-up 10) | 165.3 | 8 | 22 h | 2026-11-08 to 11-09 |

Before any run, follow-up 10 needs this row, the OSF addendum, 68 MSAs from the ColabFold server (34 sequences of
1,000 to 1,500 residues), the inputs and an implementation row. The implementation covers the plans, M1 references cut
from the deposited mmCIF instance, and `analysis/large.py`. The latest launches that still finish by 2026-11-14 24:00
UTC, with the deadline margin of 1.25: TPU V5 on 2026-11-11 (V6 follows), G on 2026-11-13. Running V5 and V6 in
parallel (37 h instead of 65 h) would need its own launch-setting row (two TPU arms at once). It is only worth doing
if the inputs are late.
