# Licences of the third-party data in records R1 and R2

Prepared on 2026-11-08 from the pages cited below; extended on 2026-11-17 to freeze `4ec185c5` (follow-ups 13 to 16,
`memlimit_ah`, `warm_th`), with the open point on the XLA dumps of follow-up 16 (last section), and on 2026-11-18 to the
copy of the code in R1. This table says what each source states; it is not legal advice. "Unclear"
means the source does not settle the point, and no guess is made.

"Derived alignments" asks whether alignments computed against the source may be redistributed. No source below speaks of
alignments by name. The column gives what follows from the licence text: CC0 places no condition, and CC BY 4.0 allows
sharing and adaptation with attribution.

## Sources

| source | what ends up in the records | licence, as stated | official statement | derived alignments | attribution |
|---|---|---|---|---|---|
| PoseBusters benchmark set (Zenodo record 8278563) | target identities and screen results only (`targets/`), no coordinates | CC BY 4.0 | https://zenodo.org/records/8278563 | not applicable, no alignment comes from it | required: Buttenschoen, M. et al. *Chemical Science* 15, 3130-3139 (2024) |
| PoseBusters V2 identifier list (`posebusters_pdb_ccd_ids.txt`, 308 identifiers, attached to a GitHub issue) | the identifiers, in `targets/posebusters_v2_screen.csv` | **unclear**: the attachment carries no licence statement of its own, and the record above has version v1 and does not use the V1 or V2 labels | https://github.com/maabuu/posebusters/files/14516485/posebusters_pdb_ccd_ids.txt | not applicable | cite as the benchmark set above |
| PDB entries and mmCIF files | sequences, identifiers and metadata (`targets/`, inputs); no deposited coordinate file is redistributed | CC0 1.0 ("Data files contained in the PDB archive") | https://www.wwpdb.org/about/usage-policies | yes (CC0) | encouraged, not required: the original authors of each structure; wwPDB consortium, *Nature Structural Biology* 10, 980 (2003) |
| Runs N' Poses (Zenodo record 14794785, `annotations.csv`; screen only) | `targets/large/rnp_screen.csv`: identifiers, chains, release dates, sequences and pocket clusters of the screened systems, no prediction | CC BY 4.0 (the record's licence field). The same record also ships AlphaFold 3 predictions with the Output Terms; none of those files was read or used | https://zenodo.org/records/14794785 | not applicable | required under CC BY 4.0: the preprint "Have protein-ligand co-folding methods moved beyond memorisation?", https://doi.org/10.1101/2025.02.03.636309 |
| ColabFold MSA server: UniRef30 (2302 in the server's history) | the MSAs of the main, follow-up 1, 10 and 11 inputs, and their provenance | CC BY 4.0 ("All files are available under") | https://colabfold.mmseqs.com/ and, for the versions, https://github.com/sokrypton/ColabFold/wiki/MSA-Server-Database-History | yes, with attribution | required: Mirdita, M. et al. *Nature Methods* 19, 679-682 (2022); UniProt Consortium |
| ColabFold MSA server: ColabFoldDB, environmental sequences (202108 in the server's history) | as above | CC BY 4.0 (same page). The page also credits MetaEuk, SMAG, TOPAZ, MGV, GPD and MetaClust2; it does not say they carry other terms, and those sources' own terms were not checked one by one | https://colabfold.mmseqs.com/ | yes, with attribution | required: Mirdita et al. above; the page lists the sub-sources without saying they must be cited |
| AlphaFold 3 pipeline: UniRef90 2022_05 | in record R2: the unpaired and paired MSAs of the official inputs (follow-up 6) | CC BY 4.0, as AlphaFold 3's README lists it; UniProt: "CC BY 4.0 License to all copyrightable parts of our databases" | https://www.uniprot.org/help/license (read through the same wording at https://www.uniprot.org/api-documentation/uniprotkb, because the help page would not load as text) and https://github.com/google-deepmind/alphafold3 | yes, with attribution | required: UniProt Consortium |
| AlphaFold 3 pipeline: UniProt 2021_04 | as above | CC BY 4.0, as above | as above | yes, with attribution | required: UniProt Consortium |
| AlphaFold 3 pipeline: MGnify 2022_05 | as above | CC0 1.0 | https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/README.txt | yes (CC0) | requested: Mitchell, A. L. et al. *Nucleic Acids Research* (2019), https://doi.org/10.1093/nar/gkz1035 |
| AlphaFold 3 pipeline: small BFD (`bfd-first_non_consensus_sequences.fasta`) | as above | CC BY 4.0; AlphaFold 3's README lists it as modified by Google DeepMind | https://bfd.mmseqs.com/ and https://github.com/google-deepmind/alphafold3 | yes, with attribution, saying that the database was modified by Google DeepMind | required: Steinegger, M. and Söding, J. *Nature Communications* (2018), https://doi.org/10.1038/s41467-018-04964-5; Steinegger, M., Mirdita, M. and Söding, J. *Nature Methods* (2019), https://doi.org/10.1038/s41592-019-0437-4 |
| AlphaFold 3 pipeline: PDB seqres (`pdb_seqres_2022_09_28.fasta`) | in record R2: the template search behind the official inputs | CC0 1.0 for "PDB" in AlphaFold 3's README; the wwPDB statement covers "data files contained in the PDB archive" and does not name this file | https://www.wwpdb.org/about/usage-policies and https://github.com/google-deepmind/alphafold3 | yes (CC0) | encouraged, as for the PDB |
| AlphaFold 3 pipeline: PDB mmCIF templates (`mmcif_files`) | in record R2: the template coordinates inside the official inputs | CC0 1.0 | https://www.wwpdb.org/about/usage-policies | yes (CC0) | encouraged, as for the PDB |
| AlphaFold 3 pipeline: NT, Rfam, RNAcentral | **nothing**: these RNA databases were not used. Every official input holds proteins and ligands only, and the pipeline logs show no `nhmmer` run | not applicable (Rfam and RNAcentral CC0 1.0 and NT not stated, in AlphaFold 3's README) | https://github.com/google-deepmind/alphafold3 | not applicable | not applicable |
| Chemical Component Dictionary (CCD) | ligand codes in targets and inputs; ligand definitions inside the AlphaFold 3 outputs and features | **unclear**: the CCD page states no licence, and the wwPDB statement covers "data files contained in the PDB archive" without naming the CCD. CC0 is the likely reading, supported only by a mailing-list message | https://www.wwpdb.org/data/ccd, https://www.wwpdb.org/about/usage-policies | not applicable | requested on the page: Westbrook, J. D. et al. *Bioinformatics* (2015), https://doi.org/10.1093/bioinformatics/btu789 |
| AlphaFold 3 outputs (predicted structures, confidences, ranking scores, embeddings, feature arrays of follow-up 5, data-pipeline results of follow-up 6, layer captures of follow-up 13) and what derives from them | record R2 | AlphaFold 3 Output Terms of Use: non-commercial use only; no training of similar structure-prediction models; conspicuous notice; no additional or conflicting terms; citation | https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md | not applicable | required: Abramson, J. et al. *Nature* 630, 493-500 (2024) |

## The licence line of each archive (proposal)

| record | archives | licence line |
|---|---|---|
| R1 | `inputs_targets_manifests.tar.gz`, `inputs_repl_large.tar.gz` | CC BY 4.0, with attribution of the ColabFold databases, UniProt, PoseBusters and Runs N' Poses; PDB data CC0 1.0 |
| R1 | `msa_provenance.tar.gz`, `freeze.tar.gz`, `session_lists_launch_records.tar.gz`, `session_records.tar.gz`, `session_records_followups.tar.gz`, `session_records_followups1316.tar.gz`, `attempt_superseded_records.tar.gz`, `outside_freeze_records.tar.gz`, `outside_freeze_records_followups1316.tar.gz`, the `release_*` files | CC BY 4.0 (our own records) |
| R1 | `code_alphafold-on-tpu_zenodo-v1.tar.gz` (the copy of the code, without `paper/assets/`) | MIT, as in the repository (`LICENSE` in the archive); the files `NOTICE` lists, copied from github.com/lorenzopazienza/alphafold-tpu-benchmark, under that repository's MIT licence (Lorenzo Pazienza and Ihab El Bani, `LICENSES/MIT-alphafold-tpu-benchmark` in the archive); the lines of AlphaFold 3's `run_alphafold.py` that `af3_tpu/af3_tpu.patch` carries, under the Apache License 2.0 (`LICENSES/Apache-2.0` in the archive). It holds no AlphaFold 3 output file and no model parameters |
| R2 | `inputs_official_pipeline.tar.gz`, `nas_naspaz8-af3_run1.tar.gz`, `nas_naspaz6-af3_run1.tar.gz`, `nas_naspaz8-af3_run2.tar.gz` | AlphaFold 3 Output Terms of Use, no Creative Commons. The databases behind the alignments are attributed in R2's README: UniRef90, UniProt and small BFD (modified by Google DeepMind), CC BY 4.0 at their sources; MGnify and the PDB, CC0 1.0 |
| R2 | `hlo_xla_dumps.tar.gz` (proposal, pending Lorenzo's decision: last section) | AlphaFold 3 Output Terms of Use, no Creative Commons; the archive also carries the Apache License 2.0 of AlphaFold 3's code it is compiled from |
| R2 | every other archive, including `af3_feat_features.tar.gz` and `analysis_outputs.tar.gz` (with the paper's `paper/assets/`) | AlphaFold 3 Output Terms of Use, no Creative Commons |

Record R1 as a whole carries CC BY 4.0 on Zenodo, which has one licence field per record: CC0 data can be redistributed inside
it, and every CC BY 4.0 source is attributed in its README. Record R2 carries a custom licence on Zenodo, title "AlphaFold 3 Output
Terms of Use", link https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.

## Third-party content of the pipeline results (record R2)

The AlphaFold 3 Output Terms apply to these files as AlphaFold 3 output. They do not restrict the third-party content the files contain: sequences and entries from UniProt, UniRef90 and BFD (CC BY 4.0) and from MGnify and the PDB (CC0 1.0) remain available under their own licences from their sources, and are attributed below.

- **UniProt**, release 2021_04 (`uniprot_all_2021_04.fa`), CC BY 4.0, https://www.uniprot.org/help/license. Cite: UniProt Consortium, UniProt: the Universal Protein Knowledgebase.
- **UniRef90**, release 2022_05 (`uniref90_2022_05.fa`), CC BY 4.0, https://www.uniprot.org/help/license. Cite: UniProt Consortium, UniProt: the Universal Protein Knowledgebase.
- **BFD**, small BFD as modified by Google DeepMind for AlphaFold 3 (`bfd-first_non_consensus_sequences.fasta`, AlphaFold 3 database set `v3.0`; no BFD release number is recorded), CC BY 4.0, https://bfd.mmseqs.com/. Cite: Steinegger, M. and Söding, J. *Nature Communications* (2018), https://doi.org/10.1038/s41467-018-04964-5; Steinegger, M., Mirdita, M. and Söding, J. *Nature Methods* (2019), https://doi.org/10.1038/s41592-019-0437-4.
- **MGnify**, release 2022_05 (`mgy_clusters_2022_05.fa`), CC0 1.0, https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/README.txt. Citation requested: Mitchell, A. L. et al. *Nucleic Acids Research* (2019), https://doi.org/10.1093/nar/gkz1035.
- **PDB**, snapshot of 2022-09-28 (`pdb_seqres_2022_09_28.fasta` and the mmCIF template files), CC0 1.0, https://www.wwpdb.org/about/usage-policies. Citation encouraged, not required: wwPDB consortium, *Nature Structural Biology* 10, 980 (2003); RCSB PDB: Berman, H. M. et al. *Nucleic Acids Research* 28, 235-242 (2000).

## Choices made where the sources are unclear

1. **Pipeline results and feature arrays are treated as Output (decision of 2026-11-08).** The Terms define Output as "the structure
   predictions and all related information provided by AlphaFold 3", and "AlphaFold 3" as the AlphaFold 3 Code and Model Parameters; the
   data pipeline is part of the Code. So the pipeline results and the feature arrays are in R2 under the Terms, although they involve no
   model inference, and R1 keeps only CC BY 4.0 material. The Terms do not name them, so this is a reading, the more restrictive one. The
   Terms govern these files as AlphaFold 3 provides them; the database entries they are built from stay available from their sources under
   the licences above.
2. **The CCD.** Ligand codes and component definitions are treated as PDB archive data, with the citation the CCD page lists. No
   CCD file is redistributed on its own.
3. **The PoseBusters V2 list.** Only identifiers are used, which are facts about public PDB entries, and the benchmark set is cited.
4. **The layer captures of follow-up 13** are intermediate activations of AlphaFold 3's trunk recorded while it ran with the
   official parameters, as the embeddings of follow-up 2 are: Output. They are not released (decision of 2026-11-17: with them
   R2 would pass 50 GB); if they ever are, it is in a record under the Output Terms, unmodified.

## XLA dumps of follow-up 16: proposal (2026-11-17), for Lorenzo to decide

What they are: 21,696 text files that XLA wrote while JAX compiled AlphaFold 3's model on the v5e and the v6e (HLO and MLIR
text before and after optimisation, buffer assignment, memory and live-range reports, compiler options and flags; 1.28 GB raw,
0.16 GB as built on 2026-11-17). They are produced by the compiler from AlphaFold 3's code and from the shapes of the inputs and
of the parameters; they carry the names and shapes of the parameters and source locations (file and function names) of
AlphaFold 3's code.

What they are not: they do not hold the model parameters. Checked on 2026-11-17 on every file: the 394 weight tensors
(1,145,669,864 bytes in the entry of each compiled `jit_apply_fn`, against a weights file of 1,020,545,840 bytes) are
arguments of the compiled function; no constant printed in the HLO text is longer than 16 characters (XLA elides large
ones); the hex-encoded constants of the MLIR files total 80,384 bytes in all and at most 5,024 bytes in one file (two
256-element float vectors and an integer table of the model code; the 1024 x 1024 constants are splats of 1.0); no MLIR
resource blob. `release.py` refuses any dump file with more than 65,536 characters of printed constants or with a resource
blob, so the VM build repeats the check. They are not predictions or confidences either.

The licences that could apply:

- **AlphaFold 3's code, Apache License 2.0** (https://github.com/google-deepmind/alphafold3/blob/main/LICENSE): the dumps
  are a compiled form of that code. Apache 2.0 allows redistribution of derivative works, in any medium, with a copy of the
  licence, the notices, and a statement of changes, and allows adding other terms for the derivative as a whole (section 4).
  The archive carries the licence text (`LICENSE_ALPHAFOLD3_CODE_APACHE-2.0.txt`, from `LICENSES/Apache-2.0` in the
  repository) and `NOTICE_XLA_DUMPS.txt` (what the files are, the copyright of the code, the redaction of home paths).
- **AlphaFold 3 Output Terms of Use**: Output is "the structure predictions and all related information provided by
  AlphaFold 3", and "AlphaFold 3" is the Code and the Model Parameters. The dumps are written while AlphaFold 3 runs with the
  official parameters, so the reading of decision (5) of 2026-11-08 (the data-pipeline results, made by the Code alone,
  treated as Output) reaches them too; the Terms do not name compiler dumps, so this is a reading, the more restrictive one.
- **The model parameters' terms** (`WEIGHTS_TERMS_OF_USE.md`): not engaged, since no parameter value is in the files; parameter
  names and shapes are also in the open-source code.
- **XLA (Apache License 2.0) and libtpu**: the compilers that wrote the files; the TPU dumps after code generation reflect
  libtpu's choices (layouts, fusion). Neither licence speaks of compiler output; no restriction found, not checked further.

Options:

1. **R2, under the Output Terms, with the Apache 2.0 text and notice inside the archive (proposed, and what `release.py`
   builds now: `HLO_DUMPS_RECORD = 'R2'`).** Consistent with decision (5); costs 2 of R2's files and about 0.16 GB; a user who
   reads the dumps for the compiled program alone is bound by the non-commercial terms, which Apache 2.0 allows a
   distributor to add.
2. **R1, under CC BY 4.0 for our part and Apache 2.0 for the code they derive from** (the same licence text and notice inside
   the archive). Fewer restrictions, closer to what the files are (a compiled program, not a prediction); it departs from the
   reading of decision (5), and R1 would no longer hold only CC BY 4.0 material. To choose it, set
   `HLO_DUMPS_RECORD = 'R1'` in `release.py` and move the row of `hlo_xla_dumps.tar.gz` from R2's README to R1's.

Either way the dumps are released as recorded, with home paths replaced by `/home/<user>` (about 6,600 files, one row each
in `REDACTIONS.tsv`).
