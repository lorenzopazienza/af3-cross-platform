# Licences of the third-party data in records R1 and R2

Prepared on 2026-11-08 from the pages cited below. This table says what each source states; it is not legal advice. "Unclear"
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
| AlphaFold 3 outputs (predicted structures, confidences, ranking scores, embeddings, feature arrays of follow-up 5, data-pipeline results of follow-up 6) and what derives from them | record R2 | AlphaFold 3 Output Terms of Use: non-commercial use only; no training of similar structure-prediction models; conspicuous notice; no additional or conflicting terms; citation | https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md | not applicable | required: Abramson, J. et al. *Nature* 630, 493-500 (2024) |

## The licence line of each archive (proposal)

| record | archives | licence line |
|---|---|---|
| R1 | `inputs_targets_manifests.tar.gz`, `inputs_repl_large.tar.gz` | CC BY 4.0, with attribution of the ColabFold databases, UniProt, PoseBusters and Runs N' Poses; PDB data CC0 1.0 |
| R1 | `msa_provenance.tar.gz`, `freeze.tar.gz`, `session_lists_launch_records.tar.gz`, `session_records.tar.gz`, `session_records_followups.tar.gz`, `attempt_superseded_records.tar.gz`, `outside_freeze_records.tar.gz`, the `release_*` files | CC BY 4.0 (our own records) |
| R2 | `inputs_official_pipeline.tar.gz`, `nas_naspaz8-af3_run1.tar.gz`, `nas_naspaz6-af3_run1.tar.gz`, `nas_naspaz8-af3_run2.tar.gz` | AlphaFold 3 Output Terms of Use, no Creative Commons. The databases behind the alignments are attributed in R2's README: UniRef90, UniProt and small BFD (modified by Google DeepMind), CC BY 4.0 at their sources; MGnify and the PDB, CC0 1.0 |
| R2 | every other archive, including `af3_feat_features.tar.gz` and `analysis_outputs.tar.gz` | AlphaFold 3 Output Terms of Use, no Creative Commons |

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
