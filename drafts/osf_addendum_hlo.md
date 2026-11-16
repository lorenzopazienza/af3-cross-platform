# Addendum to the registration: follow-up 16, the compiled program of AlphaFold 3 on TPU v5e and v6e

Registered before the first run of follow-up 16. Registered on 2026-11-DD.

## Status

This is a follow-up experiment, not part of the pre-registered study. It was specified on 2026-11-16, after the
main-study and earlier follow-up outcomes were read (the last of them the analysis of follow-up 15 and the post hoc
analyses of 2026-11-14), and before any run. It will be reported whatever it shows. The specification is the row of
2026-11-16 "Follow-up experiment 16" in section 10 of the analysis plan (`notes/analysis_plan.md`). The
implementation will be recorded in the row that follows it, before the first run. At registration, the specification
is at commit `COMMIT` of the repository (`notes/analysis_plan.md`, SHA-256 `SHA256`). The main study's registration
is https://osf.io/fegsy; follow-up 15's addendum is https://osf.io/h9yw8. At registration, no follow-up 16 run had
been made.

## Question

The study found that TPU v5e and TPU v6e give different AlphaFold 3 outputs from identical inputs, and follow-up 13
found the two chips already different at the input of the Pairformer, after the input embedding, the templates and the
MSA module. Here the compiled program itself is compared: XLA's HLO of AlphaFold 3's model after optimisation, on each
chip, located per AlphaFold 3 module.

**Pre-specified question:** do the differences between the two chips' HLO appear already in the operations that
precede the Pairformer (input embedding, templates, MSA module)?

## Settings

Each chip runs the main study's configuration for that chip (same AlphaFold 3 version and patch, weights, frozen
inputs, recycles, samples, machine type, software stack and libtpu), with one change: XLA writes the model's HLO to
text files, before and after its optimisations. The dump is set by a variable that only the study's entry script
reads; it adds XLA's dump flags, with a separate folder for each process, before JAX starts. XLA's dump flags are
debug options that only write files; JAX leaves them out of its compilation-cache key for that reason. Whether the
dump leaves the computation unchanged is tested by the run itself (see Checks).

## Design

- Two chips, one virtual machine each, on demand: TPU v5e and TPU v6e.
- Four targets, one in each token bucket of the study (256, 512, 768 and 1,024): in each bucket, the pilot target
  with the most tokens. AlphaFold 3 pads every input to fixed shapes per bucket, so the compiled program depends on
  the bucket, not on the target.
- Seed 1 only (the seed is a run-time input, not part of the program).
- Two fresh processes per target on each chip, so that each chip compiles each program twice.

## Checks

Before any HLO is compared: every process exits 0 and has one optimised and one unoptimised dump of the model; and
every process's outputs are identical bit for bit with the registered outputs of the same chip, target and seed from
the main study. The device, the software stack and the dump flags are also checked and reported.

## Measures

Every instruction of the optimised program is assigned to an AlphaFold 3 module (input embedding, templates, MSA
module, Pairformer, diffusion, confidence head, distogram head, weight reading, other) from the module path and the
source location that XLA records with it. For each bucket and module, the two chips are compared at four levels, and
the first level that differs is reported:

1. **semantic:** which arithmetic operations run, with which element types, shapes, precisions and reductions;
2. **fusion:** the same operations grouped into different kernels;
3. **layout:** the same kernels with different memory layouts or data movement;
4. **backend:** the same HLO with a different backend configuration;

or **identical**. Descriptive tables give the counts of operations, fusions, element types, matrix-multiplication
precisions and reductions per module and chip, and the module paths whose operations differ.

## Reading rules

- **Validity.** The run is valid only if both checks above pass for all 16 processes; otherwise the comparison is
  reported as invalid and not read.
- **Pre-specified question, per bucket:** "yes" if the highest level among the input embedding, the templates and the
  MSA module is semantic or fusion; "layout only" if it is layout; "backend only" if it is backend; "no" if all three
  are identical. Overall: the common answer of the four buckets, otherwise "mixed", with each bucket's answer.
- **Within-chip determinism:** the two compilations on the same chip are compared. A difference between the chips at
  or below the level at which a chip differs from itself is reported as not attributable to the chip.
- **Unoptimised program:** the HLO before XLA's optimisations is compared between the chips (expected identical).
- **Interpretation, stated in advance:** the optimised HLO is the last form of the program that XLA writes as text;
  the TPU back end's code generation below it is visible only through the backend configuration. "Yes" places the
  difference that follow-up 13 measured before the Pairformer in XLA's optimisation passes, at the modules and
  operations the tables name. "Layout only" means the same operations in the same kernels with other memory layouts
  or data movement, which are exact, so the numerical difference comes from the code generated for those layouts.
  "Backend only" or "no" place it below the HLO, in code generation. No direction is predicted.

## Budget and calendar

Expected cost about 2.7 USD (3.88 USD at worst; at most 8.32 USD before the launchers delete the machines). Cap of
follow-up 16: 15 USD, within the TPU credit read on 2026-11-16 (40.47 USD) less a floor of 25 USD. A stop rule
before each part refuses it if the projected spend would exceed the cap. The launch is planned for 2026-11-16 or
2026-11-17 and will be no later than 2026-11-17; every run must be finished by 2026-11-18, or it is reported as not
run. No run is dropped or selected on results.
