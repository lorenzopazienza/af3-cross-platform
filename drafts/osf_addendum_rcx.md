# Addendum to the registration: follow-up 15, recompilation of the float32 A100 and H100 arms and of the L4 Triton arm

Registered before the first run of follow-up 15. Registered on 2026-11-DD.

## Status

This is a follow-up experiment, not part of the pre-registered study. It was specified on 2026-11-14, after the
main-study and earlier follow-up outcomes were read (the last of them the analysis of follow-up 14 and three post hoc
analyses of 2026-11-14), and before any run. It will be reported whatever it shows. The specification is the row of
2026-11-14 "Follow-up experiment 15" in section 10 of the analysis plan (`notes/analysis_plan.md`). The
implementation will be recorded in the row that follows it, before the first run. At registration, the specification
is at commit `COMMIT` of the repository (`notes/analysis_plan.md`, SHA-256 `SHA256`). The main study's registration
is https://osf.io/fegsy; follow-up 14's addendum is https://osf.io/eafkx. At registration, no follow-up 15 run had
been made.

## Questions

In AlphaFold 3, "recompilation" (RC) means running the same prediction again in a new process on a new machine with
the same settings, so that the compiler builds the program again. The study uses the RC rate of each arm as a
yardstick for its hardware contrasts.

- **A4.** Follow-up 14 did not measure the recompilation of its float32 arms on the NVIDIA A100 and H100 (Af and Hf).
  Its comparison of the H100 with the A100 was read with the H100's float32 recompilation term set to 0, a
  conservative choice. How often does recompilation change the top-ranked sample of Af and of Hf, and is it
  bit-identical?
- **A3.** The main study's NVIDIA L4 arm with Triton attention (GT, AlphaFold 3's GPU default) has no recompilation of
  its own. How often does recompilation change GT's top-ranked sample, compared with the L4 arm with XLA attention?

## Settings

Each arm is a second repetition (r2) of an existing arm, with that arm's configuration unchanged: the same AlphaFold
3 version and patch, weights, frozen inputs, recycles, samples, machine type, software stack and NVIDIA driver, and
one fresh process per (target, seed) with an empty compile cache. No AlphaFold 3 source file and no configuration is
edited.

- **Af r2 and Hf r2:** follow-up 14's float32 trunk (`bfloat16` set to `none`, matmuls at the highest precision) on
  the A100 80 GB and the H100 80 GB.
- **GT r2:** the main study's L4 Triton configuration, bfloat16 trunk at the default precision.

Differences from the first repetition, none in a setting of the computation: Af r2 and Hf r2 run on a copy of a
Canonical GPU image that 2 of the 15 Af first-repetition parts used, while the other parts ran on a newer image of
the same family that Canonical has since withdrawn (same driver, kernel and Python packages). GT r2 runs on a copy of
the image GT's first repetition used (inferred from the image family's history). The machine setup installs the
newest Python 3.12 patch release of the day: 3.12.15 for GT's first repetition, 3.12.16 now, with the same Python
packages; follow-up 13 found this change leaves the outputs on TPU identical bit for bit. All arms run on Spot VMs;
GT's first repetition ran on demand. Provisioning, zone and image enter no contrast.

## Arms

- **Part A4 (mandatory):** Af r2 and Hf r2, the 10 pilot targets, seeds 1 to 5 (50 predictions each). Their first
  repetitions are follow-up 14's Af and Hf.
- **Part A3 (conditional):** GT r2, the 60 targets, seeds 1 to 5 (300 predictions). Its first repetition is the main
  study's GT. It runs only after part A4 has ended and only if the budget left after part A4 covers its worst case
  (see Budget); otherwise it is reported as not run.

## Checks

These are made before any result is read: every process exits 0 or its failure is recorded; one machine type per
condition, the first repetition's; completeness; the float32 setting (or its absence for GT) and the expected device
in every process; for every session, the driver, kernel, Python version and Python packages compared with the first
repetition's, any difference listed.

## Statistics

The statistics are those of section 5 of the plan: the target-clustered bootstrap with 10,000 replicates and seed
2026, every contrast paired on (target, seed), the top-1 sample taken per seed. RC(Af), RC(Hf) and RC(GT) compare
each arm's first and second repetitions: the top-1 flip rate, the success flip rate and flips that matter, the RMSD
between the two sides' structures, and the fraction of bit-identical samples.

## Reading rules

No reading is added beyond those follow-up 14 used for its recompilation arms.

- **Bit-identity.** For each arm, "recompilation bit-identical" if every (target, seed) of the second repetition is
  identical to the first, and otherwise the number that differ ("not determined" if files are missing and none
  differs). For Af and Hf every output file is compared, with the mmCIF timestamp masked. For GT, whose first
  repetition kept only the structures, ranking scores and summary confidences on disk, those files are compared.
- **Rates.** RC(Af), RC(Hf) and RC(GT) are reported with their 95% intervals, beside the registered recompilation
  rates of the bfloat16 A100 and H100 on the same targets and of follow-up 14's other arms.
- **Sensitivity of follow-up 14's H100 against A100 reading.** Follow-up 14's D_F for the H100 and the A100 is
  recomputed with the measured RC(Hf) in place of 0, on its 60 targets with the same bootstrap draws (RC(Hf) taken
  over the pilot targets among the drawn targets) and on the 10 pilot targets alone. This is descriptive; follow-up
  14's registered reading stands whatever it shows.
- **GT against the L4 with XLA attention.** RC(GT) minus RC(G) of the main study, with its interval, descriptive.

## Budget and calendar

Part A4 is expected to cost about 30 USD, 37 USD at worst; part A3 about 11 USD, 13 USD at worst. The cap of
follow-up 15 is 62 USD (the credit read on 2026-11-14, 87 USD, minus a floor of 25 USD). Before every part, a stop
rule refuses the part if the projected spend would exceed the cap, after which the author decides. Part A3 starts
only if part A4 has spent at most about 49 USD. The launch is planned for 2026-11-15 and will be no later than
2026-11-17. Every run must be finished by 2026-11-18, or the arm is reported as incomplete. The analysis is due by
2026-11-19. No run is dropped or selected on results.
