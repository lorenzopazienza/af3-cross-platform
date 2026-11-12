# Addendum to the registration: follow-up 14, the float32 trunk on every accelerator

Registered before the first run of follow-up 14. Registered on 2026-11-12.

## Status

This is a follow-up experiment, not part of the pre-registered study. It was specified on 2026-11-12, after the
main-study and earlier follow-up outcomes were read (the last of them the analysis of follow-up 13), and before any
run. It will be reported whatever it shows. The specification is the row of 2026-11-12 "Follow-up experiment 14" in
section 10 of the analysis plan (`notes/analysis_plan.md`). The implementation will be recorded in the row that
follows it, before the first run. At registration, the specification is at commit `596cb35` of the repository
(`notes/analysis_plan.md`, SHA-256 `75ead95d72804eaaeeac3c5d729a8e2fce383311d145a39ed9e328adaea196ac`). The main study's registration is https://osf.io/fegsy; follow-up 13's
addendum is https://osf.io/c4kh7. At registration, no follow-up 14 run had been made.

## Questions

Follow-up 13 found that with AlphaFold 3's trunk in float32 at the highest matmul precision, TPU v6e and the NVIDIA
L4 rank the same sample first in 299 of 300 predictions, where in bfloat16 they disagree in about one prediction in
five.

- **F.** Does this hold for the other accelerators of the study, TPU v5e, the NVIDIA A100 80 GB and the NVIDIA H100
  80 GB, and at what cost in time and memory on each?
- **D.** Does XLA's deterministic-ops flag make recompilation bit-identical on the A100 and the H100, as it does on
  the L4, and at what cost?

## Settings

Every arm keeps the main study's settings on its hardware: AlphaFold 3 v3.0.4 with the TPU patch, the same weights,
software stacks, frozen inputs, recycles, samples, machine types and image families, and one fresh process per
(target, seed, repetition) with an empty compile cache. No AlphaFold 3 source file is edited.

- **Float32 trunk**, as in follow-up 13: AlphaFold 3's own global setting `bfloat16` is changed from `all` to `none`,
  together with `JAX_DEFAULT_MATMUL_PRECISION=highest`. The trunk and the confidence head's Pairformer then run on
  float32 activations, with float32 matmuls (six-pass bfloat16 on TPU v5e, IEEE float32 without TensorFloat-32 on the
  A100 and the H100).
- **Deterministic flag**: the trunk in bfloat16 as in the main study, with XLA's `--xla_gpu_deterministic_ops=true`,
  the setting of the main study's deterministic L4 arm.

## Arms

- **V5f:** TPU v5e, the 60 main-study targets, seeds 1 to 5, one repetition, plus a second repetition of the 10 pilot
  targets, seed 1, as a check that recompilation is bit-identical.
- **Af:** A100 80 GB, the 60 targets, seeds 1 to 5, one repetition.
- **Hf:** H100 80 GB, the 60 targets, seeds 1 to 5, one repetition.
- V6f and Gf (TPU v6e and L4) are follow-up 13's arms, used as they are.
- **Ad and Hd:** the A100 and the H100 with the deterministic flag, the 10 pilot targets, seeds 1 to 5, two
  repetitions, launched only after the float32 arms have ended.

The CPU is not run: at its measured bfloat16 times the arm would take about 470 to 740 VM hours. Out-of-memory stops
and other failures are recorded with their cause and are not rerun with other settings. The provisioning model (on
demand on v5e, Spot on the A100 and the H100, on demand on the A100 only as a fallback) and the zone are recorded per
run and enter no contrast.

## Checks

These are made before any result is read: every process exits 0 or its failure is recorded; one machine type per
condition; completeness; the float32 setting, or the deterministic flag, and the expected device are recorded in
every process; the v5e second repetition is compared bit for bit with the first; the device memory of each bucket is
compared with the device's limit.

## Statistics

The statistics are those of section 5 of the plan: the target-clustered bootstrap with 10,000 replicates and seed
2026, every contrast paired on (target, seed), the top-1 sample taken per seed, seeds 1 to 5.

- **Pairs:** all ten pairs of the five float32 conditions (v5e, v6e, L4, A100, H100).
- **Against bfloat16:** for four primary pairs (v5e and v6e, v6e and the L4, the H100 and the A100, the L4 and the
  A100), D_F = [HW float32 minus RC] minus [HW bfloat16 minus RC], with the recompilation term of the earlier analysis
  of that pair. Where the float32 recompilation is not measured, its term is set to 0; since a flip rate is never
  negative, this can only make "reduced" harder to reach.
- **Against bfloat16 at the highest precision**, descriptive, where both conditions of the pair have such an arm.
- **Cost:** the ratio of the median process wall time and of the device memory, against the bfloat16 arms on the same
  hardware, overall and per token bucket.

## Reading rules

- **Reduced.** For each of the four primary pairs, the top-1 flip rate is "reduced" if the 95% interval of D_F lies
  below 0, and "not reduced" otherwise. The same rule applies to the success flip rate and to flips that matter, as
  secondary.
- **Agree.** For each of the ten pairs, "agree" if the upper limit of the 95% interval of the float32 top-1 flip rate
  is below 5.0 percentage points, and "agreement not shown" otherwise. The threshold lies below the top-1 flip rate of
  every bfloat16 GPU against its own recompilation (7.3% to 10.0%) and at the lowest point of follow-up 13's
  perturbation curve (5.0%). It was chosen after follow-up 13's analysis was read and before any follow-up 14 result
  existed. The v6e and L4 pair is known to agree from follow-up 13 and is reported as known, not as new evidence.
- **Generality.** Follow-up 13's result is stated as holding on every accelerator of the study only if all ten pairs
  read "agree", each at its own 95% level (an intersection-union test, so no multiplicity correction). Otherwise the
  pairs that agree and those that do not are named.
- **Deterministic flag.** On each of the A100 and the H100, "recompilation bit-identical" if all 50 (target, seed)
  predictions of the second repetition are identical to the first in every output file; otherwise the number that
  differ is reported, with the recompilation flip rates and the cost in time beside those of the arms without the
  flag.

## Budget and calendar

The float32 arms are expected to cost about 201 USD, 400 USD at worst; their cap is 300 USD, and before each A100
part a stop rule refuses the part if the projected spend would exceed the cap, after which Lorenzo decides. The
deterministic arms are expected to cost about 50 USD, 61 USD at worst, with a cap of 62 USD; they run only if the
credit left after the float32 arms is at least their worst case plus 25 USD. The launch is planned for 2026-11-13 and
will be no later than 2026-11-18. Every run must be finished by 2026-11-20, or the arm is reported as incomplete. The
analysis is due by 2026-11-22. No run is dropped or selected on results.
