#!/usr/bin/env bash
# Runs ON the VM, from cloud/vm_af3_job.sh. Checks the uploaded inputs against
# inputs/manifest.csv, installs AlphaFold3 at the pinned commit with
# af3_tpu.patch (the same steps as af3_tpu/vm_smoke.sh, copied here, not
# called), sets up JAX for the platform, runs build_data, puts the weights in
# place (cloud/vm_af3_weights.sh) and checks that JAX sees the right device.
#
#   PLATFORM=cpu|l4|a100|h100|v5e|v6e WEIGHTS=random|gcs|none TARGETS=ID,ID [CCD_SHA256=...] [STACK=...] [EIGEN_SOURCE=archive] bash cloud/vm_af3_setup.sh OUT
#
# EIGEN_SOURCE=archive (follow-ups 7 to 9, plan key eigen_source; cloud/af3_run.sh): the Eigen source that libcifpp's
# FetchContent (my-eigen3) would clone from gitlab.com comes instead from our bucket, as the archive of the same
# commit (cloud/eigen_pin.sh, cloud/vm_eigen_source.sh: SHA-256 and commit checked, exit 3 otherwise), passed to
# the build as CMAKE_ARGS=-DFETCHCONTENT_SOURCE_DIR_MY-EIGEN3=<folder> (scikit-build-core forwards CMAKE_ARGS); during
# uv sync any git URL on gitlab.com is rewritten to an invalid host, so a clone cannot happen silently. OUT/eigen.json
# records it. Without EIGEN_SOURCE, nothing changes.
#
# STACK (follow-up 7, a plan's stack key; cloud/af3_run.sh): pkg==version pairs, comma-separated. On TPU its jax
# and jaxlib replace 0.10.2 in the swap (its libtpu is LIBTPU_VERSION); on GPU they and the jax-cuda12 plugin and
# pjrt are installed over AlphaFold3's pinned ones after uv sync, and each installed version must be the pinned one
# (exit 5 otherwise). pip_freeze.txt and setup.json record them. Without STACK, nothing changes.
#
# CCD_SHA256 (featurise-only plans of follow-up 5): build_data reads ~/components.cif.zst, uploaded by
# cloud/af3_run.sh, which must have that SHA-256 once decompressed (exit 3 otherwise), instead of the
# components.cif the AlphaFold3 build downloads; OUT/ccd.json records both SHA-256. Without it, nothing changes.
#
# JAX per platform: TPU swaps AlphaFold3's CUDA JAX for jax/jaxlib 0.10.2 with
# libtpu LIBTPU_VERSION (default 0.0.43.2; cloud/vm_tpu_stack.sh explains why
# not the 0.0.42.* that jax[tpu]==0.10.2 pins); L4 keeps AlphaFold3's pinned CUDA JAX and must show an
# L4; CPU keeps the default install (the cpu_xla config sets JAX_PLATFORMS=cpu).
# The device check is cloud/vm_device_check.sh (JAX_PLATFORMS=cuda on L4; it
# writes OUT/device_check.txt, which explains any failure).
# Writes OUT/setup.json (on TPU with a tpu_runtime block: libtpu version,
# build label and build date), OUT/pip_freeze.txt and OUT/weights.json. Exit 3
# if an input does not match the manifest, 4 for a weights problem, 5 if JAX
# does not see the expected device or the installed libtpu is not LIBTPU_VERSION.
set -euo pipefail
OUT=${1:?output folder}
: "${PLATFORM:?}" "${WEIGHTS:?}" "${TARGETS:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/af3_tpu/pin.sh"
AF3="$ROOT/third_party/alphafold3"
PY="$AF3/.venv/bin/python"
WEIGHTS_DIR="${AF3_WEIGHTS_DIR:-$HOME/af3_weights_run}"   # outside the checkout; never fetched
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
step() { echo; echo ">> [$(date -u +%H:%M:%S)] setup: $*"; }

# The plan's manifest (plan key manifest, default inputs/manifest.csv; harness/run_plan.py).
MANIFEST=inputs/manifest.csv
if [ -n "${PLAN:-}" ]; then
  MANIFEST=$(cd "$ROOT" && python3 harness/run_plan.py --plan "$PLAN" --platform "$PLATFORM" --describe \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('manifest', 'inputs/manifest.csv'))") || exit 3
fi
step "inputs against $MANIFEST"
python3 "$ROOT/harness/verify_inputs.py" --targets "$TARGETS" --manifest "$MANIFEST" || exit 3

step "system packages (the ones AlphaFold3's docker/Dockerfile installs)"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git gcc g++ make zlib1g-dev zstd

step "uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "AlphaFold3 $AF3_TAG ($AF3_COMMIT) with af3_tpu.patch"
[ -d "$AF3/.git" ] || git clone -q "$AF3_REPO" "$AF3"
git -C "$AF3" -c advice.detachedHead=false checkout -q "$AF3_COMMIT"
bash "$ROOT/af3_tpu/apply.sh" "$AF3"

case "${EIGEN_SOURCE:-}" in
  "") ;;
  archive)
    step "Eigen for libcifpp: the pinned archive from our bucket, not a clone from gitlab.com"
    EIGEN_DIR=$(bash "$ROOT/cloud/vm_eigen_source.sh" "$OUT") || exit 3
    export CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DFETCHCONTENT_SOURCE_DIR_MY-EIGEN3=$EIGEN_DIR"
    echo "   CMAKE_ARGS=$CMAKE_ARGS" ;;
  *) echo "!! EIGEN_SOURCE must be archive or unset, not $EIGEN_SOURCE"; exit 3 ;;
esac

step "install from uv.lock (uv sync --frozen, as in docker/Dockerfile)"
cd "$AF3"
if [ "${EIGEN_SOURCE:-}" = "archive" ]; then
  # Only for the build: a git URL on gitlab.com becomes an invalid host, so a clone of Eigen fails instead of happening.
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://gitlab.invalid/.insteadOf GIT_CONFIG_VALUE_0=https://gitlab.com/ \
    uv sync --frozen --python 3.12
else
  uv sync --frozen --python 3.12
fi

# From here on, never `uv run`: it re-syncs to uv.lock and would undo the TPU
# swap. Call the venv's executables directly.
case "$PLATFORM" in
  v5e | v6e)
    STACK_JAX=$(echo "${STACK:-}" | tr ',' '\n' | sed -n 's/^jax==//p')
    step "swap CUDA JAX for jax/jaxlib ${STACK_JAX:-0.10.2} with libtpu ${LIBTPU_VERSION:-0.0.43.2}${STACK:+ (the alternative stack of the plan)}"
    PY="$PY" LIBTPU_VERSION="${LIBTPU_VERSION:-0.0.43.2}" JAX_VERSION="${STACK_JAX:-0.10.2}" \
      bash "$ROOT/cloud/vm_tpu_stack.sh" || exit $? ;;
  l4 | a100 | h100)
    step "NVIDIA driver (from the image); AlphaFold3's pinned CUDA JAX is kept${STACK:+, then the alternative stack of the plan}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if [ -n "${STACK:-}" ]; then
      echo "   installing $(echo "$STACK" | tr ',' ' ')"
      uv pip install --python "$PY" $(echo "$STACK" | tr ',' ' ') || exit 5
      for pv in $(echo "$STACK" | tr ',' ' '); do
        GOT=$("$PY" -c "import importlib.metadata as m; print(m.version('${pv%%==*}'))" 2> /dev/null || echo none)
        [ "$GOT" = "${pv#*==}" ] || { echo "!! stack check: ${pv%%==*} $GOT, expected ${pv#*==}" >&2; exit 5; }
      done
      echo "   GPU stack: $(echo "$STACK" | tr ',' ' ')"
    fi ;;
  cpu)
    step "CPU: default install kept; cpu_xla runs with JAX_PLATFORMS=cpu" ;;
esac

if [ -n "${CCD_SHA256:-}" ]; then
  step "CCD: the plan's components.cif ($(echo "$CCD_SHA256" | cut -c1-12)...), uploaded from the laptop"
  CCD_DIR="$HOME/ccd_plan"
  mkdir -p "$CCD_DIR"
  zstd -d -q -f "$HOME/components.cif.zst" -o "$CCD_DIR/components.cif" && rm -f "$HOME/components.cif.zst" || exit 3
  USED=$(sha256sum "$CCD_DIR/components.cif" | cut -d' ' -f1)
  [ "$USED" = "$CCD_SHA256" ] || { echo "!! the uploaded components.cif has SHA-256 $USED, not the plan's"; exit 3; }
  DOWNLOADED=$(find "$AF3/.venv" -path '*share/libcifpp/components.cif' -exec sha256sum {} \; | head -1 | cut -d' ' -f1)
  printf '{"used_sha256": "%s", "used_bytes": %s, "downloaded_by_build_sha256": "%s"}\n' \
    "$USED" "$(wc -c < "$CCD_DIR/components.cif" | tr -d ' ')" "${DOWNLOADED:-none}" > "$OUT/ccd.json"
  echo "   build_data reads $CCD_DIR/components.cif (LIBCIFPP_DATA_DIR); the build downloaded ${DOWNLOADED:-nothing}"
  export LIBCIFPP_DATA_DIR="$CCD_DIR"
fi

step "build_data"
"$AF3/.venv/bin/build_data"

step "weights ($WEIGHTS)"
cd "$ROOT"
WEIGHTS="$WEIGHTS" bash "$ROOT/cloud/vm_af3_weights.sh" "$WEIGHTS_DIR" "$OUT" || exit 4

step "devices and versions"
uv pip freeze --python "$PY" > "$OUT/pip_freeze.txt"
# JAX_PLATFORMS is cpu, cuda or tpu there, never "gpu" (see that script).
PLATFORM="$PLATFORM" bash "$ROOT/cloud/vm_device_check.sh" "$OUT" "$AF3_COMMIT" || exit 5
echo ">> [$(date -u +%H:%M:%S)] setup: done"
