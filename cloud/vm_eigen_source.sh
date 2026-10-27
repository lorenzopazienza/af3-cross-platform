#!/usr/bin/env bash
# Runs ON the VM, from cloud/vm_af3_setup.sh, only when the plan sets eigen_source: archive (EIGEN_SOURCE=archive).
# Installs the pinned Eigen source (cloud/eigen_pin.sh) from our bucket, so AlphaFold3's build does not clone it
# from gitlab.com; the files are those of the same commit the build would clone (tag 3.4.0), so what is compiled
# does not change, only where the source comes from.
#
#   EIGEN_SOURCE=archive bash cloud/vm_eigen_source.sh OUT     # prints the folder for FETCHCONTENT_SOURCE_DIR_MY-EIGEN3
#
# Steps: copy EIGEN_ARCHIVE_URI with gcloud storage cp; its SHA-256 must be EIGEN_ARCHIVE_SHA256 and the commit id
# git archive wrote into it must be EIGEN_COMMIT (any difference: refused, exit 3, nothing extracted); extract it
# under ~/deps/eigen-src (one top folder, eigen/, with EIGEN_FILES files); write OUT/eigen.json (source, tag,
# commit, URI, SHA-256, file count, folder). Logs go to stderr; stdout carries only the folder.
set -euo pipefail
OUT=${1:?output folder}
[ "${EIGEN_SOURCE:-}" = "archive" ] || { echo "!! vm_eigen_source.sh runs only with EIGEN_SOURCE=archive" >&2; exit 3; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/eigen_pin.sh"
DEPS="${EIGEN_DEPS_DIR:-$HOME/deps}"
SRC="$DEPS/eigen-src"
FILE="$DEPS/$EIGEN_ARCHIVE_NAME"
sha() { (sha256sum "$1" 2> /dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
refuse() { echo "!! Eigen archive refused: $*" >&2; rm -f "$FILE"; exit 3; }

mkdir -p "$DEPS"
rm -rf "$SRC" "$FILE"
echo "   Eigen $EIGEN_TAG ($EIGEN_COMMIT) from $EIGEN_ARCHIVE_URI" >&2
gcloud storage cp "$EIGEN_ARCHIVE_URI" "$DEPS/" > /dev/null 2>&1 || refuse "could not copy $EIGEN_ARCHIVE_URI"
[ -f "$FILE" ] || refuse "no $EIGEN_ARCHIVE_NAME after the copy"
GOT=$(sha "$FILE")
[ "$GOT" = "$EIGEN_ARCHIVE_SHA256" ] || refuse "SHA-256 $GOT, expected $EIGEN_ARCHIVE_SHA256"
# get-tar-commit-id reads only the tar header and exits, so gzip may die of SIGPIPE: check the id, not the status.
ID=$(gzip -dc "$FILE" 2> /dev/null | git get-tar-commit-id || true)
[ "$ID" = "$EIGEN_COMMIT" ] || refuse "the archive carries commit ${ID:-none}, expected $EIGEN_COMMIT"
mkdir -p "$SRC"
tar -xzf "$FILE" -C "$SRC"
[ -d "$SRC/eigen" ] && [ "$(ls -A "$SRC")" = "eigen" ] || refuse "the archive does not hold exactly one folder, eigen/"
N=$(find "$SRC/eigen" -type f | wc -l | tr -d ' ')
[ "$N" = "$EIGEN_FILES" ] || refuse "$N files, expected $EIGEN_FILES"
printf '{"source": "archive", "tag": "%s", "commit": "%s", "archive_uri": "%s", "archive_sha256": "%s", "files": %s, "fetchcontent_source_dir": "%s", "replaces": "FetchContent my-eigen3 (libcifpp), GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG %s"}\n' \
  "$EIGEN_TAG" "$EIGEN_COMMIT" "$EIGEN_ARCHIVE_URI" "$GOT" "$N" "$SRC/eigen" "$EIGEN_TAG" > "$OUT/eigen.json"
echo "   SHA-256 $GOT and commit checked; $N files in $SRC/eigen (OUT/eigen.json)" >&2
echo "$SRC/eigen"
