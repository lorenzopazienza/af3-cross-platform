#!/bin/bash
# Run on the Mac (Lorenzo): writes the Eigen source archive that cloud/vm_eigen_source.sh installs on the VMs, so
# their AlphaFold3 build no longer clones Eigen from gitlab.com (cloud/eigen_pin.sh says which revision and why).
#
#   bash cloud/make_eigen_archive.sh OUT_DIR             # fetch, check, archive, print the SHA-256
#   bash cloud/make_eigen_archive.sh OUT_DIR --upload    # the same, then copy it to EIGEN_ARCHIVE_URI
#
# 1. Fetches commit EIGEN_COMMIT only (git fetch --depth 1), from gitlab.com or, if gitlab.com refuses, from the
#    GitHub mirror (EIGEN_GIT_URLS overrides the list); the fetched commit must be EIGEN_COMMIT, so the content is
#    the same whatever the server (a git commit id fixes every file).
# 2. git archive --format=tar --prefix=eigen/ EIGEN_COMMIT | gzip -n -9: the files of that commit, the commit id in
#    the tar's pax header (git get-tar-commit-id reads it back), no gzip timestamp or name.
# 3. Compares the archive's SHA-256 with EIGEN_ARCHIVE_SHA256 in cloud/eigen_pin.sh and stops on a difference
#    (another git or gzip may compress differently: then check the contents and update the pin by hand).
# 4. With --upload only: gcloud storage cp to EIGEN_ARCHIVE_URI (refused if the object already exists with other
#    bytes), then reads the object's size back. Nothing else touches the cloud. macOS bash 3.2.
set -euo pipefail
OUT=${1:?usage: make_eigen_archive.sh OUT_DIR [--upload]}
UPLOAD=${2:-}
[ -z "$UPLOAD" ] || [ "$UPLOAD" = "--upload" ] || { echo "!! second argument: --upload or nothing" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/cloud/eigen_pin.sh"
URLS=${EIGEN_GIT_URLS:-"https://gitlab.com/libeigen/eigen.git https://github.com/eigen-mirror/eigen.git"}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/eigen-archive.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
git init -q "$WORK/src"
got=""
for url in $URLS; do
  echo ">> fetching $EIGEN_COMMIT from $url"
  if git -C "$WORK/src" fetch -q --depth 1 "$url" "$EIGEN_COMMIT"; then got=$url; break; fi
  echo "   failed; trying the next source"
done
[ -n "$got" ] || { echo "!! could not fetch $EIGEN_COMMIT from any of: $URLS" >&2; exit 1; }
[ "$(git -C "$WORK/src" rev-parse FETCH_HEAD)" = "$EIGEN_COMMIT" ] || { echo "!! fetched commit is not $EIGEN_COMMIT" >&2; exit 1; }
n=$(git -C "$WORK/src" ls-tree -r --name-only FETCH_HEAD | wc -l | tr -d ' ')
attrs=$(git -C "$WORK/src" ls-tree -r --name-only FETCH_HEAD | grep -c '\.gitattributes$\|\.gitmodules$' || true)
[ "$attrs" = "0" ] || { echo "!! the commit has .gitattributes or .gitmodules: git archive may differ from a clone" >&2; exit 1; }
FILE="$OUT/$EIGEN_ARCHIVE_NAME"
git -C "$WORK/src" archive --format=tar --prefix=eigen/ "$EIGEN_COMMIT" | gzip -n -9 > "$FILE"
SHA=$(shasum -a 256 "$FILE" | cut -d' ' -f1)
# get-tar-commit-id reads only the tar header and exits, so gzip may die of SIGPIPE: check the id, not the status.
ID=$(gzip -dc "$FILE" 2> /dev/null | git get-tar-commit-id || true)
[ "$ID" = "$EIGEN_COMMIT" ] || { echo "!! the archive carries commit ${ID:-none}, not $EIGEN_COMMIT" >&2; exit 1; }
echo ">> $FILE"
echo "   source $got, commit $ID, $n files, $(wc -c < "$FILE" | tr -d ' ') bytes"
echo "   sha256 $SHA"
if [ "$EIGEN_ARCHIVE_SHA256" != "$SHA" ]; then
  echo "!! cloud/eigen_pin.sh pins EIGEN_ARCHIVE_SHA256=$EIGEN_ARCHIVE_SHA256, this archive is $SHA" >&2
  exit 1
fi
echo "   matches cloud/eigen_pin.sh"
if [ "$UPLOAD" = "--upload" ]; then
  if gcloud storage objects describe "$EIGEN_ARCHIVE_URI" > /dev/null 2>&1; then
    tmp="$WORK/existing.tar.gz"
    gcloud storage cp "$EIGEN_ARCHIVE_URI" "$tmp" > /dev/null
    [ "$(shasum -a 256 "$tmp" | cut -d' ' -f1)" = "$SHA" ] || { echo "!! $EIGEN_ARCHIVE_URI exists with other bytes; not overwritten" >&2; exit 1; }
    echo ">> $EIGEN_ARCHIVE_URI already holds this archive; nothing uploaded"
  else
    gcloud storage cp "$FILE" "$EIGEN_ARCHIVE_URI"
    echo ">> uploaded: $(gcloud storage objects describe "$EIGEN_ARCHIVE_URI" --format='value(size)') bytes at $EIGEN_ARCHIVE_URI"
  fi
fi
