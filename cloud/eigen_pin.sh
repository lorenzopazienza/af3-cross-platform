# The Eigen source that AlphaFold3 v3.0.4's build compiles against, pinned so the VM setup can take it from our
# bucket instead of cloning gitlab.com (sourced by cloud/make_eigen_archive.sh and cloud/vm_eigen_source.sh).
#
# Where the build gets Eigen: AlphaFold3 v3.0.4 (third_party/alphafold3/CMakeLists.txt, FetchContent_Declare cifpp)
# builds libcifpp at commit ac98531a2fc8daf21131faa0c3d73766efa46180 (v7.0.3). libcifpp's CMakeLists.txt at that
# commit (lines 229 to 247) first tries find_package(Eigen3 3.4); if no Eigen 3.4 is installed (none is on the VM
# images), it declares FetchContent "my-eigen3" with GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git and
# GIT_TAG 3.4.0, populates it (headers only, nothing built) and compiles libcifpp with that folder on its include
# path. Tag 3.4.0 is a lightweight tag on commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c ("Change version to
# 3.4.0", tree 228ffa5204a0035e58a3e57798dcf23eda0134f2; `git ls-remote` on gitlab.com and on the GitHub mirror
# github.com/eigen-mirror/eigen, 2026-10-27). That commit has no .gitattributes and no submodules, so
# `git archive` of it holds exactly the files a clone checks out (1,784 files), without .git.
export EIGEN_TAG="3.4.0"
export EIGEN_COMMIT="3147391d946bb4b6c68edd901f2add6ac1f31f8c"
export EIGEN_ARCHIVE_NAME="eigen-3.4.0-3147391d.tar.gz"
export EIGEN_FILES="1784"   # files in the archive (and in a checkout of the commit, without .git)
export EIGEN_ARCHIVE_URI="gs://af2-tpu-benchmark-af3-results/deps/eigen-3.4.0-3147391d.tar.gz"
# SHA-256 of the archive cloud/make_eigen_archive.sh writes (git archive --format=tar | gzip -n); the VM refuses
# any other file.
export EIGEN_ARCHIVE_SHA256="fc793f0a054e84796ee4d5ab1ac6ccd6f1d7e9426362466ac46e824c3182769f"
