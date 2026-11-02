#!/usr/bin/env bash
# On the DGX, after build_image.sh, build_ws.sh and the dgx run_suite.sh: pack
# what the Jetson needs into two files next to this checkout --
#   jetson_bench_image.tar.gz   the image (docker save)
#   jetson_bench_ws.tar.gz      this checkout with its built install/, the test
#                               binary and the model it reads; no build/ or log/
#                               or thesis data the Jetson does not use
# Nothing is compiled on the Jetson: it runs exactly these binaries.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEST=${1:-..}
echo "saving the image (a few minutes)..."
docker save mujoco-pat:jetson | gzip -1 > "$DEST/jetson_bench_image.tar.gz"
echo "packing the workspace..."
# No .git goes to the Jetson: record which revision this is, for its manifests.
echo "$(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD)" \
     "$([ -n "$(git status --porcelain --untracked-files=no)" ] && echo dirty || echo clean)" \
     > tools/jetson_bench/REVISION
{
    git ls-files
    echo tools/jetson_bench/REVISION
    find install -type f -o -type l
    echo build/pat_arm_control/test_pat_arm_control
    echo build/pat_arm_control/pat_platform_planar.xml
} | grep -v '^data/thesis_eval/live/' | tar -czf "$DEST/jetson_bench_ws.tar.gz" -T -
ls -lh "$DEST"/jetson_bench_image.tar.gz "$DEST"/jetson_bench_ws.tar.gz
