#!/usr/bin/env bash
# Build the Jetson timing image, mujoco-pat:jetson (Dockerfile target `bench`),
# with BLASFEO's kernels for the Orin's Cortex-A78AE cores (A76 is BLASFEO's
# nearest target). Run on an arm64 host -- the DGX -- from anywhere in the repo.
# About 30-45 minutes; CPU-heavy but its result does not depend on load, so it
# can share the machine.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "$(uname -m)" != aarch64 ]; then
    echo "build_image.sh: build on an arm64 host (the DGX); this is $(uname -m)" >&2
    exit 1
fi
docker build -f docker/Dockerfile --target bench \
    --build-arg BLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76 \
    --build-arg ACADOS_JOBS="${ACADOS_JOBS:-8}" \
    -t mujoco-pat:jetson .
echo "built mujoco-pat:jetson $(docker image inspect -f '{{.Id}}' mujoco-pat:jetson)"
