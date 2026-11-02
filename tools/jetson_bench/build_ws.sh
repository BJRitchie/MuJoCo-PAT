#!/usr/bin/env bash
# Build the workspace for the Orin, inside the image (run through enter.sh):
# Release, -mcpu=cortex-a78 (the Orin's cores; the DGX's X925 runs it too), and
# the tests. Only what the bench needs. About 10 minutes on the DGX.
#   tools/jetson_bench/enter.sh tools/jetson_bench/build_ws.sh
set -euo pipefail
# ROS's setup scripts read unset variables: not under set -u.
set +u
source /opt/ros/humble/setup.bash
set -u
MAKEFLAGS="-j${JOBS:-8}" colcon build --parallel-workers "${WORKERS:-4}" \
    --packages-up-to pat_arm_control pat_simulation \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DPAT_ARCH_FLAGS=-mcpu=cortex-a78 \
                 -DBUILD_TESTING=ON
echo "build_ws.sh: done"
