#!/usr/bin/env bash
# Runs once, in-container, after the dev container is created.
set -euo pipefail
cd /ws

# A build/ or install/ left over from a host-native `colcon build` has host
# absolute paths baked into its CMake caches — unusable in the container.
if [ -e build ] || [ -e install ]; then
    echo "post-create: removing pre-existing build/ install/ log/ (host-built, wrong paths)"
    rm -rf build install log
fi

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

MAKEFLAGS="-j${BUILD_JOBS:-2}" colcon build --symlink-install --parallel-workers 1 \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON \
                 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

jq -s 'map(.[])' build/*/compile_commands.json > compile_commands.json 2>/dev/null || true
rosdep update 2>/dev/null || true

echo
echo "dev container ready. Rebuild one package: colcon build --packages-select pat_arm_nmpc"
echo "(MAKEFLAGS=-j${BUILD_JOBS:-2} is set). Launch files live in /ws/launch."
