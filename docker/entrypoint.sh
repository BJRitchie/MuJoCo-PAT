#!/usr/bin/env bash
# Container entrypoint: source ROS + the built workspace overlay, then exec.
set -e
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [ -f /ws/install/setup.bash ]; then
    source /ws/install/setup.bash
fi
exec "$@"
