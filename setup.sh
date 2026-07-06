#!/usr/bin/env bash
# Usage: source setup.sh
# Override MuJoCo path: MUJOCO_DIR=/your/path source setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco}"

if [ ! -d "$MUJOCO_DIR" ]; then
    echo "WARNING: MUJOCO_DIR=$MUJOCO_DIR does not exist."
    echo "  Install MuJoCo 3.x from https://github.com/google-deepmind/mujoco/releases"
else
    export MUJOCO_DIR
    export CMAKE_PREFIX_PATH="$MUJOCO_DIR:${CMAKE_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$MUJOCO_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "WARNING: ROS 2 Humble not found at /opt/ros/humble."
fi

if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/install/setup.bash"
fi

echo "MuJoCo-PAT environment active."
echo "  MUJOCO_DIR = $MUJOCO_DIR"
echo "  ROS_DISTRO = ${ROS_DISTRO:-not set}"
