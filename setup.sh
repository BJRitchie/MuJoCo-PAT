#!/usr/bin/env bash
# Usage: source setup.sh
# Override paths: MUJOCO_DIR=/your/path ACADOS_SOURCE_DIR=/your/acados source setup.sh
#
# Prefer the container (docker/Dockerfile + .devcontainer/) — it provides
# MuJoCo and acados at /opt and is the same recipe used for CI and the Jetson
# runtime image. This script is the bare-host fallback.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco}"

# The precompiled release extracts to a versioned subdir (mujoco-3.7.0/, ...).
# If MUJOCO_DIR points at the parent, descend into the newest one it contains.
if [ ! -e "$MUJOCO_DIR/include/mujoco/mujoco.h" ]; then
    _mj_cand="$(ls -d "$MUJOCO_DIR"/mujoco-* 2>/dev/null | sort -V | tail -1)"
    [ -n "$_mj_cand" ] && MUJOCO_DIR="$_mj_cand"
    unset _mj_cand
fi

if [ ! -d "$MUJOCO_DIR" ]; then
    echo "WARNING: MUJOCO_DIR=$MUJOCO_DIR does not exist."
    echo "  Install MuJoCo 3.x from https://github.com/google-deepmind/mujoco/releases"
else
    export MUJOCO_DIR
    export CMAKE_PREFIX_PATH="$MUJOCO_DIR:${CMAKE_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$MUJOCO_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

# acados (+ HPIPM + BLASFEO) — required by pat_arm_nmpc's constrained QP.
ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/acados}"
if [ -e "$ACADOS_SOURCE_DIR/include/acados_c/ocp_qp_interface.h" ]; then
    export ACADOS_SOURCE_DIR
    export CMAKE_PREFIX_PATH="$ACADOS_SOURCE_DIR:${CMAKE_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
else
    echo "WARNING: acados not found at ACADOS_SOURCE_DIR=$ACADOS_SOURCE_DIR"
    echo "  pat_arm_nmpc will fail to configure. Build acados from source, or"
    echo "  use the container (docker/Dockerfile / .devcontainer/)."
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
echo "  MUJOCO_DIR        = $MUJOCO_DIR"
echo "  ACADOS_SOURCE_DIR = ${ACADOS_SOURCE_DIR:-not set}"
echo "  ROS_DISTRO        = ${ROS_DISTRO:-not set}"
