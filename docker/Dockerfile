# syntax=docker/dockerfile:1.7
# ───────────────────────────────────────────────────────────────────────────────
# MuJoCo-PAT container build (multi-stage, multi-arch).
#
#   base    — ROS 2 Humble (ros-base) + toolchain + apt deps; common ancestor
#   acados  — acados + HPIPM + BLASFEO built from source  ->  /opt/acados
#   mujoco  — MuJoCo release tarball for $TARGETARCH        ->  /opt/mujoco
#   ws-base — base + /opt/acados + /opt/mujoco + the source tree
#   dev     — + RViz/GLFW; builds the FULL workspace (incl. pat_simulation)   [x86]
#   runtime — headless; builds only the algorithm packages (Jetson image)  [arm64+]
#
# The sim-to-real split is enforced at the image boundary: `runtime` builds
# with --packages-skip pat_simulation, so it cannot regress into depending on
# the simulator.
#
# Build:
#   docker build -f docker/Dockerfile --target dev -t mujoco-pat:dev .
#   docker buildx build -f docker/Dockerfile --target runtime \
#       --platform linux/arm64 --build-arg BLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76 \
#       -t mujoco-pat:runtime-arm64 .
# ───────────────────────────────────────────────────────────────────────────────

ARG ROS_DISTRO=humble
ARG MUJOCO_VERSION=3.7.0
ARG ACADOS_REF=v0.4.2

# ── base ─────────────────────────────────────────────────────────────────────
FROM ros:${ROS_DISTRO}-ros-base AS base
ARG ROS_DISTRO
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates jq \
        libeigen3-dev \
        python3-colcon-common-extensions python3-rosdep \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-ament-cmake-gtest \
        ros-${ROS_DISTRO}-tf2 ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-nav-msgs ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# ── acados (+ HPIPM + BLASFEO), built from source ────────────────────────────
FROM base AS acados
ARG ACADOS_REF
ARG TARGETARCH
# Override for max perf: X64_INTEL_HASWELL / ARMV8A_ARM_CORTEX_A76 (Orin) / ...
ARG BLASFEO_TARGET=""
RUN set -eux; \
    git clone --recursive --depth 1 --branch "${ACADOS_REF}" \
        https://github.com/acados/acados.git /tmp/acados; \
    t="${BLASFEO_TARGET}"; \
    if [ -z "$t" ]; then case "${TARGETARCH}" in \
        amd64) t=X64_AUTOMATIC ;; \
        arm64) t=ARMV8A_ARM_CORTEX_A57 ;; \
        *)     t=GENERIC ;; \
      esac; fi; \
    echo "BLASFEO_TARGET=$t"; \
    cmake -S /tmp/acados -B /tmp/acados/build \
        -DACADOS_INSTALL_DIR=/opt/acados \
        -DACADOS_WITH_QPOASES=OFF -DACADOS_WITH_OSQP=OFF -DACADOS_WITH_QPDUNES=OFF \
        -DACADOS_PYTHON=OFF -DACADOS_EXAMPLES=OFF -DACADOS_UNIT_TESTS=OFF \
        -DBLASFEO_TARGET="$t" -DHPIPM_TARGET=GENERIC -DCMAKE_BUILD_TYPE=Release; \
    cmake --build /tmp/acados/build --target install -j2; \
    rm -rf /tmp/acados

# ── mujoco (arch-specific release tarball) ──────────────────────────────────
FROM base AS mujoco
ARG MUJOCO_VERSION
ARG TARGETARCH
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) march=x86_64 ;; \
      arm64) march=aarch64 ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    wget -q "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${march}.tar.gz" \
        -O /tmp/mujoco.tar.gz; \
    mkdir -p /opt/mujoco; \
    tar -xzf /tmp/mujoco.tar.gz -C /opt/mujoco --strip-components=1; \
    rm /tmp/mujoco.tar.gz

# ── ws-base: deps in place + source tree ───────────────────────────────────
FROM base AS ws-base
ENV ACADOS_SOURCE_DIR=/opt/acados \
    MUJOCO_DIR=/opt/mujoco \
    CMAKE_PREFIX_PATH=/opt/mujoco:/opt/acados \
    LD_LIBRARY_PATH=/opt/acados/lib:/opt/mujoco/lib
COPY --from=acados /opt/acados /opt/acados
COPY --from=mujoco /opt/mujoco /opt/mujoco
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /ws
COPY . /ws/
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# colcon/make throttle — one package at a time, one compiler process, so
# heavy Eigen/MuJoCo/acados TUs don't OOM a memory-tight host. Bump on a
# machine with headroom: --build-arg BUILD_JOBS=4
ARG BUILD_JOBS=1

# ── dev: toolchain + deps; the workspace is built interactively against the
#    bind-mounted source (docker-compose.yml binds ./ -> /ws and colcon builds
#    into ./build + ./install on the host, VORTEX-style). A `dev` user at
#    UID/GID 1000 (override with --build-arg) gives compose's `user:` a real
#    name and $HOME.
FROM ws-base AS dev
ARG ROS_DISTRO
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglfw3-dev libglew-dev libgl1-mesa-dev \
        gdb clangd sudo \
    && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-desktop \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid ${USER_GID} dev \
 && useradd  --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash dev \
 && echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev && chmod 0440 /etc/sudoers.d/dev \
 && printf '%s\n' \
      "source /opt/ros/${ROS_DISTRO}/setup.bash" \
      "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" \
      | tee -a /home/dev/.bashrc /etc/bash.bashrc >/dev/null
USER dev

# ── runtime: headless, algorithm packages only (the Jetson image) ───────────
FROM ws-base AS runtime
ARG ROS_DISTRO
ARG BUILD_JOBS
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && MAKEFLAGS="-j${BUILD_JOBS}" colcon build --parallel-workers 1 \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-skip pat_simulation
