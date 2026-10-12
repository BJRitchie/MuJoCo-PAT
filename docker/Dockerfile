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
# Defaults to 2 because this is the longest stage and the reference dev box is
# RAM-tight (see Common failures in the root README). It is the dominant cost of a
# cold build, so raise it on a machine with headroom -- a DGX Spark or any
# many-core arm64 host doing the step-0 validation build wants --build-arg
# ACADOS_JOBS=8 as much as it wants BUILD_JOBS, and only this one shortens the
# wait.
ARG ACADOS_JOBS=2
# Override for max perf: X64_INTEL_HASWELL / ARMV8A_ARM_CORTEX_A76 (Orin) / ...
ARG BLASFEO_TARGET=""
# HPIPM_TARGET stays GENERIC by default, deliberately. Measured on an i7-13700H
# against an otherwise identical X64_INTEL_HASWELL build (same binary, swapped
# only by LD_LIBRARY_PATH): ~3% of the QP solve, ~2% of the control tick -- at the
# edge of this box's noise. BLASFEO already carries the AVX2/FMA kernels that do
# the actual linear algebra; HPIPM's target only tunes its own glue.
#
# It must be set TOGETHER WITH an explicit BLASFEO_TARGET, never alone: a
# concrete HPIPM target emits instructions for that CPU, so pairing it with
# X64_AUTOMATIC (which resolves per build host) risks SIGILL wherever the image
# later runs.
ARG HPIPM_TARGET=GENERIC
RUN set -eux; \
    git clone --recursive --depth 1 --branch "${ACADOS_REF}" \
        https://github.com/acados/acados.git /tmp/acados; \
    t="${BLASFEO_TARGET}"; \
    if [ -z "$t" ]; then case "${TARGETARCH}" in \
        amd64) t=X64_AUTOMATIC ;; \
        arm64) t=ARMV8A_ARM_CORTEX_A57 ;; \
        *)     t=GENERIC ;; \
      esac; fi; \
    echo "BLASFEO_TARGET=$t HPIPM_TARGET=${HPIPM_TARGET}"; \
    cmake -S /tmp/acados -B /tmp/acados/build \
        -DACADOS_INSTALL_DIR=/opt/acados \
        -DACADOS_WITH_QPOASES=OFF -DACADOS_WITH_OSQP=OFF -DACADOS_WITH_QPDUNES=OFF \
        -DACADOS_PYTHON=OFF -DACADOS_EXAMPLES=OFF -DACADOS_UNIT_TESTS=OFF \
        -DBLASFEO_TARGET="$t" -DHPIPM_TARGET="${HPIPM_TARGET}" \
        -DCMAKE_BUILD_TYPE=Release; \
    cmake --build /tmp/acados/build --target install -j"${ACADOS_JOBS}"; \
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
ARG MUJOCO_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000
# plotjuggler-ros: live telemetry plotting GUI for pat_telemetry's topics.
# dev-only — a GUI tool has no place on the headless Jetson `runtime` image;
# view real-hardware telemetry from a laptop's own PlotJuggler pointed at
# the Jetson's ROS 2 domain over the network instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglfw3-dev libglew-dev libgl1-mesa-dev \
        gdb clangd sudo \
    && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-desktop \
    && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-plotjuggler-ros \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
# tools/pinn_datagen/ (offline PINN training-data generator) needs the raw
# Python mujoco bindings + numpy/pandas/h5py/pyyaml — dev-only: `runtime`
# explicitly skips pat_simulation and has no data-generation use case, so
# there's no reason to carry this into the Jetson deploy image. mujoco is
# pinned to the same release as the C++ library already baked into
# /opt/mujoco (a separate install — pip's wheel bundles its own libmujoco.so)
# so MJCF-parsing behavior can't silently drift between the two.
RUN pip install --no-cache-dir "mujoco==${MUJOCO_VERSION}" numpy pandas h5py pyyaml
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
ARG TARGETARCH
# Instruction set for THIS package's own compiled code (the rollout's small dense
# Eigen algebra and nmpc_bench); MuJoCo and acados/BLASFEO are precompiled and
# carry their own targets, set in their stages above.
#
# Never "native": this image is cross-built under QEMU, where native means the
# builder's CPU and produces binaries that SIGILL on the target. Orin's cores are
# Cortex-A78AE, so -mcpu=cortex-a78 is the closest GCC knows; it is a correctness
# choice as much as a performance one, since without it the arm64 build targets
# baseline ARMv8-A. Measured on x86 the equivalent flags moved the tick by nothing
# (README's "Settled by measurement"), so do not expect much -- the point is not
# leaving the target unstated.
ARG PAT_ARCH_FLAGS=""
# OFF for a deployment image: a gtest binary has no business in one, and before
# this was set the runtime image was carrying it. ON is for validating a NEW
# ARCHITECTURE -- build this target on an arm64 host, flip it, and run the suite
# natively to check the numerics rather than trusting QEMU:
#   docker build -f docker/Dockerfile --target runtime \
#       --build-arg BUILD_TESTING=ON --build-arg BUILD_JOBS=8 -t pat:arm64-check .
ARG BUILD_TESTING=OFF
# set -e only, never -u: ROS's setup.sh reads unset variables, so -u aborts the
# build the moment it is sourced.
RUN set -e; \
    f="${PAT_ARCH_FLAGS}"; \
    if [ -z "$f" ]; then case "${TARGETARCH}" in \
        arm64) f="-mcpu=cortex-a78" ;; \
        amd64) f="" ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; fi; \
    echo "PAT_ARCH_FLAGS=$f"; \
    . /opt/ros/${ROS_DISTRO}/setup.sh; \
    MAKEFLAGS="-j${BUILD_JOBS}" colcon build --parallel-workers 1 \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
                   -DPAT_ARCH_FLAGS="$f" \
                   -DBUILD_TESTING="${BUILD_TESTING}" \
      --packages-skip pat_simulation
