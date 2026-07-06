FROM osrf/ros:humble-desktop-full
ARG MUJOCO_VERSION=3.1.6
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential wget libeigen3-dev \
        python3-colcon-common-extensions python3-rosdep \
        ros-humble-tf2 ros-humble-tf2-ros ros-humble-nav-msgs \
        ros-humble-sensor-msgs ros-humble-geometry-msgs \
        ros-humble-ament-cmake-gtest \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz \
      -O /tmp/mujoco.tar.gz \
    && tar -xzf /tmp/mujoco.tar.gz -C /opt && rm /tmp/mujoco.tar.gz
ENV MUJOCO_DIR=/opt/mujoco-${MUJOCO_VERSION}
ENV CMAKE_PREFIX_PATH="${MUJOCO_DIR}:${CMAKE_PREFIX_PATH:-}"
ENV LD_LIBRARY_PATH="${MUJOCO_DIR}/lib:${LD_LIBRARY_PATH:-}"
WORKDIR /ws
COPY . /ws/
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -y && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"
ENTRYPOINT ["/bin/bash","-c","source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && exec \"$@\"","--"]
CMD ["/bin/bash"]
