cmake_minimum_required(VERSION 3.14)
project(pat_arm_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake              REQUIRED)
find_package(Eigen3                   REQUIRED NO_MODULE)
find_package(pat_platform_description REQUIRED)   # shared piper_planar_arm.xacro

# ─── Hot-path instruction set ──────────────────────────────────────────────
# Empty by default and deliberately not "native": the runtime image is arm64, so
# a baked-in host-specific default would either fail to build or produce
# binaries that fault on the Jetson. Set it per build:
#   dev box   -DPAT_ARCH_FLAGS="-march=native -mtune=native"
#   Jetson    -DPAT_ARCH_FLAGS="-mcpu=cortex-a78"
# Without it the workspace compiles for baseline x86-64 -- SSE2 only, no AVX or
# FMA -- which is what Eigen then uses for the rollout's small dense algebra.
# acados/BLASFEO are precompiled and unaffected; they carry their own target.
set(PAT_ARCH_FLAGS "" CACHE STRING
    "Extra -march/-mtune/-mcpu flags for this package's compiled code")

# Applied to EVERY target here, never just the library. Eigen's object alignment
# (EIGEN_MAX_ALIGN_BYTES) follows the enabled instruction set, so a library built
# with AVX and an executable built without disagree about the layout of the Eigen
# types they hand each other -- an ODR violation that corrupts data rather than
# failing to link.
function(pat_arm_control_arch_flags target)
    if(PAT_ARCH_FLAGS)
        separate_arguments(_flags UNIX_COMMAND "${PAT_ARCH_FLAGS}")
        target_compile_options(${target} PRIVATE ${_flags})
    endif()
endfunction()

# The precompiled MuJoCo release ships no mujocoConfig.cmake — use our own
# Find module (copied from pat_simulation). MuJoCo is used here only as a
# rigid-body dynamics library (mass matrix, Jacobians, bias forces), never
# as the simulator.
# Neither the MuJoCo release nor the from-source acados install ships a
# usable *Config.cmake, so use our own Find modules (see cmake/). Both are
# numerical libraries here — MuJoCo for rigid-body dynamics, acados/HPIPM for
# the box/torque-constrained QP. Still zero ROS in this lib.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(mujoco REQUIRED)
find_package(acados REQUIRED)

# Layout, by role (include/pat_arm_control/<role>/ mirrors src/<role>/):
#   common/     profiling instruments every law and node shares
#   dynamics/   FloatingBaseDynamics, the plant model the control laws use
#   control/    the control laws; control/nmpc/ is the NMPC and its QP solver
#   observers/  disturbance observers
#   estimation/ per-joint Kalman filter for encoder readings
#   reference/  reference windows, trajectory shapes, teleop (no dependencies)
#   src/nodes/  the ROS nodes -- their headers are private, never installed
#   src/bench/  ROS-free benches and the support library they share with tests

# ─── Reference trajectories — no dependencies at all ──────────────────────
# Its own library so the planner (ee_target_publisher) can link the shape
# generators without pulling in MuJoCo and acados; the controller library
# links it too, so both build the same reference from the same code.
add_library(pat_arm_control_reference STATIC
    src/reference/reference_trajectory.cpp
    src/reference/teleop_reference.cpp)
target_include_directories(pat_arm_control_reference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(pat_arm_control_reference PRIVATE -Wall -Wextra -Wpedantic)
pat_arm_control_arch_flags(pat_arm_control_reference)

# ─── Algorithm library — Eigen + MuJoCo + acados, zero ROS ─────────────────
add_library(pat_arm_control_lib STATIC
    src/dynamics/floating_base_dynamics.cpp
    src/control/controller_config.cpp
    src/control/task_space_controller.cpp
    src/control/nmpc/quad_prob_solver.cpp
    src/control/nmpc/arm_nmpc.cpp
    src/control/nmpc/nmpc_config.cpp
    src/control/one_step_controller.cpp
    src/control/ompc/arm_ompc.cpp
    src/control/ompc/ompc_config.cpp
    src/control/smc/arm_smc.cpp
    src/control/smc/smc_config.cpp
    src/observers/momentum_observer.cpp
    src/estimation/joint_kalman_filter.cpp)
target_include_directories(pat_arm_control_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(pat_arm_control_lib PUBLIC
    Eigen3::Eigen mujoco::mujoco acados::acados pat_arm_control_reference)
target_compile_options(pat_arm_control_lib PRIVATE -Wall -Wextra -Wpedantic)
pat_arm_control_arch_flags(pat_arm_control_lib)

# ─── ROS 2 nodes ──────────────────────────────────────────────────────────
find_package(rclcpp        REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(trajectory_msgs REQUIRED)   # ee_trajectory
find_package(std_msgs      REQUIRED)   # ee_teleop_reset
find_package(pat_rt        REQUIRED)   # mailbox, ring, RT thread helpers
# The arm controller node, law-agnostic: subscriptions, the realtime control
# and observer threads, the report. One static library, linked into one
# executable per control law (nmpc_node, later smc_node, ompc_node), each of
# which adds only its law's factory.
add_library(pat_arm_control_node STATIC
    src/nodes/arm_control_node.cpp
    src/nodes/arm_control_loop.cpp
    src/nodes/arm_control_observer.cpp
    src/nodes/arm_control_report.cpp)
pat_arm_control_arch_flags(pat_arm_control_node)
target_link_libraries(pat_arm_control_node PUBLIC pat_arm_control_lib pat_rt::pat_rt)
ament_target_dependencies(pat_arm_control_node PUBLIC rclcpp sensor_msgs geometry_msgs
                          nav_msgs trajectory_msgs)

add_executable(nmpc_node src/nodes/nmpc_node.cpp)
pat_arm_control_arch_flags(nmpc_node)
target_link_libraries(nmpc_node pat_arm_control_node)
install(TARGETS nmpc_node RUNTIME DESTINATION lib/${PROJECT_NAME})

add_executable(ompc_node src/nodes/ompc_node.cpp)
pat_arm_control_arch_flags(ompc_node)
target_link_libraries(ompc_node pat_arm_control_node)
install(TARGETS ompc_node RUNTIME DESTINATION lib/${PROJECT_NAME})

add_executable(smc_node src/nodes/smc_node.cpp)
pat_arm_control_arch_flags(smc_node)
target_link_libraries(smc_node pat_arm_control_node)
install(TARGETS smc_node RUNTIME DESTINATION lib/${PROJECT_NAME})

# Stand-in mission node: cycles PoseStamped waypoints onto ee_setpoint, or
# follows a circle / rectangle / teleop command on ee_trajectory. Links only
# the reference library -- no MuJoCo, no acados.
add_executable(ee_target_publisher_node src/nodes/ee_target_publisher_node.cpp)
target_link_libraries(ee_target_publisher_node pat_arm_control_reference)
pat_arm_control_arch_flags(ee_target_publisher_node)
ament_target_dependencies(ee_target_publisher_node rclcpp geometry_msgs trajectory_msgs
                          std_msgs)
install(TARGETS ee_target_publisher_node RUNTIME DESTINATION lib/${PROJECT_NAME})

# Keyboard teleop: reads held keys from the X server, publishes ee_teleop for
# ee_target_publisher (pattern: teleop). Built only where X11 exists -- the
# headless runtime image has none, and nothing else depends on this node.
find_package(X11 QUIET)
if(X11_FOUND)
    add_executable(ee_teleop_node src/nodes/ee_teleop_node.cpp)
    target_include_directories(ee_teleop_node PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${X11_INCLUDE_DIR})
    target_link_libraries(ee_teleop_node ${X11_LIBRARIES})
    target_compile_options(ee_teleop_node PRIVATE -Wall -Wextra -Wpedantic)
    pat_arm_control_arch_flags(ee_teleop_node)
    ament_target_dependencies(ee_teleop_node rclcpp geometry_msgs std_msgs)
    install(TARGETS ee_teleop_node RUNTIME DESTINATION lib/${PROJECT_NAME})
else()
    message(STATUS "X11 not found: ee_teleop_node (keyboard teleop) is not built")
endif()

# ─── MuJoCo model: expand the xacro to a self-contained MJCF at build time ──
# FloatingBaseDynamics mj_loadXML's the installed .xml; the .xacro sources
# the shared PAT bus + Piper arms from pat_platform_description.
find_program(XACRO_EXECUTABLE xacro REQUIRED)
set(_desc_xacro "${pat_platform_description_DIR}/../xacro")
set(_model_xml "${CMAKE_CURRENT_BINARY_DIR}/pat_platform_planar.xml")
add_custom_command(
    OUTPUT  "${_model_xml}"
    COMMAND "${XACRO_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/pat_platform_planar.xacro"
            -o "${_model_xml}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/models/pat_platform_planar.xacro"
            "${_desc_xacro}/pat_platform_bus.xacro"
            "${_desc_xacro}/piper_planar_arm.xacro"
    COMMENT "xacro -> pat_platform_planar.xml"
    VERBATIM)
add_custom_target(pat_arm_control_models ALL DEPENDS "${_model_xml}")
install(FILES "${_model_xml}" "${CMAKE_CURRENT_SOURCE_DIR}/models/platform.xml"
        DESTINATION share/${PROJECT_NAME}/models/)

# ─── Bench support: ROS-free config loading and the contact scenario ──────
# Shared by both benches and by the tests, which score the bench outputs: one
# compiled copy, so a test checks the code a bench actually runs.
find_package(yaml-cpp REQUIRED)
# The estimator and the sensor model it is fed with, so a study can hand the arms
# a noisy measurement or the base state estimator's output. Both are ROS-free
# libraries, which keeps the bench inside the layering rules.
find_package(pat_gnc     REQUIRED)
find_package(pat_sensors REQUIRED)
add_library(pat_arm_control_bench STATIC
    src/bench/bench_yaml.cpp
    src/bench/contact_scenario.cpp
    src/bench/metrics.cpp
    src/bench/run_summary.cpp
    src/bench/state_source.cpp)
target_include_directories(pat_arm_control_bench PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bench)
target_link_libraries(pat_arm_control_bench PUBLIC pat_arm_control_lib yaml-cpp
                      pat_gnc::pat_gnc_lib pat_sensors::pat_sensors)
target_compile_options(pat_arm_control_bench PRIVATE -Wall -Wextra -Wpedantic)
pat_arm_control_arch_flags(pat_arm_control_bench)

# ─── nmpc_bench: ROS-free closed-loop timing ──────────────────────────────
# Built into the runtime image so the Orin can be characterised with one
# command: no ROS graph, no DDS, no simulator, no X11. Links the algorithm lib
# and yaml-cpp only -- yaml-cpp so it reads the REAL nmpc.yaml rather than
# carrying a second copy of the configuration.
#
# The default params/model paths are compiled in because this deliberately has
# no ROS: there is no FindPackageShare to resolve them at runtime. They point at
# the INSTALLED copies, so the binary works from anywhere once installed, and
# both are overridable with --params / --model.
add_executable(nmpc_bench src/bench/nmpc_bench.cpp)
pat_arm_control_arch_flags(nmpc_bench)
add_dependencies(nmpc_bench pat_arm_control_models)
target_link_libraries(nmpc_bench pat_arm_control_bench)
# The model too: nmpc.yaml ships mjcf_path as "" because the launch file fills it
# from FindPackageShare, which a ROS-free binary cannot call.
target_compile_definitions(nmpc_bench PRIVATE
    PAT_NMPC_BENCH_COMMON="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/arm_common.yaml"
    PAT_NMPC_BENCH_PARAMS="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/nmpc.yaml"
    PAT_NMPC_BENCH_MODEL="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/models/pat_platform_planar.xml")
install(TARGETS nmpc_bench RUNTIME DESTINATION lib/${PROJECT_NAME})

# ─── contact_bench: deterministic observer scoring against contact truth ───
# Same ROS-free pattern as nmpc_bench, and offline for the same reason the
# constraint ladder is: the live stack cannot reproduce a contact event closely
# enough to compare two configurations. Measured — two runs of one config gave
# peak-force ratios of 0.37/0.41/0.42 and 0.52/0.24/0.29.
#
# It additionally loads pat_simulation's SCENE model, because the contact being
# scored only exists there (pat_arm_control's own MJCF is built arm_contype=0).
# That is a compiled-in PATH, not a package dependency: nothing here links
# against pat_simulation, and this is a benchmark, not the library or the node,
# so the rule that algorithm packages must not depend on the simulator is not
# in play. Overridable with --scene, which is how you would point it at a
# different scene without rebuilding.
add_executable(contact_bench src/bench/contact_bench.cpp)
pat_arm_control_arch_flags(contact_bench)
add_dependencies(contact_bench pat_arm_control_models)
target_link_libraries(contact_bench pat_arm_control_bench)
target_compile_definitions(contact_bench PRIVATE
    PAT_OMPC_BENCH_PARAMS="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/ompc.yaml"
    PAT_SMC_BENCH_PARAMS="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/smc.yaml"
    PAT_NMPC_BENCH_COMMON="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/arm_common.yaml"
    PAT_NMPC_BENCH_PARAMS="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/nmpc.yaml"
    PAT_NMPC_BENCH_MODEL="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/models/pat_platform_planar.xml"
    PAT_CONTACT_BENCH_SCENE="${CMAKE_INSTALL_PREFIX}/share/pat_simulation/models/environment/air_bearing_table.xml")
install(TARGETS contact_bench RUNTIME DESTINATION lib/${PROJECT_NAME})

# ─── Tests ───────────────────────────────────────────────────────────────
# The model-fidelity tests load the generated MJCF straight out of the build
# tree (they run before install), so the path is compiled in rather than
# resolved via FindPackageShare the way the node does it.
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    # One executable, one file per topic. test/support/ holds what they share:
    # the NMPC fixtures, allocation attribution, and the one translation unit
    # that defines pat_rt's malloc interposer (alloc_counter.cpp). The bench
    # support library is linked for the bench scorers: they turn a recorded run
    # into the figures the README quotes, so an error in them produces a
    # confident wrong number rather than a failure.
    ament_add_gtest(test_pat_arm_control
        test/support/alloc_counter.cpp
        test/support/nmpc_fixtures.cpp
        test/test_nmpc_dynamics.cpp
        test/test_nmpc_model.cpp
        test/test_nmpc_constraints.cpp
        test/test_safe_mode.cpp
        test/test_feedforward.cpp
        test/test_reference_tracking.cpp
        test/test_realtime_profile.cpp
        test/test_studies.cpp
        test/test_param_loading.cpp
        test/test_ompc.cpp
        test/test_smc.cpp
        test/test_bench_metrics.cpp
        test/test_run_summary.cpp
        test/test_state_source.cpp
        test/test_momentum_observer.cpp
        test/test_joint_kalman_filter.cpp
        test/test_bench_scoring.cpp
        test/test_reference.cpp
        test/test_teleop_reference.cpp)
    target_include_directories(test_pat_arm_control PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test)
    pat_arm_control_arch_flags(test_pat_arm_control)
    add_dependencies(test_pat_arm_control pat_arm_control_models)
    # ${CMAKE_DL_LIBS}: DISABLED_AllocStudy.Attribution resolves allocation call
    # sites to their shared object with dladdr.
    target_link_libraries(test_pat_arm_control pat_arm_control_lib pat_arm_control_bench
                          pat_rt::pat_rt ${CMAKE_DL_LIBS})
    target_compile_definitions(test_pat_arm_control PRIVATE
        PAT_NMPC_TEST_MJCF="${_model_xml}"
        PAT_NMPC_TEST_COMMON="${CMAKE_CURRENT_SOURCE_DIR}/config/arm_common.yaml"
        PAT_NMPC_TEST_PARAMS="${CMAKE_CURRENT_SOURCE_DIR}/config/nmpc.yaml"
        PAT_OMPC_TEST_PARAMS="${CMAKE_CURRENT_SOURCE_DIR}/config/ompc.yaml"
        PAT_SMC_TEST_PARAMS="${CMAKE_CURRENT_SOURCE_DIR}/config/smc.yaml")
endif()

install(TARGETS pat_arm_control_lib
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config/)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)

ament_package()
