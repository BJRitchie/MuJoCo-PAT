cmake_minimum_required(VERSION 3.14)
project(pat_arm_nmpc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake              REQUIRED)
find_package(Eigen3                   REQUIRED NO_MODULE)
find_package(pat_platform_description REQUIRED)   # shared piper_planar_arm.xacro

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

# ─── Algorithm library — Eigen + MuJoCo + acados, zero ROS ─────────────────
add_library(pat_arm_nmpc_lib STATIC
    src/arm_task_space_controller.cpp
    src/quad_prob_solver.cpp
    src/arm_nmpc.cpp)
target_include_directories(pat_arm_nmpc_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(pat_arm_nmpc_lib PUBLIC
    Eigen3::Eigen mujoco::mujoco acados::acados)
target_compile_options(pat_arm_nmpc_lib PRIVATE -Wall -Wextra -Wpedantic)

# ─── ROS 2 node wrapper ───────────────────────────────────────────────────
find_package(rclcpp        REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs      REQUIRED)
add_executable(arm_nmpc_node src/arm_nmpc_node.cpp)
target_link_libraries(arm_nmpc_node pat_arm_nmpc_lib)
ament_target_dependencies(arm_nmpc_node rclcpp sensor_msgs geometry_msgs nav_msgs)
install(TARGETS arm_nmpc_node RUNTIME DESTINATION lib/${PROJECT_NAME})

# Fixed-waypoint task-space setpoint source (stand-in mission node). ROS-only,
# no algorithm lib — just cycles PoseStamped targets onto ee_setpoint.
add_executable(ee_target_publisher_node src/ee_target_publisher_node.cpp)
target_include_directories(ee_target_publisher_node PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(ee_target_publisher_node rclcpp geometry_msgs)
install(TARGETS ee_target_publisher_node RUNTIME DESTINATION lib/${PROJECT_NAME})

# ─── MuJoCo model: expand the xacro to a self-contained MJCF at build time ──
# ArmTaskSpaceController mj_loadXML's the installed .xml; the .xacro sources
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
add_custom_target(pat_arm_nmpc_models ALL DEPENDS "${_model_xml}")
install(FILES "${_model_xml}" "${CMAKE_CURRENT_SOURCE_DIR}/models/platform.xml"
        DESTINATION share/${PROJECT_NAME}/models/)

# ─── Tests ───────────────────────────────────────────────────────────────
# if(BUILD_TESTING)
#     find_package(ament_cmake_gtest REQUIRED)
#     ament_add_gtest(test_pat_arm_nmpc test/test_pat_arm_nmpc.cpp)
#     target_link_libraries(test_pat_arm_nmpc pat_arm_nmpc_lib)
# endif()

install(TARGETS pat_arm_nmpc_lib
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config/)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)

ament_package()
