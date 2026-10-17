cmake_minimum_required(VERSION 3.14)
project(pat_simulation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake   REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2           REQUIRED)
find_package(tf2_ros       REQUIRED)
find_package(pat_msgs      REQUIRED)
find_package(glfw3         REQUIRED)
find_package(pat_platform_description REQUIRED)   # shared piper_planar_arm.xacro

# The precompiled MuJoCo release tarball does not ship a mujocoConfig.cmake,
# so provide our own Find module (searches for the library/headers directly).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(mujoco        REQUIRED)

add_library(mujoco_sim STATIC
    src/mujoco_sim.cpp
    src/mujoco_vis.cpp
    src/sensor_model.cpp
    src/disturbance_model.cpp)
target_include_directories(mujoco_sim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(mujoco_sim PUBLIC mujoco::mujoco glfw)

add_executable(simulation_node src/simulation_node.cpp)
target_include_directories(simulation_node PRIVATE include)
target_link_libraries(simulation_node mujoco_sim)
ament_target_dependencies(simulation_node
    rclcpp rosgraph_msgs nav_msgs sensor_msgs geometry_msgs tf2 tf2_ros pat_msgs)

# Expand the scene xacro to a self-contained MJCF at build time. mj_loadXML
# (via the `model_path` param) loads the generated .xml; the .xacro sources the
# shared Piper arm from pat_platform_description.
find_program(XACRO_EXECUTABLE xacro REQUIRED)
set(_desc_xacro "${pat_platform_description_DIR}/../xacro")
set(_abt_xml "${CMAKE_CURRENT_BINARY_DIR}/air_bearing_table.xml")
add_custom_command(
    OUTPUT  "${_abt_xml}"
    COMMAND "${XACRO_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/models/environment/air_bearing_table.xacro"
            -o "${_abt_xml}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/models/environment/air_bearing_table.xacro"
            "${_desc_xacro}/pat_platform_bus.xacro"
            "${_desc_xacro}/piper_planar_arm.xacro"
    COMMENT "xacro -> air_bearing_table.xml"
    VERBATIM)
add_custom_target(pat_simulation_models ALL DEPENDS "${_abt_xml}")

# ─── nav_bench: the base state estimator against MuJoCo truth, offline ─────
# Drives pat_gnc's ImuEkfNavigator with this package's plant and SensorModel,
# deterministically. Depends on pat_gnc's library, never the other way round:
# the runtime image skips this package entirely.
find_package(pat_gnc  REQUIRED)
find_package(yaml-cpp REQUIRED)
add_executable(nav_bench src/nav_bench.cpp)
add_dependencies(nav_bench pat_simulation_models)
target_link_libraries(nav_bench mujoco_sim pat_gnc::pat_gnc_lib yaml-cpp)
install(TARGETS nav_bench RUNTIME DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pat_simulation
        test/test_sensor_model.cpp
        test/test_scene_sensors.cpp
        test/test_disturbance_model.cpp)
    # test_scene_sensors loads the real generated scene, so it has to wait for
    # xacro. It loads the INSTALLED copy, not the one in the build tree: the
    # MJCF's mesh paths are relative to its own directory ("../manipulators/"),
    # which only resolves under share/. That also makes the test load the exact
    # file the node loads via FindPackageShare.
    add_dependencies(test_pat_simulation pat_simulation_models)
    target_link_libraries(test_pat_simulation mujoco_sim)
    target_compile_definitions(test_pat_simulation PRIVATE
        PAT_SIM_TEST_MJCF="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/models/environment/air_bearing_table.xml")
endif()

install(TARGETS mujoco_sim simulation_node
    ARCHIVE DESTINATION lib  LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY models/  DESTINATION share/${PROJECT_NAME}/models/)
install(FILES "${_abt_xml}" DESTINATION share/${PROJECT_NAME}/models/environment/)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config/)
ament_package()
