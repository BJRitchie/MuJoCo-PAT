cmake_minimum_required(VERSION 3.14)
project(pat_simulation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake   REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2           REQUIRED)
find_package(tf2_ros       REQUIRED)
find_package(pat_msgs      REQUIRED)
find_package(glfw3         REQUIRED)

# The precompiled MuJoCo release tarball does not ship a mujocoConfig.cmake,
# so provide our own Find module (searches for the library/headers directly).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(mujoco        REQUIRED)

add_library(mujoco_sim STATIC src/mujoco_sim.cpp src/mujoco_vis.cpp)
target_include_directories(mujoco_sim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(mujoco_sim PUBLIC mujoco::mujoco glfw)

add_executable(simulation_node src/simulation_node.cpp)
target_include_directories(simulation_node PRIVATE include)
target_link_libraries(simulation_node mujoco_sim)
ament_target_dependencies(simulation_node
    rclcpp nav_msgs sensor_msgs geometry_msgs tf2 tf2_ros pat_msgs)

install(TARGETS mujoco_sim simulation_node
    ARCHIVE DESTINATION lib  LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY models/  DESTINATION share/${PROJECT_NAME}/models/)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config/)
ament_package()
