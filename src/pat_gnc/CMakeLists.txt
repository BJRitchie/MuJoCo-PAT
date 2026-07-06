cmake_minimum_required(VERSION 3.14)
project(pat_gnc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake   REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pat_msgs      REQUIRED)
find_package(Eigen3        REQUIRED NO_MODULE)

# Algorithm library — zero ROS dependency
# Add cw_guidance.cpp here in Phase 2
add_library(pat_gnc_lib STATIC
    src/planar_dynamics.cpp
    src/pid_controller.cpp
    src/planar_ekf.cpp)
target_include_directories(pat_gnc_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(pat_gnc_lib PUBLIC Eigen3::Eigen)
target_compile_options(pat_gnc_lib PRIVATE -Wall -Wextra -Wpedantic)

# ROS 2 node wrapper
add_executable(gnc_node src/gnc_node.cpp)
target_link_libraries(gnc_node pat_gnc_lib)
ament_target_dependencies(gnc_node rclcpp nav_msgs sensor_msgs geometry_msgs pat_msgs)

# GTest — no ROS dependency
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pat_gnc
        test/test_dynamics.cpp
        test/test_gnc.cpp
        test/test_navigators.cpp)
    target_link_libraries(test_pat_gnc pat_gnc_lib)
endif()

install(TARGETS pat_gnc_lib gnc_node
    ARCHIVE DESTINATION lib  LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config/)
ament_package()
