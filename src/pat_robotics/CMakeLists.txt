cmake_minimum_required(VERSION 3.14)
project(pat_robotics CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp      REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3      REQUIRED NO_MODULE)

# Algorithm library — zero ROS dependency.
# Phase 3: add pat_robotics_lib manipulator kinematics here (see
# include/pat_robotics/i_manipulator.hpp for the interface to implement).
add_library(pat_robotics_lib STATIC
    src/joint_pid.cpp
    src/arm_controller.cpp)
target_include_directories(pat_robotics_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(pat_robotics_lib PUBLIC Eigen3::Eigen)
target_compile_options(pat_robotics_lib PRIVATE -Wall -Wextra -Wpedantic)

# ROS 2 node wrapper
add_executable(arm_control_node src/arm_control_node.cpp)
target_link_libraries(arm_control_node pat_robotics_lib)
ament_target_dependencies(arm_control_node rclcpp sensor_msgs)

# GTest — no ROS dependency
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pat_robotics test/test_pat_robotics.cpp)
    target_link_libraries(test_pat_robotics pat_robotics_lib)
endif()

install(TARGETS pat_robotics_lib arm_control_node
    ARCHIVE DESTINATION lib  LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config/)
ament_package()
