cmake_minimum_required(VERSION 3.14)
project(pat_telemetry CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake   REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs      REQUIRED)

# Algorithm library — zero ROS dependency
add_library(pat_telemetry_lib STATIC
    src/telemetry_aggregator.cpp)
target_include_directories(pat_telemetry_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(pat_telemetry_lib PRIVATE -Wall -Wextra -Wpedantic)

# ROS 2 node wrapper
add_executable(telemetry_node src/telemetry_node.cpp)
target_link_libraries(telemetry_node pat_telemetry_lib)
ament_target_dependencies(telemetry_node rclcpp sensor_msgs geometry_msgs std_msgs)

# GTest — no ROS dependency
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pat_telemetry
        test/test_telemetry_aggregator.cpp)
    target_link_libraries(test_pat_telemetry pat_telemetry_lib)
endif()

install(TARGETS pat_telemetry_lib telemetry_node
    ARCHIVE DESTINATION lib  LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config/)
ament_package()
