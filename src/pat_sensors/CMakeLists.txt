cmake_minimum_required(VERSION 3.14)
project(pat_sensors CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)

# The sensor model: standard library only, deliberately free of ROS and MuJoCo.
# It started inside pat_simulation; it lives here so the offline benches in the
# algorithm packages can corrupt measurements with the same, seeded code the
# simulator uses, without depending on the simulator (and its viewer).
# Consumers link pat_sensors::pat_sensors after find_package(pat_sensors).
add_library(pat_sensors STATIC src/sensor_model.cpp)
target_include_directories(pat_sensors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(pat_sensors PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(pat_sensors PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS pat_sensors EXPORT pat_sensorsTargets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/)
ament_export_targets(pat_sensorsTargets HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pat_sensors test/test_sensor_model.cpp)
    target_link_libraries(test_pat_sensors pat_sensors)
endif()

ament_package()
