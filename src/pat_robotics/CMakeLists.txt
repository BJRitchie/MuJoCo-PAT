cmake_minimum_required(VERSION 3.14)
project(pat_robotics CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3      REQUIRED NO_MODULE)

# Phase 3: add pat_robotics_lib and manipulator_node here.
# See include/pat_robotics/i_manipulator.hpp for the interface to implement.

install(DIRECTORY include/ DESTINATION include/)
ament_package()
