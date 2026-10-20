cmake_minimum_required(VERSION 3.14)
project(pat_rt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Threads     REQUIRED)

# Header-only, and deliberately ROS-free: these are used from inside realtime
# threads and from zero-ROS algorithm libraries alike. Consumers link
# pat_rt::pat_rt after find_package(pat_rt).
add_library(pat_rt INTERFACE)
target_include_directories(pat_rt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(pat_rt INTERFACE Threads::Threads)

install(TARGETS pat_rt EXPORT pat_rtTargets)
install(DIRECTORY include/ DESTINATION include/)
ament_export_targets(pat_rtTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(Threads)

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    # alloc_counter.cpp is the one translation unit that defines the malloc
    # interposer (pat_rt/testing/alloc_counter_impl.h).
    ament_add_gtest(test_pat_rt
        test/test_rt_thread.cpp
        test/test_rt_mailbox.cpp
        test/alloc_counter.cpp)
    target_link_libraries(test_pat_rt pat_rt)
endif()

ament_package()
