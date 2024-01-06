cmake_minimum_required(VERSION 3.26)
set(NAME cuda_particle_simulation)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(${NAME})

set(CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

add_subdirectory(thirdparty/raylib)
include_directories(thirdparty)
include_directories(thirdparty/src)

file (GLOB_RECURSE PROJECT_HEADERS
        src/*.hpp
        src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES
        src/*.cpp
        src/*.c)

add_executable(${NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(${NAME} raylib OpenMP::OpenMP_CXX)