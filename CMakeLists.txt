cmake_minimum_required(VERSION 3.26)
set(NAME cuda_particle_simulation)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(${NAME} CXX CUDA)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(thirdparty/raylib)
include_directories(thirdparty)
include_directories(thirdparty/src)

file (GLOB_RECURSE PROJECT_HEADERS
        src/*.hpp
        src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES
        src/*.cpp
        src/*.cxx
        src/*.cc
        src/*.c
        src/*.cu)


add_executable(${NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(${NAME} CUDA::cudart raylib)



