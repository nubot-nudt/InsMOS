cmake_minimum_required(VERSION 3.10)
project(Array_Index)

add_subdirectory(third_party/pybind11)
pybind11_add_module(Array_Index Array_Index.cpp)

# openMP settings
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

