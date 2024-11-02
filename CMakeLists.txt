cmake_minimum_required(VERSION 3.18)
project(MATRIX LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(MATRIX matrix_test.cu
        CUTRIX.cuh
        MATRIX.cuh)

set_target_properties(MATRIX PROPERTIES
        CUDA_ARCHITECTURES 86
        CUDA_SEPARABLE_COMPILATION ON)
