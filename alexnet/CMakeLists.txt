cmake_minimum_required(VERSION 2.6)

project(alexnet)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(COMMON_INCLUDE ../includes/common)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/workspace/TensorRT-8.2.4.2/include)
link_directories(/workspace/TensorRT-8.2.4.2/lib)

# common lib
include_directories(${COMMON_INCLUDE})

add_executable(alexnet ${PROJECT_SOURCE_DIR}/alex.cpp)
target_link_libraries(alexnet nvinfer)
target_link_libraries(alexnet cudart)
target_link_libraries(alexnet nvonnxparser)

add_definitions(-O2 -pthread)